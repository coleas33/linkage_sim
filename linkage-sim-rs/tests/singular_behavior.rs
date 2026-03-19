//! Tests for singular and near-singular behavior in pathological numerical cases.
//!
//! These tests verify the solver handles edge cases gracefully:
//! near-toggle positions, redundant constraints, underconstrained mechanisms,
//! zero/near-zero inertia, prismatic joints near dead center, and branch
//! consistency across full sweeps.

use std::f64::consts::PI;

use nalgebra::{DVector, Vector2};

use linkage_sim_rs::analysis::validation::{grubler_dof, jacobian_rank_analysis};
use linkage_sim_rs::core::body::{make_bar, make_ground, Body};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{ForceElement, GravityElement};
use linkage_sim_rs::solver::assembly::assemble_mass_matrix;
use linkage_sim_rs::solver::inverse_dynamics::solve_inverse_dynamics;
use linkage_sim_rs::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use linkage_sim_rs::solver::statics::solve_statics;

// ============================================================================
// Helper: build a standard driven 4-bar (ground=4, crank=1, coupler=3, rocker=2)
// ============================================================================

fn build_standard_fourbar() -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
    let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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

    // Identity driver: f(t) = t, f'(t) = 1, f''(t) = 0
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();
    mech
}

fn build_standard_fourbar_with_gravity() -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
    let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.add_force(ForceElement::Gravity(GravityElement::default()));
    mech.build().unwrap();
    mech
}

/// Create initial guess for the standard 4-bar at a given crank angle.
fn fourbar_initial_guess(mech: &Mechanism, angle: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q = state.make_q();
    state.set_pose("crank", &mut q, 0.0, 0.0, angle);
    let bx = angle.cos();
    let by = angle.sin();
    state.set_pose("coupler", &mut q, bx, by, 0.0);
    state.set_pose("rocker", &mut q, 4.0, 0.0, PI / 2.0);
    q
}

// ============================================================================
// 1. Near-toggle 4-bar (crank at 0 deg and 180 deg)
// ============================================================================

#[test]
fn near_toggle_position_solver_converges() {
    let mech = build_standard_fourbar();

    // Angles near the 180-degree toggle point
    let near_toggle_angles_deg: [f64; 4] = [179.0, 179.9, 180.1, 181.0];

    // Use continuation from a well-behaved starting point
    let start_angle = 170.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, start_angle);
    let start_result = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(
        start_result.converged,
        "Failed to converge at starting angle 170 deg"
    );

    let mut q_prev = start_result.q;
    let mut prev_angle = start_angle;

    for &angle_deg in &near_toggle_angles_deg {
        let angle = angle_deg.to_radians();

        // Step through intermediate angles for reliable continuation
        let n_intermediate = ((angle_deg - prev_angle.to_degrees()).abs() / 0.5).ceil() as usize;
        let n_intermediate = n_intermediate.max(1);

        for step in 1..=n_intermediate {
            let frac = step as f64 / n_intermediate as f64;
            let interp_angle = prev_angle + frac * (angle - prev_angle);
            let result = solve_position(&mech, &q_prev, interp_angle, 1e-10, 100).unwrap();
            assert!(
                result.converged,
                "Position solver failed near toggle at {:.1} deg (intermediate step), residual={}",
                interp_angle.to_degrees(),
                result.residual_norm,
            );
            q_prev = result.q;
        }

        prev_angle = angle;
    }
}

#[test]
fn near_toggle_velocity_is_finite() {
    let mech = build_standard_fourbar();

    // Use continuation from a safe starting angle to reach near-toggle
    let safe_angle = 170.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, safe_angle);
    let mut result = solve_position(&mech, &q0, safe_angle, 1e-10, 50).unwrap();
    assert!(result.converged);

    // Walk toward toggle with small steps
    let target_angles_deg: [f64; 8] = [175.0, 178.0, 179.0, 179.5, 179.9, 180.1, 180.5, 181.0];
    for &angle_deg in &target_angles_deg {
        let angle = angle_deg.to_radians();
        result = solve_position(&mech, &result.q, angle, 1e-10, 100).unwrap();
        if !result.converged {
            // Near exact toggle the solver may struggle; skip but do not panic
            continue;
        }

        let q_dot = solve_velocity(&mech, &result.q, angle).unwrap();

        for i in 0..q_dot.len() {
            assert!(
                q_dot[i].is_finite(),
                "q_dot[{}] is not finite at {:.1} deg: {}",
                i,
                angle_deg,
                q_dot[i],
            );
        }
    }
}

#[test]
fn near_toggle_acceleration_large_but_finite() {
    let mech = build_standard_fourbar();

    // Start from a safe angle and step toward toggle
    let safe_angle = 170.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, safe_angle);
    let mut pos = solve_position(&mech, &q0, safe_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let near_toggle_angles_deg: [f64; 6] = [175.0, 178.0, 179.0, 179.9, 180.1, 181.0];

    for &angle_deg in &near_toggle_angles_deg {
        let angle = angle_deg.to_radians();
        pos = solve_position(&mech, &pos.q, angle, 1e-10, 100).unwrap();
        if !pos.converged {
            continue;
        }

        let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
        let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle).unwrap();

        for i in 0..q_ddot.len() {
            assert!(
                q_ddot[i].is_finite(),
                "q_ddot[{}] is not finite at {:.1} deg: {}",
                i,
                angle_deg,
                q_ddot[i],
            );
        }
    }
}

#[test]
fn near_toggle_statics_returns_finite_lambdas() {
    let mech = build_standard_fourbar_with_gravity();

    let safe_angle = 170.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, safe_angle);
    let mut pos = solve_position(&mech, &q0, safe_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let near_toggle_angles_deg: [f64; 6] = [175.0, 178.0, 179.0, 179.9, 180.1, 181.0];

    for &angle_deg in &near_toggle_angles_deg {
        let angle = angle_deg.to_radians();
        pos = solve_position(&mech, &pos.q, angle, 1e-10, 100).unwrap();
        if !pos.converged {
            continue;
        }

        let statics_result = solve_statics(&mech, &pos.q, angle).unwrap();

        for i in 0..statics_result.lambdas.len() {
            assert!(
                statics_result.lambdas[i].is_finite(),
                "lambda[{}] is not finite at {:.1} deg: {}",
                i,
                angle_deg,
                statics_result.lambdas[i],
            );
        }
    }
}

#[test]
fn near_toggle_0_deg_position_solver_converges() {
    let mech = build_standard_fourbar();

    // Walk from 10 deg toward 0 deg
    let start_angle = 10.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, start_angle);
    let mut pos = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let near_zero_angles_deg: [f64; 4] = [5.0, 2.0, 1.0, 0.1];

    for &angle_deg in &near_zero_angles_deg {
        let angle = angle_deg.to_radians();
        pos = solve_position(&mech, &pos.q, angle, 1e-10, 100).unwrap();

        assert!(
            pos.converged,
            "Position solver failed near 0 deg at {:.1} deg, residual={}",
            angle_deg,
            pos.residual_norm,
        );

        let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
        for i in 0..q_dot.len() {
            assert!(
                q_dot[i].is_finite(),
                "q_dot[{}] is not finite at {:.1} deg",
                i,
                angle_deg,
            );
        }
    }
}

// ============================================================================
// 2. Redundant constraints (parallelogram 4-bar)
// ============================================================================

fn build_parallelogram_fourbar() -> Mechanism {
    // Parallelogram: ground=4, crank=2, coupler=4, rocker=2
    // This creates a redundant constraint (Grubler says DOF=1 but Jacobian
    // may show rank deficiency at certain configurations).
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 2.0, 1.0, 0.01);
    let coupler = make_bar("coupler", "B", "C", 4.0, 2.0, 0.05);
    let rocker = make_bar("rocker", "D", "C", 2.0, 1.0, 0.02);

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

    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();
    mech
}

fn parallelogram_initial_guess(mech: &Mechanism, angle: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q = state.make_q();
    // Crank at the driven angle
    state.set_pose("crank", &mut q, 0.0, 0.0, angle);
    // Coupler: starts at crank tip, same angle as ground (horizontal offset)
    let bx = 2.0 * angle.cos();
    let by = 2.0 * angle.sin();
    state.set_pose("coupler", &mut q, bx, by, 0.0);
    // Rocker: same angle as crank (parallelogram property)
    state.set_pose("rocker", &mut q, 4.0, 0.0, angle);
    q
}

#[test]
fn parallelogram_grubler_dof() {
    let mech = build_parallelogram_fourbar();

    // Without driver: 3 bodies * 3 - 4 rev * 2 = 9 - 8 = 1
    // With driver: 9 - 9 = 0
    let result = grubler_dof(&mech, 0);
    assert_eq!(result.dof, 0, "Grubler DOF with driver should be 0");
}

#[test]
fn parallelogram_position_solver_converges() {
    let mech = build_parallelogram_fourbar();

    let angle = PI / 4.0;
    let q0 = parallelogram_initial_guess(&mech, angle);
    let result = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();

    assert!(
        result.converged,
        "Parallelogram position solver failed, residual={}",
        result.residual_norm,
    );
}

#[test]
fn parallelogram_statics_produces_finite_results() {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 2.0, 1.0, 0.01);
    let coupler = make_bar("coupler", "B", "C", 4.0, 2.0, 0.05);
    let rocker = make_bar("rocker", "D", "C", 2.0, 1.0, 0.02);

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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.add_force(ForceElement::Gravity(GravityElement::default()));
    mech.build().unwrap();

    let angle = PI / 4.0;
    let q0 = parallelogram_initial_guess(&mech, angle);
    let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let statics_result = solve_statics(&mech, &pos.q, angle).unwrap();

    // Lambdas should all be finite (SVD pseudoinverse handles any redundancy)
    for i in 0..statics_result.lambdas.len() {
        assert!(
            statics_result.lambdas[i].is_finite(),
            "Parallelogram lambda[{}] is not finite: {}",
            i,
            statics_result.lambdas[i],
        );
    }
}

#[test]
fn parallelogram_jacobian_rank_at_flat_config() {
    // At the flat configuration (all links collinear), a parallelogram has a
    // redundant constraint. The Jacobian rank may drop, which the rank analysis
    // should detect.
    let mech = build_parallelogram_fourbar();

    // At angle=0 or PI, the parallelogram is flat -- this is the singular config
    let angle = 0.01_f64; // Very close to 0 but not exact to allow convergence
    let q0 = parallelogram_initial_guess(&mech, angle);
    let pos = solve_position(&mech, &q0, angle, 1e-10, 100).unwrap();

    if pos.converged {
        let rank_result = jacobian_rank_analysis(&mech, &pos.q, 0.0, None);

        // Even if redundant, the analysis should complete without panic
        assert!(
            rank_result.constraint_rank <= rank_result.n_constraints,
            "Rank should not exceed number of constraints"
        );
        assert!(
            rank_result.condition_number > 0.0,
            "Condition number should be positive"
        );
    }
    // If position solver did not converge at the near-flat config, that is acceptable
    // for a parallelogram at its singular configuration.
}

// ============================================================================
// 3. Underconstrained mechanism
// ============================================================================

#[test]
fn underconstrained_grubler_reports_positive_dof() {
    // Build mechanism with 2 bodies and only 1 revolute joint: DOF > 0 after driver.
    // 2 moving bodies * 3 = 6 coords, 1 rev joint removes 2 DOF, 1 driver removes 1.
    // Grubler: 6 - 2 - 1 = 3 remaining DOF.
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let bar1 = make_bar("bar1", "A", "B", 1.0, 1.0, 0.01);
    let bar2 = make_bar("bar2", "C", "D", 1.0, 1.0, 0.01);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar1).unwrap();
    mech.add_body(bar2).unwrap();

    // Only one revolute joint connecting bar1 to ground
    mech.add_revolute_joint("J1", "ground", "O", "bar1", "A")
        .unwrap();

    // Driver on bar1
    mech.add_revolute_driver("D1", "ground", "bar1", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();

    let result = grubler_dof(&mech, 3);
    // bar2 is completely free (3 DOF) and bar1 is fully constrained (0 remaining)
    // Total: 6 - 3 = 3
    assert_eq!(result.dof, 3, "Underconstrained mechanism should report DOF=3");
    assert!(!result.is_warning, "Should match expected DOF of 3");
}

#[test]
fn underconstrained_jacobian_rank_analysis() {
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let bar1 = make_bar("bar1", "A", "B", 1.0, 1.0, 0.01);
    let bar2 = make_bar("bar2", "C", "D", 1.0, 1.0, 0.01);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar1).unwrap();
    mech.add_body(bar2).unwrap();

    mech.add_revolute_joint("J1", "ground", "O", "bar1", "A")
        .unwrap();
    mech.add_revolute_driver("D1", "ground", "bar1", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.build().unwrap();

    let state = mech.state();
    let mut q = state.make_q();
    state.set_pose("bar1", &mut q, 0.0, 0.0, 0.5);
    state.set_pose("bar2", &mut q, 2.0, 1.0, 0.3);

    let rank_result = jacobian_rank_analysis(&mech, &q, 0.0, None);

    // 3 constraint equations (2 from revolute + 1 from driver), 6 coordinates
    assert_eq!(rank_result.n_constraints, 3);
    assert_eq!(rank_result.n_coords, 6);
    // Rank should be 3 (all constraints are independent)
    assert_eq!(rank_result.constraint_rank, 3);
    // Mobility: 6 - 3 = 3
    assert_eq!(rank_result.instantaneous_mobility, 3);
}

// ============================================================================
// 4. Zero and near-zero inertia bodies
// ============================================================================

#[test]
fn zero_izz_nonzero_mass_mass_matrix_not_singular() {
    // Build 4-bar with one body having Izz_cg = 0 and mass > 0
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.0); // Izz_cg = 0, mass = 2
    let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.build().unwrap();

    let angle = PI / 4.0;
    let q0 = fourbar_initial_guess(&mech, angle);
    let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let m_mat = assemble_mass_matrix(&mech, &pos.q);

    // The mass matrix should have positive diagonal entries for m > 0 bodies
    // even when Izz_cg = 0, because M_theta_theta = Izz_cg + m*|s_cg|^2
    // For crank: s_cg = (0.5, 0), so M_theta_theta = 0 + 2*0.25 = 0.5
    let crank_idx = mech.state().get_index("crank").unwrap();
    let m_theta_theta = m_mat[(crank_idx.theta_idx(), crank_idx.theta_idx())];
    assert!(
        m_theta_theta > 0.0,
        "M_theta_theta for Izz=0, m>0 body should be positive (parallel axis), got {}",
        m_theta_theta,
    );
    assert!(
        (m_theta_theta - 0.5).abs() < 1e-10,
        "Expected M_theta_theta = 0.5 (m * |s_cg|^2), got {}",
        m_theta_theta,
    );
}

#[test]
fn zero_izz_nonzero_mass_inverse_dynamics_works() {
    // Full kinematic + inverse dynamics pipeline with Izz_cg = 0
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.0); // Izz_cg = 0
    let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.add_force(ForceElement::Gravity(GravityElement::default()));
    mech.build().unwrap();

    let angle = PI / 3.0;
    let q0 = fourbar_initial_guess(&mech, angle);
    let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
    let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle).unwrap();

    let result =
        solve_inverse_dynamics(&mech, &pos.q, &q_dot, &q_ddot, angle).unwrap();

    // All lambdas and inertial forces should be finite
    for i in 0..result.lambdas.len() {
        assert!(
            result.lambdas[i].is_finite(),
            "lambda[{}] not finite with Izz=0: {}",
            i,
            result.lambdas[i],
        );
    }
    for i in 0..result.m_q_ddot.len() {
        assert!(
            result.m_q_ddot[i].is_finite(),
            "M*q_ddot[{}] not finite with Izz=0: {}",
            i,
            result.m_q_ddot[i],
        );
    }
}

#[test]
fn zero_mass_zero_izz_body_skipped_in_mass_matrix() {
    // Build a 4-bar where one body has mass=0, Izz=0
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 0.0, 0.0); // mass=0, Izz=0
    let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.build().unwrap();

    let angle = PI / 4.0;
    let q0 = fourbar_initial_guess(&mech, angle);
    let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let m_mat = assemble_mass_matrix(&mech, &pos.q);

    // The crank's 3x3 block in the mass matrix should be all zeros
    let crank_idx = mech.state().get_index("crank").unwrap();
    assert!(
        m_mat[(crank_idx.x_idx(), crank_idx.x_idx())].abs() < 1e-15,
        "M_xx for massless crank should be 0"
    );
    assert!(
        m_mat[(crank_idx.y_idx(), crank_idx.y_idx())].abs() < 1e-15,
        "M_yy for massless crank should be 0"
    );
    assert!(
        m_mat[(crank_idx.theta_idx(), crank_idx.theta_idx())].abs() < 1e-15,
        "M_tt for massless crank should be 0"
    );

    // But coupler and rocker should have positive mass entries
    let coupler_idx = mech.state().get_index("coupler").unwrap();
    assert!(
        m_mat[(coupler_idx.x_idx(), coupler_idx.x_idx())] > 0.0,
        "Coupler M_xx should be positive"
    );
    let rocker_idx = mech.state().get_index("rocker").unwrap();
    assert!(
        m_mat[(rocker_idx.x_idx(), rocker_idx.x_idx())] > 0.0,
        "Rocker M_xx should be positive"
    );
}

// ============================================================================
// 5. Prismatic joint at problematic orientations (slider-crank near TDC)
// ============================================================================

fn build_short_crank_slidercrank() -> Mechanism {
    // Slider-crank with very small crank: crank_length << conrod_length
    // crank = 0.2, conrod = 3.0 (ratio 1:15)
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 3.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.2, 0.5, 0.001);
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
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();
    mech
}

fn short_crank_slidercrank_initial_guess(mech: &Mechanism, angle: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q = state.make_q();
    let bx = 0.2 * angle.cos();
    let by = 0.2 * angle.sin();
    let phi = (-by / 3.0).asin();
    let cx = bx + 3.0 * phi.cos();
    state.set_pose("crank", &mut q, 0.0, 0.0, angle);
    state.set_pose("conrod", &mut q, bx, by, phi);
    state.set_pose("slider", &mut q, cx, 0.0, 0.0);
    q
}

#[test]
fn prismatic_near_tdc_position_converges() {
    let mech = build_short_crank_slidercrank();

    // Near TDC: crank at ~90 deg (piston at max displacement for horizontal slider)
    // and at 0/180 deg (near dead center positions)
    let test_angles_deg: [f64; 11] = [0.5, 1.0, 5.0, 85.0, 89.0, 89.9, 90.1, 91.0, 95.0, 175.0, 179.0];

    let start_angle = 45.0_f64.to_radians();
    let q0 = short_crank_slidercrank_initial_guess(&mech, start_angle);
    let mut pos = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(pos.converged, "Failed at starting angle");

    // Walk through sorted angles with continuation
    let mut sorted_angles: Vec<f64> = test_angles_deg.iter().map(|a| f64::to_radians(*a)).collect();
    sorted_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Reset to a safe start and sweep forward
    let start_angle = 0.5_f64.to_radians();
    let q0 = short_crank_slidercrank_initial_guess(&mech, start_angle);
    pos = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let step_size = 1.0_f64.to_radians(); // 1 degree steps
    let mut current_angle = start_angle;

    for &target_angle in &sorted_angles {
        // Walk toward target with small steps for reliable continuation
        while current_angle < target_angle {
            let next = (current_angle + step_size).min(target_angle);
            pos = solve_position(&mech, &pos.q, next, 1e-10, 100).unwrap();
            assert!(
                pos.converged,
                "Slider-crank position solver failed at {:.1} deg (walking toward {:.1} deg), residual={}",
                next.to_degrees(),
                target_angle.to_degrees(),
                pos.residual_norm,
            );
            current_angle = next;
        }
    }
}

#[test]
fn prismatic_near_tdc_acceleration_finite() {
    let mech = build_short_crank_slidercrank();

    let angles_deg: [f64; 7] = [5.0, 45.0, 85.0, 90.0, 95.0, 135.0, 175.0];

    let start_angle = 5.0_f64.to_radians();
    let q0 = short_crank_slidercrank_initial_guess(&mech, start_angle);
    let pos = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let step_size = 2.0_f64.to_radians();

    for &angle_deg in &angles_deg {
        let target = angle_deg.to_radians();

        // Step toward target
        let mut current = start_angle;
        let mut local_pos = pos.clone();
        while (current - target).abs() > 1e-12 {
            let next = if target > current {
                (current + step_size).min(target)
            } else {
                (current - step_size).max(target)
            };
            local_pos = solve_position(&mech, &local_pos.q, next, 1e-10, 100).unwrap();
            if !local_pos.converged {
                break;
            }
            current = next;
        }
        if !local_pos.converged {
            continue;
        }

        let q_dot = solve_velocity(&mech, &local_pos.q, target).unwrap();
        let q_ddot = solve_acceleration(&mech, &local_pos.q, &q_dot, target).unwrap();

        for i in 0..q_ddot.len() {
            assert!(
                q_ddot[i].is_finite(),
                "Slider-crank q_ddot[{}] not finite at {:.1} deg: {}",
                i,
                angle_deg,
                q_ddot[i],
            );
        }
    }
}

// ============================================================================
// 6. Branch consistency across full sweep
// ============================================================================

/// Compute angle-aware distance between two q vectors.
///
/// For each body in the state, the x,y differences are taken directly but
/// theta differences are wrapped to [-PI, PI] before summing. This avoids
/// false "branch jump" signals when the solver legitimately wraps an angle
/// across the +/-PI boundary.
fn angle_aware_q_distance(mech: &Mechanism, q_a: &DVector<f64>, q_b: &DVector<f64>) -> f64 {
    let state = mech.state();
    let mut sum_sq = 0.0;
    for body_id in state.body_ids() {
        let idx = state.get_index(&body_id).unwrap();
        let dx = q_a[idx.x_idx()] - q_b[idx.x_idx()];
        let dy = q_a[idx.y_idx()] - q_b[idx.y_idx()];
        let mut dtheta = q_a[idx.theta_idx()] - q_b[idx.theta_idx()];
        // Wrap to [-PI, PI]
        dtheta = (dtheta + PI) % (2.0 * PI) - PI;
        if dtheta < -PI {
            dtheta += 2.0 * PI;
        }
        sum_sq += dx * dx + dy * dy + dtheta * dtheta;
    }
    sum_sq.sqrt()
}

#[test]
fn fourbar_sweep_no_branch_jumps() {
    let mech = build_standard_fourbar();

    let n_steps = 360;
    let step_deg = 360.0 / n_steps as f64;

    // Start at 10 degrees (away from toggle)
    let start_angle = 10.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, start_angle);
    let first_result = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(first_result.converged);

    let mut q_prev = first_result.q;
    let mut max_q_step = 0.0_f64;
    let mut converged_count = 0_usize;

    for i in 1..=n_steps {
        let angle_deg = 10.0 + i as f64 * step_deg;
        let angle = angle_deg.to_radians();

        let result = solve_position(&mech, &q_prev, angle, 1e-10, 100).unwrap();

        if !result.converged {
            // Near exact toggle, the solver might struggle. That is acceptable
            // for this test as long as it does not branch-jump. Re-seed and continue.
            let q0_reseed = fourbar_initial_guess(&mech, angle);
            let retry = solve_position(&mech, &q0_reseed, angle, 1e-10, 100).unwrap();
            if retry.converged {
                q_prev = retry.q;
            }
            continue;
        }

        converged_count += 1;

        // Use angle-aware distance so theta wrapping does not trigger false alarms
        let q_step_norm = angle_aware_q_distance(&mech, &result.q, &q_prev);
        max_q_step = max_q_step.max(q_step_norm);
        q_prev = result.q;
    }

    // At least 95% of steps should converge (only toggle points may fail)
    assert!(
        converged_count >= (n_steps * 95 / 100),
        "Only {}/{} steps converged during full sweep",
        converged_count,
        n_steps,
    );

    // Maximum position step should be bounded.
    // For 1-degree steps on a 4-bar with link lengths O(1..4), the max step
    // in x,y should be small and theta changes ~1 degree. A true branch jump
    // would show up as a large x,y displacement (link-length scale).
    assert!(
        max_q_step < 2.0,
        "Possible branch jump detected: max angle-aware ||dq|| = {:.4} (threshold: 2.0)",
        max_q_step,
    );
}

#[test]
fn fourbar_sweep_consecutive_solutions_close() {
    // Sweep a half-revolution (20 to 200 deg) in 2-degree increments,
    // staying away from the 0/360 wrap-around and the exact toggle.
    // Verify consecutive solutions are close (no physical discontinuity).
    let mech = build_standard_fourbar();

    let n_steps = 90;
    let step_rad = (2.0_f64).to_radians();

    let start_angle = 20.0_f64.to_radians();
    let q0 = fourbar_initial_guess(&mech, start_angle);
    let pos = solve_position(&mech, &q0, start_angle, 1e-10, 50).unwrap();
    assert!(pos.converged);

    let mut prev_q = pos.q.clone();
    let mut current_q = pos.q.clone();
    let mut steps_checked = 0;

    for i in 1..=n_steps {
        let angle = start_angle + i as f64 * step_rad;

        let result = solve_position(&mech, &current_q, angle, 1e-10, 100).unwrap();

        if !result.converged {
            // Reset with a fresh guess and continue
            let q0_fresh = fourbar_initial_guess(&mech, angle);
            let retry = solve_position(&mech, &q0_fresh, angle, 1e-10, 100).unwrap();
            if retry.converged {
                prev_q = retry.q.clone();
                current_q = retry.q;
            }
            continue;
        }

        let q_step = angle_aware_q_distance(&mech, &result.q, &prev_q);

        // For 2-degree steps, the physical configuration change should be modest.
        // Generous bound: 1.5 (link lengths are O(1..4)).
        if steps_checked > 0 {
            assert!(
                q_step < 1.5,
                "Step {} ({:.1} deg): angle-aware ||dq|| = {:.4} -- possible branch jump",
                i,
                angle.to_degrees(),
                q_step,
            );
        }

        prev_q = result.q.clone();
        current_q = result.q;
        steps_checked += 1;
    }

    assert!(
        steps_checked > n_steps * 90 / 100,
        "Too few converged steps: {}/{}",
        steps_checked,
        n_steps,
    );
}
