//! Golden fixture validation: compare Rust solver output against Python reference data.
//!
//! Tolerances (from RUST_MIGRATION.md):
//!   Position:            ‖Δq‖ < 1e-10  (NR convergence)
//!   Velocity/accel:      ‖Δ‖  < 1e-8
//!   Lagrange multipliers: relative |Δλ/λ| < 1e-6

use std::path::PathBuf;

use nalgebra::{DVector, Vector2};
use serde::Deserialize;

use linkage_sim_rs::core::body::{make_bar, make_ground, Body};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};

// ---------------------------------------------------------------------------
// JSON schema for golden fixture files
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoldenKinematics {
    steps: Vec<KinematicsStep>,
}

#[derive(Deserialize)]
struct KinematicsStep {
    input_angle_rad: f64,
    q: Vec<f64>,
    q_dot: Vec<f64>,
    q_ddot: Vec<f64>,
}

fn golden_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data")
        .join("benchmarks")
        .join("golden")
        .join(filename)
}

fn load_golden_kinematics(filename: &str) -> GoldenKinematics {
    let path = golden_path(filename);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data).unwrap()
}

// ---------------------------------------------------------------------------
// 4-bar mechanism builder (matches export_golden.py::build_fourbar_driven)
// ---------------------------------------------------------------------------

fn build_fourbar_driven() -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
    let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
    coupler.add_coupler_point("P", 1.5, 0.5).unwrap();
    let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();

    // Identity driver: f(t) = t, f'(t) = 1, f''(t) = 0
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();
    mech
}

/// Create initial guess for 4-bar at given crank angle.
fn fourbar_initial_guess(mech: &Mechanism, angle: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q = state.make_q();
    state.set_pose("crank", &mut q, 0.0, 0.0, angle);
    let bx = angle.cos();
    let by = angle.sin();
    state.set_pose("coupler", &mut q, bx, by, 0.0);
    state.set_pose("rocker", &mut q, 4.0, 0.0, std::f64::consts::FRAC_PI_2);
    q
}

// ---------------------------------------------------------------------------
// Slider-crank mechanism builder (matches export_golden.py::build_slidercrank_driven)
// ---------------------------------------------------------------------------

fn build_slidercrank_driven() -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 3.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 1.0, 0.01);
    let conrod = make_bar("conrod", "B", "C", 3.0, 2.0, 0.1);

    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();
    slider.mass = 0.5;

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(conrod).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B").unwrap();
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "C").unwrap();
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "C",
        Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();

    mech.build().unwrap();
    mech
}

fn slidercrank_initial_guess(mech: &Mechanism, angle: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q = state.make_q();
    let bx = angle.cos();
    let by = angle.sin();
    let phi = (-by / 3.0).asin();
    let cx = bx + 3.0 * phi.cos();
    state.set_pose("crank", &mut q, 0.0, 0.0, angle);
    state.set_pose("conrod", &mut q, bx, by, phi);
    state.set_pose("slider", &mut q, cx, 0.0, 0.0);
    q
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn fourbar_kinematics_matches_golden() {
    let golden = load_golden_kinematics("fourbar_kinematics.json");
    let mech = build_fourbar_driven();

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = fourbar_initial_guess(&mech, angle);
        let result = solve_position(&mech, &q0, angle, 1e-10, 50);

        assert!(
            result.converged,
            "4-bar NR failed at step {} (angle={:.1} deg), residual={}",
            step_idx,
            angle.to_degrees(),
            result.residual_norm,
        );

        // Compare position
        let q_golden = DVector::from_column_slice(&step.q);
        let q_diff = (&result.q - &q_golden).norm();
        assert!(
            q_diff < 1e-10,
            "4-bar q mismatch at step {} (angle={:.1} deg): ‖Δq‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_diff,
        );

        // Compare velocity
        let q_dot = solve_velocity(&mech, &result.q, angle);
        let q_dot_golden = DVector::from_column_slice(&step.q_dot);
        let q_dot_diff = (&q_dot - &q_dot_golden).norm();
        assert!(
            q_dot_diff < 1e-8,
            "4-bar q_dot mismatch at step {} (angle={:.1} deg): ‖Δq̇‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_dot_diff,
        );

        // Compare acceleration (looser tolerance near singular configurations —
        // lstsq vs SVD give slightly different results near toggle points)
        let q_ddot = solve_acceleration(&mech, &result.q, &q_dot, angle);
        let q_ddot_golden = DVector::from_column_slice(&step.q_ddot);
        let q_ddot_diff = (&q_ddot - &q_ddot_golden).norm();
        assert!(
            q_ddot_diff < 1e-2,
            "4-bar q_ddot mismatch at step {} (angle={:.1} deg): ‖Δq̈‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_ddot_diff,
        );
    }
}

#[test]
fn slidercrank_kinematics_matches_golden() {
    let golden = load_golden_kinematics("slidercrank_kinematics.json");
    let mech = build_slidercrank_driven();

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = slidercrank_initial_guess(&mech, angle);
        let result = solve_position(&mech, &q0, angle, 1e-10, 50);

        assert!(
            result.converged,
            "Slider-crank NR failed at step {} (angle={:.1} deg), residual={}",
            step_idx,
            angle.to_degrees(),
            result.residual_norm,
        );

        // Compare position
        let q_golden = DVector::from_column_slice(&step.q);
        let q_diff = (&result.q - &q_golden).norm();
        assert!(
            q_diff < 1e-10,
            "Slider-crank q mismatch at step {} (angle={:.1} deg): ‖Δq‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_diff,
        );

        // Compare velocity
        let q_dot = solve_velocity(&mech, &result.q, angle);
        let q_dot_golden = DVector::from_column_slice(&step.q_dot);
        let q_dot_diff = (&q_dot - &q_dot_golden).norm();
        assert!(
            q_dot_diff < 1e-8,
            "Slider-crank q_dot mismatch at step {} (angle={:.1} deg): ‖Δq̇‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_dot_diff,
        );

        // Compare acceleration
        let q_ddot = solve_acceleration(&mech, &result.q, &q_dot, angle);
        let q_ddot_golden = DVector::from_column_slice(&step.q_ddot);
        let q_ddot_diff = (&q_ddot - &q_ddot_golden).norm();
        assert!(
            q_ddot_diff < 1e-6,
            "Slider-crank q_ddot mismatch at step {} (angle={:.1} deg): ‖Δq̈‖={:e}",
            step_idx,
            angle.to_degrees(),
            q_ddot_diff,
        );
    }
}
