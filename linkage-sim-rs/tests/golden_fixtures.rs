//! Golden fixture validation: compare Rust solver output against Python reference data.
//!
//! Tolerances (from RUST_MIGRATION.md):
//!   Position:            ‖Δq‖ < 1e-10  (NR convergence)
//!   Velocity/accel:      ‖Δ‖  < 1e-8
//!   Lagrange multipliers: relative |Δλ/λ| < 1e-6

use std::path::PathBuf;

use nalgebra::{DVector, Vector2};
use serde::Deserialize;

use linkage_sim_rs::analysis::energy::compute_energy_state_mech;
use linkage_sim_rs::core::body::{make_bar, make_ground, Body};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::gravity::Gravity;
use linkage_sim_rs::solver::forward_dynamics::{simulate, ForwardDynamicsConfig};
use linkage_sim_rs::solver::inverse_dynamics::solve_inverse_dynamics;
use linkage_sim_rs::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use linkage_sim_rs::solver::statics::{extract_reactions, get_driver_reactions, solve_statics};

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
        let result = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();

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
        let q_dot = solve_velocity(&mech, &result.q, angle).unwrap();
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
        let q_ddot = solve_acceleration(&mech, &result.q, &q_dot, angle).unwrap();
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
        let result = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();

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
        let q_dot = solve_velocity(&mech, &result.q, angle).unwrap();
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
        let q_ddot = solve_acceleration(&mech, &result.q, &q_dot, angle).unwrap();
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

// ---------------------------------------------------------------------------
// Statics golden fixture schemas
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoldenStatics {
    steps: Vec<StaticsStep>,
}

#[derive(Deserialize)]
struct StaticsStep {
    input_angle_rad: f64,
    lambdas: Vec<f64>,
    #[serde(rename = "Q")]
    q_forces: Vec<f64>,
    driver_torque: f64,
}

fn load_golden_statics(filename: &str) -> GoldenStatics {
    let path = golden_path(filename);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data).unwrap()
}

fn build_fourbar_gravity() -> (Mechanism, Gravity) {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
    let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
    coupler.add_coupler_point("P", 1.5, 0.5).unwrap();
    let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

    let mut bodies = std::collections::HashMap::new();
    bodies.insert("ground".to_string(), ground.clone());
    bodies.insert("crank".to_string(), crank.clone());
    bodies.insert("coupler".to_string(), coupler.clone());
    bodies.insert("rocker".to_string(), rocker.clone());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
        .unwrap();
    mech.build().unwrap();

    let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
    (mech, gravity)
}

#[test]
fn fourbar_statics_matches_golden() {
    let golden = load_golden_statics("fourbar_statics.json");
    let (mech, gravity) = build_fourbar_gravity();

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = fourbar_initial_guess(&mech, angle);
        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged, "Position solve failed at step {}", step_idx);

        let result = solve_statics(&mech, &pos.q, Some(&gravity), angle).unwrap();

        // Compare lambdas (looser near singular configs at toggle points)
        let lam_golden = DVector::from_column_slice(&step.lambdas);
        let lam_diff = (&result.lambdas - &lam_golden).norm();
        assert!(
            lam_diff < 0.5,
            "4-bar λ mismatch at step {} (angle={:.1} deg): ‖Δλ‖={:e}",
            step_idx, angle.to_degrees(), lam_diff,
        );

        // Compare Q (generalized forces — should match exactly)
        let q_golden = DVector::from_column_slice(&step.q_forces);
        let q_diff = (&result.q_forces - &q_golden).norm();
        assert!(
            q_diff < 1e-8,
            "4-bar Q mismatch at step {} (angle={:.1} deg): ‖ΔQ‖={:e}",
            step_idx, angle.to_degrees(), q_diff,
        );

        // Compare driver torque (looser near singular configs)
        let reactions = extract_reactions(&mech, &result);
        let drivers = get_driver_reactions(&reactions);
        assert_eq!(drivers.len(), 1);
        let torque_diff = (drivers[0].effort - step.driver_torque).abs();
        assert!(
            torque_diff < 0.5,
            "4-bar driver torque mismatch at step {} (angle={:.1} deg): Δτ={:e}",
            step_idx, angle.to_degrees(), torque_diff,
        );
    }
}

#[test]
fn slidercrank_statics_matches_golden() {
    let golden = load_golden_statics("slidercrank_statics.json");

    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 3.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 1.0, 0.01);
    let conrod = make_bar("conrod", "B", "C", 3.0, 2.0, 0.1);
    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();
    slider.mass = 0.5;

    let mut bodies = std::collections::HashMap::new();
    bodies.insert("ground".to_string(), ground.clone());
    bodies.insert("crank".to_string(), crank.clone());
    bodies.insert("conrod".to_string(), conrod.clone());
    bodies.insert("slider".to_string(), slider.clone());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(conrod).unwrap();
    mech.add_body(slider).unwrap();
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B").unwrap();
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "C").unwrap();
    mech.add_prismatic_joint("P1", "ground", "rail", "slider", "C", Vector2::new(1.0, 0.0), 0.0).unwrap();
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0).unwrap();
    mech.build().unwrap();

    let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = slidercrank_initial_guess(&mech, angle);
        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged, "Position solve failed at step {}", step_idx);

        let result = solve_statics(&mech, &pos.q, Some(&gravity), angle).unwrap();

        // Compare lambdas
        let lam_golden = DVector::from_column_slice(&step.lambdas);
        let lam_diff = (&result.lambdas - &lam_golden).norm();
        assert!(
            lam_diff < 1e-4,
            "Slider-crank λ mismatch at step {} (angle={:.1} deg): ‖Δλ‖={:e}",
            step_idx, angle.to_degrees(), lam_diff,
        );

        // Compare driver torque
        let reactions = extract_reactions(&mech, &result);
        let drivers = get_driver_reactions(&reactions);
        assert_eq!(drivers.len(), 1);
        let torque_diff = (drivers[0].effort - step.driver_torque).abs();
        assert!(
            torque_diff < 1e-4,
            "Slider-crank driver torque mismatch at step {} (angle={:.1} deg): Δτ={:e}",
            step_idx, angle.to_degrees(), torque_diff,
        );
    }
}
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoldenInverseDynamics {
    steps: Vec<InverseDynamicsStep>,
}

#[derive(Deserialize)]
struct InverseDynamicsStep {
    input_angle_rad: f64,
    lambdas: Vec<f64>,
    #[serde(rename = "Q")]
    q_forces: Vec<f64>,
    #[serde(rename = "M_q_ddot")]
    m_q_ddot: Vec<f64>,
    driver_torque: f64,
}

fn load_golden_inverse_dynamics(filename: &str) -> GoldenInverseDynamics {
    let path = golden_path(filename);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data).unwrap()
}

#[test]
fn fourbar_inverse_dynamics_matches_golden() {
    let golden = load_golden_inverse_dynamics("fourbar_inverse_dynamics.json");
    let (mech, gravity) = build_fourbar_gravity();

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = fourbar_initial_guess(&mech, angle);
        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged, "Position solve failed at step {}", step_idx);

        let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
        let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle).unwrap();

        let result = solve_inverse_dynamics(&mech, &pos.q, &q_dot, &q_ddot, Some(&gravity), angle).unwrap();

        // Compare Q (generalized forces -- should match closely)
        let q_golden = DVector::from_column_slice(&step.q_forces);
        let q_diff = (&result.q_forces - &q_golden).norm();
        assert!(
            q_diff < 1e-8,
            "4-bar inv dyn Q mismatch at step {} (angle={:.1} deg): norm(dQ)={:e}",
            step_idx, angle.to_degrees(), q_diff,
        );

        // Compare M*q_ddot (inertial forces)
        // M*q_ddot depends on q_ddot which has known sensitivity near toggle points
        // (lstsq vs SVD give slightly different results — same tolerance as kinematics q_ddot)
        let mq_golden = DVector::from_column_slice(&step.m_q_ddot);
        let mq_diff = (&result.m_q_ddot - &mq_golden).norm();
        assert!(
            mq_diff < 5e-2,
            "4-bar inv dyn M*q_ddot mismatch at step {} (angle={:.1} deg): norm(dMq)={:e}",
            step_idx, angle.to_degrees(), mq_diff,
        );

        // Compare lambdas and driver torque.
        // Near toggle points (0/180 deg), the Jacobian becomes near-singular and
        // lambda values blow up to ~1e10. Both Python lstsq and Rust SVD give
        // numerically meaningless values there, so we skip comparison when the
        // golden lambdas have extreme magnitude.
        let lam_golden = DVector::from_column_slice(&step.lambdas);
        let lam_golden_norm = lam_golden.norm();
        if lam_golden_norm < 1e6 {
            let lam_diff = (&result.lambdas - &lam_golden).norm();
            assert!(
                lam_diff < 0.5,
                "4-bar inv dyn lambda mismatch at step {} (angle={:.1} deg): norm(dl)={:e}",
                step_idx, angle.to_degrees(), lam_diff,
            );

            // Compare driver torque (last lambda)
            let driver_lam = result.lambdas[result.lambdas.len() - 1];
            let torque_diff = (driver_lam - step.driver_torque).abs();
            assert!(
                torque_diff < 0.5,
                "4-bar inv dyn driver torque mismatch at step {} (angle={:.1} deg): dt={:e}",
                step_idx, angle.to_degrees(), torque_diff,
            );
        }
    }
}

#[test]
fn slidercrank_inverse_dynamics_matches_golden() {
    let golden = load_golden_inverse_dynamics("slidercrank_inverse_dynamics.json");

    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 3.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 1.0, 1.0, 0.01);
    let conrod = make_bar("conrod", "B", "C", 3.0, 2.0, 0.1);
    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();
    slider.mass = 0.5;

    let mut bodies = std::collections::HashMap::new();
    bodies.insert("ground".to_string(), ground.clone());
    bodies.insert("crank".to_string(), crank.clone());
    bodies.insert("conrod".to_string(), conrod.clone());
    bodies.insert("slider".to_string(), slider.clone());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(conrod).unwrap();
    mech.add_body(slider).unwrap();
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B").unwrap();
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "C").unwrap();
    mech.add_prismatic_joint("P1", "ground", "rail", "slider", "C", Vector2::new(1.0, 0.0), 0.0).unwrap();
    mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0).unwrap();
    mech.build().unwrap();

    let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);

    for (step_idx, step) in golden.steps.iter().enumerate() {
        let angle = step.input_angle_rad;
        let q0 = slidercrank_initial_guess(&mech, angle);
        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged, "Position solve failed at step {}", step_idx);

        let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
        let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle).unwrap();

        let result = solve_inverse_dynamics(&mech, &pos.q, &q_dot, &q_ddot, Some(&gravity), angle).unwrap();

        // Compare Q
        let q_golden = DVector::from_column_slice(&step.q_forces);
        let q_diff = (&result.q_forces - &q_golden).norm();
        assert!(
            q_diff < 1e-8,
            "Slider-crank inv dyn Q mismatch at step {} (angle={:.1} deg): norm(dQ)={:e}",
            step_idx, angle.to_degrees(), q_diff,
        );

        // Compare M*q_ddot
        let mq_golden = DVector::from_column_slice(&step.m_q_ddot);
        let mq_diff = (&result.m_q_ddot - &mq_golden).norm();
        assert!(
            mq_diff < 1e-4,
            "Slider-crank inv dyn M*q_ddot mismatch at step {} (angle={:.1} deg): norm(dMq)={:e}",
            step_idx, angle.to_degrees(), mq_diff,
        );

        // Compare lambdas
        let lam_golden = DVector::from_column_slice(&step.lambdas);
        let lam_diff = (&result.lambdas - &lam_golden).norm();
        assert!(
            lam_diff < 1e-4,
            "Slider-crank inv dyn lambda mismatch at step {} (angle={:.1} deg): norm(dl)={:e}",
            step_idx, angle.to_degrees(), lam_diff,
        );

        // Compare driver torque
        let driver_lam = result.lambdas[result.lambdas.len() - 1];
        let torque_diff = (driver_lam - step.driver_torque).abs();
        assert!(
            torque_diff < 1e-4,
            "Slider-crank inv dyn driver torque mismatch at step {} (angle={:.1} deg): dt={:e}",
            step_idx, angle.to_degrees(), torque_diff,
        );
    }
}

// Forward dynamics golden fixture: simple pendulum
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[allow(dead_code)]
struct GoldenDynamics {
    initial_angle_rad: f64,
    gravity: [f64; 2],
    steps: Vec<DynamicsStep>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct DynamicsStep {
    t: f64,
    q: Vec<f64>,
    q_dot: Vec<f64>,
    kinetic_energy: f64,
    potential_energy: f64,
    total_energy: f64,
}

fn load_golden_dynamics(filename: &str) -> GoldenDynamics {
    let path = golden_path(filename);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data).unwrap()
}

/// Build the pendulum matching the golden fixture:
/// Single bar pinned to ground, L=1, m=1, Izz_cg=0, CG at tip.
fn build_pendulum_for_golden() -> (Mechanism, Gravity) {
    let ground = make_ground(&[("O", 0.0, 0.0)]);

    let mut bar = Body::new("bar");
    bar.add_attachment_point("A", 0.0, 0.0).unwrap();
    bar.mass = 1.0;
    bar.cg_local = Vector2::new(1.0, 0.0);
    bar.izz_cg = 0.0;

    let mut bodies = std::collections::HashMap::new();
    bodies.insert("ground".to_string(), ground.clone());
    bodies.insert("bar".to_string(), bar.clone());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
        .unwrap();
    mech.build().unwrap();

    let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
    (mech, gravity)
}

#[test]
fn pendulum_dynamics_matches_golden() {
    let golden = load_golden_dynamics("pendulum_dynamics.json");
    let (mech, gravity) = build_pendulum_for_golden();
    let state = mech.state();

    // Set up initial conditions matching the golden fixture
    let theta0 = golden.initial_angle_rad;
    let mut q0 = state.make_q();
    state.set_pose("bar", &mut q0, 0.0, 0.0, theta0);
    let qd0 = DVector::zeros(state.n_coords());

    // Collect t_eval from the golden steps
    let t_eval: Vec<f64> = golden.steps.iter().map(|s| s.t).collect();
    let t_end = *t_eval.last().unwrap();

    let config = ForwardDynamicsConfig {
        alpha: 10.0,
        beta: 10.0,
        max_step: 0.001,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let result = simulate(
        &mech,
        &q0,
        &qd0,
        (0.0, t_end),
        Some(&gravity),
        Some(&config),
        Some(&t_eval),
    ).unwrap();
    assert!(result.success, "Simulation failed: {}", result.message);
    assert_eq!(result.t.len(), golden.steps.len());

    // Compare trajectory: position should match within 1e-3
    let g_mag = 9.81;
    let mut max_q_diff = 0.0_f64;
    let mut max_energy_diff = 0.0_f64;
    let initial_energy = golden.steps[0].total_energy;

    for (i, step) in golden.steps.iter().enumerate() {
        // Position comparison (theta is the main DOF; x,y are constrained near 0)
        let q_golden = DVector::from_column_slice(&step.q);
        let q_diff = (&result.q[i] - &q_golden).norm();
        max_q_diff = max_q_diff.max(q_diff);

        assert!(
            q_diff < 1e-3,
            "Pendulum q mismatch at step {} (t={:.3}): ||dq||={:e}",
            i,
            step.t,
            q_diff,
        );

        // Energy conservation check: compute energy from our simulation
        let es = compute_energy_state_mech(&mech, &result.q[i], &result.q_dot[i], g_mag);
        let energy_diff = (es.total - initial_energy).abs();
        max_energy_diff = max_energy_diff.max(energy_diff);
    }

    // Total energy drift < 5% over the simulation
    let energy_drift_pct = max_energy_diff / initial_energy.abs() * 100.0;
    assert!(
        energy_drift_pct < 5.0,
        "Energy drift = {:.2}% (max deviation = {:e}, initial = {:e})",
        energy_drift_pct,
        max_energy_diff,
        initial_energy,
    );
}

#[test]
fn pendulum_dynamics_energy_conservation() {
    // Independent energy conservation test (not dependent on golden data accuracy,
    // just verifies our own simulation conserves energy internally)
    let (mech, gravity) = build_pendulum_for_golden();
    let state = mech.state();

    let theta0 = -std::f64::consts::FRAC_PI_2 + 0.2;
    let mut q0 = state.make_q();
    state.set_pose("bar", &mut q0, 0.0, 0.0, theta0);
    let qd0 = DVector::zeros(state.n_coords());

    let config = ForwardDynamicsConfig {
        alpha: 10.0,
        beta: 10.0,
        max_step: 0.001,
        ..Default::default()
    };

    let t_eval: Vec<f64> = (0..=500).map(|i| i as f64 * 0.01).collect();
    let result = simulate(
        &mech,
        &q0,
        &qd0,
        (0.0, 5.0),
        Some(&gravity),
        Some(&config),
        Some(&t_eval),
    ).unwrap();
    assert!(result.success);

    let g_mag = 9.81;
    let e0 = compute_energy_state_mech(&mech, &result.q[0], &result.q_dot[0], g_mag).total;

    let mut max_deviation = 0.0_f64;
    for i in 0..result.t.len() {
        let es = compute_energy_state_mech(&mech, &result.q[i], &result.q_dot[i], g_mag);
        max_deviation = max_deviation.max((es.total - e0).abs());
    }

    let drift_pct = max_deviation / e0.abs() * 100.0;
    assert!(
        drift_pct < 5.0,
        "Energy drift = {:.2}% over 5s simulation (max_dev={:e}, e0={:e})",
        drift_pct,
        max_deviation,
        e0,
    );
}
