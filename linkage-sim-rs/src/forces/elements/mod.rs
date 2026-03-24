//! Force element types for planar mechanisms.
//!
//! Each variant represents a force element that contributes to the
//! generalized force vector Q via the virtual work principle.

mod time_modulation;
mod element_types;
mod evaluation;
mod force_element;

pub use time_modulation::*;
pub use element_types::*;
pub use evaluation::*;
pub use force_element::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;
    use std::collections::HashMap;
    use nalgebra::DVector;

    fn setup_single_bar() -> (crate::core::state::State, HashMap<String, crate::core::body::Body>) {
        let mut state = crate::core::state::State::new();
        state.register_body("bar").unwrap();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.1);
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar".to_string(), bar);
        (state, bodies)
    }

    fn setup_two_bars() -> (crate::core::state::State, HashMap<String, crate::core::body::Body>) {
        let mut state = crate::core::state::State::new();
        state.register_body("bar1").unwrap();
        state.register_body("bar2").unwrap();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar1 = make_bar("bar1", "A", "B", 1.0, 2.0, 0.1);
        let bar2 = make_bar("bar2", "C", "D", 1.0, 2.0, 0.1);
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar1".to_string(), bar1);
        bodies.insert("bar2".to_string(), bar2);
        (state, bodies)
    }

    #[test]
    fn gravity_element_matches_old_gravity() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::Gravity(GravityElement::default());
        let result = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // F = mg = 2.0 * 9.81 = 19.62 downward
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], -19.62, epsilon = 1e-10);
        // Torque from gravity at CG=(0.5, 0) with θ=0: B(0)·(0.5,0) = (0,0.5), dot (0,-19.62) = -9.81
        assert_abs_diff_eq!(result[2], -9.81, epsilon = 1e-10);
    }

    #[test]
    fn linear_spring_zero_at_free_length() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // Place bars 1.0 apart (free length = 1.0 → no force)
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar1".into(),
            point_a: [1.0, 0.0], // tip of bar1
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0], // base of bar2
            point_b_name: None,
            stiffness: 500.0,
            free_length: 1.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At free length → zero force
        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn linear_spring_produces_restoring_force() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // bar1 origin at (0,0), bar2 origin at (2,0)
        // Attach spring at body origins → global distance = 2.0
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 2.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0], // body origin
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0], // body origin
            point_b_name: None,
            stiffness: 100.0,
            free_length: 1.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Extension = 2.0 - 1.0 = 1.0, stiffness = 100
        // Force on bar1 = +100 N in x (toward bar2)
        // Force on bar2 = -100 N in x (toward bar1)
        assert_abs_diff_eq!(result[0], 100.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], -100.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn torsion_spring_produces_restoring_torque() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.0, 0.0, PI / 4.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::TorsionSpring(TorsionSpringElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stiffness: 10.0,
            free_angle: 0.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Relative angle = π/4 - 0 = π/4
        // Torque on bar2 = -10 * (π/4) (restoring)
        // Torque on bar1 = +10 * (π/4) (reaction)
        let expected = -10.0 * PI / 4.0;
        assert_abs_diff_eq!(result[5], expected, epsilon = 1e-10); // bar2 θ
        assert_abs_diff_eq!(result[2], -expected, epsilon = 1e-10); // bar1 θ
    }

    #[test]
    fn external_force_produces_correct_q() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let ext = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0], // CG
            local_point_name: None,
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        });

        let result = ext.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        assert_abs_diff_eq!(result[0], 10.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], -5.0, epsilon = 1e-14);
        // B(0) · (0.5, 0) = (0, 0.5), dot (10, -5) = -2.5
        assert_abs_diff_eq!(result[2], -2.5, epsilon = 1e-14);
    }

    #[test]
    fn external_torque_produces_correct_q() {
        let (state, bodies) = setup_single_bar();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let ext = ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "bar".into(),
            torque: 7.5,
            modulation: TimeModulation::Constant,
        });

        let result = ext.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[2], 7.5, epsilon = 1e-15);
    }

    #[test]
    fn rotary_damper_resists_relative_motion() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar1 angular velocity = 0, bar2 angular velocity = 2 rad/s
        q_dot[5] = 2.0; // bar2 theta_dot

        let damper = ForceElement::RotaryDamper(RotaryDamperElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            damping: 5.0,
        });

        let result = damper.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Relative rate = 2.0 - 0.0 = 2.0
        // Torque on bar2 = -5 * 2 = -10 (opposes motion)
        // Torque on bar1 = +10 (reaction)
        assert_abs_diff_eq!(result[5], -10.0, epsilon = 1e-14); // bar2 θ
        assert_abs_diff_eq!(result[2], 10.0, epsilon = 1e-14); // bar1 θ
    }

    #[test]
    fn type_name_returns_correct_strings() {
        assert_eq!(
            ForceElement::Gravity(GravityElement::default()).type_name(),
            "Gravity"
        );
        assert_eq!(
            ForceElement::LinearSpring(LinearSpringElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                point_a_name: None,
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                point_b_name: None,
                stiffness: 1.0,
                free_length: 1.0,
            })
            .type_name(),
            "Linear Spring"
        );
    }

    #[test]
    fn serde_roundtrip_gravity() {
        let elem = ForceElement::Gravity(GravityElement::default());
        let json = serde_json::to_string(&elem).unwrap();
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        assert_eq!(back.type_name(), "Gravity");
    }

    #[test]
    fn serde_roundtrip_linear_spring() {
        let elem = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "crank".into(),
            point_a: [0.1, 0.0],
            point_a_name: None,
            body_b: "rocker".into(),
            point_b: [-0.1, 0.0],
            point_b_name: None,
            stiffness: 500.0,
            free_length: 0.2,
        });
        let json = serde_json::to_string(&elem).unwrap();
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::LinearSpring(s) => {
                assert_abs_diff_eq!(s.stiffness, 500.0, epsilon = 1e-15);
                assert_eq!(s.body_a, "crank");
            }
            _ => panic!("Expected LinearSpring"),
        }
    }

    #[test]
    fn serde_tagged_format() {
        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0],
            local_point_name: None,
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"ExternalForce\""));
    }

    // ── Gas Spring tests ─────────────────────────────────────────────────────

    #[test]
    fn gas_spring_force_at_extended_length() {
        // At extended length, compression=0 → gas_column=stroke → force=initial_force
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // Place bodies so attachment distance = extended_length
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.5, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At extended length: compression=0, gas_column=stroke, ratio=1.0
        // force = 200.0 * 1.0 = 200.0, pushes apart along +x
        // bar1 gets pushed in -x, bar2 in +x
        assert_abs_diff_eq!(result[0], -200.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 200.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn gas_spring_force_increases_with_compression() {
        // When compressed, gas column shrinks → force increases
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // extended_length=0.5, stroke=0.2. Place bodies 0.4 apart → compression=0.1
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.4, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // compression = 0.5 - 0.4 = 0.1
        // gas_column = 0.2 - 0.1 = 0.1
        // ratio = (0.2 / 0.1)^1.0 = 2.0
        // force = 200.0 * 2.0 = 400.0 (pushes apart)
        assert_abs_diff_eq!(result[0], -400.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 400.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn gas_spring_compression_clamped_beyond_stroke() {
        // If spring compressed beyond stroke, compression clamps to stroke
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // extended_length=0.5, stroke=0.2. Place bodies 0.1 apart → compression would be 0.4 but clamps to 0.2
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.1, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // compression clamped to 0.2 (=stroke), gas_column = max(0.2-0.2, 1e-10) = 1e-10
        // ratio = (0.2 / 1e-10)^1.0 = very large
        // This tests the singularity protection (gas_column floored at 1e-10)
        assert!(result[3] > 200.0); // force should be much larger than initial
    }

    #[test]
    fn gas_spring_stroke_zero_produces_initial_force() {
        // A degenerate gas spring with stroke=0 should act as a constant-force
        // element, producing initial_force magnitude (not zero).
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.5, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            initial_force: 150.0,
            extended_length: 0.5,
            stroke: 0.0, // degenerate: zero stroke
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // stroke=0 → constant-force element with magnitude = initial_force
        // Pushes apart along +x: bar1 gets -150, bar2 gets +150
        assert_abs_diff_eq!(result[0], -150.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 150.0, epsilon = 1e-10); // bar2 Fx
    }

    // ── Bearing Friction tests ───────────────────────────────────────────────

    #[test]
    fn bearing_friction_opposes_relative_rotation() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 spinning at 5 rad/s, bar1 stationary
        q_dot[5] = 5.0;

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 2.0,
            viscous_coeff: 0.5,
            coulomb_coeff: 0.0,
            pin_radius: 0.0,
            radial_load: 0.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // omega_rel = 5.0, tanh(5.0/0.01) ~ 1.0
        // magnitude = 2.0 + 0.5 * 5.0 = 4.5
        // torque on bar2 = -4.5 * 1.0 = -4.5 (opposes positive rotation)
        assert_abs_diff_eq!(result[5], -4.5, epsilon = 1e-3); // bar2 θ
        assert_abs_diff_eq!(result[2], 4.5, epsilon = 1e-3); // bar1 θ (reaction)
    }

    #[test]
    fn bearing_friction_smooth_near_zero_speed() {
        // Near zero speed, tanh regularization produces small, smooth torque
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[5] = 0.001; // very slow

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 2.0,
            viscous_coeff: 0.0,
            coulomb_coeff: 0.0,
            pin_radius: 0.0,
            radial_load: 0.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // omega_rel = 0.001, direction = tanh(0.001/0.01) = tanh(0.1) ~ 0.0997
        // torque = -2.0 * 0.0997 ~ -0.1993
        // Torque magnitude should be much less than constant_drag due to regularization
        assert!(result[5].abs() < 2.0);
        assert!(result[5] < 0.0); // still opposes positive motion
    }

    #[test]
    fn bearing_friction_with_coulomb_component() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[5] = 10.0; // fast enough that tanh ~ 1.0

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 1.0,
            viscous_coeff: 0.0,
            coulomb_coeff: 0.3,
            pin_radius: 0.01,
            radial_load: 1000.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // magnitude = 1.0 + 0.0 + 0.3 * 0.01 * 1000.0 = 1.0 + 3.0 = 4.0
        // torque on bar2 = -4.0 (opposes motion)
        assert_abs_diff_eq!(result[5], -4.0, epsilon = 1e-3);
    }

    // ── Joint Limit tests ────────────────────────────────────────────────────

    #[test]
    fn joint_limit_no_torque_within_range() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.0, 0.0, PI / 4.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -PI / 2.0,
            angle_max: PI / 2.0,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // theta_rel = π/4 is within [-π/2, π/2] → no torque
        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn joint_limit_restoring_torque_above_max() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        // theta_rel = 2.0 > angle_max = 1.5
        state.set_pose("bar2", &mut q, 0.0, 0.0, 2.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -1.5,
            angle_max: 1.5,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // penetration = 2.0 - 1.5 = 0.5
        // torque on bar2 = -(1000 * 0.5) = -500 (pushes back CW)
        assert_abs_diff_eq!(result[5], -500.0, epsilon = 1e-10); // bar2 θ
        assert_abs_diff_eq!(result[2], 500.0, epsilon = 1e-10); // bar1 θ (reaction)
    }

    #[test]
    fn joint_limit_restoring_torque_below_min() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        // theta_rel = -2.0 < angle_min = -1.5
        state.set_pose("bar2", &mut q, 0.0, 0.0, -2.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -1.5,
            angle_max: 1.5,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // penetration = -1.5 - (-2.0) = 0.5
        // torque on bar2 = +1000 * 0.5 = +500 (pushes back CCW)
        assert_abs_diff_eq!(result[5], 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], -500.0, epsilon = 1e-10);
    }

    // ── Motor tests ──────────────────────────────────────────────────────────

    #[test]
    fn motor_stall_torque_at_zero_speed() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 100.0,
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At zero speed, torque_fraction = 1.0
        // torque = 10.0 * 1.0 * 1.0 = 10.0
        assert_abs_diff_eq!(result[5], 10.0, epsilon = 1e-14); // bar2 θ
        assert_abs_diff_eq!(result[2], -10.0, epsilon = 1e-14); // bar1 θ (reaction)
    }

    #[test]
    fn motor_zero_torque_at_no_load_speed() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 at no-load speed
        q_dot[5] = 100.0;

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 100.0,
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At no-load speed, torque_fraction = 0.0
        assert_abs_diff_eq!(result[5], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn motor_zero_when_no_load_speed_invalid() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 0.0, // invalid
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-15);
        }
    }

    // ── Linear Actuator tests ────────────────────────────────────────────────

    #[test]
    fn linear_actuator_constant_force_no_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 0.0,
            stroke_min: 0.0, stroke_max: 0.0,
            end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Positive force = push apart along +x
        assert_abs_diff_eq!(result[0], -50.0, epsilon = 1e-10); // bar1 Fx (pushed left)
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10); // bar2 Fx (pushed right)
    }

    #[test]
    fn linear_actuator_force_reduced_at_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 moving at speed_limit → force should be zero
        q_dot[3] = 2.0; // bar2 vx = speed_limit

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 2.0,
            stroke_min: 0.0, stroke_max: 0.0,
            end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // v_along = 2.0, speed_ratio = 2.0/2.0 = 1.0 → force = 0
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_actuator_force_halved_at_half_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[3] = 1.0; // bar2 vx = half of speed_limit

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 2.0,
            stroke_min: 0.0, stroke_max: 0.0,
            end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // v_along = 1.0, speed_ratio = 0.5, force = 50 * 0.5 = 25.0
        assert_abs_diff_eq!(result[0], -25.0, epsilon = 1e-10); // bar1 pushed left
        assert_abs_diff_eq!(result[3], 25.0, epsilon = 1e-10); // bar2 pushed right
    }

    // ── Serde roundtrip tests for new elements ───────────────────────────────

    #[test]
    fn serde_roundtrip_gas_spring() {
        let elem = ForceElement::GasSpring(GasSpringElement {
            body_a: "link1".into(),
            point_a: [0.1, 0.0],
            point_a_name: None,
            body_b: "link2".into(),
            point_b: [-0.1, 0.0],
            point_b_name: None,
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 10.0,
            polytropic_exp: 1.3,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"GasSpring\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::GasSpring(g) => {
                assert_abs_diff_eq!(g.initial_force, 200.0, epsilon = 1e-15);
                assert_abs_diff_eq!(g.stroke, 0.2, epsilon = 1e-15);
                assert_abs_diff_eq!(g.polytropic_exp, 1.3, epsilon = 1e-15);
                assert_eq!(g.body_a, "link1");
            }
            _ => panic!("Expected GasSpring"),
        }
    }

    #[test]
    fn serde_roundtrip_gas_spring_defaults() {
        // Test that default fields deserialize correctly when omitted
        let json = r#"{"type":"GasSpring","body_a":"a","point_a":[0,0],"body_b":"b","point_b":[0,0],"initial_force":100,"extended_length":0.5,"stroke":0.2}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::GasSpring(g) => {
                assert_abs_diff_eq!(g.damping, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(g.polytropic_exp, 1.0, epsilon = 1e-15);
            }
            _ => panic!("Expected GasSpring"),
        }
    }

    #[test]
    fn serde_roundtrip_bearing_friction() {
        let elem = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "arm".into(),
            body_j: "ground".into(),
            constant_drag: 1.5,
            viscous_coeff: 0.1,
            coulomb_coeff: 0.3,
            pin_radius: 0.01,
            radial_load: 500.0,
            v_threshold: 0.02,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"BearingFriction\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.constant_drag, 1.5, epsilon = 1e-15);
                assert_abs_diff_eq!(b.v_threshold, 0.02, epsilon = 1e-15);
                assert_eq!(b.body_i, "arm");
            }
            _ => panic!("Expected BearingFriction"),
        }
    }

    #[test]
    fn serde_roundtrip_bearing_friction_defaults() {
        let json = r#"{"type":"BearingFriction","body_i":"a","body_j":"b","constant_drag":1,"viscous_coeff":0,"coulomb_coeff":0,"pin_radius":0,"radial_load":0}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.v_threshold, 0.01, epsilon = 1e-15);
            }
            _ => panic!("Expected BearingFriction"),
        }
    }

    #[test]
    fn serde_roundtrip_joint_limit() {
        let elem = ForceElement::JointLimit(JointLimitElement {
            body_i: "crank".into(),
            body_j: "rocker".into(),
            angle_min: -1.0,
            angle_max: 1.0,
            stiffness: 5000.0,
            damping: 50.0,
            restitution: 0.3,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"JointLimit\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::JointLimit(j) => {
                assert_abs_diff_eq!(j.stiffness, 5000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(j.restitution, 0.3, epsilon = 1e-15);
                assert_eq!(j.body_i, "crank");
            }
            _ => panic!("Expected JointLimit"),
        }
    }

    #[test]
    fn serde_roundtrip_joint_limit_defaults() {
        let json = r#"{"type":"JointLimit","body_i":"a","body_j":"b","angle_min":-1,"angle_max":1,"stiffness":1000}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::JointLimit(j) => {
                assert_abs_diff_eq!(j.damping, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(j.restitution, 0.5, epsilon = 1e-15);
            }
            _ => panic!("Expected JointLimit"),
        }
    }

    #[test]
    fn serde_roundtrip_motor() {
        let elem = ForceElement::Motor(MotorElement {
            body_i: "ground".into(),
            body_j: "wheel".into(),
            stall_torque: 50.0,
            no_load_speed: 300.0,
            direction: -1.0,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"Motor\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::Motor(m) => {
                assert_abs_diff_eq!(m.stall_torque, 50.0, epsilon = 1e-15);
                assert_abs_diff_eq!(m.direction, -1.0, epsilon = 1e-15);
                assert_eq!(m.body_j, "wheel");
            }
            _ => panic!("Expected Motor"),
        }
    }

    #[test]
    fn serde_roundtrip_motor_defaults() {
        let json = r#"{"type":"Motor","body_i":"a","body_j":"b","stall_torque":10,"no_load_speed":100}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::Motor(m) => {
                assert_abs_diff_eq!(m.direction, 1.0, epsilon = 1e-15);
            }
            _ => panic!("Expected Motor"),
        }
    }

    #[test]
    fn serde_roundtrip_linear_actuator() {
        let elem = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "piston".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "cylinder".into(),
            point_b: [0.5, 0.0],
            point_b_name: None,
            force: 1000.0,
            speed_limit: 0.5,
            stroke_min: 0.0, stroke_max: 0.0,
            end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"LinearActuator\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::LinearActuator(a) => {
                assert_abs_diff_eq!(a.force, 1000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(a.speed_limit, 0.5, epsilon = 1e-15);
                assert_eq!(a.body_a, "piston");
            }
            _ => panic!("Expected LinearActuator"),
        }
    }

    #[test]
    fn serde_roundtrip_linear_actuator_defaults() {
        let json = r#"{"type":"LinearActuator","body_a":"a","point_a":[0,0],"body_b":"b","point_b":[0,0],"force":100}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::LinearActuator(a) => {
                assert_abs_diff_eq!(a.speed_limit, 0.0, epsilon = 1e-15);
            }
            _ => panic!("Expected LinearActuator"),
        }
    }

    // ── Type name tests for new elements ─────────────────────────────────────

    #[test]
    fn type_names_for_new_elements() {
        assert_eq!(
            ForceElement::GasSpring(GasSpringElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                point_a_name: None,
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                point_b_name: None,
                initial_force: 1.0,
                extended_length: 1.0,
                stroke: 0.5,
                damping: 0.0,
                polytropic_exp: 1.0,
            })
            .type_name(),
            "Gas Spring"
        );
        assert_eq!(
            ForceElement::BearingFriction(BearingFrictionElement {
                body_i: "a".into(),
                body_j: "b".into(),
                constant_drag: 0.0,
                viscous_coeff: 0.0,
                coulomb_coeff: 0.0,
                pin_radius: 0.0,
                radial_load: 0.0,
                v_threshold: 0.01,
            })
            .type_name(),
            "Bearing Friction"
        );
        assert_eq!(
            ForceElement::JointLimit(JointLimitElement {
                body_i: "a".into(),
                body_j: "b".into(),
                angle_min: -1.0,
                angle_max: 1.0,
                stiffness: 1.0,
                damping: 0.0,
                restitution: 0.5,
            })
            .type_name(),
            "Joint Limit"
        );
        assert_eq!(
            ForceElement::Motor(MotorElement {
                body_i: "a".into(),
                body_j: "b".into(),
                stall_torque: 1.0,
                no_load_speed: 1.0,
                direction: 1.0,
            })
            .type_name(),
            "Motor"
        );
        assert_eq!(
            ForceElement::LinearActuator(LinearActuatorElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                point_a_name: None,
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                point_b_name: None,
                force: 1.0,
                speed_limit: 0.0,
                stroke_min: 0.0, stroke_max: 0.0,
                end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
            })
            .type_name(),
            "Linear Actuator"
        );
    }

    // ── Coulomb friction convenience constructor ────────────────────────────

    #[test]
    fn coulomb_friction_produces_correct_torque() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 spinning at 10 rad/s (fast enough that tanh ~ 1.0)
        q_dot[5] = 10.0;

        let elem = ForceElement::coulomb_friction("bar1", "bar2", 0.3, 0.01, 1000.0);
        let result = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // μ * R * F_n = 0.3 * 0.01 * 1000 = 3.0
        // tanh(10.0 / 0.01) ~ 1.0
        // torque on bar2 = -3.0 (opposes positive rotation)
        assert_abs_diff_eq!(result[5], -3.0, epsilon = 1e-3); // bar2 θ
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-3); // bar1 θ (reaction)

        // Verify this is a pure Coulomb element (no drag, no viscous)
        match &elem {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.constant_drag, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(b.viscous_coeff, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(b.coulomb_coeff, 0.3, epsilon = 1e-15);
            }
            _ => panic!("Expected BearingFriction variant"),
        }
    }

    // ── Time modulation tests ──────────────────────────────────────────────

    #[test]
    fn sinusoidal_modulation_external_force() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            local_point_name: None,
            force: [100.0, 0.0],
            modulation: TimeModulation::Sinusoidal {
                omega: PI,
                phase: 0.0,
            },
        });

        // At t=0: sin(0) = 0 → force = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-14);

        // At t=0.5: sin(π * 0.5) = 1.0 → force = 100
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r1[0], 100.0, epsilon = 1e-10);

        // At t=1.0: sin(π * 1.0) ~ 0 → force ~ 0
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 1.0);
        assert_abs_diff_eq!(r2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn step_modulation_external_force() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            local_point_name: None,
            force: [50.0, 0.0],
            modulation: TimeModulation::Step { t_on: 1.0 },
        });

        // Before step: t=0.5 → factor = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-15);

        // At step: t=1.0 → factor = 1
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 1.0);
        assert_abs_diff_eq!(r1[0], 50.0, epsilon = 1e-15);

        // After step: t=2.0 → factor = 1
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 2.0);
        assert_abs_diff_eq!(r2[0], 50.0, epsilon = 1e-15);
    }

    #[test]
    fn expression_modulation_evaluates_correctly() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            local_point_name: None,
            force: [100.0, 0.0],
            modulation: TimeModulation::Expression {
                expr: "sin(2*pi*t)".into(),
            },
        });

        // At t=0: sin(0) = 0 -> force = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-10);

        // At t=0.25: sin(pi/2) = 1.0 -> force = 100
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.25);
        assert_abs_diff_eq!(r1[0], 100.0, epsilon = 1e-10);

        // At t=0.5: sin(pi) ~ 0 -> force ~ 0
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r2[0], 0.0, epsilon = 1e-10);

        // At t=0.75: sin(3*pi/2) = -1.0 -> force = -100
        let r3 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.75);
        assert_abs_diff_eq!(r3[0], -100.0, epsilon = 1e-10);
    }

    #[test]
    fn expression_modulation_invalid_expr_returns_0() {
        // An invalid expression disables the force (factor = 0.0)
        let modulation = TimeModulation::Expression {
            expr: "not_a_valid_expr!!!".into(),
        };
        assert_abs_diff_eq!(modulation.factor(1.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn expression_modulation_exp_decay() {
        // Test exponential decay: 1 - exp(-t/0.5)
        let modulation = TimeModulation::Expression {
            expr: "1 - exp(-t/0.5)".into(),
        };

        // At t=0: 1 - exp(0) = 0
        assert_abs_diff_eq!(modulation.factor(0.0), 0.0, epsilon = 1e-10);

        // At t=0.5: 1 - exp(-1) ~ 0.6321
        assert_abs_diff_eq!(
            modulation.factor(0.5),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1e-10
        );

        // At large t: should approach 1.0
        assert_abs_diff_eq!(modulation.factor(10.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn serde_roundtrip_expression_modulation() {
        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0],
            local_point_name: None,
            force: [10.0, -5.0],
            modulation: TimeModulation::Expression {
                expr: "sin(2*pi*t)".into(),
            },
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"modulation_type\":\"Expression\""));
        assert!(json.contains("sin(2*pi*t)"));

        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::ExternalForce(f) => {
                match &f.modulation {
                    TimeModulation::Expression { expr } => {
                        assert_eq!(expr, "sin(2*pi*t)");
                        // Verify the deserialized expression evaluates correctly
                        assert_abs_diff_eq!(f.modulation.factor(0.25), 1.0, epsilon = 1e-10);
                    }
                    _ => panic!("Expected Expression modulation"),
                }
            }
            _ => panic!("Expected ExternalForce"),
        }
    }

    #[test]
    fn ramp_modulation_external_torque() {
        let (state, bodies) = setup_single_bar();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "bar".into(),
            torque: 20.0,
            modulation: TimeModulation::Ramp {
                t_start: 1.0,
                t_end: 3.0,
            },
        });

        // Before ramp: t=0.5 → factor = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r0[2], 0.0, epsilon = 1e-15);

        // Mid-ramp: t=2.0 → factor = (2-1)/(3-1) = 0.5 → torque = 10
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 2.0);
        assert_abs_diff_eq!(r1[2], 10.0, epsilon = 1e-10);

        // After ramp: t=4.0 → factor = 1.0 → torque = 20
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 4.0);
        assert_abs_diff_eq!(r2[2], 20.0, epsilon = 1e-15);
    }

    #[test]
    fn linear_spring_with_point_names_round_trips() {
        let spring = LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.02, 0.01],
            point_a_name: Some("M1".to_string()),
            body_b: "crank".to_string(),
            point_b: [0.0, 0.0],
            point_b_name: Some("A".to_string()),
            stiffness: 500.0,
            free_length: 0.05,
        };
        let fe = ForceElement::LinearSpring(spring);
        let json = serde_json::to_string(&fe).unwrap();
        let loaded: ForceElement = serde_json::from_str(&json).unwrap();
        if let ForceElement::LinearSpring(s) = loaded {
            assert_eq!(s.point_a_name, Some("M1".to_string()));
            assert_eq!(s.point_b_name, Some("A".to_string()));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn linear_spring_without_point_names_defaults_to_none() {
        let json = r#"{"type":"LinearSpring","body_a":"ground","point_a":[0.0,0.0],"body_b":"crank","point_b":[0.1,0.0],"stiffness":500.0,"free_length":0.05}"#;
        let loaded: ForceElement = serde_json::from_str(json).unwrap();
        if let ForceElement::LinearSpring(s) = loaded {
            assert_eq!(s.point_a_name, None);
            assert_eq!(s.point_b_name, None);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn external_force_with_local_point_name_round_trips() {
        let ef = ExternalForceElement {
            body_id: "crank".to_string(),
            local_point: [0.01, 0.0],
            local_point_name: Some("A".to_string()),
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        };
        let fe = ForceElement::ExternalForce(ef);
        let json = serde_json::to_string(&fe).unwrap();
        let loaded: ForceElement = serde_json::from_str(&json).unwrap();
        if let ForceElement::ExternalForce(e) = loaded {
            assert_eq!(e.local_point_name, Some("A".to_string()));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn resolve_named_points_caches_coordinates() {
        use crate::core::body::Body;

        let mut ground = Body::new("ground");
        ground.add_mount_point("M1", 0.02, 0.01).unwrap();

        let mut crank = Body::new("crank");
        crank.add_attachment_point("A", 0.015, 0.0).unwrap();

        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("crank".to_string(), crank);

        let spring = LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.0, 0.0],
            point_a_name: Some("M1".to_string()),
            body_b: "crank".to_string(),
            point_b: [0.0, 0.0],
            point_b_name: Some("A".to_string()),
            stiffness: 500.0,
            free_length: 0.05,
        };

        let fe = ForceElement::LinearSpring(spring);
        let resolved = fe.resolve_named_points(&bodies).unwrap();

        if let ForceElement::LinearSpring(s) = &resolved {
            assert_abs_diff_eq!(s.point_a[0], 0.02, epsilon = 1e-15);
            assert_abs_diff_eq!(s.point_a[1], 0.01, epsilon = 1e-15);
            assert_abs_diff_eq!(s.point_b[0], 0.015, epsilon = 1e-15);
            assert_abs_diff_eq!(s.point_b[1], 0.0, epsilon = 1e-15);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn resolve_named_points_none_preserves_raw_coords() {
        let bodies = HashMap::new();
        let spring = LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.05, 0.03],
            point_a_name: None,
            body_b: "crank".to_string(),
            point_b: [0.01, 0.0],
            point_b_name: None,
            stiffness: 500.0,
            free_length: 0.05,
        };
        let fe = ForceElement::LinearSpring(spring);
        let resolved = fe.resolve_named_points(&bodies).unwrap();
        if let ForceElement::LinearSpring(s) = &resolved {
            assert_abs_diff_eq!(s.point_a[0], 0.05, epsilon = 1e-15);
            assert_abs_diff_eq!(s.point_b[0], 0.01, epsilon = 1e-15);
        } else {
            panic!("wrong variant");
        }
    }
}
