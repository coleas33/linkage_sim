//! Energy balance tracking for mechanism dynamics.
//!
//! Computes kinetic energy, gravitational potential energy, and total
//! mechanical energy for energy conservation verification.
//!
//! KE = 0.5 * sum_i (m_i * v_cg_i^2 + Izz_cg_i * theta_dot_i^2)
//! PE = sum_i (m_i * g * y_cg_i)

use std::collections::HashMap;

use nalgebra::DVector;

use crate::core::body::Body;
use crate::core::mechanism::Mechanism;
use crate::core::state::{State, GROUND_ID};

/// Energy components at one instant.
#[derive(Debug, Clone)]
pub struct EnergyState {
    /// Translational + rotational kinetic energy (J).
    pub kinetic: f64,
    /// Gravitational PE relative to y=0 (J).
    pub potential_gravity: f64,
    /// KE + PE_gravity.
    pub total: f64,
}

/// Compute total kinetic energy for the mechanism.
///
/// KE = sum_i 0.5 * (m_i * |v_cg|^2 + Izz_cg * theta_dot^2)
///
/// This computes KE directly from body velocities rather than using
/// a mass matrix, which avoids depending on the mass matrix assembly
/// module.
pub fn compute_kinetic_energy(
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> f64 {
    let mut ke = 0.0;

    for (body_id, body) in bodies {
        if body_id == GROUND_ID || body.mass <= 0.0 {
            continue;
        }

        let Ok(idx) = state.get_index(body_id) else { continue; };

        // CG velocity in global frame
        let v_cg = state.body_point_velocity(body_id, &body.cg_local, q, q_dot);
        let v_sq = v_cg.x * v_cg.x + v_cg.y * v_cg.y;

        // Angular velocity
        let theta_dot = q_dot[idx.theta_idx()];

        // KE = 0.5 * m * |v_cg|^2 + 0.5 * Izz_cg * theta_dot^2
        ke += 0.5 * body.mass * v_sq + 0.5 * body.izz_cg * theta_dot * theta_dot;
    }

    ke
}

/// Compute gravitational potential energy: PE = sum m_i * g * y_cg_i.
///
/// Reference: y = 0.
pub fn compute_gravity_pe(
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
    g_magnitude: f64,
) -> f64 {
    let mut pe = 0.0;

    for (body_id, body) in bodies {
        if body_id == GROUND_ID || body.mass <= 0.0 {
            continue;
        }

        let r_cg = state.body_point_global(body_id, &body.cg_local, q);
        pe += body.mass * g_magnitude * r_cg.y;
    }

    pe
}

/// Compute all energy components at one instant.
pub fn compute_energy_state(
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    g_magnitude: f64,
) -> EnergyState {
    let ke = compute_kinetic_energy(state, bodies, q, q_dot);
    let pe = compute_gravity_pe(state, bodies, q, g_magnitude);
    EnergyState {
        kinetic: ke,
        potential_gravity: pe,
        total: ke + pe,
    }
}

/// Convenience: compute energy state from a Mechanism directly.
pub fn compute_energy_state_mech(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    g_magnitude: f64,
) -> EnergyState {
    compute_energy_state(mech.state(), mech.bodies(), q, q_dot, g_magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn single_pendulum_setup() -> (State, HashMap<String, Body>) {
        let mut state = State::new();
        state.register_body("pendulum").unwrap();

        let ground = make_ground(&[("O", 0.0, 0.0)]);
        // pendulum: length=1m, mass=1kg, Izz about CG
        let pendulum = make_bar("pendulum", "A", "B", 1.0, 1.0, 1.0 / 12.0);

        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("pendulum".to_string(), pendulum);

        (state, bodies)
    }

    #[test]
    fn kinetic_energy_at_rest_is_zero() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        state.set_pose("pendulum", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(3);

        let ke = compute_kinetic_energy(&state, &bodies, &q, &q_dot);
        assert_abs_diff_eq!(ke, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn kinetic_energy_translating_body() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        state.set_pose("pendulum", &mut q, 0.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(3);
        // Pure translation at 2 m/s in x
        q_dot[0] = 2.0;

        let ke = compute_kinetic_energy(&state, &bodies, &q, &q_dot);
        // KE = 0.5 * 1.0 * (2^2) = 2.0
        // But CG is at (0.5, 0) local, and there's coupling.
        // At theta=0, B(0)*(0.5,0) = (0, 0.5)
        // v_cg = (2.0, 0) + (0, 0.5) * 0 = (2.0, 0)
        // KE = 0.5 * 1.0 * 4.0 = 2.0
        assert_abs_diff_eq!(ke, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn kinetic_energy_rotating_body() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        state.set_pose("pendulum", &mut q, 0.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(3);
        // Pure rotation at 3 rad/s about body origin
        q_dot[2] = 3.0;

        let ke = compute_kinetic_energy(&state, &bodies, &q, &q_dot);
        // CG at (0.5, 0) local, theta_dot = 3
        // B(0)*(0.5,0) = (0, 0.5)
        // v_cg = (0, 0) + (0, 0.5) * 3 = (0, 1.5)
        // KE_trans = 0.5 * 1.0 * 2.25 = 1.125
        // KE_rot = 0.5 * (1/12) * 9 = 0.375
        // KE_total = 1.5
        assert_abs_diff_eq!(ke, 1.5, epsilon = 1e-14);
    }

    #[test]
    fn gravity_pe_at_height() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        // Place pendulum origin at y=5, angle=0 => CG at (0.5, 5.0)
        state.set_pose("pendulum", &mut q, 0.0, 5.0, 0.0);

        let pe = compute_gravity_pe(&state, &bodies, &q, 9.81);
        // PE = 1.0 * 9.81 * 5.0 = 49.05
        assert_abs_diff_eq!(pe, 49.05, epsilon = 1e-10);
    }

    #[test]
    fn gravity_pe_at_y_zero_is_zero() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        // Place pendulum so CG is at y=0: origin at (0, 0), angle=0 => CG at (0.5, 0)
        state.set_pose("pendulum", &mut q, 0.0, 0.0, 0.0);

        let pe = compute_gravity_pe(&state, &bodies, &q, 9.81);
        assert_abs_diff_eq!(pe, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn energy_conservation_pendulum_top_vs_bottom() {
        // A pendulum released from horizontal position:
        // At top (theta=0, horizontal): KE=0, PE = m*g*y_cg
        // At bottom (theta=-90 deg, vertical): KE = PE_initial, PE_bottom
        // Total energy should be conserved.
        let (state, bodies) = single_pendulum_setup();

        // Configuration 1: horizontal, at rest
        let mut q_top = state.make_q();
        state.set_pose("pendulum", &mut q_top, 0.0, 0.0, 0.0);
        // CG is at (0.5, 0) => PE = 0
        let q_dot_top = DVector::zeros(3);

        let energy_top = compute_energy_state(&state, &bodies, &q_top, &q_dot_top, 9.81);
        assert_abs_diff_eq!(energy_top.kinetic, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(energy_top.potential_gravity, 0.0, epsilon = 1e-15);

        // Configuration 2: vertical down, CG at (0, -0.5)
        let mut q_bottom = state.make_q();
        state.set_pose("pendulum", &mut q_bottom, 0.0, 0.0, -PI / 2.0);
        // CG at A(theta)*(0.5,0) = (0, -0.5) for theta=-pi/2
        // PE = 1.0 * 9.81 * (-0.5) = -4.905

        let pe_bottom = compute_gravity_pe(&state, &bodies, &q_bottom, 9.81);
        assert_abs_diff_eq!(pe_bottom, -4.905, epsilon = 1e-10);

        // If the pendulum falls from horizontal to vertical (no friction),
        // KE_bottom = -PE_bottom = 4.905 (energy conservation)
        // total_bottom = KE_bottom + PE_bottom = 0 = total_top
        let mut q_dot_bottom = DVector::zeros(3);
        // We need theta_dot such that KE = 4.905
        // KE = 0.5 * m * |v_cg|^2 + 0.5 * Izz * theta_dot^2
        // v_cg from rotation about origin: v_cg = B*s_cg * theta_dot
        // At theta=-pi/2: B(-pi/2)*(0.5,0) = (sin(pi/2)*0.5, cos(-pi/2)*0.5) = ...
        // B(theta) = [[-sin theta, -cos theta], [cos theta, -sin theta]]
        // B(-pi/2) = [[sin(pi/2), -cos(pi/2)], [cos(-pi/2), sin(pi/2)]]
        //          = [[1, 0], [0, 1]]
        // Wait: B(theta)*(0.5,0): x_comp = -sin(theta)*0.5, y_comp = cos(theta)*0.5
        // At theta=-pi/2: x = -sin(-pi/2)*0.5 = 0.5, y = cos(-pi/2)*0.5 = 0
        // v_cg = (0.5*w, 0) if only rotating
        // |v_cg|^2 = 0.25*w^2
        // KE = 0.5*1*0.25*w^2 + 0.5*(1/12)*w^2 = (0.125 + 1/24)*w^2 = (1/6)*w^2
        // Wait, let me just compute:
        // (3/24 + 1/24) = 4/24 = 1/6
        // 1/6 * w^2 = 4.905 => w^2 = 29.43 => w = 5.4249...
        // Actually for this test we just verify the energy state struct works
        let omega = (4.905 * 6.0_f64).sqrt();
        q_dot_bottom[2] = omega;

        let energy_bottom =
            compute_energy_state(&state, &bodies, &q_bottom, &q_dot_bottom, 9.81);

        assert_abs_diff_eq!(
            energy_top.total,
            energy_bottom.total,
            epsilon = 1e-10
        );
    }

    #[test]
    fn energy_state_total_is_sum() {
        let (state, bodies) = single_pendulum_setup();
        let mut q = state.make_q();
        state.set_pose("pendulum", &mut q, 0.0, 3.0, 0.5);
        let mut q_dot = DVector::zeros(3);
        q_dot[0] = 1.0;
        q_dot[2] = 2.0;

        let es = compute_energy_state(&state, &bodies, &q, &q_dot, 9.81);
        assert_abs_diff_eq!(es.total, es.kinetic + es.potential_gravity, epsilon = 1e-15);
    }
}
