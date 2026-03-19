//! Virtual work cross-check for static/inverse dynamics solver validation.
//!
//! For a 1-DOF mechanism, the required input torque can be computed
//! independently via the virtual work principle:
//!
//! ```text
//! tau_input * delta_theta_input = -sum(Q_i * delta_q_i)
//! ```
//!
//! where delta_q is the virtual displacement from the velocity solution.
//! This provides a cross-check against the Lagrange multiplier result.

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::kinematics::solve_velocity;

/// Virtual work cross-check result.
#[derive(Debug, Clone)]
pub struct VirtualWorkResult {
    /// Input torque computed via virtual work principle (N*m).
    pub input_torque: f64,
    /// Input torque from the Lagrange multiplier (for comparison) (N*m).
    pub lagrange_torque: f64,
    /// Relative error between the two methods.
    pub relative_error: f64,
    /// Whether the two methods agree within the specified tolerance.
    pub agrees: bool,
}

/// Compute input torque via virtual work and compare with Lagrange multiplier result.
///
/// Virtual work principle at static equilibrium:
///     tau_input * omega_input + Q^T * q_dot = 0
/// So:
///     tau_input = -Q^T * q_dot / omega_input
///
/// where Q is the applied generalized force vector (excluding driver reactions)
/// and q_dot is the velocity solution from the kinematic constraints.
///
/// # Arguments
/// * `mech` - A built Mechanism with at least one driver.
/// * `q` - Solved configuration vector.
/// * `t` - Time at which to evaluate.
/// * `lagrange_torque` - Driver torque from the Lagrange multiplier solver.
/// * `tol` - Tolerance for the agreement check.
///
/// # Errors
/// Returns `LinkageError::MechanismNotBuilt` if the mechanism has not been built.
pub fn virtual_work_check(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    lagrange_torque: f64,
    tol: f64,
) -> Result<VirtualWorkResult, LinkageError> {
    if !mech.is_built() {
        return Err(LinkageError::MechanismNotBuilt);
    }

    // 1. Solve velocity: Phi_q * q_dot = -Phi_t
    let q_dot = solve_velocity(mech, q, t)?;

    // 2. Assemble applied forces Q (with zero velocity for static analysis).
    let q_dot_zero = DVector::zeros(q.len());
    let q_forces = mech.assemble_forces(q, &q_dot_zero, t);

    // 3. Get input angular velocity from the driver body pair.
    let (body_i, body_j) = mech
        .driver_body_pair()
        .ok_or(LinkageError::MechanismNotBuilt)?;

    let state = mech.state();

    let omega_i = if state.is_ground(body_i) {
        0.0
    } else {
        let idx = state
            .get_index(body_i)
            .map_err(|_| LinkageError::MechanismNotBuilt)?;
        q_dot[idx.theta_idx()]
    };
    let omega_j = if state.is_ground(body_j) {
        0.0
    } else {
        let idx = state
            .get_index(body_j)
            .map_err(|_| LinkageError::MechanismNotBuilt)?;
        q_dot[idx.theta_idx()]
    };
    let omega_input = omega_j - omega_i;

    // At zero input speed, virtual work cannot determine torque.
    if omega_input.abs() < 1e-15 {
        return Ok(VirtualWorkResult {
            input_torque: f64::NAN,
            lagrange_torque,
            relative_error: f64::NAN,
            agrees: false,
        });
    }

    // 4. Virtual work: P = Q^T * q_dot, then tau_input = -P / omega_input
    let power = q_forces.dot(&q_dot);
    let vw_torque = -power / omega_input;

    let rel_err = if lagrange_torque.abs() > 1e-15 {
        (vw_torque - lagrange_torque).abs() / lagrange_torque.abs()
    } else {
        (vw_torque - lagrange_torque).abs()
    };

    Ok(VirtualWorkResult {
        input_torque: vw_torque,
        lagrange_torque,
        relative_error: rel_err,
        agrees: rel_err < tol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::elements::{ForceElement, GravityElement};
    use crate::solver::kinematics::solve_position;
    use crate::solver::statics::solve_statics;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Build a standard 4-bar with gravity for virtual work testing.
    ///
    /// Link lengths: crank=1, coupler=3, rocker=2, ground=4
    /// All links have mass=1 kg, Izz_cg=1/12 (unit bar approximation).
    fn build_fourbar_with_gravity() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 1.0, 1.0 / 12.0);
        let coupler = make_bar("coupler", "B", "C", 3.0, 1.0, 1.0 / 12.0);
        let rocker = make_bar("rocker", "D", "C", 2.0, 1.0, 1.0 / 12.0);

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

    /// Solve position for the 4-bar at a given crank angle.
    fn solve_at_angle(mech: &Mechanism, angle: f64) -> DVector<f64> {
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        let bx = angle.cos();
        let by = angle.sin();
        state.set_pose("coupler", &mut q0, bx, by, 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let result = solve_position(mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Position solve failed at angle={} (residual={})",
            angle,
            result.residual_norm
        );
        result.q
    }

    #[test]
    fn virtual_work_agrees_with_lagrange_at_60_deg() {
        let mech = build_fourbar_with_gravity();
        let angle = PI / 3.0;
        let q = solve_at_angle(&mech, angle);

        // Get Lagrange multiplier torque from statics.
        let statics = solve_statics(&mech, &q, angle).unwrap();
        // Driver is the last constraint; its multiplier is the driver torque.
        let n_lam = statics.lambdas.len();
        let lagrange_torque = statics.lambdas[n_lam - 1];

        let result = virtual_work_check(&mech, &q, angle, lagrange_torque, 1e-6).unwrap();

        assert!(
            result.agrees,
            "Virtual work torque ({:.6}) and Lagrange torque ({:.6}) disagree: rel_err={:.2e}",
            result.input_torque,
            result.lagrange_torque,
            result.relative_error
        );
        assert_abs_diff_eq!(result.input_torque, lagrange_torque, epsilon = 1e-4);
    }

    #[test]
    fn virtual_work_agrees_at_multiple_angles() {
        let mech = build_fourbar_with_gravity();

        // Test at several angles throughout the rotation.
        for &angle_deg in &[30.0_f64, 90.0, 150.0, 210.0, 300.0] {
            let angle = angle_deg.to_radians();
            let q = solve_at_angle(&mech, angle);

            let statics = solve_statics(&mech, &q, angle).unwrap();
            let n_lam = statics.lambdas.len();
            let lagrange_torque = statics.lambdas[n_lam - 1];

            let result = virtual_work_check(&mech, &q, angle, lagrange_torque, 1e-4).unwrap();

            assert!(
                result.agrees,
                "Disagreement at {}deg: vw={:.6}, lagrange={:.6}, rel_err={:.2e}",
                angle_deg,
                result.input_torque,
                result.lagrange_torque,
                result.relative_error
            );
        }
    }

    #[test]
    fn virtual_work_returns_nan_at_zero_input_speed() {
        // Build a mechanism where the driver prescribes constant angle
        // (omega_input = 0). Virtual work cannot determine torque.
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 1.0, 1.0 / 12.0);
        let coupler = make_bar("coupler", "B", "C", 3.0, 1.0, 1.0 / 12.0);
        let rocker = make_bar("rocker", "D", "C", 2.0, 1.0, 1.0 / 12.0);

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

        // Driver that prescribes a constant angle (f'(t) = 0).
        let fixed_angle = PI / 3.0;
        mech.add_revolute_driver(
            "D1",
            "ground",
            "crank",
            move |_t| fixed_angle,
            |_t| 0.0,
            |_t| 0.0,
        )
        .unwrap();

        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();

        let q = solve_at_angle(&mech, fixed_angle);
        let result = virtual_work_check(&mech, &q, 0.0, 5.0, 1e-6).unwrap();

        assert!(result.input_torque.is_nan());
        assert!(!result.agrees);
    }
}
