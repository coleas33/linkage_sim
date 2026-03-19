//! Static equilibrium solver: Φ_q^T · λ = −Q
//!
//! Given a mechanism at a solved position with applied forces,
//! solve for Lagrange multipliers (joint reactions + driver efforts).

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::assemble_jacobian;

/// Result of a static force solve.
#[derive(Debug, Clone)]
pub struct StaticSolveResult {
    /// Lagrange multiplier vector (m,).
    pub lambdas: DVector<f64>,
    /// Assembled generalized force (n_coords,).
    pub q_forces: DVector<f64>,
    /// Residual: ‖Φ_q^T · λ + Q‖.
    pub residual_norm: f64,
    /// True if pseudoinverse was used (overconstrained system).
    pub is_overconstrained: bool,
    /// Condition number of Φ_q^T.
    pub condition_number: f64,
}

/// Solve static equilibrium: Φ_q^T · λ = −Q for λ.
///
/// The multipliers represent:
/// - For joint constraints: reaction forces at that joint
/// - For driver constraints: required input torque/force
pub fn solve_statics(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
) -> Result<StaticSolveResult, LinkageError> {
    if !mech.is_built() {
        return Err(LinkageError::MechanismNotBuilt);
    }

    let state = mech.state();

    // Assemble Φ_q and Q
    let phi_q = assemble_jacobian(mech, q, t);
    let q_dot = DVector::zeros(state.n_coords());
    let q_forces = mech.assemble_forces(q, &q_dot, t);

    // Φ_q^T is (n_coords × m). We solve Φ_q^T · λ = −Q
    let phi_q_t = phi_q.transpose();
    let m = phi_q.nrows(); // n_constraints
    let rhs = -&q_forces;

    // Single SVD of Φ_q^T — reused for both conditioning and solve.
    // Singular values of A^T are the same as those of A, so this gives
    // the same condition number as SVD(Φ_q).
    let svd_t = phi_q_t.clone().svd(true, true);
    let sv = &svd_t.singular_values;

    let (condition_number, is_overconstrained) = if !sv.is_empty() && sv[0] > 0.0 {
        let rank_tol = 1e-10 * sv[0];
        let rank = sv.iter().filter(|&&s| s > rank_tol).count();
        let sigma_min = if rank > 0 {
            sv[rank.min(sv.len()) - 1]
        } else {
            0.0
        };
        let cond = if sigma_min > 0.0 {
            sv[0] / sigma_min
        } else {
            f64::INFINITY
        };
        (cond, rank < m)
    } else {
        (f64::INFINITY, true)
    };

    // Solve Φ_q^T · λ = rhs using the already-computed SVD
    let lambdas = svd_t
        .solve(&rhs, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;

    // Compute residual
    let residual = &phi_q_t * &lambdas + &q_forces;
    let residual_norm = residual.norm();

    Ok(StaticSolveResult {
        lambdas,
        q_forces,
        residual_norm,
        is_overconstrained,
        condition_number,
    })
}

/// Transform a global-frame force vector to body-local frame.
///
/// Applies the inverse rotation: F_local = A(θ)^T · F_global
pub fn reaction_to_local(force_global: [f64; 2], theta: f64) -> [f64; 2] {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    [
        cos_t * force_global[0] + sin_t * force_global[1],
        -sin_t * force_global[0] + cos_t * force_global[1],
    ]
}

/// Reaction data for a single joint or driver.
#[derive(Debug, Clone)]
pub struct JointReaction {
    pub joint_id: String,
    pub body_i_id: String,
    pub body_j_id: String,
    pub n_equations: usize,
    /// Raw multiplier values for this joint.
    pub lambdas: DVector<f64>,
    /// Reaction force in global frame [Fx, Fy].
    pub force_global: [f64; 2],
    /// Reaction force in body-i local frame [Fx_local, Fy_local].
    pub force_local_i: [f64; 2],
    /// Reaction force in body-j local frame [Fx_local, Fy_local].
    pub force_local_j: [f64; 2],
    /// Reaction moment Mz (N·m) — nonzero only for fixed joints.
    pub moment: f64,
    /// For drivers: required input torque/force.
    pub effort: f64,
    /// Magnitude of reaction force.
    pub resultant: f64,
}

/// Extract individual joint reactions from a static solve result.
///
/// Maps rows of the λ vector back to each joint/driver constraint.
pub fn extract_reactions(mech: &Mechanism, result: &StaticSolveResult) -> Vec<JointReaction> {
    let mut reactions = Vec::new();

    for (constraint, range) in mech.all_constraints().iter().zip(mech.constraint_ranges()) {
        let n_eq = range.n_equations;
        let lam = result.lambdas.rows(range.row_start, n_eq).clone_owned();

        let (force_global, moment, effort, resultant) = match n_eq {
            2 => {
                // Revolute or prismatic: [Fx, Fy]
                let fx = lam[0];
                let fy = lam[1];
                let res = (fx * fx + fy * fy).sqrt();
                ([fx, fy], 0.0, 0.0, res)
            }
            3 => {
                // Fixed joint: [Fx, Fy, Mz]
                let fx = lam[0];
                let fy = lam[1];
                let mz = lam[2];
                let res = (fx * fx + fy * fy).sqrt();
                ([fx, fy], mz, 0.0, res)
            }
            1 => {
                // Driver: scalar effort
                ([0.0, 0.0], 0.0, lam[0], 0.0)
            }
            _ => ([0.0, 0.0], 0.0, 0.0, 0.0),
        };

        reactions.push(JointReaction {
            joint_id: constraint.id().to_string(),
            body_i_id: constraint.body_i_id().to_string(),
            body_j_id: constraint.body_j_id().to_string(),
            n_equations: n_eq,
            lambdas: lam,
            force_global,
            force_local_i: [0.0, 0.0],
            force_local_j: [0.0, 0.0],
            moment,
            effort,
            resultant,
        });
    }

    reactions
}

/// Extract reactions with local-frame forces.
///
/// Like `extract_reactions` but also computes body-local force vectors
/// by rotating each reaction force through the inverse body rotation.
pub fn extract_reactions_with_local(
    mech: &Mechanism,
    result: &StaticSolveResult,
    q: &DVector<f64>,
) -> Vec<JointReaction> {
    let mut reactions = extract_reactions(mech, result);
    let state = mech.state();
    for reaction in &mut reactions {
        let theta_i = state.get_angle(&reaction.body_i_id, q);
        let theta_j = state.get_angle(&reaction.body_j_id, q);
        reaction.force_local_i = reaction_to_local(reaction.force_global, theta_i);
        reaction.force_local_j = reaction_to_local(reaction.force_global, theta_j);
    }
    reactions
}

/// Filter reactions to only driver constraints (n_equations == 1).
pub fn get_driver_reactions(reactions: &[JointReaction]) -> Vec<&JointReaction> {
    reactions.iter().filter(|r| r.n_equations == 1).collect()
}

/// Filter reactions to only joint constraints (n_equations > 1).
pub fn get_joint_reactions(reactions: &[JointReaction]) -> Vec<&JointReaction> {
    reactions.iter().filter(|r| r.n_equations > 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::elements::{ForceElement, GravityElement};
    use crate::solver::kinematics::solve_position;
    use std::f64::consts::PI;

    fn build_fourbar_with_gravity() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
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
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
    }

    #[test]
    fn fourbar_statics_solves() {
        let mech = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0; // 60 degrees
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        let bx = angle.cos();
        let by = angle.sin();
        state.set_pose("coupler", &mut q0, bx, by, 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged);

        let result = solve_statics(&mech, &pos.q, angle).unwrap();
        assert!(result.residual_norm < 1e-8, "residual = {}", result.residual_norm);
    }

    #[test]
    fn fourbar_reactions_extract() {
        let mech = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged);

        let statics = solve_statics(&mech, &pos.q, angle).unwrap();
        let reactions = extract_reactions(&mech, &statics);

        // 4 joints + 1 driver = 5 reactions
        assert_eq!(reactions.len(), 5);

        // Last one is driver
        let drivers = get_driver_reactions(&reactions);
        assert_eq!(drivers.len(), 1);
        assert_eq!(drivers[0].joint_id, "D1");
        // Driver effort should be finite
        assert!(drivers[0].effort.is_finite());

        // Joint reactions should have nonzero resultant (gravity loading)
        let joints = get_joint_reactions(&reactions);
        assert_eq!(joints.len(), 4);
        for jr in &joints {
            assert!(jr.resultant >= 0.0);
            assert!(jr.force_global[0].is_finite());
            assert!(jr.force_global[1].is_finite());
        }
    }

    #[test]
    fn reaction_to_local_at_zero_angle_matches_global() {
        // At θ=0 the body-local frame coincides with global, so local == global
        let force_global = [10.0, -5.0];
        let local = reaction_to_local(force_global, 0.0);
        assert_abs_diff_eq!(local[0], force_global[0], epsilon = 1e-14);
        assert_abs_diff_eq!(local[1], force_global[1], epsilon = 1e-14);
    }

    #[test]
    fn reaction_to_local_at_90_degrees_rotates_correctly() {
        // At θ=π/2 the rotation matrix is [0 1; -1 0]^T = [0 -1; 1 0]
        // F_local = A^T · F_global = [cos sin; -sin cos] · F_global
        // With θ=π/2: [0 1; -1 0] · [10, 0] = [0, -10]
        let force_global = [10.0, 0.0];
        let local = reaction_to_local(force_global, PI / 2.0);
        assert_abs_diff_eq!(local[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(local[1], -10.0, epsilon = 1e-14);

        // General case: [1, 1] at θ=π/2 → [cos(π/2)*1+sin(π/2)*1, -sin(π/2)*1+cos(π/2)*1] = [1, -1]
        let force2 = [1.0, 1.0];
        let local2 = reaction_to_local(force2, PI / 2.0);
        assert_abs_diff_eq!(local2[0], 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(local2[1], -1.0, epsilon = 1e-14);
    }

    #[test]
    fn extract_reactions_with_local_populates_local_forces() {
        let mech = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged);

        let statics = solve_statics(&mech, &pos.q, angle).unwrap();
        let reactions = extract_reactions_with_local(&mech, &statics, &pos.q);

        // Verify local forces are populated and have same magnitude as global
        let joints = get_joint_reactions(&reactions);
        for jr in &joints {
            let mag_global = (jr.force_global[0].powi(2) + jr.force_global[1].powi(2)).sqrt();
            let mag_local_i = (jr.force_local_i[0].powi(2) + jr.force_local_i[1].powi(2)).sqrt();
            let mag_local_j = (jr.force_local_j[0].powi(2) + jr.force_local_j[1].powi(2)).sqrt();
            // Rotation preserves magnitude
            assert_abs_diff_eq!(mag_local_i, mag_global, epsilon = 1e-10);
            assert_abs_diff_eq!(mag_local_j, mag_global, epsilon = 1e-10);
        }
    }
}
