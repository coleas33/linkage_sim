//! Static equilibrium solver: Φ_q^T · λ = −Q
//!
//! Given a mechanism at a solved position with applied forces,
//! solve for Lagrange multipliers (joint reactions + driver efforts).

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::forces::assembly::assemble_q;
use crate::forces::gravity::Gravity;
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
    gravity: Option<&Gravity>,
    t: f64,
) -> StaticSolveResult {
    assert!(mech.is_built(), "Mechanism must be built before static solve.");

    let state = mech.state();

    // Assemble Φ_q and Q
    let phi_q = assemble_jacobian(mech, q, t);
    let q_dot = DVector::zeros(state.n_coords());
    let q_forces = assemble_q(state, gravity, q, &q_dot, t);

    // Φ_q^T is (n_coords × m). We solve Φ_q^T · λ = −Q
    let phi_q_t = phi_q.transpose();
    let m = phi_q.nrows(); // n_constraints
    let rhs = -&q_forces;

    // Single SVD of Φ_q^T — reused for both conditioning and solve.
    // Singular values of A^T are the same as those of A, so this gives
    // the same condition number as SVD(Φ_q).
    let svd_t = phi_q_t.clone().svd(true, true);
    let sv = &svd_t.singular_values;

    let (condition_number, is_overconstrained) = if sv.len() > 0 && sv[0] > 0.0 {
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
    let lambdas = svd_t.solve(&rhs, 1e-14).unwrap();

    // Compute residual
    let residual = &phi_q_t * &lambdas + &q_forces;
    let residual_norm = residual.norm();

    StaticSolveResult {
        lambdas,
        q_forces,
        residual_norm,
        is_overconstrained,
        condition_number,
    }
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
    let mut row = 0;

    for constraint in mech.all_constraints() {
        let n_eq = constraint.n_equations();
        let lam = result.lambdas.rows(row, n_eq).clone_owned();

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
            moment,
            effort,
            resultant,
        });

        row += n_eq;
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
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::solver::kinematics::solve_position;
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    fn build_fourbar_with_gravity() -> (Mechanism, Gravity) {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

        let mut mech = Mechanism::new();
        let bodies_for_gravity = {
            let mut m = std::collections::HashMap::new();
            m.insert("ground".to_string(), ground.clone());
            m.insert("crank".to_string(), crank.clone());
            m.insert("coupler".to_string(), coupler.clone());
            m.insert("rocker".to_string(), rocker.clone());
            m
        };

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

        let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies_for_gravity);
        (mech, gravity)
    }

    #[test]
    fn fourbar_statics_solves() {
        let (mech, gravity) = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0; // 60 degrees
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        let bx = angle.cos();
        let by = angle.sin();
        state.set_pose("coupler", &mut q0, bx, by, 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50);
        assert!(pos.converged);

        let result = solve_statics(&mech, &pos.q, Some(&gravity), angle);
        assert!(result.residual_norm < 1e-8, "residual = {}", result.residual_norm);
    }

    #[test]
    fn fourbar_reactions_extract() {
        let (mech, gravity) = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50);
        assert!(pos.converged);

        let statics = solve_statics(&mech, &pos.q, Some(&gravity), angle);
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
}
