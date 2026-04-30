//! Unified error types for the linkage solver public APIs.

use thiserror::Error;

/// Errors returned by the public solver APIs.
///
/// Variants are grouped into two categories:
/// - **Model errors**: caused by user input (e.g., calling a solver on an unbuilt mechanism).
/// - **Numerical errors**: caused by solver failures (e.g., SVD did not converge).
#[derive(Debug, Error)]
pub enum LinkageError {
    // -- Model errors (user input) --
    /// The mechanism has not been built (call `mech.build()` first).
    #[error("Mechanism must be built before solving.")]
    MechanismNotBuilt,

    // -- Numerical errors (solver failures) --
    /// The position Newton-Raphson iteration did not converge.
    #[error(
        "Position solve did not converge after {iterations} iterations (residual = {residual:.2e})."
    )]
    PositionSolveNotConverged {
        /// Number of iterations performed.
        iterations: usize,
        /// Final ||Phi(q, t)|| at the returned q.
        residual: f64,
    },

    /// The constraint Jacobian is singular (or near-singular).
    #[error("Singular Jacobian (condition number = {condition:.2e}).")]
    SingularJacobian {
        /// Condition number of the Jacobian.
        condition: f64,
    },

    /// The mass matrix is singular (zero or negative mass body).
    #[error("Singular mass matrix.")]
    SingularMassMatrix,

    /// SVD solve failed (nalgebra returned Err).
    #[error("SVD solve failed.")]
    SvdSolveFailed,

    // -- Forward dynamics --
    /// Time integration failed.
    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    /// Constraint drift exceeded the acceptable tolerance.
    #[error(
        "Constraint drift exceeded tolerance (drift = {drift:.2e}, tolerance = {tolerance:.2e})."
    )]
    ConstraintDriftExceeded {
        /// Observed constraint drift.
        drift: f64,
        /// Configured tolerance.
        tolerance: f64,
    },

    // -- Trajectory-mode failures --
    /// Target value is outside the reachable workspace.
    #[error(
        "Trajectory unreachable at target = {target:.4} \
         (workspace [{min:.4}, {max:.4}], closest reachable = {achieved_clamp:.4})."
    )]
    TrajectoryUnreachable {
        target: f64,
        achieved_clamp: f64,
        min: f64,
        max: f64,
    },

    /// `|dg/du|` fell below the singularity threshold.
    #[error("Trajectory singular: |dg/du| = {dg_du:.2e} below threshold.")]
    TrajectorySingular { dg_du: f64 },

    /// Trajectory crossed an assembly-mode boundary mid-solve.
    #[error("Trajectory branch jump: ‖Δq‖ = {delta_q_norm:.4} exceeded threshold.")]
    TrajectoryBranchJump { delta_q_norm: f64 },

    /// Inverse Newton did not converge in `max_iter` iterations.
    #[error(
        "Trajectory inverse Newton did not converge after {iterations} iterations \
         (residual = {residual:.2e})."
    )]
    TrajectoryNonConvergent { iterations: usize, residual: f64 },
}

impl From<crate::solver::inverse_kinematics::InverseSolveStatus> for LinkageError {
    fn from(status: crate::solver::inverse_kinematics::InverseSolveStatus) -> Self {
        use crate::solver::inverse_kinematics::InverseSolveStatus;
        match status {
            InverseSolveStatus::Converged => {
                unreachable!(
                    "LinkageError::from(InverseSolveStatus::Converged) — caller bug; \
                     convert only failure variants"
                );
            }
            InverseSolveStatus::Reachability {
                target,
                achieved_clamp,
                workspace_min,
                workspace_max,
            } => LinkageError::TrajectoryUnreachable {
                target,
                achieved_clamp,
                min: workspace_min.unwrap_or(f64::NEG_INFINITY),
                max: workspace_max.unwrap_or(f64::INFINITY),
            },
            InverseSolveStatus::Singularity { dg_du } => {
                LinkageError::TrajectorySingular { dg_du }
            }
            InverseSolveStatus::BranchJump { delta_q_norm } => {
                LinkageError::TrajectoryBranchJump { delta_q_norm }
            }
            InverseSolveStatus::NonConvergent {
                iterations,
                residual,
            } => LinkageError::TrajectoryNonConvergent {
                iterations,
                residual,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse_kinematics::InverseSolveStatus;

    #[test]
    fn from_inverse_solve_status_to_linkage_error() {
        let status = InverseSolveStatus::Reachability {
            target: 1.0,
            achieved_clamp: 0.5,
            workspace_min: Some(0.0),
            workspace_max: Some(0.5),
        };
        let err: LinkageError = status.into();
        match err {
            LinkageError::TrajectoryUnreachable { target, .. } => {
                assert_eq!(target, 1.0);
            }
            _ => panic!("expected TrajectoryUnreachable"),
        }
    }

    #[test]
    fn from_singularity_status() {
        let status = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let err: LinkageError = status.into();
        assert!(matches!(err, LinkageError::TrajectorySingular { .. }));
    }

    #[test]
    #[should_panic(expected = "caller bug")]
    fn from_converged_status_panics_as_contract_violation() {
        let status = InverseSolveStatus::Converged;
        let _err: LinkageError = status.into();
    }

    #[test]
    fn unreachable_error_message_includes_achieved_clamp() {
        let status = InverseSolveStatus::Reachability {
            target: 0.092,
            achieved_clamp: 0.087,
            workspace_min: Some(-0.05),
            workspace_max: Some(0.087),
        };
        let err: LinkageError = status.into();
        let msg = format!("{}", err);
        assert!(msg.contains("0.0920"), "missing target in: {}", msg);
        assert!(msg.contains("0.0870"), "missing achieved_clamp/max in: {}", msg);
        assert!(msg.contains("closest reachable"), "missing label in: {}", msg);
    }
}
