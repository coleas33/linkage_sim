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
}
