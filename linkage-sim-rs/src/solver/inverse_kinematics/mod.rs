//! Inverse-kinematics solver for trajectory-mode position control.
//!
//! Given a desired output observable `g(q)` and target value `h`, back-solves
//! the actuator input parameter `u` such that `g(q(u)) = h`, where `q(u)` is
//! the forward solution from the existing kinematics solvers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md

pub mod control_target;
pub mod derivatives;
pub mod severity;
pub mod solver;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use control_target::ControlTarget;
pub use derivatives::{inverse_acceleration_fd, inverse_velocity};
pub use severity::{InverseSolveStatus, Severity};
pub use solver::{solve_for_target, workspace_probe, InverseSolveResult, WorkspaceProbe};
