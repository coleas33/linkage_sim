//! Severity controls how `solve_for_target` reports per-sample failures.
//!
//! `Severity::Strict` returns `Err(LinkageError::...)` on any failure mode.
//! `Severity::Analysis` returns `Ok(InverseSolveResult { status: <failure variant>, ... })`
//! so the GUI can render failed samples in red rather than aborting the trajectory.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §6.5

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Strict,
    Analysis,
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Analysis
    }
}

/// Per-sample diagnostic produced by `solve_for_target`.
///
/// In `Severity::Analysis` mode this is returned in the `InverseSolveResult.status`
/// field. In `Severity::Strict` mode it is converted to `LinkageError` and bubbled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InverseSolveStatus {
    /// Newton converged within tolerance.
    Converged,
    /// Target value is outside the workspace `[g_min, g_max]`.
    Reachability {
        target: f64,
        achieved_clamp: f64,
        workspace_min: Option<f64>,
        workspace_max: Option<f64>,
    },
    /// `|dg/du|` fell below the singularity threshold (mechanism has lost
    /// authority over the target locally — toggle, transmission angle ≈ 90°).
    Singularity { dg_du: f64 },
    /// `‖q_k − q_{k-1}‖` exceeded the branch-jump threshold (assembly mode flip).
    BranchJump { delta_q_norm: f64 },
    /// Newton did not converge in `max_iter` iterations.
    NonConvergent { iterations: usize, residual: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_default_is_analysis() {
        assert_eq!(Severity::default(), Severity::Analysis);
    }

    #[test]
    fn severity_serde_round_trip() {
        let strict = Severity::Strict;
        let json = serde_json::to_string(&strict).unwrap();
        let back: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(strict, back);
    }

    #[test]
    fn inverse_solve_status_constructs_all_variants() {
        let _converged = InverseSolveStatus::Converged;
        let _reach = InverseSolveStatus::Reachability {
            target: 1.0,
            achieved_clamp: 0.5,
            workspace_min: Some(-0.5),
            workspace_max: Some(0.5),
        };
        let _sing = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let _branch = InverseSolveStatus::BranchJump { delta_q_norm: 0.1 };
        let _nc = InverseSolveStatus::NonConvergent {
            iterations: 50,
            residual: 1e-3,
        };
    }

    #[test]
    fn inverse_solve_status_is_clone_and_debug() {
        let s = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let _cloned = s.clone();
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Singularity"));
    }
}
