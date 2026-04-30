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
}
