//! Trajectory-mode interactive solving on AppState.
//!
//! Sibling to solve_at_angle / solve_at_stroke; called by the per-frame
//! "current target" slider in `gui/trajectory_panel/mod.rs`.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §7.6

use crate::solver::inverse_kinematics::{
    solve_for_target, ControlTarget, Severity,
};

use super::AppState;

impl AppState {
    /// Drive the mechanism to the given (target, h) — the inverse-kinematics
    /// equivalent of the existing `solve_at_angle` and `solve_at_stroke`.
    /// Updates `self.q` on success; leaves state unchanged on failure.
    pub fn solve_for_trajectory_target(
        &mut self,
        target: &ControlTarget,
        h: f64,
    ) {
        let Some(mech) = self.mechanism.as_ref() else {
            return;
        };
        let nominal_rate = if self.driver_omega.abs() > 1e-12 {
            self.driver_omega
        } else {
            return;
        };
        let u_0 = self.driver_theta_0;
        let u_range = (
            self.sweep_angle_min_deg.to_radians(),
            self.sweep_angle_max_deg.to_radians(),
        );

        match solve_for_target(
            mech, &self.q, target, h, Severity::Analysis,
            u_range, u_0, nominal_rate, 1e-8, 50, 64,
        ) {
            Ok(res) => {
                self.q = res.q;
                self.driver_angle = res.u; // for revolute; trajectory mode uses radians
                self.last_good_q = self.q.clone();
            }
            Err(_) => {
                // Out-of-range / singularity: leave state unchanged.
            }
        }
    }
}
