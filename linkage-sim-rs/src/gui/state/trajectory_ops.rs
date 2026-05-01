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
        let omega = self.driver_omega();
        let nominal_rate = if omega.abs() > 1e-12 {
            omega
        } else {
            return;
        };
        let u_0 = self.driver_theta_0();
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

    /// Advance trajectory playback by `dt` seconds (real time). Returns
    /// `true` when playback is active and the canvas should be redrawn.
    ///
    /// Solves the canvas pose at the new trajectory time. When looping
    /// is off and the end is reached, playback stops at `duration`.
    /// When looping is on, playback wraps to 0.0. Caller is expected to
    /// gate via `state.trajectory_playback_active` for clarity, but this
    /// method also short-circuits internally so it's safe to call
    /// unconditionally.
    pub fn step_trajectory_playback(&mut self, dt: f64) -> bool {
        use crate::gui::sweep::SweepMode;

        if !self.trajectory_playback_active {
            return false;
        }
        // Pull duration + clone target/trajectory off self so we can
        // call &mut self below.
        let (duration, target, trajectory) =
            if let SweepMode::Trajectory { trajectory, target, .. } = &self.sweep_mode {
                (trajectory.duration(), target.clone(), trajectory.clone())
            } else {
                // Mode changed out from under us — stop.
                self.trajectory_playback_active = false;
                return false;
            };
        if duration <= 0.0 {
            self.trajectory_playback_active = false;
            return false;
        }

        self.trajectory_playback_t += dt * self.trajectory_playback_speed;

        if self.trajectory_playback_t >= duration {
            if self.trajectory_playback_loop {
                // Wrap. Modulo by duration so very large dt jumps still land in-range.
                self.trajectory_playback_t = self.trajectory_playback_t.rem_euclid(duration);
            } else {
                self.trajectory_playback_t = duration;
                self.trajectory_playback_active = false;
            }
        } else if self.trajectory_playback_t < 0.0 {
            // Defensive: if speed went negative, clamp to 0.
            self.trajectory_playback_t = 0.0;
        }

        // Solve at the new t and sync the plot cursor.
        let (h, _, _) = trajectory.evaluate(self.trajectory_playback_t);
        self.solve_for_trajectory_target(&target, h);
        self.last_trajectory_scrub_t = Some(self.trajectory_playback_t);

        true
    }
}
