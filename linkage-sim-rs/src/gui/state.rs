//! Application state: mechanism, solver results, selection, view transform.

use nalgebra::DVector;
use std::f64::consts::PI;
use std::path::Path;

use crate::core::constraint::Constraint;
use crate::core::driver::DriverMeta;
use crate::core::mechanism::Mechanism;
use crate::gui::samples::{build_sample, SampleMechanism};
use crate::io::serialization::{load_mechanism_unbuilt, save_mechanism};
use crate::solver::kinematics::solve_position;

// ── Selection ─────────────────────────────────────────────────────────────────

/// Which entity in the mechanism is currently selected for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

// ── Solver status ─────────────────────────────────────────────────────────────

/// Summary of the most recent solver call.
#[derive(Debug, Clone)]
pub struct SolverStatus {
    pub converged: bool,
    pub residual_norm: f64,
    pub iterations: usize,
}

impl Default for SolverStatus {
    fn default() -> Self {
        Self {
            converged: false,
            residual_norm: 0.0,
            iterations: 0,
        }
    }
}

// ── View transform ────────────────────────────────────────────────────────────

/// Maps between world coordinates (meters) and screen coordinates (pixels).
///
/// Screen Y grows downward, so world Y is flipped.
pub struct ViewTransform {
    /// Screen-space offset of the world origin, in pixels.
    pub offset: [f32; 2],
    /// Pixels per meter.
    pub scale: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            offset: [400.0, 400.0],
            scale: 5000.0,
        }
    }
}

impl ViewTransform {
    /// Convert world coordinates (meters) to screen coordinates (pixels).
    ///
    /// Screen Y is flipped relative to world Y.
    pub fn world_to_screen(&self, wx: f64, wy: f64) -> [f32; 2] {
        let sx = self.offset[0] + (wx as f32) * self.scale;
        let sy = self.offset[1] - (wy as f32) * self.scale;
        [sx, sy]
    }

    /// Convert screen coordinates (pixels) back to world coordinates (meters).
    pub fn screen_to_world(&self, sx: f32, sy: f32) -> [f64; 2] {
        let wx = ((sx - self.offset[0]) / self.scale) as f64;
        let wy = (-(sy - self.offset[1]) / self.scale) as f64;
        [wx, wy]
    }
}

// ── AppState ──────────────────────────────────────────────────────────────────

/// All mutable application state in one place.
pub struct AppState {
    /// The currently loaded mechanism, if any.
    pub mechanism: Option<Mechanism>,
    /// Current generalized coordinate vector.
    pub q: DVector<f64>,
    /// Current driver angle in radians (the "knob" controlled by the slider).
    pub driver_angle: f64,
    /// Last successfully solved q — used as fallback when the solver fails.
    pub last_good_q: DVector<f64>,
    /// Status of the most recent solver call.
    pub solver_status: SolverStatus,
    /// Currently selected entity (for the property panel).
    pub selected: Option<SelectedEntity>,
    /// View / zoom transform.
    pub view: ViewTransform,
    /// Show the debug overlay (defaults true in debug builds).
    pub show_debug_overlay: bool,
    /// Which sample is currently loaded.
    pub current_sample: Option<SampleMechanism>,
    /// Angular velocity of the driver (rad/s).
    pub driver_omega: f64,
    /// Initial driver angle (rad) at t=0.
    pub driver_theta_0: f64,
    // ── Animation ────────────────────────────────────────────────────────
    pub playing: bool,
    pub animation_speed_deg_per_sec: f64,
    pub loop_mode: bool,
    pub animation_direction: f64,
    // ── Driver ───────────────────────────────────────────────────────────
    pub driver_joint_id: Option<String>,
    pub pending_driver_reassignment: Option<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mechanism: None,
            q: DVector::zeros(0),
            driver_angle: 0.0,
            last_good_q: DVector::zeros(0),
            solver_status: SolverStatus::default(),
            selected: None,
            view: ViewTransform::default(),
            show_debug_overlay: cfg!(debug_assertions),
            current_sample: None,
            driver_omega: 2.0 * PI,
            driver_theta_0: 0.0,
            playing: false,
            animation_speed_deg_per_sec: 90.0,
            loop_mode: true,
            animation_direction: 1.0,
            driver_joint_id: None,
            pending_driver_reassignment: None,
        }
    }
}

impl AppState {
    /// Load a named sample: build mechanism, solve at t=0, and store state.
    pub fn load_sample(&mut self, sample: SampleMechanism) {
        let (mech, q0) = build_sample(sample);

        // Extract driver parameters from the sample before storing mechanism.
        // All current samples use omega=2π, theta_0=0.
        self.driver_omega = 2.0 * PI;
        self.driver_theta_0 = 0.0;

        match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
            Ok(result) => {
                self.solver_status = SolverStatus {
                    converged: result.converged,
                    residual_norm: result.residual_norm,
                    iterations: result.iterations,
                };

                if result.converged {
                    self.q = result.q.clone();
                    self.last_good_q = result.q;
                } else {
                    self.q = q0.clone();
                    self.last_good_q = q0;
                }
            }
            Err(_) => {
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                self.q = q0.clone();
                self.last_good_q = q0;
            }
        }

        self.driver_angle = self.driver_theta_0;
        self.mechanism = Some(mech);
        self.current_sample = Some(sample);
        self.selected = None;

        // Detect which joint is currently driven
        if let Some(mech) = &self.mechanism {
            if let Some(pair) = mech.driver_body_pair() {
                self.driver_joint_id = mech.joints().iter()
                    .find(|j| {
                        j.is_revolute()
                            && ((j.body_i_id() == pair.0 && j.body_j_id() == pair.1)
                                || (j.body_i_id() == pair.1 && j.body_j_id() == pair.0))
                    })
                    .map(|j| j.id().to_string());
            } else {
                self.driver_joint_id = None;
            }
        }
        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;
    }

    /// Solve the position problem for the given driver angle (radians).
    ///
    /// Uses `last_good_q` as the initial guess for Newton-Raphson.
    /// On success, updates both `q` and `last_good_q`.
    /// On failure, keeps `last_good_q` unchanged and reports the failure in
    /// `solver_status`.
    pub fn solve_at_angle(&mut self, angle_rad: f64) {
        let Some(mech) = &self.mechanism else {
            return;
        };

        // Convert driver angle to time using the constant-speed relationship:
        //   angle = theta_0 + omega * t  →  t = (angle - theta_0) / omega
        let t = (angle_rad - self.driver_theta_0) / self.driver_omega;

        match solve_position(mech, &self.last_good_q, t, 1e-10, 50) {
            Ok(result) => {
                self.solver_status = SolverStatus {
                    converged: result.converged,
                    residual_norm: result.residual_norm,
                    iterations: result.iterations,
                };

                if result.converged {
                    self.last_good_q = result.q.clone();
                    self.q = result.q;
                    self.driver_angle = angle_rad;
                }
                // On failure, q and driver_angle are NOT updated; the UI retains the
                // last valid pose.
            }
            Err(_) => {
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                // On error, retain last valid pose.
            }
        }
    }

    /// Returns true if a mechanism has been loaded.
    pub fn has_mechanism(&self) -> bool {
        self.mechanism.is_some()
    }

    /// Serialize the current mechanism to a JSON file at the given path.
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let mech = self
            .mechanism
            .as_ref()
            .ok_or_else(|| "No mechanism loaded".to_string())?;

        let json = save_mechanism(mech).map_err(|e| e.to_string())?;

        std::fs::write(path, json).map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    /// Load a mechanism from a JSON file, solve at t=0, and update all state.
    ///
    /// After loading, the driver is restored from the JSON. If the file has no
    /// driver, the mechanism is loaded without one (the user can assign one via
    /// the Driver panel).
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn load_from_file(&mut self, path: &Path) -> Result<(), String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

        let mut mech = load_mechanism_unbuilt(&json).map_err(|e| e.to_string())?;
        mech.build().map_err(|e| e.to_string())?;

        // Extract driver parameters before we move mech into self.
        let (driver_omega, driver_theta_0) = if let Some(driver) = mech.drivers().first() {
            match driver.meta() {
                Some(DriverMeta::ConstantSpeed { omega, theta_0 }) => (*omega, *theta_0),
                None => (2.0 * PI, 0.0),
            }
        } else {
            (2.0 * PI, 0.0)
        };

        // Detect the driven joint ID.
        let driver_joint_id = if let Some(pair) = mech.driver_body_pair() {
            mech.joints()
                .iter()
                .find(|j| {
                    j.is_revolute()
                        && ((j.body_i_id() == pair.0 && j.body_j_id() == pair.1)
                            || (j.body_i_id() == pair.1 && j.body_j_id() == pair.0))
                })
                .map(|j| j.id().to_string())
        } else {
            None
        };

        // Build a zero initial guess and solve at t=0.
        let q0 = mech.state().make_q();
        let solve_result = solve_position(&mech, &q0, 0.0, 1e-10, 50);

        match solve_result {
            Ok(result) => {
                self.solver_status = SolverStatus {
                    converged: result.converged,
                    residual_norm: result.residual_norm,
                    iterations: result.iterations,
                };
                if result.converged {
                    self.q = result.q.clone();
                    self.last_good_q = result.q;
                } else {
                    self.q = q0.clone();
                    self.last_good_q = q0;
                }
            }
            Err(_) => {
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                self.q = q0.clone();
                self.last_good_q = q0;
            }
        }

        self.driver_omega = driver_omega;
        self.driver_theta_0 = driver_theta_0;
        self.driver_angle = driver_theta_0;
        self.driver_joint_id = driver_joint_id;
        self.mechanism = Some(mech);
        self.current_sample = None;
        self.selected = None;
        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;

        Ok(())
    }

    /// Rebuild the mechanism with a different driver joint.
    pub fn reassign_driver(&mut self, joint_id: &str) {
        let Some(sample) = self.current_sample else { return };

        match crate::gui::samples::build_sample_with_driver(sample, Some(joint_id)) {
            Ok((mech, q0)) => {
                // Extract driver theta_0 from the built mechanism
                // The builder sets theta_0 to match q0's driven body angle
                self.driver_omega = 2.0 * PI;
                // We need to figure out what theta_0 was used. Since the driver
                // constraint at t=0 is: theta_driven - theta_ground - theta_0 = 0,
                // and the builder set theta_0 = driven body's angle in q0,
                // we can read it from q0 after solving at t=0.
                // For simplicity, solve at t=0 first, then set driver_angle = theta_0.

                match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
                    Ok(result) => {
                        self.solver_status = SolverStatus {
                            converged: result.converged,
                            residual_norm: result.residual_norm,
                            iterations: result.iterations,
                        };
                        if result.converged {
                            self.q = result.q.clone();
                            self.last_good_q = result.q;
                        } else {
                            self.q = q0.clone();
                            self.last_good_q = q0;
                        }
                    }
                    Err(_) => {
                        self.solver_status = SolverStatus {
                            converged: false,
                            residual_norm: f64::NAN,
                            iterations: 0,
                        };
                        self.q = q0.clone();
                        self.last_good_q = q0;
                    }
                }

                // Reset to angle = theta_0 (the starting angle for this driver)
                // Since the builder used theta_0 matching q0, and we solved at t=0,
                // driver_angle should be theta_0.
                self.driver_theta_0 = 0.0; // The builder maps t=0 to the start config
                self.driver_angle = 0.0;
                self.mechanism = Some(mech);
                self.driver_joint_id = Some(joint_id.to_string());
                self.selected = None;
                self.playing = false;
                self.animation_direction = 1.0;
                self.pending_driver_reassignment = None;
            }
            Err(msg) => {
                log::warn!("Driver reassignment failed: {}", msg);
            }
        }
    }

    /// Advance animation by one frame. Returns true if animation is active.
    pub fn step_animation(&mut self, dt: f64) -> bool {
        if !self.playing || !self.has_mechanism() {
            return false;
        }

        let step_deg = self.animation_speed_deg_per_sec * dt * self.animation_direction;
        let mut new_angle_deg = self.driver_angle.to_degrees() + step_deg;

        if self.loop_mode {
            // Wrap around
            if new_angle_deg >= 360.0 {
                new_angle_deg -= 360.0;
            } else if new_angle_deg < 0.0 {
                new_angle_deg += 360.0;
            }
        } else {
            // Once mode: always forward, stop at 360
            if new_angle_deg >= 360.0 {
                new_angle_deg = 360.0;
                self.playing = false;
            }
            if new_angle_deg < 0.0 {
                new_angle_deg = 0.0;
                self.playing = false;
            }
        }

        let prev_converged = self.solver_status.converged;
        self.solve_at_angle(new_angle_deg.to_radians());

        // Ping-pong: reverse direction on solver failure in loop mode
        if self.loop_mode && !self.solver_status.converged && prev_converged {
            self.animation_direction *= -1.0;
        }

        // Stop on failure in once mode
        if !self.loop_mode && !self.solver_status.converged {
            self.playing = false;
        }

        self.playing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_no_mechanism() {
        let state = AppState::default();
        assert!(!state.has_mechanism());
        assert!(state.current_sample.is_none());
        assert_eq!(state.q.len(), 0);
    }

    #[test]
    fn load_sample_solves_initial() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.has_mechanism());
        assert!(
            state.solver_status.converged,
            "Initial solve did not converge, residual = {}",
            state.solver_status.residual_norm
        );
        assert!(!state.q.is_empty());
    }

    #[test]
    fn solve_at_angle_updates_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let q_initial = state.q.clone();

        state.solve_at_angle(0.5);

        assert!(
            state.solver_status.converged,
            "solve_at_angle(0.5) did not converge, residual = {}",
            state.solver_status.residual_norm
        );
        // q should have changed from the initial position
        assert_ne!(state.q, q_initial, "q did not change after solve_at_angle");
    }

    #[test]
    fn step_animation_advances_angle() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = true;
        state.animation_speed_deg_per_sec = 180.0;

        let angle_before = state.driver_angle;
        let still_playing = state.step_animation(1.0 / 60.0);
        assert!(still_playing);
        assert!(state.driver_angle > angle_before);
    }

    #[test]
    fn step_animation_noop_when_paused() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = false;

        let angle_before = state.driver_angle;
        assert!(!state.step_animation(1.0 / 60.0));
        assert_eq!(state.driver_angle, angle_before);
    }

    #[test]
    fn reassign_driver_changes_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(state.solver_status.converged);
        assert!(!state.playing);
    }

    #[test]
    fn world_to_screen_roundtrip() {
        let view = ViewTransform::default();
        let wx = 0.019;
        let wy = 0.015;

        let [sx, sy] = view.world_to_screen(wx, wy);
        let [wx2, wy2] = view.screen_to_world(sx, sy);

        assert!(
            (wx - wx2).abs() < 1e-6,
            "X roundtrip error: {} vs {}",
            wx,
            wx2
        );
        assert!(
            (wy - wy2).abs() < 1e-6,
            "Y roundtrip error: {} vs {}",
            wy,
            wy2
        );
    }
}
