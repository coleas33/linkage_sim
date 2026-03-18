//! Application state: mechanism, solver results, selection, view transform.

use nalgebra::DVector;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use crate::analysis::transmission::transmission_angle_fourbar;
use crate::core::constraint::Constraint;
use crate::core::driver::DriverMeta;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::gui::samples::{build_sample, SampleMechanism};
use crate::gui::undo::{MechanismSnapshot, UndoHistory};
use crate::io::serialization::{
    load_mechanism_unbuilt, load_mechanism_unbuilt_from_json, mechanism_to_json, save_mechanism,
    DriverJson, MechanismJson,
};
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

// ── Sweep data ───────────────────────────────────────────────────────────────

/// Pre-computed sweep results for the full driver rotation (0-360 degrees).
///
/// Computed once when a mechanism is loaded or the driver changes. Cached
/// to avoid recomputing every frame. Used by the plot panel and canvas.
#[derive(Debug, Clone)]
pub struct SweepData {
    /// Driver angles in degrees at which solutions were obtained.
    pub angles_deg: Vec<f64>,
    /// Body orientation angles (degrees) keyed by body ID.
    pub body_angles: HashMap<String, Vec<f64>>,
    /// Coupler point traces keyed by "body_id.point_name", each entry
    /// is a sequence of [x, y] world-coordinate pairs.
    pub coupler_traces: HashMap<String, Vec<[f64; 2]>>,
    /// Transmission angle (degrees) at each step, if the mechanism is a
    /// 4-bar linkage with identifiable link lengths.
    pub transmission_angles: Option<Vec<f64>>,
}

// ── AppState ──────────────────────────────────────────────────────────────────

/// All mutable application state in one place.
pub struct AppState {
    /// The editable blueprint -- source of truth for the mechanism definition.
    /// Edits mutate this, then rebuild() reconstructs the Mechanism.
    pub blueprint: Option<MechanismJson>,
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
    // ── Undo/Redo ────────────────────────────────────────────────────────
    pub undo_history: UndoHistory,
    // ── Sweep / Plots ────────────────────────────────────────────────────
    /// Cached sweep data for the full driver rotation.
    pub sweep_data: Option<SweepData>,
    /// Whether the plot panel is visible.
    pub show_plots: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            blueprint: None,
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
            undo_history: UndoHistory::new(50),
            sweep_data: None,
            show_plots: false,
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

        // Create blueprint from the built mechanism
        self.blueprint = mechanism_to_json(&mech).ok();

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
        self.undo_history.clear();
        self.compute_sweep();
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
        let json_str =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

        // Parse JSON and store as blueprint before building
        let json_struct: MechanismJson =
            serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
        self.blueprint = Some(json_struct);

        let mut mech =
            load_mechanism_unbuilt(&json_str).map_err(|e| e.to_string())?;
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
        self.undo_history.clear();
        self.compute_sweep();

        Ok(())
    }

    // ── Blueprint rebuild pipeline ───────────────────────────────────────

    /// Rebuild Mechanism from the current blueprint, solve at current angle.
    /// Called after every edit operation.
    pub fn rebuild(&mut self) {
        let Some(bp) = &self.blueprint else { return };

        // Build mechanism from blueprint
        let mut mech = match load_mechanism_unbuilt_from_json(bp) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Blueprint rebuild failed: {}", e);
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                return;
            }
        };

        if let Err(e) = mech.build() {
            log::warn!("Mechanism build failed: {}", e);
            self.solver_status = SolverStatus {
                converged: false,
                residual_norm: f64::NAN,
                iterations: 0,
            };
            return;
        }

        // Extract driver params from blueprint
        // (look for the first constant_speed driver)
        self.driver_omega = 2.0 * PI;
        self.driver_theta_0 = 0.0;
        // Check blueprint drivers for actual values
        for (_id, driver) in &bp.drivers {
            match driver {
                DriverJson::ConstantSpeed { omega, theta_0, .. } => {
                    self.driver_omega = *omega;
                    self.driver_theta_0 = *theta_0;
                    break;
                }
            }
        }

        // Detect driven joint
        if let Some(pair) = mech.driver_body_pair() {
            self.driver_joint_id = mech
                .joints()
                .iter()
                .find(|j| {
                    j.is_revolute()
                        && ((j.body_i_id() == pair.0 && j.body_j_id() == pair.1)
                            || (j.body_i_id() == pair.1 && j.body_j_id() == pair.0))
                })
                .map(|j| j.id().to_string());
        }

        // Solve at current angle using last_good_q as initial guess
        let t = if self.driver_omega.abs() > f64::EPSILON {
            (self.driver_angle - self.driver_theta_0) / self.driver_omega
        } else {
            0.0
        };

        // Try solving with last_good_q if it has the right dimension
        let try_q = if self.last_good_q.len() == mech.state().n_coords() {
            &self.last_good_q
        } else {
            &mech.state().make_q()
        };

        match solve_position(&mech, try_q, t, 1e-10, 50) {
            Ok(result) => {
                self.solver_status = SolverStatus {
                    converged: result.converged,
                    residual_norm: result.residual_norm,
                    iterations: result.iterations,
                };
                if result.converged {
                    self.q = result.q.clone();
                    self.last_good_q = result.q;
                }
            }
            Err(_) => {
                // If solve fails with last_good_q, try from zeros
                let q0 = mech.state().make_q();
                match solve_position(&mech, &q0, t, 1e-10, 100) {
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
                            self.q = q0;
                        }
                    }
                    Err(_) => {
                        self.solver_status = SolverStatus {
                            converged: false,
                            residual_norm: f64::NAN,
                            iterations: 0,
                        };
                    }
                }
            }
        }

        self.mechanism = Some(mech);
    }

    // ── Blueprint edit operations ────────────────────────────────────────

    /// Move an attachment point on a body in the blueprint.
    /// This is the core drag operation for the editor.
    pub fn move_attachment_point(
        &mut self,
        body_id: &str,
        point_name: &str,
        new_x: f64,
        new_y: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_name) {
                *pt = [new_x, new_y];
            }
        }
        self.rebuild();
    }

    /// Set mass property on a body in the blueprint.
    pub fn set_body_mass(&mut self, body_id: &str, mass: f64) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.mass = mass;
        }
        self.rebuild();
    }

    /// Set moment of inertia on a body in the blueprint.
    pub fn set_body_izz(&mut self, body_id: &str, izz: f64) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.izz_cg = izz;
        }
        self.rebuild();
    }

    /// Rebuild the mechanism with a different driver joint.
    pub fn reassign_driver(&mut self, joint_id: &str) {
        let Some(sample) = self.current_sample else { return };

        self.push_undo();

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
                // Update blueprint to reflect the reassigned driver
                self.blueprint = mechanism_to_json(&mech).ok();
                self.mechanism = Some(mech);
                self.driver_joint_id = Some(joint_id.to_string());
                self.selected = None;
                self.playing = false;
                self.animation_direction = 1.0;
                self.pending_driver_reassignment = None;
                self.compute_sweep();
            }
            Err(msg) => {
                log::warn!("Driver reassignment failed: {}", msg);
            }
        }
    }

    // ── Undo / Redo ────────────────────────────────────────────────────────

    /// Create a snapshot of the current mechanism document state.
    ///
    /// Returns `None` if no mechanism is loaded or serialization fails.
    pub fn take_snapshot(&self) -> Option<MechanismSnapshot> {
        let mech = self.mechanism.as_ref()?;
        let json = mechanism_to_json(mech).ok()?;
        let json_str = serde_json::to_string(&json).ok()?;
        Some(MechanismSnapshot {
            mechanism_json: json_str,
            driver_angle: self.driver_angle,
            driver_omega: self.driver_omega,
            driver_theta_0: self.driver_theta_0,
            driver_joint_id: self.driver_joint_id.clone(),
            q: self.q.iter().copied().collect(),
        })
    }

    /// Restore mechanism state from a snapshot.
    ///
    /// Deserializes the mechanism JSON, builds, solves at t=0, and updates all
    /// document-level state. View transform, selection, and animation state are
    /// left unchanged.
    pub fn restore_snapshot(&mut self, snapshot: &MechanismSnapshot) {
        let Ok(mut mech) = load_mechanism_unbuilt(&snapshot.mechanism_json) else {
            log::warn!("Undo/redo: failed to deserialize mechanism snapshot");
            return;
        };
        if mech.build().is_err() {
            log::warn!("Undo/redo: failed to build mechanism from snapshot");
            return;
        }

        // Solve at the snapshot's driver angle, using the stored q as the initial guess.
        // Falls back to make_q() (all zeros) if the snapshot has no q data.
        let t = (snapshot.driver_angle - snapshot.driver_theta_0) / snapshot.driver_omega;
        let q0 = if snapshot.q.len() == mech.state().n_coords() {
            DVector::from_vec(snapshot.q.clone())
        } else {
            mech.state().make_q()
        };
        match solve_position(&mech, &q0, t, 1e-10, 50) {
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

        self.mechanism = Some(mech);
        self.driver_angle = snapshot.driver_angle;
        self.driver_omega = snapshot.driver_omega;
        self.driver_theta_0 = snapshot.driver_theta_0;
        self.driver_joint_id = snapshot.driver_joint_id.clone();
        self.playing = false;
    }

    /// Push the current state onto the undo stack before an undoable action.
    ///
    /// No-op if no mechanism is loaded (nothing to snapshot).
    pub fn push_undo(&mut self) {
        if let Some(snapshot) = self.take_snapshot() {
            self.undo_history.push(snapshot);
        }
    }

    /// Undo the last action: restore the previous mechanism state.
    pub fn undo(&mut self) {
        let Some(current) = self.take_snapshot() else {
            return;
        };
        if let Some(previous) = self.undo_history.undo(current) {
            self.restore_snapshot(&previous);
        }
    }

    /// Redo the last undone action: restore the next mechanism state.
    pub fn redo(&mut self) {
        let Some(current) = self.take_snapshot() else {
            return;
        };
        if let Some(next) = self.undo_history.redo(current) {
            self.restore_snapshot(&next);
        }
    }

    /// Returns true if there is at least one state to undo.
    pub fn can_undo(&self) -> bool {
        self.undo_history.can_undo()
    }

    /// Returns true if there is at least one state to redo.
    pub fn can_redo(&self) -> bool {
        self.undo_history.can_redo()
    }

    // ── Sweep computation ─────────────────────────────────────────────────

    /// Compute a full 0-360 degree sweep and cache the results.
    ///
    /// Solves the position problem at each degree using continuation
    /// (previous solution as initial guess). Extracts body angles, coupler
    /// point traces, and transmission angle if the mechanism is a simple
    /// 4-bar.
    pub fn compute_sweep(&mut self) {
        if self.mechanism.is_none() {
            self.sweep_data = None;
            return;
        }

        // Copy values we need from self before taking a reference to the mechanism,
        // to avoid borrow-checker conflicts between &self.mechanism and &mut self.sweep_data.
        let q_start = self.last_good_q.clone();
        let omega = self.driver_omega;
        let theta_0 = self.driver_theta_0;

        let data = compute_sweep_data(self.mechanism.as_ref().unwrap(), &q_start, omega, theta_0);
        self.sweep_data = Some(data);
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

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Perform the sweep computation (0-360 degrees) and return `SweepData`.
///
/// Extracted as a free function so it borrows only `&Mechanism` and not
/// `&mut AppState`, avoiding borrow-checker conflicts.
fn compute_sweep_data(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
) -> SweepData {
    let mut data = SweepData {
        angles_deg: Vec::with_capacity(361),
        body_angles: HashMap::new(),
        coupler_traces: HashMap::new(),
        transmission_angles: None,
    };

    // Pre-allocate body angle vectors.
    let body_order: Vec<String> = mech.body_order().to_vec();
    for body_id in &body_order {
        data.body_angles
            .insert(body_id.clone(), Vec::with_capacity(361));
    }

    // Pre-allocate coupler trace vectors.
    // Collect coupler point keys: "body_id.point_name"
    let mut coupler_keys: Vec<(String, String, nalgebra::Vector2<f64>)> = Vec::new();
    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID {
            continue;
        }
        for (point_name, local) in &body.coupler_points {
            let key = format!("{}.{}", body_id, point_name);
            coupler_keys.push((key.clone(), body_id.clone(), *local));
            data.coupler_traces.insert(key, Vec::with_capacity(361));
        }
        // Also trace attachment points on non-ground bodies (useful
        // for visualization even if no explicit coupler points exist).
        for (point_name, local) in &body.attachment_points {
            let key = format!("{}.{}", body_id, point_name);
            if !data.coupler_traces.contains_key(&key) {
                coupler_keys.push((key.clone(), body_id.clone(), *local));
                data.coupler_traces.insert(key, Vec::with_capacity(361));
            }
        }
    }

    // Detect 4-bar link lengths for transmission angle.
    let fourbar_links = detect_fourbar_links(mech);
    if fourbar_links.is_some() {
        data.transmission_angles = Some(Vec::with_capacity(361));
    }

    // Sweep from 0 to 360 degrees in 1-degree steps.
    let mut q = q_start.clone();

    for i in 0..=360 {
        let angle_deg = i as f64;
        let t = (angle_deg.to_radians() - theta_0) / omega;

        match solve_position(mech, &q, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                q = result.q.clone();
                data.angles_deg.push(angle_deg);

                let mech_state = mech.state();

                // Extract body angles.
                for body_id in &body_order {
                    let theta = mech_state.get_angle(body_id, &q);
                    data.body_angles
                        .get_mut(body_id)
                        .unwrap()
                        .push(theta.to_degrees());
                }

                // Extract coupler traces.
                for (key, body_id, local) in &coupler_keys {
                    let global = mech_state.body_point_global(body_id, local, &q);
                    data.coupler_traces
                        .get_mut(key)
                        .unwrap()
                        .push([global.x, global.y]);
                }

                // Transmission angle (4-bar only).
                if let Some((a, b, c, d)) = fourbar_links {
                    let theta_crank = angle_deg.to_radians();
                    let ta = transmission_angle_fourbar(a, b, c, d, theta_crank);
                    data.transmission_angles.as_mut().unwrap().push(ta.angle_deg);
                }
            }
            _ => {
                // Solver failed at this angle -- stop sweep.
                // The mechanism likely cannot complete a full rotation.
                break;
            }
        }
    }

    data
}

/// Try to detect a classic 4-bar linkage and return (crank, coupler, rocker,
/// ground) link lengths for transmission angle computation.
///
/// A 4-bar is identified by:
/// - Exactly 3 moving bodies
/// - Exactly 4 revolute joints
/// - Each moving body is a binary bar (exactly 2 attachment points)
/// - One of the moving bodies is the driven body (crank)
///
/// Returns `None` for non-4-bar mechanisms.
fn detect_fourbar_links(mech: &Mechanism) -> Option<(f64, f64, f64, f64)> {
    use crate::core::constraint::Constraint;

    let body_order = mech.body_order();
    if body_order.len() != 3 {
        return None;
    }

    let joints = mech.joints();
    let revolute_joints: Vec<_> = joints.iter().filter(|j| j.is_revolute()).collect();
    if revolute_joints.len() != 4 {
        return None;
    }

    // Identify which body is the crank (driven body).
    let driver_pair = mech.driver_body_pair()?;
    let driven_body = if driver_pair.0 == GROUND_ID {
        driver_pair.1
    } else {
        driver_pair.0
    };

    let bodies = mech.bodies();

    // Find the crank length (distance between its two attachment points).
    let crank_body = bodies.get(driven_body)?;
    if crank_body.attachment_points.len() != 2 {
        return None;
    }
    let crank_pts: Vec<_> = crank_body.attachment_points.values().collect();
    let crank_len = (crank_pts[0] - crank_pts[1]).norm();

    // Find the coupler and rocker. The coupler connects to the crank at a
    // non-ground joint, and the rocker connects the coupler to ground.
    // We identify them by finding which bodies connect to the crank vs ground.
    let other_bodies: Vec<&str> = body_order
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != driven_body)
        .collect();

    if other_bodies.len() != 2 {
        return None;
    }

    // Check which of the two bodies connects to ground (rocker).
    let mut coupler_id = None;
    let mut rocker_id = None;
    for &body_id in &other_bodies {
        let connects_to_ground = revolute_joints.iter().any(|j| {
            (j.body_i_id() == body_id && j.body_j_id() == GROUND_ID)
                || (j.body_j_id() == body_id && j.body_i_id() == GROUND_ID)
        });
        let connects_to_crank = revolute_joints.iter().any(|j| {
            (j.body_i_id() == body_id && j.body_j_id() == driven_body)
                || (j.body_j_id() == body_id && j.body_i_id() == driven_body)
        });

        if connects_to_ground && !connects_to_crank {
            rocker_id = Some(body_id);
        } else if connects_to_crank && !connects_to_ground {
            coupler_id = Some(body_id);
        } else if connects_to_crank && connects_to_ground {
            // This body connects to both -- could be either in a parallelogram.
            // Treat as rocker if we haven't assigned one yet.
            if rocker_id.is_none() {
                rocker_id = Some(body_id);
            } else {
                coupler_id = Some(body_id);
            }
        }
    }

    let coupler_body = bodies.get(coupler_id?)?;
    let rocker_body = bodies.get(rocker_id?)?;

    if coupler_body.attachment_points.len() != 2 || rocker_body.attachment_points.len() != 2 {
        return None;
    }

    let coupler_pts: Vec<_> = coupler_body.attachment_points.values().collect();
    let coupler_len = (coupler_pts[0] - coupler_pts[1]).norm();

    let rocker_pts: Vec<_> = rocker_body.attachment_points.values().collect();
    let rocker_len = (rocker_pts[0] - rocker_pts[1]).norm();

    // Ground length: distance between the two ground pivots.
    let ground = bodies.get(GROUND_ID)?;
    if ground.attachment_points.len() != 2 {
        return None;
    }
    let ground_pts: Vec<_> = ground.attachment_points.values().collect();
    let ground_len = (ground_pts[0] - ground_pts[1]).norm();

    Some((crank_len, coupler_len, rocker_len, ground_len))
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

    // ── Undo / Redo integration tests ──────────────────────────────────

    #[test]
    fn take_snapshot_returns_none_without_mechanism() {
        let state = AppState::default();
        assert!(state.take_snapshot().is_none());
    }

    #[test]
    fn take_snapshot_returns_some_with_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.take_snapshot().is_some());
    }

    #[test]
    fn snapshot_roundtrip_preserves_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let snapshot = state.take_snapshot().unwrap();
        let body_count = state.mechanism.as_ref().unwrap().bodies().len();
        let joint_count = state.mechanism.as_ref().unwrap().joints().len();

        // Clobber state then restore
        state.load_sample(SampleMechanism::SliderCrank);
        state.restore_snapshot(&snapshot);

        let mech = state.mechanism.as_ref().unwrap();
        assert_eq!(mech.bodies().len(), body_count);
        assert_eq!(mech.joints().len(), joint_count);
        assert!(state.solver_status.converged);
    }

    #[test]
    fn snapshot_roundtrip_preserves_driver_fields() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.driver_omega = 5.0;
        state.driver_theta_0 = 1.0;
        state.driver_angle = 2.0;
        state.driver_joint_id = Some("J1".to_string());

        let snapshot = state.take_snapshot().unwrap();
        state.restore_snapshot(&snapshot);

        assert_eq!(state.driver_omega, 5.0);
        assert_eq!(state.driver_theta_0, 1.0);
        assert_eq!(state.driver_angle, 2.0);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
    }

    #[test]
    fn load_sample_clears_undo_history() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.push_undo();
        assert!(state.can_undo());

        state.load_sample(SampleMechanism::SliderCrank);
        assert!(!state.can_undo());
        assert!(!state.can_redo());
    }

    #[test]
    fn push_undo_noop_without_mechanism() {
        let mut state = AppState::default();
        state.push_undo();
        assert!(!state.can_undo());
    }

    #[test]
    fn undo_noop_without_mechanism() {
        let mut state = AppState::default();
        state.undo();
        assert!(!state.can_undo());
    }

    #[test]
    fn undo_after_driver_reassignment_restores_previous_driver() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(state.can_undo());

        state.undo();
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
        assert!(state.solver_status.converged);
    }

    #[test]
    fn redo_after_undo_restores_forward_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));

        state.undo();
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
        assert!(state.can_redo());

        state.redo();
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(!state.can_redo());
    }

    #[test]
    fn new_push_clears_redo_stack() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.reassign_driver("J4");
        state.undo();
        assert!(state.can_redo());

        // A new undoable action should clear redo
        state.push_undo();
        assert!(!state.can_redo());
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

    // ── Sweep tests ──────────────────────────────────────────────────────

    #[test]
    fn compute_sweep_produces_data_for_fourbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // load_sample calls compute_sweep internally
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some after load_sample");

        // FourBar is a Grashof crank-rocker -- full 361 points (0-360 inclusive).
        assert_eq!(
            sweep.angles_deg.len(),
            361,
            "Expected 361 sweep points, got {}",
            sweep.angles_deg.len()
        );

        // Body angles should exist for all 3 moving bodies.
        assert_eq!(sweep.body_angles.len(), 3);
        for (body_id, angles) in &sweep.body_angles {
            assert_eq!(
                angles.len(),
                361,
                "Body '{}' has {} angle entries, expected 361",
                body_id,
                angles.len()
            );
        }
    }

    #[test]
    fn compute_sweep_coupler_traces_non_empty() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let sweep = state.sweep_data.as_ref().unwrap();

        assert!(
            !sweep.coupler_traces.is_empty(),
            "coupler_traces should not be empty"
        );
        for (key, trace) in &sweep.coupler_traces {
            assert!(
                !trace.is_empty(),
                "trace for '{}' should not be empty",
                key
            );
        }
    }

    #[test]
    fn compute_sweep_fourbar_has_transmission_angle() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let sweep = state.sweep_data.as_ref().unwrap();

        let ta = sweep
            .transmission_angles
            .as_ref()
            .expect("FourBar should have transmission angles");
        assert_eq!(ta.len(), 361);
        // All transmission angles should be in (0, 180).
        for &angle in ta {
            assert!(
                angle > 0.0 && angle < 180.0,
                "transmission angle {} out of range",
                angle
            );
        }
    }

    #[test]
    fn compute_sweep_partial_for_non_crank() {
        // Double-rocker cannot complete full 360 rotation.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::DoubleRocker);
        let sweep = state.sweep_data.as_ref().unwrap();

        // Should have fewer than 361 points.
        assert!(
            sweep.angles_deg.len() < 361,
            "Double-rocker sweep should stop before 360, got {} points",
            sweep.angles_deg.len()
        );
        // But should have at least some data.
        assert!(
            !sweep.angles_deg.is_empty(),
            "Sweep should have at least some points"
        );
    }

    #[test]
    fn detect_fourbar_links_returns_some_for_fourbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let mech = state.mechanism.as_ref().unwrap();
        let links = detect_fourbar_links(mech);
        assert!(links.is_some(), "Should detect 4-bar link lengths");
        let (a, b, c, d) = links.unwrap();
        assert!(a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0);
    }

    #[test]
    fn detect_fourbar_links_returns_none_for_sixbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::SixBarB1);
        let mech = state.mechanism.as_ref().unwrap();
        let links = detect_fourbar_links(mech);
        assert!(links.is_none(), "Should not detect 4-bar links for 6-bar");
    }

    // ── Blueprint + rebuild tests ────────────────────────────────────────

    #[test]
    fn load_sample_populates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.blueprint.is_some(), "blueprint should be Some after load_sample");
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn blueprint_bodies_match_mechanism_bodies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let bp = state.blueprint.as_ref().unwrap();
        let mech = state.mechanism.as_ref().unwrap();
        assert_eq!(
            bp.bodies.len(),
            mech.bodies().len(),
            "Blueprint and mechanism should have the same number of bodies"
        );
        for body_id in mech.bodies().keys() {
            assert!(
                bp.bodies.contains_key(body_id),
                "Blueprint should contain body '{}'",
                body_id
            );
        }
    }

    #[test]
    fn rebuild_produces_valid_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Manually trigger rebuild
        state.rebuild();

        assert!(state.mechanism.is_some());
        assert!(
            state.solver_status.converged,
            "rebuild() should produce a converged mechanism, residual = {}",
            state.solver_status.residual_norm
        );
    }

    #[test]
    fn set_body_mass_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.set_body_mass("crank", 1.5);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(
            (crank.mass - 1.5).abs() < f64::EPSILON,
            "Blueprint mass should be 1.5, got {}",
            crank.mass
        );
        // Mechanism should still be valid after rebuild
        assert!(state.mechanism.is_some());
    }

    #[test]
    fn set_body_izz_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.set_body_izz("crank", 0.05);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(
            (crank.izz_cg - 0.05).abs() < f64::EPSILON,
            "Blueprint izz_cg should be 0.05, got {}",
            crank.izz_cg
        );
    }

    #[test]
    fn move_attachment_point_updates_blueprint_and_rebuilds() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Read original position of an attachment point
        let bp = state.blueprint.as_ref().unwrap();
        let orig = bp.bodies.get("crank").unwrap().attachment_points.get("B").unwrap();
        let orig_x = orig[0];
        let orig_y = orig[1];

        // Move it slightly
        let new_x = orig_x + 0.001;
        let new_y = orig_y + 0.001;
        state.move_attachment_point("crank", "B", new_x, new_y);

        // Check blueprint was updated
        let bp = state.blueprint.as_ref().unwrap();
        let moved = bp.bodies.get("crank").unwrap().attachment_points.get("B").unwrap();
        assert!(
            (moved[0] - new_x).abs() < f64::EPSILON,
            "Blueprint point X should be {}, got {}",
            new_x,
            moved[0]
        );
        assert!(
            (moved[1] - new_y).abs() < f64::EPSILON,
            "Blueprint point Y should be {}, got {}",
            new_y,
            moved[1]
        );

        // Mechanism should still be present (rebuild happened)
        assert!(state.mechanism.is_some());
    }

    #[test]
    fn load_from_file_populates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Save to file, then reload
        let path = std::env::temp_dir().join("linkage_test_blueprint.json");
        state.save_to_file(&path).expect("save_to_file failed");

        let mut state2 = AppState::default();
        state2.load_from_file(&path).expect("load_from_file failed");

        assert!(
            state2.blueprint.is_some(),
            "blueprint should be Some after load_from_file"
        );
        assert!(state2.mechanism.is_some());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn edit_nonexistent_body_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Edit a body that doesn't exist -- should not crash
        state.set_body_mass("nonexistent", 999.0);

        // Mechanism should still be valid (rebuild ran but nothing changed)
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn edit_nonexistent_point_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Edit a point that doesn't exist on the body -- should not crash
        state.move_attachment_point("crank", "nonexistent", 0.0, 0.0);

        // The rebuild still happens but the blueprint wasn't changed
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn rebuild_all_samples_have_blueprints() {
        for sample in SampleMechanism::all() {
            let mut state = AppState::default();
            state.load_sample(*sample);
            assert!(
                state.blueprint.is_some(),
                "Sample {:?} should have a blueprint after load_sample",
                sample
            );
            assert!(
                state.mechanism.is_some(),
                "Sample {:?} should have a mechanism after load_sample",
                sample
            );
        }
    }
}
