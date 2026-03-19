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
    load_mechanism_unbuilt, load_mechanism_unbuilt_from_json, mechanism_to_json,
    BodyJson, DriverJson, JointJson, LoadCaseJson, MechanismJson,
};
use crate::solver::kinematics::solve_position;
use crate::solver::statics::{
    extract_reactions, get_driver_reactions, get_joint_reactions, solve_statics,
};

// ── Display units ─────────────────────────────────────────────────────────────

/// Length unit preference for display. Solvers always use SI (meters) internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthUnit {
    Meters,
    Millimeters,
}

/// Angle unit preference for display. Solvers always use radians internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleUnit {
    Radians,
    Degrees,
}

/// Display unit preferences. All conversion happens at the display boundary —
/// solvers never see converted values.
pub struct DisplayUnits {
    pub length: LengthUnit,
    pub angle: AngleUnit,
}

impl Default for DisplayUnits {
    fn default() -> Self {
        Self {
            length: LengthUnit::Millimeters, // default to mm for engineering use
            angle: AngleUnit::Degrees,       // default to degrees
        }
    }
}

impl DisplayUnits {
    /// Convert meters to display length.
    pub fn length(&self, meters: f64) -> f64 {
        match self.length {
            LengthUnit::Meters => meters,
            LengthUnit::Millimeters => meters * 1000.0,
        }
    }

    /// Convert display length back to meters.
    pub fn length_to_si(&self, display: f64) -> f64 {
        match self.length {
            LengthUnit::Meters => display,
            LengthUnit::Millimeters => display / 1000.0,
        }
    }

    /// Length unit suffix string.
    pub fn length_suffix(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => " m",
            LengthUnit::Millimeters => " mm",
        }
    }

    /// Convert radians to display angle.
    pub fn angle(&self, radians: f64) -> f64 {
        match self.angle {
            AngleUnit::Radians => radians,
            AngleUnit::Degrees => radians.to_degrees(),
        }
    }

    /// Angle unit suffix string.
    pub fn angle_suffix(&self) -> &'static str {
        match self.angle {
            AngleUnit::Radians => " rad",
            AngleUnit::Degrees => "\u{00b0}",
        }
    }

    /// X/Y axis label for length plots.
    pub fn length_axis_label(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => "m",
            LengthUnit::Millimeters => "mm",
        }
    }
}

// ── Grid settings ─────────────────────────────────────────────────────────────

/// Grid display and snap-to-grid settings. Spacing is stored in meters (SI);
/// the UI converts to/from the active display unit at the display boundary.
pub struct GridSettings {
    /// Whether snap-to-grid is active during drag operations.
    pub snap_enabled: bool,
    /// Whether to draw the grid on the canvas.
    pub show_grid: bool,
    /// Grid spacing in meters (SI).
    pub spacing_m: f64,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 1.0,
        }
    }
}

impl GridSettings {
    /// Snap a single world-coordinate value to the nearest grid point.
    ///
    /// Returns the value unchanged when snapping is disabled or spacing is
    /// non-positive.
    pub fn snap(&self, value: f64) -> f64 {
        if !self.snap_enabled || self.spacing_m <= 0.0 {
            return value;
        }
        (value / self.spacing_m).round() * self.spacing_m
    }

    /// Snap an (x, y) world-coordinate pair to the nearest grid point.
    pub fn snap_point(&self, x: f64, y: f64) -> (f64, f64) {
        (self.snap(x), self.snap(y))
    }
}

// ── Load cases ────────────────────────────────────────────────────────────────

/// A named driver configuration on the same mechanism geometry.
///
/// Engineers use load cases to compare different operating conditions
/// (which joint is driven, speed, direction) without rebuilding the mechanism.
///
/// Type alias for `LoadCaseJson` — the canonical definition lives in
/// `io::serialization` so load cases can be serialized/deserialized
/// alongside the mechanism JSON.
pub type LoadCase = LoadCaseJson;

/// Manages multiple load cases for the current mechanism.
#[derive(Debug, Clone)]
pub struct LoadCaseManager {
    pub cases: Vec<LoadCase>,
    pub active_index: usize,
}

impl Default for LoadCaseManager {
    fn default() -> Self {
        Self {
            cases: Vec::new(),
            active_index: 0,
        }
    }
}

impl LoadCaseManager {
    /// Create a manager with a single default load case from the current driver settings.
    pub fn new_default(driver_joint_id: &str, omega: f64, theta_0: f64) -> Self {
        Self {
            cases: vec![LoadCase {
                name: "Default".to_string(),
                driver_joint_id: driver_joint_id.to_string(),
                omega,
                theta_0,
            }],
            active_index: 0,
        }
    }

    /// Add a new load case by copying the current driver settings.
    ///
    /// Returns the index of the newly added case.
    pub fn add_case(&mut self, driver_joint_id: &str, omega: f64, theta_0: f64) -> usize {
        let n = self.cases.len() + 1;
        self.cases.push(LoadCase {
            name: format!("Case {}", n),
            driver_joint_id: driver_joint_id.to_string(),
            omega,
            theta_0,
        });
        self.cases.len() - 1
    }

    /// Remove the load case at the given index.
    ///
    /// Returns false (no-op) if there is only one case remaining.
    pub fn remove_case(&mut self, index: usize) -> bool {
        if self.cases.len() <= 1 || index >= self.cases.len() {
            return false;
        }
        self.cases.remove(index);
        // Adjust active_index if it's out of bounds or was pointing at the removed case
        if self.active_index >= self.cases.len() {
            self.active_index = self.cases.len() - 1;
        }
        true
    }
}

// ── Selection ─────────────────────────────────────────────────────────────────

/// Which entity in the mechanism is currently selected for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

// ── Drag target ──────────────────────────────────────────────────────────────

/// Tracks which attachment point is being dragged on the canvas.
#[derive(Clone, Debug)]
pub struct DragTarget {
    /// Body that owns the attachment point being dragged.
    pub body_id: String,
    /// Name of the attachment point being dragged.
    pub point_name: String,
    /// True after the first movement (undo is pushed once at drag start).
    pub started: bool,
}

// ── Validation warnings ──────────────────────────────────────────────────────

/// Lightweight validation warnings computed after each rebuild.
#[derive(Clone, Debug, Default)]
pub struct ValidationWarnings {
    /// Grubler DOF mismatch: expected 0 for a fully-constrained driven mechanism.
    pub dof_warning: Option<String>,
    /// Bodies not connected by any joint.
    pub disconnected_bodies: Vec<String>,
    /// Mechanism has no driver.
    pub missing_driver: bool,
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

// ── Force results ─────────────────────────────────────────────────────────────

/// Results from static/inverse-dynamics force computation at the current pose.
#[derive(Debug, Clone, Default)]
pub struct ForceResults {
    /// Driver torque (N*m) -- the required input effort from the driver constraint.
    pub driver_torque: Option<f64>,
    /// Per-joint reaction forces in global frame: joint_id -> (Fx, Fy) in Newtons.
    pub joint_reactions: HashMap<String, (f64, f64)>,
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
    /// Driver torque (N*m) at each step, computed from statics.
    pub driver_torques: Option<Vec<f64>>,
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
    // ── Drag interaction ─────────────────────────────────────────────────
    /// Currently active drag target (attachment point being dragged).
    pub drag_target: Option<DragTarget>,
    // ── Joint creation mode ──────────────────────────────────────────────
    /// First click of a two-click joint creation: (body_id, point_name).
    pub creating_joint: Option<(String, String)>,
    // ── Validation ───────────────────────────────────────────────────────
    /// Validation warnings computed after each rebuild.
    pub validation_warnings: ValidationWarnings,
    // ── Display units ────────────────────────────────────────────────────
    /// Unit preferences for display. Solvers remain SI internally.
    pub display_units: DisplayUnits,
    // ── Grid ─────────────────────────────────────────────────────────────
    /// Grid display and snap-to-grid settings.
    pub grid: GridSettings,
    // ── Force visualization ─────────────────────────────────────────────
    /// Force computation results at the current pose.
    pub force_results: ForceResults,
    /// Whether to draw force arrows on the canvas.
    pub show_forces: bool,
    // ── Load cases ──────────────────────────────────────────────────────
    /// Named driver configurations for comparing operating conditions.
    pub load_cases: LoadCaseManager,
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
            drag_target: None,
            creating_joint: None,
            validation_warnings: ValidationWarnings::default(),
            display_units: DisplayUnits::default(),
            grid: GridSettings::default(),
            force_results: ForceResults::default(),
            show_forces: false,
            load_cases: LoadCaseManager::default(),
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
        // Initialize default load case from current driver settings
        self.load_cases = if let Some(ref joint_id) = self.driver_joint_id {
            LoadCaseManager::new_default(joint_id, self.driver_omega, self.driver_theta_0)
        } else {
            LoadCaseManager::default()
        };

        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;
        self.undo_history.clear();
        self.auto_grid_spacing();
        self.compute_forces(0.0);
        self.compute_sweep();
        self.compute_validation();
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
                    self.compute_forces(t);
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

    /// Compute static force results (joint reactions + driver torque) at the
    /// current pose. Called after each successful position solve.
    ///
    /// Uses the statics solver (no inertial effects) since the GUI currently
    /// operates at quasi-static conditions. Falls back gracefully on failure,
    /// clearing force results rather than propagating errors.
    fn compute_forces(&mut self, t: f64) {
        let Some(mech) = &self.mechanism else {
            self.force_results = ForceResults::default();
            return;
        };

        // Guard: only compute forces when the solver converged and q has the
        // correct dimension (prevents panics during partial rebuilds).
        if !self.solver_status.converged
            || self.q.len() != mech.state().n_coords()
            || mech.n_drivers() == 0
        {
            self.force_results = ForceResults::default();
            return;
        }

        // Statics solve: no gravity for now (Q = 0), gives constraint-force-only results.
        // The driver torque is always meaningful even without external loads.
        let statics_result = match solve_statics(mech, &self.q, None, t) {
            Ok(r) => r,
            Err(_) => {
                self.force_results = ForceResults::default();
                return;
            }
        };

        let reactions = extract_reactions(mech, &statics_result);

        // Extract driver torque.
        let driver_torque = get_driver_reactions(&reactions)
            .first()
            .map(|r| r.effort);

        // Extract per-joint reaction forces.
        let mut joint_reactions = HashMap::new();
        for jr in get_joint_reactions(&reactions) {
            joint_reactions.insert(
                jr.joint_id.clone(),
                (jr.force_global[0], jr.force_global[1]),
            );
        }

        self.force_results = ForceResults {
            driver_torque,
            joint_reactions,
        };
    }

    /// Returns true if a mechanism has been loaded.
    pub fn has_mechanism(&self) -> bool {
        self.mechanism.is_some()
    }

    /// Serialize the current mechanism to a JSON file at the given path.
    ///
    /// Load cases are included in the saved JSON so they persist across
    /// save/load cycles.
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let mech = self
            .mechanism
            .as_ref()
            .ok_or_else(|| "No mechanism loaded".to_string())?;

        let mut json_struct = mechanism_to_json(mech).map_err(|e| e.to_string())?;

        // Persist load cases into the JSON structure
        json_struct.load_cases = self.load_cases.cases.clone();

        let json = serde_json::to_string_pretty(&json_struct).map_err(|e| e.to_string())?;
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

        // Restore load cases from the blueprint, or create a default one
        if let Some(ref bp) = self.blueprint {
            if !bp.load_cases.is_empty() {
                self.load_cases = LoadCaseManager {
                    cases: bp.load_cases.clone(),
                    active_index: 0,
                };
            } else if let Some(ref joint_id) = self.driver_joint_id {
                self.load_cases = LoadCaseManager::new_default(
                    joint_id,
                    self.driver_omega,
                    self.driver_theta_0,
                );
            } else {
                self.load_cases = LoadCaseManager::default();
            }
        } else if let Some(ref joint_id) = self.driver_joint_id {
            self.load_cases =
                LoadCaseManager::new_default(joint_id, self.driver_omega, self.driver_theta_0);
        } else {
            self.load_cases = LoadCaseManager::default();
        }

        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;
        self.undo_history.clear();
        self.auto_grid_spacing();
        self.compute_forces(0.0);
        self.compute_sweep();
        self.compute_validation();

        Ok(())
    }

    // ── Grid auto-spacing ─────────────────────────────────────────────────

    /// Set grid spacing based on the bounding box of all attachment points
    /// in the current blueprint. Picks a "clean" spacing that gives roughly
    /// 10-20 grid cells across the largest dimension.
    pub fn auto_grid_spacing(&mut self) {
        let Some(bp) = &self.blueprint else { return };

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut count = 0usize;

        for body in bp.bodies.values() {
            for pt in body.attachment_points.values() {
                x_min = x_min.min(pt[0]);
                x_max = x_max.max(pt[0]);
                y_min = y_min.min(pt[1]);
                y_max = y_max.max(pt[1]);
                count += 1;
            }
        }

        if count < 2 {
            return; // Not enough points to determine scale
        }

        let extent = (x_max - x_min).max(y_max - y_min);
        if extent <= 0.0 || !extent.is_finite() {
            return;
        }

        // Target ~10 grid cells across the largest dimension.
        // Round down to the nearest "clean" value from a fixed set.
        let raw = extent / 10.0;
        const CLEAN: [f64; 12] = [
            10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001,
        ];
        self.grid.spacing_m = CLEAN
            .iter()
            .copied()
            .find(|&c| c <= raw)
            .unwrap_or(0.001);
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
        self.compute_forces(t);
        self.compute_validation();
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

    // ── Create / delete operations ──────────────────────────────────────

    /// Add a new ground pivot (attachment point on the ground body).
    ///
    /// Pushes undo, adds the point, and rebuilds.
    pub fn add_ground_pivot(&mut self, name: &str, x: f64, y: f64) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let ground = bp.bodies.entry(GROUND_ID.to_string()).or_insert_with(|| BodyJson {
            attachment_points: HashMap::new(),
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            coupler_points: HashMap::new(),
        });
        ground.attachment_points.insert(name.to_string(), [x, y]);
        self.rebuild();
    }

    /// Add a new binary body with two attachment points.
    ///
    /// Creates a body with two points at `p1` and `p2`, default mass properties.
    /// Pushes undo, adds the body, and rebuilds.
    pub fn add_body(&mut self, body_id: &str, p1: (&str, f64, f64), p2: (&str, f64, f64)) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let mut attachment_points = HashMap::new();
        attachment_points.insert(p1.0.to_string(), [p1.1, p1.2]);
        attachment_points.insert(p2.0.to_string(), [p2.1, p2.2]);
        let cx = (p1.1 + p2.1) / 2.0;
        let cy = (p1.2 + p2.2) / 2.0;
        let body = BodyJson {
            attachment_points,
            mass: 1.0,
            cg_local: [cx, cy],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
        };
        bp.bodies.insert(body_id.to_string(), body);
        self.rebuild();
    }

    /// Remove a body and all joints/drivers that reference it.
    ///
    /// Pushes undo, removes the body and cascading references, and rebuilds.
    pub fn remove_body(&mut self, body_id: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };

        bp.bodies.remove(body_id);

        // Remove joints that reference this body.
        bp.joints.retain(|_id, joint| {
            let (bi, bj) = joint_body_ids(joint);
            bi != body_id && bj != body_id
        });

        // Remove drivers that reference this body.
        bp.drivers.retain(|_id, driver| {
            let (bi, bj) = driver_body_ids(driver);
            bi != body_id && bj != body_id
        });

        self.rebuild();
    }

    /// Add a revolute joint between two body attachment points.
    ///
    /// Generates a unique joint ID. Pushes undo, adds joint, and rebuilds.
    pub fn add_revolute_joint(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };

        let joint_id = generate_unique_id("J", &bp.joints);
        bp.joints.insert(
            joint_id,
            JointJson::Revolute {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
            },
        );
        self.rebuild();
    }

    /// Remove a joint by ID.
    ///
    /// Pushes undo, removes the joint, and rebuilds.
    pub fn remove_joint(&mut self, joint_id: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        bp.joints.remove(joint_id);
        self.rebuild();
    }

    /// Generate a unique body ID (e.g. "body_1", "body_2", ...).
    pub fn next_body_id(&self) -> String {
        let Some(bp) = &self.blueprint else {
            return "body_1".to_string();
        };
        generate_unique_id("body_", &bp.bodies)
    }

    /// Generate a unique ground pivot name (e.g. "P1", "P2", ...).
    pub fn next_ground_pivot_name(&self) -> String {
        let Some(bp) = &self.blueprint else {
            return "P1".to_string();
        };
        if let Some(ground) = bp.bodies.get(GROUND_ID) {
            let mut i = 1;
            loop {
                let name = format!("P{}", i);
                if !ground.attachment_points.contains_key(&name) {
                    return name;
                }
                i += 1;
            }
        } else {
            "P1".to_string()
        }
    }

    // ── Validation ──────────────────────────────────────────────────────

    /// Compute validation warnings from the current mechanism state.
    ///
    /// Called after each rebuild to update `self.validation_warnings`.
    pub fn compute_validation(&mut self) {
        let mut warnings = ValidationWarnings::default();

        if let Some(mech) = &self.mechanism {
            // DOF check: 3 * n_moving - n_constraints should be 0
            let n_coords = mech.state().n_coords() as isize;
            let n_constraints = mech.n_constraints() as isize;
            let dof = n_coords - n_constraints;
            if dof != 0 {
                warnings.dof_warning = Some(format!(
                    "DOF = {} (coords={}, constraints={})",
                    dof, n_coords, n_constraints
                ));
            }

            // Missing driver check
            warnings.missing_driver = mech.n_drivers() == 0;

            // Disconnected body check: a body that has no joints connecting to it
            if let Some(bp) = &self.blueprint {
                for body_id in bp.bodies.keys() {
                    if body_id == GROUND_ID {
                        continue;
                    }
                    let connected = bp.joints.values().any(|j| {
                        let (bi, bj) = joint_body_ids(j);
                        bi == body_id || bj == body_id
                    });
                    if !connected {
                        warnings.disconnected_bodies.push(body_id.clone());
                    }
                }
                warnings.disconnected_bodies.sort();
            }
        }

        self.validation_warnings = warnings;
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

                // Reset load cases to a single default for the new driver
                self.load_cases = LoadCaseManager::new_default(
                    joint_id,
                    self.driver_omega,
                    self.driver_theta_0,
                );

                self.playing = false;
                self.animation_direction = 1.0;
                self.pending_driver_reassignment = None;
                self.compute_sweep();
                self.compute_validation();
            }
            Err(msg) => {
                log::warn!("Driver reassignment failed: {}", msg);
            }
        }
    }

    // ── Load case operations ──────────────────────────────────────────────

    /// Add a new load case by copying the current driver settings.
    pub fn add_load_case(&mut self) {
        let driver_joint_id = self
            .driver_joint_id
            .clone()
            .unwrap_or_default();
        self.load_cases.add_case(&driver_joint_id, self.driver_omega, self.driver_theta_0);
    }

    /// Remove the currently active load case.
    ///
    /// No-op if only one case remains. After removal, applies the new active case.
    pub fn remove_active_load_case(&mut self) {
        let index = self.load_cases.active_index;
        if self.load_cases.remove_case(index) {
            self.apply_load_case(self.load_cases.active_index);
        }
    }

    /// Switch to and apply the load case at the given index.
    ///
    /// Updates driver settings from the load case. If the driver joint differs
    /// from the current one, triggers a driver reassignment via the pending
    /// mechanism. Otherwise just updates omega/theta_0 and re-solves.
    pub fn apply_load_case(&mut self, index: usize) {
        if index >= self.load_cases.cases.len() {
            return;
        }

        self.push_undo();
        self.load_cases.active_index = index;

        let case = self.load_cases.cases[index].clone();

        let current_joint = self.driver_joint_id.clone().unwrap_or_default();

        if case.driver_joint_id != current_joint {
            // Different driver joint -- need to reassign.
            // Store the load case driver params so they survive reassignment,
            // then trigger the rebuild via the pending reassignment path.
            self.driver_omega = case.omega;
            self.driver_theta_0 = case.theta_0;
            self.pending_driver_reassignment = Some(case.driver_joint_id.clone());
        } else {
            // Same driver joint -- just update speed and angle.
            self.driver_omega = case.omega;
            self.driver_theta_0 = case.theta_0;
            self.driver_angle = case.theta_0;
            self.solve_at_angle(case.theta_0);
            self.compute_sweep();
        }
    }

    /// Sync the active load case from the current driver state.
    ///
    /// Called when the user changes driver settings (omega, theta_0, or joint)
    /// so the active load case stays in sync.
    pub fn sync_active_load_case(&mut self) {
        if let Some(case) = self.load_cases.cases.get_mut(self.load_cases.active_index) {
            if let Some(ref joint_id) = self.driver_joint_id {
                case.driver_joint_id = joint_id.clone();
            }
            case.omega = self.driver_omega;
            case.theta_0 = self.driver_theta_0;
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

        // Restore the blueprint from the snapshot JSON so it stays in sync.
        self.blueprint = serde_json::from_str(&snapshot.mechanism_json).ok();

        self.mechanism = Some(mech);
        self.driver_angle = snapshot.driver_angle;
        self.driver_omega = snapshot.driver_omega;
        self.driver_theta_0 = snapshot.driver_theta_0;
        self.driver_joint_id = snapshot.driver_joint_id.clone();
        self.playing = false;
        self.compute_validation();
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

// ── Blueprint helper functions ────────────────────────────────────────────────

/// Extract the body_i and body_j IDs from a JointJson.
fn joint_body_ids(joint: &JointJson) -> (&str, &str) {
    match joint {
        JointJson::Revolute { body_i, body_j, .. }
        | JointJson::Fixed { body_i, body_j, .. }
        | JointJson::Prismatic { body_i, body_j, .. }
        | JointJson::RevoluteDriver { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
    }
}

/// Extract the body_i and body_j IDs from a DriverJson.
fn driver_body_ids(driver: &DriverJson) -> (&str, &str) {
    match driver {
        DriverJson::ConstantSpeed { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
    }
}

/// Generate a unique ID with the given prefix in a HashMap.
///
/// Tries prefix + "1", prefix + "2", ... until an unused key is found.
fn generate_unique_id<V>(prefix: &str, map: &HashMap<String, V>) -> String {
    let mut i = 1;
    loop {
        let id = format!("{}{}", prefix, i);
        if !map.contains_key(&id) {
            return id;
        }
        i += 1;
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
        driver_torques: Some(Vec::with_capacity(361)),
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

                // Driver torque from statics solve.
                if let Ok(statics) = solve_statics(mech, &q, None, t) {
                    let reactions = extract_reactions(mech, &statics);
                    let torque = get_driver_reactions(&reactions)
                        .first()
                        .map(|r| r.effort)
                        .unwrap_or(0.0);
                    data.driver_torques.as_mut().unwrap().push(torque);
                } else {
                    data.driver_torques.as_mut().unwrap().push(0.0);
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

    // ── Create / delete tests ───────────────────────────────────────────

    #[test]
    fn add_ground_pivot_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let n_before = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        state.add_ground_pivot("O5", 5.0, 0.0);
        let n_after = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        assert_eq!(n_after, n_before + 1);
        // The new point should exist with the right coordinates.
        let pt = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .get("O5")
            .unwrap();
        assert!((pt[0] - 5.0).abs() < f64::EPSILON);
        assert!((pt[1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn add_ground_pivot_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_before = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();

        state.add_ground_pivot("P99", 1.0, 2.0);
        assert!(state.can_undo());

        state.undo();
        let n_after = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        assert_eq!(n_after, n_before);
    }

    #[test]
    fn add_body_creates_new_body_in_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_bodies_before = state.blueprint.as_ref().unwrap().bodies.len();

        state.add_body("new_link", ("X", 0.0, 0.0), ("Y", 0.05, 0.0));

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), n_bodies_before + 1);
        assert!(bp.bodies.contains_key("new_link"));
        let body = bp.bodies.get("new_link").unwrap();
        assert_eq!(body.attachment_points.len(), 2);
        assert!(body.attachment_points.contains_key("X"));
        assert!(body.attachment_points.contains_key("Y"));
    }

    #[test]
    fn remove_body_cascades_to_joints() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let joints_before = state.blueprint.as_ref().unwrap().joints.len();

        state.remove_body("coupler");

        let bp = state.blueprint.as_ref().unwrap();
        assert!(!bp.bodies.contains_key("coupler"));
        // Joints connected to coupler should be removed.
        assert!(
            bp.joints.len() < joints_before,
            "Expected fewer joints after removing coupler, got {} (was {})",
            bp.joints.len(),
            joints_before
        );
        // No remaining joint should reference "coupler".
        for (_id, joint) in &bp.joints {
            let (bi, bj) = joint_body_ids(joint);
            assert_ne!(bi, "coupler", "Joint still references removed body");
            assert_ne!(bj, "coupler", "Joint still references removed body");
        }
    }

    #[test]
    fn remove_body_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_bodies = state.blueprint.as_ref().unwrap().bodies.len();
        let n_joints = state.blueprint.as_ref().unwrap().joints.len();

        state.remove_body("crank");
        assert!(state.can_undo());

        state.undo();
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), n_bodies);
        assert_eq!(bp.joints.len(), n_joints);
    }

    #[test]
    fn add_revolute_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints_before = state.blueprint.as_ref().unwrap().joints.len();

        // Add a new body first so we have a valid target.
        state.add_body("extra", ("E1", 0.0, 0.0), ("E2", 0.01, 0.0));

        // Add joint between ground and the new body.
        state.add_revolute_joint("ground", "O2", "extra", "E1");

        let bp = state.blueprint.as_ref().unwrap();
        // +1 from the new joint (the add_body doesn't add joints).
        assert_eq!(bp.joints.len(), n_joints_before + 1);
    }

    #[test]
    fn remove_joint_removes_from_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints_before = state.blueprint.as_ref().unwrap().joints.len();

        // Get the first joint ID.
        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), n_joints_before - 1);
        assert!(!bp.joints.contains_key(&joint_id));
    }

    #[test]
    fn remove_joint_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints = state.blueprint.as_ref().unwrap().joints.len();

        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);
        assert!(state.can_undo());

        state.undo();
        assert_eq!(state.blueprint.as_ref().unwrap().joints.len(), n_joints);
    }

    #[test]
    fn next_body_id_generates_unique_ids() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let id1 = state.next_body_id();
        state.add_body(&id1, ("A", 0.0, 0.0), ("B", 0.01, 0.0));

        let id2 = state.next_body_id();
        assert_ne!(id1, id2, "Second ID should differ from first");
        assert!(
            !state.blueprint.as_ref().unwrap().bodies.contains_key(&id2),
            "Generated ID should not already exist in blueprint"
        );
    }

    #[test]
    fn next_ground_pivot_name_generates_unique_names() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let name1 = state.next_ground_pivot_name();
        state.add_ground_pivot(&name1, 1.0, 0.0);

        let name2 = state.next_ground_pivot_name();
        assert_ne!(name1, name2);
    }

    // ── Validation tests ────────────────────────────────────────────────

    #[test]
    fn validation_no_warnings_for_valid_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar has driver, correct DOF, no disconnected bodies.
        state.compute_validation();

        assert!(
            state.validation_warnings.dof_warning.is_none(),
            "FourBar should have DOF=0, got: {:?}",
            state.validation_warnings.dof_warning
        );
        assert!(
            !state.validation_warnings.missing_driver,
            "FourBar should have a driver"
        );
        assert!(
            state.validation_warnings.disconnected_bodies.is_empty(),
            "FourBar should have no disconnected bodies"
        );
    }

    #[test]
    fn validation_disconnected_body_detected() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Add a body with no joints.
        state.add_body("floating", ("F1", 0.0, 0.5), ("F2", 0.01, 0.5));
        state.compute_validation();

        assert!(
            state.validation_warnings.disconnected_bodies.contains(&"floating".to_string()),
            "Should detect 'floating' as disconnected"
        );
    }

    #[test]
    fn validation_dof_warning_after_removing_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Remove a joint -- DOF should no longer be 0.
        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);

        // Validation is computed during rebuild.
        assert!(
            state.validation_warnings.dof_warning.is_some(),
            "Should have DOF warning after removing a joint"
        );
    }

    #[test]
    fn remove_body_cascades_to_drivers() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // The driver references ground + crank. Removing crank should
        // also remove the driver.
        let drivers_before = state.blueprint.as_ref().unwrap().drivers.len();
        assert!(drivers_before > 0, "FourBar should have a driver");

        state.remove_body("crank");

        let bp = state.blueprint.as_ref().unwrap();
        assert!(
            bp.drivers.is_empty(),
            "Drivers referencing removed body should be cascaded"
        );
    }

    // ── Grid snap tests ──────────────────────────────────────────────────

    #[test]
    fn snap_to_grid_rounds_to_nearest() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 1.0,
        };
        assert_eq!(grid.snap(2.3), 2.0);
        assert_eq!(grid.snap(2.7), 3.0);
        assert_eq!(grid.snap(-0.4), 0.0);
        assert_eq!(grid.snap(0.5), 1.0); // half-away-from-zero: 0.5 rounds to 1
        assert_eq!(grid.snap(1.5), 2.0); // half-away-from-zero: 1.5 rounds to 2
    }

    #[test]
    fn snap_disabled_passes_through() {
        let grid = GridSettings {
            snap_enabled: false,
            show_grid: true,
            spacing_m: 1.0,
        };
        assert_eq!(grid.snap(2.3), 2.3);
        assert_eq!(grid.snap(-7.777), -7.777);
    }

    #[test]
    fn snap_zero_spacing_passes_through() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.0,
        };
        assert_eq!(grid.snap(2.3), 2.3);
    }

    #[test]
    fn snap_point_snaps_both_axes() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.5,
        };
        let (sx, sy) = grid.snap_point(1.3, -0.2);
        assert!((sx - 1.5).abs() < 1e-12);
        assert!((sy - 0.0).abs() < 1e-12);
    }

    #[test]
    fn snap_fine_spacing() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.005,
        };
        // 0.0123 is closest to 0.010 (2.46 grid units, rounds to 2)
        // Actually 0.0123 / 0.005 = 2.46, rounds to 2 -> 0.010
        let result = grid.snap(0.0123);
        assert!((result - 0.010).abs() < 1e-12, "got {}", result);
    }

    #[test]
    fn auto_grid_spacing_fourbar_small_scale() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar has ground=0.038m. Expect a fine grid spacing.
        assert!(
            state.grid.spacing_m >= 0.001 && state.grid.spacing_m <= 0.01,
            "FourBar grid spacing should be 1-10 mm, got {} m",
            state.grid.spacing_m
        );
    }

    #[test]
    fn auto_grid_spacing_crank_rocker_large_scale() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        // CrankRocker has d=4m ground. Expect a coarser grid.
        assert!(
            state.grid.spacing_m >= 0.1 && state.grid.spacing_m <= 2.0,
            "CrankRocker grid spacing should be 0.1-2.0 m, got {} m",
            state.grid.spacing_m
        );
    }

    // ── Load case tests ───────────────────────────────────────────────

    #[test]
    fn default_load_case_created_on_sample_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases.len(), 1);
        assert_eq!(state.load_cases.cases[0].name, "Default");
        assert_eq!(state.load_cases.active_index, 0);
    }

    #[test]
    fn add_load_case_copies_current() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);
        assert_eq!(state.load_cases.cases[1].name, "Case 2");
        // New case should have same driver params as current
        assert_eq!(
            state.load_cases.cases[1].omega,
            state.driver_omega,
        );
    }

    #[test]
    fn remove_load_case_prevents_last_removal() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases.len(), 1);
        // Should not be able to remove the last case
        state.remove_active_load_case();
        assert_eq!(state.load_cases.cases.len(), 1);
    }

    #[test]
    fn remove_load_case_works_with_multiple() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);
        state.remove_active_load_case();
        assert_eq!(state.load_cases.cases.len(), 1);
    }

    #[test]
    fn switch_load_case_changes_omega() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);

        let original_omega = state.driver_omega;
        let new_omega = original_omega * 2.0;

        // Add a case with different omega
        state.load_cases.cases.push(LoadCase {
            name: "Fast".to_string(),
            driver_joint_id: state.driver_joint_id.clone().unwrap(),
            omega: new_omega,
            theta_0: 0.0,
        });

        state.apply_load_case(1);
        assert_eq!(state.driver_omega, new_omega);
        assert_eq!(state.load_cases.active_index, 1);
    }

    #[test]
    fn switch_load_case_changes_driver_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);

        // Add a case targeting a different joint (J4 is the rocker pivot)
        state.load_cases.cases.push(LoadCase {
            name: "Rocker Drive".to_string(),
            driver_joint_id: "J4".to_string(),
            omega: 2.0 * PI,
            theta_0: 0.0,
        });

        state.apply_load_case(1);
        // The pending_driver_reassignment should be set for the next frame
        assert_eq!(
            state.pending_driver_reassignment,
            Some("J4".to_string()),
        );
    }

    #[test]
    fn load_case_manager_new_default() {
        let mgr = LoadCaseManager::new_default("J1", 2.0 * PI, 0.5);
        assert_eq!(mgr.cases.len(), 1);
        assert_eq!(mgr.cases[0].name, "Default");
        assert_eq!(mgr.cases[0].driver_joint_id, "J1");
        assert_eq!(mgr.cases[0].omega, 2.0 * PI);
        assert_eq!(mgr.cases[0].theta_0, 0.5);
        assert_eq!(mgr.active_index, 0);
    }

    #[test]
    fn load_case_manager_add_case_returns_index() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        let idx = mgr.add_case("J2", 2.0 * PI, 1.0);
        assert_eq!(idx, 1);
        assert_eq!(mgr.cases.len(), 2);
        assert_eq!(mgr.cases[1].driver_joint_id, "J2");
    }

    #[test]
    fn load_case_manager_remove_adjusts_active_index() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        mgr.add_case("J2", 2.0 * PI, 0.0);
        mgr.add_case("J3", 3.0 * PI, 0.0);
        mgr.active_index = 2; // point to last

        mgr.remove_case(2);
        // active_index should be clamped to the last valid index
        assert_eq!(mgr.active_index, 1);
        assert_eq!(mgr.cases.len(), 2);
    }

    #[test]
    fn load_case_manager_remove_single_case_is_noop() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        assert!(!mgr.remove_case(0));
        assert_eq!(mgr.cases.len(), 1);
    }

    #[test]
    fn sync_active_load_case_updates_params() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases[0].omega, 2.0 * PI);

        state.driver_omega = 10.0;
        state.sync_active_load_case();
        assert_eq!(state.load_cases.cases[0].omega, 10.0);
    }

    #[test]
    fn load_cases_persist_through_save_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_load_case();
        state.load_cases.cases[1].name = "High Speed".to_string();
        state.load_cases.cases[1].omega = 10.0 * PI;
        assert_eq!(state.load_cases.cases.len(), 2);

        let path = std::env::temp_dir().join("linkage_test_load_cases.json");
        state.save_to_file(&path).expect("save_to_file failed");

        let mut state2 = AppState::default();
        state2.load_from_file(&path).expect("load_from_file failed");

        assert_eq!(state2.load_cases.cases.len(), 2);
        assert_eq!(state2.load_cases.cases[0].name, "Default");
        assert_eq!(state2.load_cases.cases[1].name, "High Speed");
        assert!(
            (state2.load_cases.cases[1].omega - 10.0 * PI).abs() < 1e-10,
            "omega should be preserved, got {}",
            state2.load_cases.cases[1].omega,
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reassign_driver_resets_load_cases() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);

        state.reassign_driver("J4");
        // After reassignment, load cases should be reset to a single default
        assert_eq!(state.load_cases.cases.len(), 1);
        assert_eq!(state.load_cases.cases[0].name, "Default");
        assert_eq!(state.load_cases.cases[0].driver_joint_id, "J4");
    }

    #[test]
    fn load_sample_no_driver_has_empty_load_cases() {
        // Build a state, check that samples without a driver don't crash
        let state = AppState::default();
        assert!(state.load_cases.cases.is_empty());
    }

    #[test]
    fn apply_load_case_out_of_bounds_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let omega_before = state.driver_omega;
        state.apply_load_case(999);
        assert_eq!(state.driver_omega, omega_before);
    }
}
