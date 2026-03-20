//! Application state: mechanism, solver results, selection, view transform.

use nalgebra::DVector;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use crate::analysis::grashof::{check_grashof, GrashofResult};
use crate::analysis::transmission::{
    mechanical_advantage, transmission_angle_fourbar, VelocityCoord,
};
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
use crate::forces::elements::{ForceElement, GravityElement};
use crate::analysis::coupler::eval_coupler_point;
use crate::analysis::energy::compute_energy_state_mech;
use crate::analysis::force_breakdown::evaluate_contributions;
use crate::analysis::validation::check_toggle;
use crate::analysis::virtual_work::virtual_work_check;
use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use crate::solver::inverse_dynamics::solve_inverse_dynamics;
use crate::solver::forward_dynamics::{simulate, ForwardDynamicsConfig};
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

// ── Editor tool ───────────────────────────────────────────────────────────────

/// Joint type for the two-click creation flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingJointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// Active editor tool — determines what happens on canvas clicks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorTool {
    /// Default: click to select entities. Drag empty space to pan.
    Select,
    /// Draw Link: click a point (or empty space for ground pivot), drag to
    /// another point → creates bar + auto-creates revolute joints at both
    /// ends if they connect to existing points.
    DrawLink,
    /// Multi-click body placement: click to place attachment points, then
    /// confirm to create a body with those points.
    AddBody,
    /// Click canvas to place a new ground pivot.
    AddGroundPivot,
}

// ── Context menu target ──────────────────────────────────────────────────────

/// Stores what was under the cursor when a right-click occurred.
///
/// egui's `context_menu()` closure runs every frame while the menu is open,
/// but `secondary_clicked()` is only true on the trigger frame. This struct
/// persists the hit-test result so the menu content stays correct.
#[derive(Debug, Clone, Default)]
pub struct ContextMenuTarget {
    /// Joint ID if right-click landed on a joint.
    pub joint_id: Option<String>,
    /// Attachment point under cursor (body_id, point_name).
    /// Takes priority over body_area.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) -- only set when no
    /// attachment point is within HIT_RADIUS.
    pub body_area: Option<String>,
    /// World coordinates of the right-click position.
    pub world_pos: Option<[f64; 2]>,
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
    /// Condition number of the constraint Jacobian at this pose.
    pub condition_number: Option<f64>,
    /// True if the statics solver detected an overconstrained system.
    pub is_overconstrained: bool,
    /// Mechanical advantage (output/input angular velocity ratio) at current pose.
    pub mechanical_advantage: Option<f64>,
    /// Per-element force contribution norms at the current pose.
    pub force_contributions: Vec<(String, f64)>,
    /// Virtual work cross-check result: (vw_torque, lagrange_torque, agrees).
    pub virtual_work_check: Option<(f64, f64, bool)>,
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
    /// Kinetic energy at each sweep step (Joules). Requires velocity solve.
    pub kinetic_energy: Vec<f64>,
    /// Gravitational potential energy at each sweep step (Joules).
    pub potential_energy: Vec<f64>,
    /// Total mechanical energy (KE + PE) at each sweep step (Joules).
    pub total_energy: Vec<f64>,
    /// Inverse dynamics driver torque at each sweep step (N·m).
    /// Includes inertial effects (unlike the statics-based driver_torques).
    pub inverse_dynamics_torques: Vec<f64>,
    /// Mechanical advantage (output/input angular velocity ratio) at each
    /// sweep step. Requires velocity solve and a detectable driver body pair.
    pub mechanical_advantage: Vec<f64>,
    /// Per-joint reaction force magnitudes (N) over the sweep.
    /// Key: joint_id, Value: vec of resultant force magnitudes at each step.
    pub joint_reaction_magnitudes: HashMap<String, Vec<f64>>,
    /// Coupler point velocity magnitudes over the sweep.
    /// Key: trace name (same as coupler_traces), Value: velocity magnitude (m/s) at each step.
    pub coupler_velocities: HashMap<String, Vec<f64>>,
    /// Coupler point acceleration magnitudes over the sweep.
    /// Key: trace name, Value: acceleration magnitude (m/s^2) at each step.
    pub coupler_accelerations: HashMap<String, Vec<f64>>,
    /// Angles (degrees) at which toggle/dead points were detected.
    pub toggle_angles: Vec<f64>,
}

// ── Parametric study ──────────────────────────────────────────────────────────

/// A parameter that can be swept in a parametric study.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepParameter {
    /// Body mass (kg). Value: body_id.
    BodyMass(String),
    /// Body moment of inertia about CG (kg*m^2). Value: body_id.
    BodyIzz(String),
    /// Attachment point X coordinate (m). Value: (body_id, point_name).
    AttachmentX(String, String),
    /// Attachment point Y coordinate (m). Value: (body_id, point_name).
    AttachmentY(String, String),
    /// Force element scalar parameter. Value: (force_index, field_name).
    ForceParam(usize, String),
    /// Driver angular velocity (rad/s).
    DriverOmega,
}

impl SweepParameter {
    /// Human-readable label for the parameter.
    pub fn label(&self) -> String {
        match self {
            Self::BodyMass(id) => format!("{} mass (kg)", id),
            Self::BodyIzz(id) => format!("{} Izz (kg*m^2)", id),
            Self::AttachmentX(body, pt) => format!("{}.{} x (m)", body, pt),
            Self::AttachmentY(body, pt) => format!("{}.{} y (m)", body, pt),
            Self::ForceParam(idx, field) => format!("Force[{}].{}", idx, field),
            Self::DriverOmega => "Driver omega (rad/s)".to_string(),
        }
    }
}

/// Which output metric to plot on the Y-axis of a parametric study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametricMetric {
    PeakDriverTorque,
    RmsDriverTorque,
    MinTransmissionAngle,
    MaxTransmissionAngle,
    PeakReaction,
    PeakKineticEnergy,
    MeanMechanicalAdvantage,
}

impl ParametricMetric {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PeakDriverTorque => "Peak Driver Torque (N*m)",
            Self::RmsDriverTorque => "RMS Driver Torque (N*m)",
            Self::MinTransmissionAngle => "Min Transmission Angle (deg)",
            Self::MaxTransmissionAngle => "Max Transmission Angle (deg)",
            Self::PeakReaction => "Peak Joint Reaction (N)",
            Self::PeakKineticEnergy => "Peak Kinetic Energy (J)",
            Self::MeanMechanicalAdvantage => "Mean Mechanical Advantage",
        }
    }

    pub fn all() -> &'static [ParametricMetric] {
        &[
            Self::PeakDriverTorque,
            Self::RmsDriverTorque,
            Self::MinTransmissionAngle,
            Self::MaxTransmissionAngle,
            Self::PeakReaction,
            Self::PeakKineticEnergy,
            Self::MeanMechanicalAdvantage,
        ]
    }

    /// Extract a scalar value from a sweep dataset for this metric.
    pub fn extract(&self, sweep: &SweepData) -> f64 {
        match self {
            Self::PeakDriverTorque => sweep
                .driver_torques
                .as_ref()
                .map(|v| v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max))
                .unwrap_or(0.0),
            Self::RmsDriverTorque => sweep
                .driver_torques
                .as_ref()
                .map(|v| {
                    let n = v.len() as f64;
                    if n == 0.0 { return 0.0; }
                    (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
                })
                .unwrap_or(0.0),
            Self::MinTransmissionAngle => sweep
                .transmission_angles
                .as_ref()
                .map(|v| v.iter().copied().fold(f64::INFINITY, f64::min))
                .unwrap_or(0.0),
            Self::MaxTransmissionAngle => sweep
                .transmission_angles
                .as_ref()
                .map(|v| v.iter().copied().fold(0.0_f64, f64::max))
                .unwrap_or(0.0),
            Self::PeakReaction => sweep
                .joint_reaction_magnitudes
                .values()
                .flat_map(|v| v.iter().copied())
                .fold(0.0_f64, f64::max),
            Self::PeakKineticEnergy => sweep
                .kinetic_energy
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            Self::MeanMechanicalAdvantage => {
                let v = &sweep.mechanical_advantage;
                if v.is_empty() { return 0.0; }
                // Filter out extreme values near toggle
                let filtered: Vec<f64> = v.iter().copied().filter(|x| x.abs() < 1e6).collect();
                if filtered.is_empty() { return 0.0; }
                filtered.iter().sum::<f64>() / filtered.len() as f64
            }
        }
    }
}

/// Configuration for a parametric study.
#[derive(Debug, Clone)]
pub struct ParametricStudyConfig {
    pub parameter: SweepParameter,
    pub min_value: f64,
    pub max_value: f64,
    pub num_steps: usize,
    pub metric: ParametricMetric,
}

/// Results of a parametric study.
#[derive(Debug, Clone)]
pub struct ParametricStudyResult {
    pub config: ParametricStudyConfig,
    /// Parameter values at each step.
    pub parameter_values: Vec<f64>,
    /// Extracted metric value at each step.
    pub metric_values: Vec<f64>,
    /// Full sweep data for the selected (hovered/clicked) parameter value, if any.
    pub selected_sweep: Option<(f64, SweepData)>,
}

// ── Counterbalance assistant ──────────────────────────────────────────────────

/// Configuration for a counterbalance spring optimization.
#[derive(Debug, Clone)]
pub struct CounterbalanceConfig {
    /// Body A for the spring attachment (e.g., ground).
    pub body_a: String,
    /// Attachment point name on body A.
    pub point_a: String,
    /// Body B for the spring attachment (e.g., coupler).
    pub body_b: String,
    /// Attachment point name on body B.
    pub point_b: String,
    /// Minimum spring stiffness to search (N/m).
    pub k_min: f64,
    /// Maximum spring stiffness to search (N/m).
    pub k_max: f64,
    /// Number of stiffness steps.
    pub k_steps: usize,
    /// Minimum free length to search (m).
    pub free_length_min: f64,
    /// Maximum free length to search (m).
    pub free_length_max: f64,
    /// Number of free length steps.
    pub free_length_steps: usize,
}

/// Results from a counterbalance optimization.
#[derive(Debug, Clone)]
pub struct CounterbalanceResult {
    /// Optimal spring stiffness (N/m).
    pub best_k: f64,
    /// Optimal free length (m).
    pub best_free_length: f64,
    /// Peak-to-peak torque with the optimal spring (N*m).
    pub best_peak_to_peak: f64,
    /// Peak-to-peak torque without any counterbalance spring (N*m).
    pub baseline_peak_to_peak: f64,
    /// Driver torques over the sweep WITHOUT the spring (baseline).
    pub baseline_torques: Vec<f64>,
    /// Driver torques over the sweep WITH the optimal spring.
    pub optimized_torques: Vec<f64>,
    /// Sweep angles in degrees (shared by both torque curves).
    pub angles_deg: Vec<f64>,
    /// Full grid of peak-to-peak values: [k_idx][fl_idx].
    pub grid: Vec<Vec<f64>>,
    /// K values searched.
    pub k_values: Vec<f64>,
    /// Free length values searched.
    pub fl_values: Vec<f64>,
}

// ── Simulation state ──────────────────────────────────────────────────────────

/// Forward dynamics simulation result and playback state.
pub struct SimulationState {
    /// Time history from the simulation.
    pub times: Vec<f64>,
    /// Position history (one q vector per time step).
    pub positions: Vec<DVector<f64>>,
    /// Current playback index into the trajectory.
    pub time_index: usize,
    /// Whether simulation playback is active.
    pub playing: bool,
    /// Playback speed multiplier.
    pub speed: f64,
    /// Accumulated time for playback interpolation.
    pub elapsed: f64,
    /// Constraint drift at each step.
    pub drift: Vec<f64>,
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
    /// Solved q at driver angle = 0 — used to reset initial guess on animation wrap.
    pub q_at_zero: DVector<f64>,
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
    /// Whether sweep data needs recomputation (set by rebuild, gravity change, etc.).
    pub sweep_dirty: bool,
    /// Timestamp (egui time in seconds) when sweep was last marked dirty (for debounce).
    pub sweep_dirty_since: Option<f64>,
    // ── Drag interaction ─────────────────────────────────────────────────
    /// Currently active drag target (attachment point being dragged).
    pub drag_target: Option<DragTarget>,
    // ── Joint creation mode ──────────────────────────────────────────────
    /// First click of a two-click joint creation: (body_id, point_name, type).
    pub creating_joint: Option<(String, String, PendingJointType)>,
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
    /// Whether to show link length dimensions on the canvas.
    pub show_dimensions: bool,
    /// Gravity magnitude in m/s² (0 = disabled, 9.81 = Earth standard).
    pub gravity_magnitude: f64,
    // ── Load cases ──────────────────────────────────────────────────────
    /// Named driver configurations for comparing operating conditions.
    pub load_cases: LoadCaseManager,
    // ── Editor tool ─────────────────────────────────────────────────────
    /// Active editor tool (Select, AddBody, AddGroundPivot, AddJoint).
    pub active_tool: EditorTool,
    // ── Context menu ────────────────────────────────────────────────────
    /// Persisted right-click target for context menu rendering across frames.
    pub context_menu_target: ContextMenuTarget,
    // ── Draw Link state ──────────────────────────────────────────────────
    /// Start of a Draw Link gesture: world position and optional existing
    /// attachment point (body_id, point_name). If None, a new ground pivot
    /// was created at the start position.
    pub draw_link_start: Option<DrawLinkStart>,
    // ── Add Body state ──────────────────────────────────────────────────
    /// Multi-click body placement state. None when not in AddBody mode.
    pub add_body_state: Option<AddBodyState>,
    // ── Diagnostics ─────────────────────────────────────────────────────
    /// Cached Grashof classification for 4-bar mechanisms.
    pub grashof_result: Option<GrashofResult>,
    /// Cached crank recommendation for 4-bar mechanisms.
    pub crank_recommendation: Option<crate::analysis::crank_selection::CrankRecommendation>,
    // ── Forward dynamics simulation ─────────────────────────────────────
    /// Forward dynamics simulation result and playback state.
    pub simulation: Option<SimulationState>,
    /// Duration for forward dynamics simulation (seconds).
    pub simulation_duration: f64,
    /// Error messages from simulation and solver failures.
    pub error_log: Vec<String>,
    /// Whether the error panel is visible.
    pub show_error_panel: bool,
    // ── Parametric study ──────────────────────────────────────────────
    /// Cached parametric study results.
    pub parametric_result: Option<ParametricStudyResult>,
    /// Whether the parametric study panel is visible.
    pub show_parametric: bool,
    /// Active parametric study configuration (persists across panel close/open).
    pub parametric_config: ParametricStudyConfig,
    /// Cached counterbalance study results.
    pub counterbalance_result: Option<CounterbalanceResult>,
    /// Active counterbalance configuration.
    pub counterbalance_config: CounterbalanceConfig,
    // ── Expression driver editor ────────────────────────────────────────
    /// Text buffer for f(t) expression being edited.
    pub expr_buf: String,
    /// Text buffer for f'(t) expression being edited.
    pub expr_dot_buf: String,
    /// Text buffer for f''(t) expression being edited.
    pub expr_ddot_buf: String,
    /// Whether the expression editor is currently showing parse errors.
    pub expr_error: Option<String>,
    // ── Help dialog ─────────────────────────────────────────────────
    /// Whether the keyboard shortcuts help window is open.
    pub show_shortcuts: bool,
    // ── Autosave ────────────────────────────────────────────────────
    /// Accumulated time since last autosave (seconds).
    pub autosave_timer: f64,
    /// Path of the last manual save (used for autosave naming).
    pub last_save_path: Option<std::path::PathBuf>,
    /// Whether unsaved changes exist since last manual save or load.
    pub dirty: bool,
    // ── Recent files ────────────────────────────────────────────────
    /// Recently opened/saved file paths (most recent first, max 5).
    pub recent_files: Vec<std::path::PathBuf>,
    // ── Autosave recovery ───────────────────────────────────────────
    /// Path to a recoverable autosave file found on startup (if any).
    pub recovery_path: Option<std::path::PathBuf>,
}

/// Tracks placement state for the Add Body tool.
#[derive(Debug, Clone)]
pub struct AddBodyState {
    /// Points placed so far: (name, world_position).
    pub points: Vec<(String, [f64; 2])>,
}

/// Tracks the start of a Draw Link gesture.
#[derive(Debug, Clone)]
pub struct DrawLinkStart {
    /// World coordinates of the start point.
    pub world_pos: [f64; 2],
    /// If the start landed on an existing attachment point: (body_id, point_name).
    /// If None, a new ground pivot was created at this position.
    pub attachment: Option<(String, String)>,
}

impl Default for AppState {
    fn default() -> Self {
        // Start with an empty mechanism (just a ground body) so the canvas
        // is immediately editable without loading a sample first.
        let empty_blueprint = MechanismJson {
            schema_version: "1.0.0".to_string(),
            bodies: {
                let mut m = HashMap::new();
                m.insert(
                    GROUND_ID.to_string(),
                    BodyJson {
                        attachment_points: HashMap::new(),
                        mass: 0.0,
                        cg_local: [0.0, 0.0],
                        izz_cg: 0.0,
                        coupler_points: HashMap::new(),
                        point_masses: Vec::new(),
                    },
                );
                m
            },
            joints: HashMap::new(),
            drivers: HashMap::new(),
            load_cases: Vec::new(),
            forces: Vec::new(),
        };

        let mut state = Self {
            blueprint: Some(empty_blueprint),
            mechanism: None,
            q: DVector::zeros(0),
            driver_angle: 0.0,
            last_good_q: DVector::zeros(0),
            q_at_zero: DVector::zeros(0),
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
            sweep_dirty: false,
            sweep_dirty_since: None,
            drag_target: None,
            creating_joint: None,
            validation_warnings: ValidationWarnings::default(),
            display_units: DisplayUnits::default(),
            grid: GridSettings::default(),
            force_results: ForceResults::default(),
            show_forces: true,
            show_dimensions: false,
            gravity_magnitude: 9.81,
            load_cases: LoadCaseManager::default(),
            active_tool: EditorTool::Select,
            context_menu_target: ContextMenuTarget::default(),
            draw_link_start: None,
            add_body_state: None,
            grashof_result: None,
            crank_recommendation: None,
            simulation: None,
            simulation_duration: 5.0,
            error_log: Vec::new(),
            show_error_panel: false,
            parametric_result: None,
            show_parametric: false,
            parametric_config: ParametricStudyConfig {
                parameter: SweepParameter::DriverOmega,
                min_value: 1.0,
                max_value: 10.0,
                num_steps: 5,
                metric: ParametricMetric::PeakDriverTorque,
            },
            counterbalance_result: None,
            counterbalance_config: CounterbalanceConfig {
                body_a: GROUND_ID.to_string(),
                point_a: String::new(),
                body_b: String::new(),
                point_b: String::new(),
                k_min: 10.0,
                k_max: 1000.0,
                k_steps: 10,
                free_length_min: 0.01,
                free_length_max: 0.10,
                free_length_steps: 5,
            },
            expr_buf: String::new(),
            expr_dot_buf: String::new(),
            expr_ddot_buf: String::new(),
            expr_error: None,
            show_shortcuts: false,
            autosave_timer: 0.0,
            last_save_path: None,
            dirty: false,
            recent_files: Self::load_recent_files(),
            recovery_path: Self::check_autosave_recovery(),
        };
        state.rebuild();
        state
    }
}

impl AppState {
    /// Reset to an empty mechanism (just ground body), clearing all state.
    pub fn new_empty_mechanism(&mut self) {
        let fresh = AppState::default();
        // Preserve user preferences across reset.
        let recent = std::mem::take(&mut self.recent_files);
        let units = DisplayUnits {
            length: self.display_units.length,
            angle: self.display_units.angle,
        };
        *self = fresh;
        self.recent_files = recent;
        self.display_units = units;
    }

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
        self.q_at_zero = self.q.clone();

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
        self.update_grashof();
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

    /// Sync gravity force element on the mechanism with the `gravity_magnitude` value.
    /// Adds or updates `ForceElement::Gravity` when magnitude > 0; removes it when 0.
    pub fn sync_gravity(&mut self) {
        let Some(mech) = &mut self.mechanism else {
            return;
        };
        let has_gravity = mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_)));
        if self.gravity_magnitude > 0.0 {
            let g_elem = GravityElement {
                g_vector: [0.0, -self.gravity_magnitude],
            };
            if has_gravity {
                if let Some(idx) = mech
                    .forces()
                    .iter()
                    .position(|f| matches!(f, ForceElement::Gravity(_)))
                {
                    mech.replace_force(idx, ForceElement::Gravity(g_elem));
                }
            } else {
                mech.add_force(ForceElement::Gravity(g_elem));
            }
        } else if has_gravity {
            if let Some(idx) = mech
                .forces()
                .iter()
                .position(|f| matches!(f, ForceElement::Gravity(_)))
            {
                mech.remove_force(idx);
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
        self.sync_gravity();
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

        let statics_result = match solve_statics(mech, &self.q, t) {
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

        // Compute mechanical advantage via velocity solve.
        let ma = if let Ok(q_dot) = solve_velocity(mech, &self.q, t) {
            // The driver body pair gives (body_i, body_j) where body_j is
            // the driven body (crank). Find the last non-driver moving body
            // as the output body.
            if let Some((_body_i, driver_body)) = mech.driver_body_pair() {
                let output_body = mech.body_order().iter()
                    .filter(|b| b.as_str() != driver_body)
                    .last();
                if let Some(out_id) = output_body {
                    mechanical_advantage(
                        mech.state(), &q_dot,
                        driver_body, out_id,
                        VelocityCoord::Theta, VelocityCoord::Theta,
                    ).map(|r| r.ma)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Per-element force contribution breakdown.
        let n = self.q.len();
        let q_dot_zero = DVector::zeros(n);
        let mut contribs: Vec<(String, f64)> = evaluate_contributions(mech, &self.q, &q_dot_zero, t)
            .iter()
            .map(|c| (c.type_name.clone(), c.q_norm))
            .collect();
        contribs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Virtual work cross-check.
        let vw_check = if let Some(dt) = driver_torque {
            virtual_work_check(mech, &self.q, t, dt, 1e-4).ok().map(|vw| {
                (vw.input_torque, vw.lagrange_torque, vw.agrees)
            })
        } else {
            None
        };

        self.force_results = ForceResults {
            driver_torque,
            joint_reactions,
            condition_number: Some(statics_result.condition_number),
            is_overconstrained: statics_result.is_overconstrained,
            mechanical_advantage: ma,
            force_contributions: contribs,
            virtual_work_check: vw_check,
        };
    }

    /// Update cached Grashof classification and crank recommendation from
    /// the current mechanism.
    ///
    /// Only produces results for 4-bar mechanisms (3 moving bodies + ground,
    /// 4 revolute joints). Sets both `grashof_result` and
    /// `crank_recommendation` to `None` for all other mechanism topologies.
    fn update_grashof(&mut self) {
        let Some(mech) = &self.mechanism else {
            self.grashof_result = None;
            self.crank_recommendation = None;
            return;
        };

        match detect_fourbar_links(mech) {
            Some((crank_len, coupler_len, rocker_len, ground_len)) => {
                self.grashof_result = Some(check_grashof(
                    ground_len,
                    crank_len,
                    coupler_len,
                    rocker_len,
                    1e-10,
                ));
                self.crank_recommendation = Some(
                    crate::analysis::crank_selection::recommend_crank(
                        ground_len,
                        crank_len,
                        coupler_len,
                        rocker_len,
                    ),
                );
            }
            None => {
                self.grashof_result = None;
                self.crank_recommendation = None;
            }
        }
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
    pub fn save_to_file(&mut self, path: &Path) -> Result<(), String> {
        let mech = self
            .mechanism
            .as_ref()
            .ok_or_else(|| "No mechanism loaded".to_string())?;

        let mut json_struct = mechanism_to_json(mech).map_err(|e| e.to_string())?;

        // Persist load cases into the JSON structure
        json_struct.load_cases = self.load_cases.cases.clone();

        // Preserve blueprint-only data that mechanism_to_json can't reconstruct:
        // point masses are baked into mass/CG/Izz at build time, so copy them
        // from the blueprint to ensure they persist through save/load.
        if let Some(ref bp) = self.blueprint {
            for (body_id, bp_body) in &bp.bodies {
                if let Some(json_body) = json_struct.bodies.get_mut(body_id) {
                    json_body.point_masses = bp_body.point_masses.clone();
                }
            }
        }

        let json = serde_json::to_string_pretty(&json_struct).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write file: {}", e))?;

        self.last_save_path = Some(path.to_path_buf());
        self.dirty = false;
        self.add_recent_file(path);
        Ok(())
    }

    /// Perform periodic autosave to a temp file alongside the last save path.
    ///
    /// Called from the update loop with accumulated dt. Saves every 30 seconds
    /// if there are unsaved changes and a mechanism is loaded.
    #[cfg(feature = "native")]
    pub fn tick_autosave(&mut self, dt: f64) {
        const AUTOSAVE_INTERVAL: f64 = 30.0;

        if !self.dirty || self.mechanism.is_none() {
            return;
        }

        self.autosave_timer += dt;
        if self.autosave_timer < AUTOSAVE_INTERVAL {
            return;
        }
        self.autosave_timer = 0.0;

        let autosave_path = self.autosave_path();
        if let Some(path) = autosave_path {
            if let Err(e) = self.write_json_to(&path) {
                log::warn!("Autosave failed: {}", e);
            } else {
                log::debug!("Autosaved to {:?}", path);
            }
        }
    }

    /// Compute the autosave file path (sibling to last save, or temp dir).
    #[cfg(feature = "native")]
    fn autosave_path(&self) -> Option<std::path::PathBuf> {
        if let Some(ref save_path) = self.last_save_path {
            let mut p = save_path.clone();
            let stem = p.file_stem()?.to_string_lossy().to_string();
            p.set_file_name(format!(".{}.autosave.json", stem));
            Some(p)
        } else {
            let mut p = std::env::temp_dir();
            p.push("linkage_simulator_autosave.json");
            Some(p)
        }
    }

    /// Write mechanism JSON to an arbitrary path (for autosave — doesn't clear dirty flag).
    #[cfg(feature = "native")]
    fn write_json_to(&self, path: &Path) -> Result<(), String> {
        let mech = self
            .mechanism
            .as_ref()
            .ok_or_else(|| "No mechanism loaded".to_string())?;
        let mut json_struct = mechanism_to_json(mech).map_err(|e| e.to_string())?;
        json_struct.load_cases = self.load_cases.cases.clone();
        // Preserve blueprint point masses (baked into mass/CG/Izz at build time).
        if let Some(ref bp) = self.blueprint {
            for (body_id, bp_body) in &bp.bodies {
                if let Some(json_body) = json_struct.bodies.get_mut(body_id) {
                    json_body.point_masses = bp_body.point_masses.clone();
                }
            }
        }
        let json = serde_json::to_string_pretty(&json_struct).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write: {}", e))?;
        Ok(())
    }

    // ── Recent files ─────────────────────────────────────────────────────

    /// Add a path to the recent files list (deduplicates, keeps max 5).
    pub fn add_recent_file(&mut self, path: &Path) {
        let canonical = path.to_path_buf();
        self.recent_files.retain(|p| p != &canonical);
        self.recent_files.insert(0, canonical);
        self.recent_files.truncate(5);
        self.save_recent_files();
    }

    /// Path to the recent files JSON in the user's temp directory.
    fn recent_files_path() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push("linkage_simulator_recent.json");
        p
    }

    /// Load recent files list from disk (returns empty Vec on any failure).
    fn load_recent_files() -> Vec<std::path::PathBuf> {
        let path = Self::recent_files_path();
        let Ok(json) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        serde_json::from_str(&json).unwrap_or_default()
    }

    /// Check for an autosave file in the temp directory on startup.
    /// Returns the path if a recoverable file exists (< 1 hour old).
    fn check_autosave_recovery() -> Option<std::path::PathBuf> {
        let mut p = std::env::temp_dir();
        p.push("linkage_simulator_autosave.json");
        if !p.exists() {
            return None;
        }
        // Only offer recovery for recent autosaves (< 1 hour).
        if let Ok(metadata) = std::fs::metadata(&p) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    if elapsed.as_secs() < 3600 {
                        return Some(p);
                    }
                }
            }
        }
        None
    }

    /// Save recent files list to disk.
    fn save_recent_files(&self) {
        let path = Self::recent_files_path();
        if let Ok(json) = serde_json::to_string(&self.recent_files) {
            let _ = std::fs::write(&path, json);
        }
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
                Some(DriverMeta::Expression { .. }) => (2.0 * PI, 0.0),
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
        self.q_at_zero = self.q.clone();
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
        self.last_save_path = Some(path.to_path_buf());
        self.dirty = false;
        self.autosave_timer = 0.0;
        self.add_recent_file(path);

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
    /// Called after every edit operation. Pauses animation to prevent the solver
    /// from fighting with mid-edit mechanism state.
    pub fn rebuild(&mut self) {
        self.playing = false;
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
        self.driver_omega = 2.0 * PI;
        self.driver_theta_0 = 0.0;
        // Check blueprint drivers for actual values
        if let Some(driver) = bp.drivers.values().next() {
            match driver {
                DriverJson::ConstantSpeed { omega, theta_0, .. } => {
                    self.driver_omega = *omega;
                    self.driver_theta_0 = *theta_0;
                }
                DriverJson::Expression { .. } => {
                    // Expression drivers use f(t) directly; omega/theta_0
                    // aren't meaningful, so keep defaults for angle-slider
                    // mapping (omega=2*pi means 1 rev/s, theta_0=0).
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

        // Ensure q always matches the new mechanism's dimension.
        let n = mech.state().n_coords();
        if self.q.len() != n {
            self.q = mech.state().make_q();
            self.last_good_q = self.q.clone();
            self.q_at_zero = self.q.clone();
        }

        self.mechanism = Some(mech);
        self.compute_forces(t);
        self.update_grashof();
        self.compute_validation();
        self.mark_sweep_dirty();
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

    /// Set the distance between two attachment points on a body, maintaining direction.
    ///
    /// Moves `point_b` along the vector from `point_a` to `point_b` so that
    /// the new distance equals `new_length`. If the points are coincident,
    /// moves `point_b` along the +X direction.
    pub fn set_link_length(
        &mut self,
        body_id: &str,
        point_a: &str,
        point_b: &str,
        new_length: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get(body_id) else { return };
        let Some(&pa) = body.attachment_points.get(point_a) else { return };
        let Some(&pb) = body.attachment_points.get(point_b) else { return };

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let current_len = (dx * dx + dy * dy).sqrt();

        let (ux, uy) = if current_len > 1e-12 {
            (dx / current_len, dy / current_len)
        } else {
            (1.0, 0.0)
        };

        let new_pb = [pa[0] + ux * new_length, pa[1] + uy * new_length];

        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_b) {
                *pt = new_pb;
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

    /// Add a force element to the blueprint.
    ///
    /// Pushes undo, appends the element, and rebuilds.
    pub fn add_force_element(&mut self, force: ForceElement) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        bp.forces.push(force);
        self.rebuild();
    }

    /// Remove a force element from the blueprint by index.
    ///
    /// Pushes undo, removes the element, and rebuilds.
    /// No-op if `index` is out of bounds.
    pub fn remove_force_element(&mut self, index: usize) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if index >= bp.forces.len() {
            return;
        }
        bp.forces.remove(index);
        self.rebuild();
    }

    /// Add a point mass to a body in the blueprint.
    ///
    /// Pushes undo, appends the point mass, and rebuilds (which recomputes
    /// composite mass, CG, and Izz via parallel axis theorem).
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.push(crate::io::serialization::PointMassJson {
            mass,
            local_pos,
        });
        self.rebuild();
    }

    /// Remove a point mass from a body in the blueprint by index.
    ///
    /// Pushes undo, removes the point mass, and rebuilds.
    pub fn remove_point_mass(&mut self, body_id: &str, index: usize) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        if index < body.point_masses.len() {
            body.point_masses.remove(index);
            self.rebuild();
        }
    }

    /// Update the slide axis of a prismatic joint in the blueprint.
    ///
    /// Continuous parameter tweak — no undo snapshot pushed.
    pub fn update_prismatic_axis(&mut self, joint_id: &str, axis: [f64; 2]) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(joint) = bp.joints.get_mut(joint_id) {
            if let JointJson::Prismatic { axis_local_i, .. } = joint {
                *axis_local_i = axis;
            }
        }
        self.rebuild();
    }

    /// Replace a force element in the blueprint at the given index.
    ///
    /// Intended for continuous parameter tweaks (e.g. DragValue), so no undo
    /// snapshot is pushed.
    /// No-op if `index` is out of bounds.
    pub fn update_force_element(&mut self, index: usize, force: ForceElement) {
        let Some(bp) = &mut self.blueprint else { return };
        if index >= bp.forces.len() {
            return;
        }
        bp.forces[index] = force;
        self.rebuild();
    }

    // ── Parametric study ──────────────────────────────────────────────

    /// Apply a parameter value to a blueprint clone. Returns None if the
    /// parameter path doesn't resolve.
    fn set_parameter_on_blueprint(
        bp: &mut MechanismJson,
        param: &SweepParameter,
        value: f64,
        omega: &mut f64,
    ) -> bool {
        match param {
            SweepParameter::BodyMass(id) => {
                if let Some(body) = bp.bodies.get_mut(id) {
                    body.mass = value;
                    return true;
                }
            }
            SweepParameter::BodyIzz(id) => {
                if let Some(body) = bp.bodies.get_mut(id) {
                    body.izz_cg = value;
                    return true;
                }
            }
            SweepParameter::AttachmentX(body_id, point_name) => {
                if let Some(body) = bp.bodies.get_mut(body_id) {
                    if let Some(pt) = body.attachment_points.get_mut(point_name) {
                        pt[0] = value;
                        return true;
                    }
                }
            }
            SweepParameter::AttachmentY(body_id, point_name) => {
                if let Some(body) = bp.bodies.get_mut(body_id) {
                    if let Some(pt) = body.attachment_points.get_mut(point_name) {
                        pt[1] = value;
                        return true;
                    }
                }
            }
            SweepParameter::ForceParam(idx, field) => {
                if let Some(force) = bp.forces.get_mut(*idx) {
                    return set_force_field(force, field, value);
                }
            }
            SweepParameter::DriverOmega => {
                *omega = value;
                return true;
            }
        }
        false
    }

    /// Run a parametric study: sweep a parameter across a range, compute a full
    /// kinematic/force sweep at each value, and extract the selected metric.
    ///
    /// Clones the blueprint for each parameter value — the user's current
    /// mechanism is not modified.
    pub fn run_parametric_study(&mut self) {
        let Some(ref base_bp) = self.blueprint else { return };
        let config = &self.parametric_config;
        if config.num_steps < 2 {
            return;
        }

        let step_size = (config.max_value - config.min_value) / (config.num_steps - 1) as f64;
        let mut param_values = Vec::with_capacity(config.num_steps);
        let mut metric_values = Vec::with_capacity(config.num_steps);

        for i in 0..config.num_steps {
            let value = config.min_value + i as f64 * step_size;
            param_values.push(value);

            // Clone blueprint and apply parameter
            let mut bp = base_bp.clone();
            let mut omega = self.driver_omega;
            if !Self::set_parameter_on_blueprint(&mut bp, &config.parameter, value, &mut omega) {
                metric_values.push(f64::NAN);
                continue;
            }

            // Build mechanism from modified blueprint
            let Ok(mut mech) = load_mechanism_unbuilt_from_json(&bp) else {
                metric_values.push(f64::NAN);
                continue;
            };
            if mech.build().is_err() {
                metric_values.push(f64::NAN);
                continue;
            }

            // Use current q as initial guess when dimensions match
            let q0 = if self.q.len() == mech.state().n_coords() { self.q.clone() } else { mech.state().make_q() };
            let theta_0 = self.driver_theta_0;
            let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude);

            // Extract the selected metric
            metric_values.push(config.metric.extract(&sweep));
        }

        self.parametric_result = Some(ParametricStudyResult {
            config: config.clone(),
            parameter_values: param_values,
            metric_values,
            selected_sweep: None,
        });
    }

    /// Run a counterbalance optimization: grid search over spring (k, free_length)
    /// to minimize driver torque peak-to-peak variation.
    ///
    /// The baseline torque curve (without the spring) is computed first, then
    /// each (k, free_length) combination is evaluated.
    pub fn run_counterbalance_study(&mut self) {
        use crate::analysis::envelopes::compute_envelope;

        let Some(ref base_bp) = self.blueprint else { return };
        let config = &self.counterbalance_config;
        if config.k_steps < 2 || config.body_a.is_empty() || config.body_b.is_empty() {
            return;
        }

        let omega = self.driver_omega;
        let theta_0 = self.driver_theta_0;

        // Use current q as the initial guess (the geometric guess from sample loading
        // converges much better than all-zeros for the first step).
        let q_init = self.q.clone();

        // 1. Compute baseline (no spring)
        let baseline_sweep = {
            let Ok(mut mech) = load_mechanism_unbuilt_from_json(base_bp) else { return };
            if mech.build().is_err() { return; }
            let q0 = if q_init.len() == mech.state().n_coords() { q_init.clone() } else { mech.state().make_q() };
            let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude);
            sweep
        };
        let baseline_torques = baseline_sweep.driver_torques.clone().unwrap_or_default();
        let baseline_pp = compute_envelope(&baseline_torques)
            .map(|e| e.peak_to_peak)
            .unwrap_or(0.0);

        // 2. Grid search
        let k_step = if config.k_steps > 1 {
            (config.k_max - config.k_min) / (config.k_steps - 1) as f64
        } else { 0.0 };
        let fl_steps = config.free_length_steps.max(1);
        let fl_step = if fl_steps > 1 {
            (config.free_length_max - config.free_length_min) / (fl_steps - 1) as f64
        } else { 0.0 };

        let mut k_values = Vec::with_capacity(config.k_steps);
        let mut fl_values = Vec::with_capacity(fl_steps);
        for i in 0..config.k_steps {
            k_values.push(config.k_min + i as f64 * k_step);
        }
        for j in 0..fl_steps {
            fl_values.push(config.free_length_min + j as f64 * fl_step);
        }

        let mut grid: Vec<Vec<f64>> = vec![vec![f64::INFINITY; fl_steps]; config.k_steps];
        let mut best_k = config.k_min;
        let mut best_fl = config.free_length_min;
        let mut best_pp = f64::INFINITY;
        let mut best_torques: Option<Vec<f64>> = None;

        for (ki, &k) in k_values.iter().enumerate() {
            for (fi, &fl) in fl_values.iter().enumerate() {
                let mut bp = base_bp.clone();
                // Add a linear spring to the blueprint
                bp.forces.push(ForceElement::LinearSpring(
                    crate::forces::elements::LinearSpringElement {
                        body_a: config.body_a.clone(),
                        point_a: Self::resolve_point_coords(&bp, &config.body_a, &config.point_a),
                        body_b: config.body_b.clone(),
                        point_b: Self::resolve_point_coords(&bp, &config.body_b, &config.point_b),
                        stiffness: k,
                        free_length: fl,
                    },
                ));

                let Ok(mut mech) = load_mechanism_unbuilt_from_json(&bp) else {
                    grid[ki][fi] = f64::NAN;
                    continue;
                };
                if mech.build().is_err() {
                    grid[ki][fi] = f64::NAN;
                    continue;
                }

                let q0 = if q_init.len() == mech.state().n_coords() { q_init.clone() } else { mech.state().make_q() };
                let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude);
                let pp = sweep.driver_torques.as_ref()
                    .and_then(|t| compute_envelope(t))
                    .map(|e| e.peak_to_peak)
                    .unwrap_or(f64::INFINITY);
                grid[ki][fi] = pp;

                if pp < best_pp {
                    best_pp = pp;
                    best_k = k;
                    best_fl = fl;
                    best_torques = sweep.driver_torques.clone();
                }
            }
        }

        self.counterbalance_result = Some(CounterbalanceResult {
            best_k,
            best_free_length: best_fl,
            best_peak_to_peak: best_pp,
            baseline_peak_to_peak: baseline_pp,
            baseline_torques,
            optimized_torques: best_torques.unwrap_or_default(),
            angles_deg: baseline_sweep.angles_deg,
            grid,
            k_values,
            fl_values,
        });
    }

    /// Helper: look up attachment point local coordinates from the blueprint.
    fn resolve_point_coords(bp: &MechanismJson, body_id: &str, point_name: &str) -> [f64; 2] {
        bp.bodies
            .get(body_id)
            .and_then(|b| b.attachment_points.get(point_name))
            .copied()
            .unwrap_or([0.0, 0.0])
    }

    /// Enumerate all sweepable parameters from the current blueprint.
    pub fn available_parameters(&self) -> Vec<SweepParameter> {
        let Some(ref bp) = self.blueprint else { return Vec::new() };
        let mut params = Vec::new();

        // Body parameters (skip ground)
        for (body_id, body) in &bp.bodies {
            if body_id == GROUND_ID { continue; }
            params.push(SweepParameter::BodyMass(body_id.clone()));
            params.push(SweepParameter::BodyIzz(body_id.clone()));
            let mut pts: Vec<_> = body.attachment_points.keys().collect();
            pts.sort();
            for pt_name in pts {
                params.push(SweepParameter::AttachmentX(body_id.clone(), pt_name.clone()));
                params.push(SweepParameter::AttachmentY(body_id.clone(), pt_name.clone()));
            }
        }

        // Ground attachment point positions
        if let Some(ground) = bp.bodies.get(GROUND_ID) {
            let mut pts: Vec<_> = ground.attachment_points.keys().collect();
            pts.sort();
            for pt_name in pts {
                params.push(SweepParameter::AttachmentX(GROUND_ID.to_string(), pt_name.clone()));
                params.push(SweepParameter::AttachmentY(GROUND_ID.to_string(), pt_name.clone()));
            }
        }

        // Force element parameters
        for (idx, force) in bp.forces.iter().enumerate() {
            for field in force_sweepable_fields(force) {
                params.push(SweepParameter::ForceParam(idx, field));
            }
        }

        // Driver omega
        params.push(SweepParameter::DriverOmega);

        params
    }

    // ── Raw blueprint helpers (no undo / no rebuild) ──────────────────

    /// Add a ground pivot (attachment point on the ground body).
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    /// Use this inside compound operations that batch a single undo + rebuild.
    pub(crate) fn add_ground_pivot_raw(&mut self, name: &str, x: f64, y: f64) {
        let Some(bp) = &mut self.blueprint else { return };
        let ground = bp.bodies.entry(GROUND_ID.to_string()).or_insert_with(|| BodyJson {
            attachment_points: HashMap::new(),
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        });
        ground.attachment_points.insert(name.to_string(), [x, y]);
    }

    /// Add a revolute joint between two body attachment points.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    pub(crate) fn add_revolute_joint_raw(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
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
    }

    /// Create a body from N world-coordinate points.
    ///
    /// The first point becomes local (0, 0); all other points are stored
    /// relative to the first. CG is set to the centroid of all local points.
    /// Default mass properties (mass=1, Izz=0.01) are assigned.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    pub(crate) fn add_body_with_points_raw(
        &mut self,
        body_id: &str,
        points: &[(String, [f64; 2])],
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if points.is_empty() {
            return;
        }

        // First point becomes the body-local origin.
        let origin = points[0].1;
        let mut attachment_points = HashMap::new();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for (name, world) in points {
            let local_x = world[0] - origin[0];
            let local_y = world[1] - origin[1];
            attachment_points.insert(name.clone(), [local_x, local_y]);
            sum_x += local_x;
            sum_y += local_y;
        }

        let n = points.len() as f64;
        let body = BodyJson {
            attachment_points,
            mass: 1.0,
            cg_local: [sum_x / n, sum_y / n],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        };
        bp.bodies.insert(body_id.to_string(), body);
    }

    /// Add an attachment point to an existing body in body-local coordinates.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    /// No-op if the body does not exist.
    pub(crate) fn add_attachment_point_local_raw(
        &mut self,
        body_id: &str,
        name: &str,
        local_x: f64,
        local_y: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.attachment_points.insert(name.to_string(), [local_x, local_y]);
        }
    }

    // ── Create / delete operations ──────────────────────────────────────

    /// Add a named attachment point to a body at a world-coordinate position.
    ///
    /// Converts the world coordinates to body-local using the current pose, then
    /// delegates to `add_attachment_point_local_raw`. Pushes undo and rebuilds.
    /// No-op if the body does not exist or there is no blueprint.
    pub fn add_attachment_point_to_body(
        &mut self,
        body_id: &str,
        name: &str,
        world_x: f64,
        world_y: f64,
    ) {
        self.push_undo();
        let [lx, ly] = self.world_to_body_local(body_id, world_x, world_y);
        self.add_attachment_point_local_raw(body_id, name, lx, ly);
        self.rebuild();
    }

    /// Remove a named attachment point from a body.
    ///
    /// Cascades: any joint that references `(body_id, point_name)` is also
    /// removed. Pushes undo and rebuilds.
    /// No-op if the body or point does not exist, or there is no blueprint.
    pub fn remove_attachment_point(&mut self, body_id: &str, point_name: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.attachment_points.remove(point_name);
        }
        bp.joints
            .retain(|_id, joint| !joint_references_point(joint, body_id, point_name));
        self.rebuild();
    }

    /// Add a new ground pivot (attachment point on the ground body).
    ///
    /// Pushes undo, adds the point, and rebuilds.
    pub fn add_ground_pivot(&mut self, name: &str, x: f64, y: f64) {
        self.push_undo();
        self.add_ground_pivot_raw(name, x, y);
        self.rebuild();
    }

    /// Add a new body from N world-coordinate points.
    ///
    /// Auto-generates a body ID via `next_body_id()`. The first point becomes
    /// local (0, 0); CG is at the centroid. Pushes undo, adds the body, and rebuilds.
    ///
    /// Returns the generated body ID.
    pub fn add_body_with_points(&mut self, points: &[(String, [f64; 2])]) -> String {
        self.push_undo();
        let body_id = self.next_body_id();
        self.add_body_with_points_raw(&body_id, points);
        self.rebuild();
        body_id
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
        self.add_revolute_joint_raw(body_i, point_i, body_j, point_j);
        self.rebuild();
    }

    /// Add a prismatic joint between two bodies at the given attachment points.
    ///
    /// The slide axis defaults to the vector from point_i to point_j in body_i's
    /// local frame (normalized). delta_theta_0 defaults to 0.
    pub fn add_prismatic_joint(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };

        // Compute default axis from the direction between the two points.
        // Use the blueprint coordinates (local frame of body_i).
        let axis = if let (Some(bi), Some(bj)) = (bp.bodies.get(body_i), bp.bodies.get(body_j)) {
            if let (Some(pi), Some(pj)) = (
                bi.attachment_points.get(point_i),
                bj.attachment_points.get(point_j),
            ) {
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-12 {
                    [dx / len, dy / len]
                } else {
                    [1.0, 0.0]
                }
            } else {
                [1.0, 0.0]
            }
        } else {
            [1.0, 0.0]
        };

        let joint_id = generate_unique_id("J", &bp.joints);
        bp.joints.insert(
            joint_id,
            JointJson::Prismatic {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
                axis_local_i: axis,
                delta_theta_0: 0.0,
            },
        );
        self.rebuild();
    }

    /// Add a fixed joint between two bodies at the given attachment points.
    pub fn add_fixed_joint(
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
            JointJson::Fixed {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
                delta_theta_0: 0.0,
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

    /// Generate the next unused attachment point name for a body.
    ///
    /// Names follow the sequence A, B, ..., Z, AA, AB, ..., AZ, BA, ...
    /// (bijective base-26, always uppercase). Returns "A" when the body
    /// does not exist or the blueprint is absent.
    pub fn next_attachment_point_name(&self, body_id: &str) -> String {
        let Some(bp) = &self.blueprint else {
            return "A".to_string();
        };
        let existing: std::collections::HashSet<&String> = bp
            .bodies
            .get(body_id)
            .map(|b| b.attachment_points.keys().collect())
            .unwrap_or_default();

        let mut name = String::new();
        let mut n: usize = 0;
        loop {
            name.clear();
            let mut val = n;
            loop {
                name.push((b'A' + (val % 26) as u8) as char);
                val /= 26;
                if val == 0 {
                    break;
                }
                val -= 1;
            }
            let name_rev: String = name.chars().rev().collect();
            if !existing.contains(&name_rev) {
                return name_rev;
            }
            n += 1;
        }
    }

    /// Convert world coordinates to body-local coordinates using the body's
    /// current pose from `self.q`.
    ///
    /// Returns world coordinates unchanged for the ground body (which has no
    /// pose in `q`) or when the mechanism is not built.
    pub fn world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2] {
        if body_id == GROUND_ID {
            return [world_x, world_y];
        }
        let q_start = match &self.mechanism {
            Some(mech) => match mech.state().get_index(body_id) {
                Ok(idx) => idx.q_start,
                Err(_) => return [world_x, world_y],
            },
            None => return [world_x, world_y],
        };
        let bx = self.q[q_start];
        let by = self.q[q_start + 1];
        let theta = self.q[q_start + 2];
        let dx = world_x - bx;
        let dy = world_y - by;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        [cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy]
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
    ///
    /// Works for both sample mechanisms (via sample builder) and blueprint-based
    /// mechanisms (via blueprint driver mutation + rebuild).
    pub fn reassign_driver(&mut self, joint_id: &str) {
        // Try sample-based reassignment first (preserves sample-specific builder logic).
        if let Some(sample) = self.current_sample {
            self.push_undo();
            match crate::gui::samples::build_sample_with_driver(sample, Some(joint_id)) {
                Ok((mech, q0)) => {
                    self.driver_omega = 2.0 * PI;
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
                    self.driver_theta_0 = 0.0;
                    self.driver_angle = 0.0;
                    self.q_at_zero = self.q.clone();
                    self.blueprint = mechanism_to_json(&mech).ok();
                    self.mechanism = Some(mech);
                    self.driver_joint_id = Some(joint_id.to_string());
                    self.selected = None;
                    self.load_cases = LoadCaseManager::new_default(
                        joint_id,
                        self.driver_omega,
                        self.driver_theta_0,
                    );
                    self.playing = false;
                    self.animation_direction = 1.0;
                    self.pending_driver_reassignment = None;
                    self.mark_sweep_dirty();
                    self.compute_validation();
                    return;
                }
                Err(msg) => {
                    log::warn!("Sample-based driver reassignment failed: {}", msg);
                    // Fall through to blueprint-based reassignment.
                }
            }
        }

        // Blueprint-based reassignment: modify the driver in the blueprint and rebuild.
        let Some(bp) = &self.blueprint else { return };

        // Find the joint and its body pair.
        let Some(joint) = bp.joints.get(joint_id) else {
            log::warn!("Joint '{}' not found in blueprint", joint_id);
            return;
        };
        let (body_i, body_j) = joint_body_ids(joint);
        let body_i = body_i.to_string();
        let body_j = body_j.to_string();

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();

        // Remove all existing drivers.
        bp.drivers.clear();

        // Add new constant-speed driver for the target joint's body pair.
        let driver_id = generate_unique_id("D", &bp.drivers);
        bp.drivers.insert(
            driver_id,
            DriverJson::ConstantSpeed {
                body_i,
                body_j,
                omega: self.driver_omega,
                theta_0: 0.0,
            },
        );

        self.driver_theta_0 = 0.0;
        self.driver_angle = 0.0;
        self.driver_joint_id = Some(joint_id.to_string());
        self.selected = None;
        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;

        self.load_cases = LoadCaseManager::new_default(
            joint_id,
            self.driver_omega,
            self.driver_theta_0,
        );

        self.rebuild();
        self.q_at_zero = self.q.clone();
        // rebuild() already calls mark_sweep_dirty()
    }

    /// Switch the current driver to an expression-based driver.
    ///
    /// Replaces the first driver in the blueprint with an `Expression` variant,
    /// preserving the body pair. Pushes undo and rebuilds.
    pub fn set_expression_driver(&mut self, expr: &str, expr_dot: &str, expr_ddot: &str) {
        let Some(bp) = &self.blueprint else { return };

        // Find the first driver and its ID + body pair
        let Some((id, existing)) = bp.drivers.iter().next() else { return };
        let id = id.clone();
        let (body_i, body_j) = match existing {
            DriverJson::ConstantSpeed { body_i, body_j, .. }
            | DriverJson::Expression { body_i, body_j, .. } => {
                (body_i.clone(), body_j.clone())
            }
        };

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();
        bp.drivers.insert(
            id,
            DriverJson::Expression {
                body_i,
                body_j,
                expr: expr.to_string(),
                expr_dot: expr_dot.to_string(),
                expr_ddot: expr_ddot.to_string(),
            },
        );
        self.rebuild();
    }

    /// Switch the current driver back to a constant-speed driver.
    ///
    /// Replaces the first driver in the blueprint with a `ConstantSpeed` variant,
    /// preserving the body pair. Pushes undo and rebuilds.
    pub fn set_constant_speed_driver(&mut self, omega: f64, theta_0: f64) {
        let Some(bp) = &self.blueprint else { return };

        // Find the first driver and its ID + body pair
        let Some((id, existing)) = bp.drivers.iter().next() else { return };
        let id = id.clone();
        let (body_i, body_j) = match existing {
            DriverJson::ConstantSpeed { body_i, body_j, .. }
            | DriverJson::Expression { body_i, body_j, .. } => {
                (body_i.clone(), body_j.clone())
            }
        };

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();
        bp.drivers.insert(
            id,
            DriverJson::ConstantSpeed {
                body_i,
                body_j,
                omega,
                theta_0,
            },
        );
        self.driver_omega = omega;
        self.driver_theta_0 = theta_0;
        self.rebuild();
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
            self.mark_sweep_dirty();
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
            self.dirty = true;
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
    /// Mark sweep data as stale, starting the debounce timer.
    ///
    /// The actual recomputation happens in the update loop after a 200ms
    /// debounce delay, so rapid edits don't cause jank.
    pub fn mark_sweep_dirty(&mut self) {
        self.sweep_dirty = true;
        // sweep_dirty_since is set in the update loop using egui time
        // (we can't use std::time::Instant because it panics on WASM).
        // If it's already set, keep the existing timestamp for proper debounce.
    }

    pub fn compute_sweep(&mut self) {
        self.sweep_dirty = false;
        self.sweep_dirty_since = None;

        if self.mechanism.is_none() {
            self.sweep_data = None;
            return;
        }
        self.sync_gravity();

        // Copy values we need from self before taking a reference to the mechanism,
        // to avoid borrow-checker conflicts between &self.mechanism and &mut self.sweep_data.
        let q_start = self.last_good_q.clone();
        let omega = self.driver_omega;
        let theta_0 = self.driver_theta_0;

        let (data, q_zero) = compute_sweep_data(self.mechanism.as_ref().unwrap(), &q_start, omega, theta_0, self.gravity_magnitude);
        self.sweep_data = Some(data);
        self.q_at_zero = q_zero;
    }

    /// Run a forward dynamics simulation from the current pose.
    ///
    /// Builds a copy of the mechanism without driver constraints (free motion),
    /// runs RK4 + Baumgarte integration, and stores the trajectory for playback.
    pub fn run_simulation(&mut self, duration: f64) {
        let Some(bp) = &self.blueprint else { return };

        // Build mechanism WITHOUT the driver constraint
        let mut mech_json = bp.clone();
        mech_json.drivers.clear();

        let mut mech = match load_mechanism_unbuilt_from_json(&mech_json) {
            Ok(m) => m,
            Err(e) => {
                self.error_log.push(format!("Simulation: failed to build mechanism: {}", e));
                self.show_error_panel = true;
                return;
            }
        };
        if let Err(e) = mech.build() {
            self.error_log.push(format!("Simulation: mechanism assembly failed: {}", e));
            self.show_error_panel = true;
            return;
        }

        // Sync gravity
        if self.gravity_magnitude > 0.0 {
            mech.add_force(ForceElement::Gravity(GravityElement {
                g_vector: [0.0, -self.gravity_magnitude],
            }));
        }
        // Copy non-gravity force elements from blueprint
        for force in &bp.forces {
            if !matches!(force, ForceElement::Gravity(_)) {
                mech.add_force(force.clone());
            }
        }

        // Use current position as initial conditions (zero velocity).
        // Dimension of q stays the same -- drivers add constraint equations,
        // not coordinates.
        let q0 = self.q.clone();
        let q_dot0 = DVector::zeros(q0.len());

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            project_interval: 10,
            project_tol: 1e-10,
            max_project_iter: 10,
            ..Default::default()
        };

        // Generate evaluation times (60 fps)
        let n_frames = (duration * 60.0) as usize;
        let t_eval: Vec<f64> = (0..=n_frames)
            .map(|i| i as f64 * duration / n_frames as f64)
            .collect();

        match simulate(
            &mech,
            &q0,
            &q_dot0,
            (0.0, duration),
            Some(&config),
            Some(&t_eval),
        ) {
            Ok(result) if result.success => {
                self.simulation = Some(SimulationState {
                    times: result.t,
                    positions: result.q,
                    time_index: 0,
                    playing: true,
                    speed: 1.0,
                    elapsed: 0.0,
                    drift: result.constraint_drift,
                });
                // Stop kinematic animation
                self.playing = false;
            }
            Ok(result) => {
                self.error_log.push(format!(
                    "Simulation did not converge: {}",
                    result.message
                ));
                self.show_error_panel = true;
            }
            Err(e) => {
                self.error_log.push(format!("Simulation failed: {}", e));
                self.show_error_panel = true;
            }
        }
    }

    /// Advance simulation playback by dt. Returns true if playback is active.
    pub fn step_simulation(&mut self, dt: f64) -> bool {
        // Compute the new time index without holding a mutable borrow across
        // the assignment to self.q.
        let new_q = {
            let Some(sim) = &mut self.simulation else {
                return false;
            };
            if !sim.playing || sim.positions.is_empty() {
                return false;
            }

            sim.elapsed += dt * sim.speed;

            // Find the time index closest to elapsed time
            let target_t = sim.elapsed;
            if target_t >= *sim.times.last().unwrap_or(&0.0) {
                // Simulation ended
                sim.playing = false;
                sim.time_index = sim.positions.len() - 1;
            } else {
                // Find first index where t >= target_t
                sim.time_index = sim
                    .times
                    .iter()
                    .position(|&t| t >= target_t)
                    .unwrap_or(sim.positions.len() - 1);
            }

            // Clone the position vector so we can drop the sim borrow
            let idx = sim.time_index;
            if idx < sim.positions.len() {
                Some(sim.positions[idx].clone())
            } else {
                None
            }
        };

        // Update q from simulation trajectory (sim borrow is dropped)
        if let Some(q) = new_q {
            self.q = q;
        }

        true // request repaint
    }

    /// Advance animation by one frame. Returns true if animation is active.
    pub fn step_animation(&mut self, dt: f64) -> bool {
        if !self.playing || !self.has_mechanism() {
            return false;
        }

        let step_deg = self.animation_speed_deg_per_sec * dt * self.animation_direction;
        let mut new_angle_deg = self.driver_angle.to_degrees() + step_deg;

        if self.loop_mode {
            // Wrap around — reset initial guess to the solved q at angle 0
            // so the solver stays on the same assembly configuration branch.
            if new_angle_deg >= 360.0 {
                new_angle_deg -= 360.0;
                self.last_good_q = self.q_at_zero.clone();
            } else if new_angle_deg < 0.0 {
                new_angle_deg += 360.0;
                self.last_good_q = self.q_at_zero.clone();
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
        | JointJson::CamFollower { body_i, body_j, .. }
        | JointJson::RevoluteDriver { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
    }
}

/// Returns true if the joint references the given (body_id, point_name) pair.
///
/// Used by `remove_attachment_point` to cascade-delete joints that depend on
/// the removed pivot. RevoluteDriver joints reference bodies but not specific
/// attachment points, so they are never matched.
fn joint_references_point(joint: &JointJson, body_id: &str, point_name: &str) -> bool {
    match joint {
        JointJson::Revolute { body_i, point_i, body_j, point_j, .. }
        | JointJson::Prismatic { body_i, point_i, body_j, point_j, .. }
        | JointJson::Fixed { body_i, point_i, body_j, point_j, .. }
        | JointJson::CamFollower { body_i, point_i, body_j, point_j, .. } => {
            (body_i == body_id && point_i == point_name)
                || (body_j == body_id && point_j == point_name)
        }
        // RevoluteDriver has body_i/body_j but no point_i/point_j fields.
        // It references bodies, not specific attachment points.
        JointJson::RevoluteDriver { .. } => false,
    }
}

/// Extract the body_i and body_j IDs from a DriverJson.
fn driver_body_ids(driver: &DriverJson) -> (&str, &str) {
    match driver {
        DriverJson::ConstantSpeed { body_i, body_j, .. }
        | DriverJson::Expression { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
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

/// Perform the sweep computation (0-360 degrees) and return `(SweepData, q_at_zero)`.
///
/// Extracted as a free function so it borrows only `&Mechanism` and not
/// `&mut AppState`, avoiding borrow-checker conflicts.
///
/// Returns the sweep data and the solved q at angle 0 (used to reset the
/// initial guess when animation wraps around 360→0).
/// Set a single scalar field on a force element by field name. Returns true if successful.
fn set_force_field(force: &mut ForceElement, field: &str, value: f64) -> bool {
    match force {
        ForceElement::LinearSpring(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            "free_length" => { e.free_length = value; true }
            _ => false,
        },
        ForceElement::TorsionSpring(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            "free_angle" => { e.free_angle = value; true }
            _ => false,
        },
        ForceElement::LinearDamper(e) => match field {
            "damping" => { e.damping = value; true }
            _ => false,
        },
        ForceElement::RotaryDamper(e) => match field {
            "damping" => { e.damping = value; true }
            _ => false,
        },
        ForceElement::GasSpring(e) => match field {
            "initial_force" => { e.initial_force = value; true }
            "extended_length" => { e.extended_length = value; true }
            "stroke" => { e.stroke = value; true }
            _ => false,
        },
        ForceElement::Motor(e) => match field {
            "stall_torque" => { e.stall_torque = value; true }
            "no_load_speed" => { e.no_load_speed = value; true }
            _ => false,
        },
        ForceElement::ExternalForce(e) => match field {
            "force_x" => { e.force[0] = value; true }
            "force_y" => { e.force[1] = value; true }
            _ => false,
        },
        ForceElement::ExternalTorque(e) => match field {
            "torque" => { e.torque = value; true }
            _ => false,
        },
        ForceElement::BearingFriction(e) => match field {
            "constant_drag" => { e.constant_drag = value; true }
            "viscous_coeff" => { e.viscous_coeff = value; true }
            "coulomb_coeff" => { e.coulomb_coeff = value; true }
            _ => false,
        },
        ForceElement::JointLimit(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            _ => false,
        },
        ForceElement::LinearActuator(e) => match field {
            "force" => { e.force = value; true }
            "speed_limit" => { e.speed_limit = value; true }
            _ => false,
        },
        _ => false,
    }
}

/// List sweepable field names for a force element.
fn force_sweepable_fields(force: &ForceElement) -> Vec<String> {
    match force {
        ForceElement::LinearSpring(_) => vec!["stiffness".into(), "free_length".into()],
        ForceElement::TorsionSpring(_) => vec!["stiffness".into(), "free_angle".into()],
        ForceElement::LinearDamper(_) => vec!["damping".into()],
        ForceElement::RotaryDamper(_) => vec!["damping".into()],
        ForceElement::GasSpring(_) => vec!["initial_force".into(), "extended_length".into(), "stroke".into()],
        ForceElement::Motor(_) => vec!["stall_torque".into(), "no_load_speed".into()],
        ForceElement::ExternalForce(_) => vec!["force_x".into(), "force_y".into()],
        ForceElement::ExternalTorque(_) => vec!["torque".into()],
        ForceElement::BearingFriction(_) => vec!["constant_drag".into(), "viscous_coeff".into(), "coulomb_coeff".into()],
        ForceElement::JointLimit(_) => vec!["stiffness".into()],
        ForceElement::LinearActuator(_) => vec!["force".into(), "speed_limit".into()],
        _ => Vec::new(),
    }
}

pub(crate) fn compute_sweep_data(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
    gravity_magnitude: f64,
) -> (SweepData, DVector<f64>) {
    let mut data = SweepData {
        angles_deg: Vec::with_capacity(361),
        body_angles: HashMap::new(),
        coupler_traces: HashMap::new(),
        transmission_angles: None,
        driver_torques: Some(Vec::with_capacity(361)),
        kinetic_energy: Vec::with_capacity(361),
        potential_energy: Vec::with_capacity(361),
        total_energy: Vec::with_capacity(361),
        inverse_dynamics_torques: Vec::with_capacity(361),
        mechanical_advantage: Vec::with_capacity(361),
        joint_reaction_magnitudes: HashMap::new(),
        coupler_velocities: HashMap::new(),
        coupler_accelerations: HashMap::new(),
        toggle_angles: Vec::new(),
    };

    // Temporary accumulator for reaction data (filled during sweep,
    // then moved into `data` at the end).
    let mut reaction_data: HashMap<String, Vec<f64>> = HashMap::new();

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

    // Pre-allocate coupler velocity and acceleration vectors.
    let mut coupler_vel_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut coupler_accel_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (key, _, _) in &coupler_keys {
        coupler_vel_data.insert(key.clone(), Vec::with_capacity(361));
        coupler_accel_data.insert(key.clone(), Vec::with_capacity(361));
    }

    // Detect 4-bar link lengths for transmission angle.
    let fourbar_links = detect_fourbar_links(mech);
    if fourbar_links.is_some() {
        data.transmission_angles = Some(Vec::with_capacity(361));
    }

    // Detect driver/output body pair for mechanical advantage.
    let ma_bodies: Option<(String, String)> = mech.driver_body_pair().and_then(|(_bi, driver)| {
        let output = mech.body_order().iter()
            .filter(|b| b.as_str() != driver)
            .last()
            .cloned();
        output.map(|out| (driver.to_string(), out))
    });

    // Sweep from 0 to 360 degrees in 1-degree steps.
    let mut q = q_start.clone();
    let mut q_at_zero = q_start.clone();

    for i in 0..=360 {
        let angle_deg = i as f64;
        let t = (angle_deg.to_radians() - theta_0) / omega;

        match solve_position(mech, &q, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                q = result.q.clone();
                if i == 0 {
                    q_at_zero = q.clone();
                }
                data.angles_deg.push(angle_deg);

                // Toggle/dead-point detection.
                let toggle = check_toggle(mech, &q, t, 1e-6);
                if toggle.is_near_toggle {
                    data.toggle_angles.push(angle_deg);
                }

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

                // Driver torque and joint reactions from statics solve.
                if let Ok(statics) = solve_statics(mech, &q, t) {
                    let reactions = extract_reactions(mech, &statics);
                    let torque = get_driver_reactions(&reactions)
                        .first()
                        .map(|r| r.effort)
                        .unwrap_or(0.0);
                    data.driver_torques.as_mut().unwrap().push(torque);

                    // Per-joint reaction magnitudes.
                    for jr in &reactions {
                        if jr.n_equations > 1 {
                            reaction_data
                                .entry(jr.joint_id.clone())
                                .or_insert_with(|| Vec::with_capacity(361))
                                .push(jr.resultant);
                        }
                    }
                } else {
                    data.driver_torques.as_mut().unwrap().push(0.0);

                    // Push NaN for all tracked joints when statics fails.
                    for values in reaction_data.values_mut() {
                        values.push(f64::NAN);
                    }
                }

                // Velocity solve for energy and mechanical advantage.
                if let Ok(q_dot) = solve_velocity(mech, &q, t) {
                    let energy = compute_energy_state_mech(mech, &q, &q_dot, gravity_magnitude);
                    data.kinetic_energy.push(energy.kinetic);
                    data.potential_energy.push(energy.potential_gravity);
                    data.total_energy.push(energy.total);

                    // Mechanical advantage from velocity ratio.
                    if let Some((ref input_id, ref output_id)) = ma_bodies {
                        let ma_val = mechanical_advantage(
                            mech.state(), &q_dot,
                            input_id, output_id,
                            VelocityCoord::Theta, VelocityCoord::Theta,
                        ).map(|r| r.ma).unwrap_or(f64::NAN);
                        data.mechanical_advantage.push(ma_val);
                    } else {
                        data.mechanical_advantage.push(f64::NAN);
                    }

                    // Acceleration solve + inverse dynamics for torque including inertial effects
                    let accel_result = solve_acceleration(mech, &q, &q_dot, t);
                    if let Ok(ref q_ddot) = accel_result {
                        if let Ok(inv_dyn) = solve_inverse_dynamics(mech, &q, &q_dot, q_ddot, t) {
                            // Extract driver torque from the last lambda (driver is last constraint)
                            let n_lam = inv_dyn.lambdas.len();
                            if n_lam > 0 {
                                data.inverse_dynamics_torques.push(inv_dyn.lambdas[n_lam - 1]);
                            } else {
                                data.inverse_dynamics_torques.push(f64::NAN);
                            }
                        } else {
                            data.inverse_dynamics_torques.push(f64::NAN);
                        }
                    } else {
                        data.inverse_dynamics_torques.push(f64::NAN);
                    }

                    // Coupler point velocities and accelerations.
                    let n = q.len();
                    for (key, body_id, local) in &coupler_keys {
                        if let Ok(ref q_ddot) = accel_result {
                            let (_pos, vel, acc) = eval_coupler_point(
                                mech.state(), body_id, local, &q, &q_dot, q_ddot,
                            );
                            coupler_vel_data.get_mut(key).unwrap().push(vel.norm());
                            coupler_accel_data.get_mut(key).unwrap().push(acc.norm());
                        } else {
                            // No acceleration -- still store velocity.
                            let zero = DVector::zeros(n);
                            let (_pos, vel, _acc) = eval_coupler_point(
                                mech.state(), body_id, local, &q, &q_dot, &zero,
                            );
                            coupler_vel_data.get_mut(key).unwrap().push(vel.norm());
                            coupler_accel_data.get_mut(key).unwrap().push(f64::NAN);
                        }
                    }
                } else {
                    data.kinetic_energy.push(f64::NAN);
                    data.potential_energy.push(f64::NAN);
                    data.total_energy.push(f64::NAN);
                    data.inverse_dynamics_torques.push(f64::NAN);
                    data.mechanical_advantage.push(f64::NAN);

                    // No velocity solve -- push NaN for coupler vel/accel.
                    for (key, _, _) in &coupler_keys {
                        coupler_vel_data.get_mut(key).unwrap().push(f64::NAN);
                        coupler_accel_data.get_mut(key).unwrap().push(f64::NAN);
                    }
                }
            }
            _ => {
                // Solver failed at this angle -- stop sweep.
                // The mechanism likely cannot complete a full rotation.
                break;
            }
        }
    }

    data.joint_reaction_magnitudes = reaction_data;
    data.coupler_velocities = coupler_vel_data;
    data.coupler_accelerations = coupler_accel_data;

    (data, q_at_zero)
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
    fn default_state_has_empty_mechanism() {
        let state = AppState::default();
        // Default state starts with an empty mechanism (ground only).
        assert!(state.has_mechanism());
        assert!(state.current_sample.is_none());
        assert!(state.blueprint.is_some());
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
    fn take_snapshot_returns_some_for_default_state() {
        let state = AppState::default();
        // Default state has an empty mechanism, so snapshots should work.
        assert!(state.take_snapshot().is_some());
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
    fn push_undo_works_on_default_empty_mechanism() {
        let mut state = AppState::default();
        state.push_undo();
        // Default state has an empty mechanism, so undo should work.
        assert!(state.can_undo());
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
    fn add_body_with_points_creates_new_body_in_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_bodies_before = state.blueprint.as_ref().unwrap().bodies.len();

        let points = vec![
            ("X".to_string(), [0.0, 0.0]),
            ("Y".to_string(), [0.05, 0.0]),
        ];
        let body_id = state.add_body_with_points(&points);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), n_bodies_before + 1);
        assert!(bp.bodies.contains_key(&body_id));
        let body = bp.bodies.get(&body_id).unwrap();
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
        let points = vec![
            ("E1".to_string(), [0.0, 0.0]),
            ("E2".to_string(), [0.01, 0.0]),
        ];
        let extra_id = state.add_body_with_points(&points);

        // Add joint between ground and the new body.
        state.add_revolute_joint("ground", "O2", &extra_id, "E1");

        let bp = state.blueprint.as_ref().unwrap();
        // +1 from the new joint (add_body_with_points doesn't add joints).
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
        let points = vec![
            ("A".to_string(), [0.0, 0.0]),
            ("B".to_string(), [0.01, 0.0]),
        ];
        state.add_body_with_points(&points);

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
        let points = vec![
            ("F1".to_string(), [0.0, 0.5]),
            ("F2".to_string(), [0.01, 0.5]),
        ];
        let floating_id = state.add_body_with_points(&points);
        state.compute_validation();

        assert!(
            state.validation_warnings.disconnected_bodies.contains(&floating_id),
            "Should detect '{}' as disconnected",
            floating_id,
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

    // ── next_attachment_point_name tests ─────────────────────────────────

    #[test]
    fn next_attachment_point_name_skips_existing() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar sample uses body names "crank", "coupler", "rocker" with points A/B
        let name = state.next_attachment_point_name("crank");
        assert_eq!(name, "C");
    }

    #[test]
    fn next_attachment_point_name_empty_body() {
        let mut state = AppState::default();
        let bp = state.blueprint.as_mut().unwrap();
        bp.bodies.insert("empty".to_string(), BodyJson {
            attachment_points: HashMap::new(),
            mass: 1.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        });
        let name = state.next_attachment_point_name("empty");
        assert_eq!(name, "A");
    }

    #[test]
    fn next_attachment_point_name_overflow_past_z() {
        let mut state = AppState::default();
        let bp = state.blueprint.as_mut().unwrap();
        let mut pts = HashMap::new();
        for c in b'A'..=b'Z' {
            pts.insert(String::from(c as char), [0.0, 0.0]);
        }
        bp.bodies.insert("full".to_string(), BodyJson {
            attachment_points: pts,
            mass: 1.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        });
        let name = state.next_attachment_point_name("full");
        assert_eq!(name, "AA");
    }

    // ── world_to_body_local tests ─────────────────────────────────────────

    #[test]
    fn world_to_body_local_identity_pose() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let [lx, ly] = state.world_to_body_local("ground", 0.05, 0.03);
        assert!((lx - 0.05).abs() < 1e-10);
        assert!((ly - 0.03).abs() < 1e-10);
    }

    // ── Raw helper tests ─────────────────────────────────────────────────

    #[test]
    fn add_ground_pivot_raw_mutates_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot_raw("P1", 0.1, 0.2);
        let bp = state.blueprint.as_ref().unwrap();
        let ground = bp.bodies.get("ground").unwrap();
        assert!(ground.attachment_points.contains_key("P1"));
        let pt = ground.attachment_points.get("P1").unwrap();
        assert!((pt[0] - 0.1).abs() < 1e-15);
        assert!((pt[1] - 0.2).abs() < 1e-15);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_revolute_joint_raw_mutates_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot_raw("O", 0.0, 0.0);
        let bp = state.blueprint.as_mut().unwrap();
        let mut pts = HashMap::new();
        pts.insert("A".to_string(), [0.0, 0.0]);
        pts.insert("B".to_string(), [0.1, 0.0]);
        bp.bodies.insert("link".to_string(), BodyJson {
            attachment_points: pts,
            mass: 1.0,
            cg_local: [0.05, 0.0],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        });
        state.add_revolute_joint_raw("ground", "O", "link", "A");
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), 1);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_body_with_points_raw_first_point_is_local_origin() {
        let mut state = AppState::default();
        let points = vec![
            ("A".to_string(), [0.05, 0.03]),
            ("B".to_string(), [0.15, 0.03]),
            ("C".to_string(), [0.10, 0.08]),
        ];
        let new_body_id = state.next_body_id();
        state.add_body_with_points_raw(&new_body_id, &points);
        let bp = state.blueprint.as_ref().unwrap();
        let body = bp.bodies.values()
            .find(|b| b.attachment_points.contains_key("A") && b.attachment_points.contains_key("C"))
            .expect("body with A, B, C");
        let a = body.attachment_points.get("A").unwrap();
        assert!((a[0]).abs() < 1e-15);
        assert!((a[1]).abs() < 1e-15);
        let b = body.attachment_points.get("B").unwrap();
        assert!((b[0] - 0.10).abs() < 1e-15);
        assert!((b[1] - 0.00).abs() < 1e-15);
        let c = body.attachment_points.get("C").unwrap();
        assert!((c[0] - 0.05).abs() < 1e-15);
        assert!((c[1] - 0.05).abs() < 1e-15);
        let expected_cg_x = (0.0 + 0.10 + 0.05) / 3.0;
        let expected_cg_y = (0.0 + 0.0 + 0.05) / 3.0;
        assert!((body.cg_local[0] - expected_cg_x).abs() < 1e-10);
        assert!((body.cg_local[1] - expected_cg_y).abs() < 1e-10);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_attachment_point_local_raw_adds_to_existing_body() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_attachment_point_local_raw("crank", "C", 0.005, 0.003);
        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(crank.attachment_points.contains_key("C"));
        let c = crank.attachment_points.get("C").unwrap();
        assert!((c[0] - 0.005).abs() < 1e-15);
        assert!((c[1] - 0.003).abs() < 1e-15);
    }

    // ── add_attachment_point_to_body / remove_attachment_point tests ──────────

    #[test]
    fn add_attachment_point_to_body_converts_world_to_local() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Ground is at (0,0,0), so world = local for ground
        state.add_attachment_point_to_body("ground", "PX", 0.05, 0.02);
        let bp = state.blueprint.as_ref().unwrap();
        let ground = bp.bodies.get("ground").unwrap();
        let px = ground.attachment_points.get("PX").unwrap();
        assert!((px[0] - 0.05).abs() < 1e-10);
        assert!((px[1] - 0.02).abs() < 1e-10);
        assert!(state.can_undo());
    }

    #[test]
    fn remove_attachment_point_cascades_to_joints() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Crank has points A and B. J1 connects ground:O2 to crank:A.
        // Removing crank:A should also remove J1.
        let bp = state.blueprint.as_ref().unwrap();
        let joint_count_before = bp.joints.len();
        state.remove_attachment_point("crank", "A");
        let bp = state.blueprint.as_ref().unwrap();
        assert!(!bp.bodies.get("crank").unwrap().attachment_points.contains_key("A"));
        assert!(bp.joints.len() < joint_count_before);
        assert!(state.can_undo());
    }

    // ── Integration tests: multi-pivot body editing workflows ─────────────────

    #[test]
    fn create_ternary_body_via_add_body_with_points() {
        let mut state = AppState::default();
        let points = vec![
            ("A".to_string(), [0.0, 0.0]),
            ("B".to_string(), [0.1, 0.0]),
            ("C".to_string(), [0.05, 0.08]),
        ];
        state.add_body_with_points(&points);
        let bp = state.blueprint.as_ref().unwrap();
        // Should have ground + the new body
        assert_eq!(bp.bodies.len(), 2);
        let body = bp.bodies.iter()
            .find(|(id, _)| *id != "ground")
            .unwrap().1;
        assert_eq!(body.attachment_points.len(), 3);
        assert!(state.can_undo());
    }

    #[test]
    fn add_pivot_then_joint_creates_ternary_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Add a third pivot to the coupler body
        // (FourBar coupler has attachment points B and C)
        state.add_attachment_point_to_body("coupler", "P", 0.02, 0.01);
        let bp = state.blueprint.as_ref().unwrap();
        let coupler = bp.bodies.get("coupler").unwrap();
        assert_eq!(coupler.attachment_points.len(), 3); // B, C, P
    }

    #[test]
    fn remove_attachment_point_with_min_two_points_survives() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Crank has A and B. Removing B should leave just A.
        state.remove_attachment_point("crank", "B");
        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert_eq!(crank.attachment_points.len(), 1);
        assert!(crank.attachment_points.contains_key("A"));
    }

    #[test]
    fn compound_draw_link_is_single_undo_step() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp_before = state.blueprint.as_ref().unwrap().clone();
        let bodies_before = bp_before.bodies.len();
        let joints_before = bp_before.joints.len();

        // Simulate a compound Draw Link operation
        state.push_undo();
        state.add_ground_pivot_raw("PX", 0.1, 0.1);
        let new_body_id = state.next_body_id();
        let points = vec![
            ("A".to_string(), [0.1, 0.1]),
            ("B".to_string(), [0.2, 0.1]),
        ];
        state.add_body_with_points_raw(&new_body_id, &points);
        state.add_revolute_joint_raw("ground", "PX", &new_body_id, "A");
        state.rebuild();

        // Verify operation added bodies/joints
        let bp_after = state.blueprint.as_ref().unwrap();
        assert!(bp_after.bodies.len() > bodies_before);
        assert!(bp_after.joints.len() > joints_before);

        // One undo should restore the entire previous state
        assert!(state.can_undo());
        state.undo();
        let bp_undone = state.blueprint.as_ref().unwrap();
        assert_eq!(bp_undone.bodies.len(), bodies_before);
        assert_eq!(bp_undone.joints.len(), joints_before);
    }

    #[test]
    fn dirty_flag_set_on_push_undo() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(!state.dirty, "freshly loaded sample should not be dirty");
        state.push_undo();
        assert!(state.dirty, "push_undo should set dirty flag");
    }

    #[test]
    fn show_shortcuts_defaults_false() {
        let state = AppState::default();
        assert!(!state.show_shortcuts);
    }

    #[test]
    fn show_dimensions_defaults_false() {
        let state = AppState::default();
        assert!(!state.show_dimensions);
    }

    #[cfg(feature = "native")]
    #[test]
    fn autosave_path_with_no_save_uses_temp_dir() {
        let state = AppState::default();
        let path = state.autosave_path();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.to_string_lossy().contains("linkage_simulator_autosave"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn autosave_path_with_save_path_creates_sibling() {
        let mut state = AppState::default();
        state.last_save_path = Some(std::path::PathBuf::from("/tmp/my_mechanism.json"));
        let path = state.autosave_path();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(
            p.to_string_lossy().contains(".my_mechanism.autosave.json"),
            "Expected sibling autosave path, got {:?}",
            p
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn save_to_file_clears_dirty_flag() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.push_undo(); // sets dirty=true
        assert!(state.dirty);

        let tmp = std::env::temp_dir().join("linkage_test_save.json");
        state.save_to_file(&tmp).expect("save should succeed");
        assert!(!state.dirty, "save_to_file should clear dirty flag");
        assert_eq!(state.last_save_path.as_deref(), Some(tmp.as_path()));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[cfg(feature = "native")]
    #[test]
    fn load_from_file_clears_dirty_flag() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Save first
        let tmp = std::env::temp_dir().join("linkage_test_load.json");
        state.save_to_file(&tmp).expect("save should succeed");

        // Dirty it, then reload
        state.push_undo();
        assert!(state.dirty);

        state.load_from_file(&tmp).expect("load should succeed");
        assert!(!state.dirty, "load_from_file should clear dirty flag");
        assert_eq!(state.last_save_path.as_deref(), Some(tmp.as_path()));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn add_point_mass_modifies_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find a non-ground body
        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "should start with no point masses"
        );

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies[&body_id].point_masses.len(), 1);
        assert!((bp.bodies[&body_id].point_masses[0].mass - 0.5).abs() < 1e-10);
    }

    #[test]
    fn remove_point_mass_modifies_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        state.remove_point_mass(&body_id, 0);
        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "point mass should be removed"
        );
    }

    #[test]
    fn add_point_mass_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        state.undo();
        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "undo should remove the point mass"
        );
    }

    #[test]
    fn point_mass_affects_built_mechanism_mass() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        let mass_before = state.mechanism.as_ref().unwrap().bodies()[&body_id].mass;

        state.add_point_mass(&body_id, 2.0, [0.01, 0.0]);

        let mass_after = state.mechanism.as_ref().unwrap().bodies()[&body_id].mass;
        assert!(
            (mass_after - mass_before - 2.0).abs() < 1e-10,
            "built mechanism mass should increase by 2.0 kg, was {} now {}",
            mass_before,
            mass_after
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn point_mass_persists_through_save_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        state.add_point_mass(&body_id, 1.5, [0.02, -0.01]);

        // Save
        let tmp = std::env::temp_dir().join("linkage_test_pm.json");
        state.save_to_file(&tmp).expect("save should succeed");

        // Reload into fresh state
        let mut state2 = AppState::default();
        state2.load_from_file(&tmp).expect("load should succeed");

        let bp2 = state2.blueprint.as_ref().unwrap();
        assert_eq!(
            bp2.bodies[&body_id].point_masses.len(),
            1,
            "point mass should persist through save/load"
        );
        assert!((bp2.bodies[&body_id].point_masses[0].mass - 1.5).abs() < 1e-10);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn new_empty_mechanism_resets_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.current_sample.is_some());

        // Add a recent file so we can verify it persists
        state.recent_files.push(std::path::PathBuf::from("/fake/file.json"));
        state.display_units.length = LengthUnit::Meters;

        state.new_empty_mechanism();

        // Mechanism should be ground-only
        assert!(state.current_sample.is_none());
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), 1, "should have only ground body");
        assert!(bp.bodies.contains_key(GROUND_ID));
        assert!(bp.joints.is_empty());

        // Preferences should persist
        assert!(
            state.recent_files.iter().any(|p| p.to_string_lossy().contains("fake")),
            "recent_files should contain the fake entry we added"
        );
        assert_eq!(state.display_units.length, LengthUnit::Meters);
    }

    #[test]
    fn add_prismatic_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        // Build a minimal mechanism with two bodies + ground pivots.
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        let joints_before = state.blueprint.as_ref().unwrap().joints.len();
        state.add_prismatic_joint(GROUND_ID, "PX", &body_id, &body_point);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), joints_before + 1);
        // Find the new joint and verify it's prismatic
        let new_joint = bp.joints.values().last().unwrap();
        match new_joint {
            JointJson::Prismatic { axis_local_i, .. } => {
                // Axis should be a unit vector
                let len = (axis_local_i[0].powi(2) + axis_local_i[1].powi(2)).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-6 || len < 1e-12,
                    "axis should be normalized or zero, got length {}",
                    len
                );
            }
            _ => panic!("Expected Prismatic joint, got {:?}", new_joint),
        }
    }

    #[test]
    fn add_fixed_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        state.add_fixed_joint(GROUND_ID, "PX", &body_id, &body_point);

        let bp = state.blueprint.as_ref().unwrap();
        let has_fixed = bp.joints.values().any(|j| matches!(j, JointJson::Fixed { .. }));
        assert!(has_fixed, "Should have a Fixed joint in the blueprint");
    }

    #[test]
    fn add_prismatic_joint_is_undoable() {
        let mut state = AppState::default();
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        let joints_before = state.blueprint.as_ref().unwrap().joints.len();
        state.add_prismatic_joint(GROUND_ID, "PX", &body_id, &body_point);
        assert_eq!(state.blueprint.as_ref().unwrap().joints.len(), joints_before + 1);

        state.undo();
        assert_eq!(
            state.blueprint.as_ref().unwrap().joints.len(),
            joints_before,
            "undo should remove the prismatic joint"
        );
    }

    // ── Parametric study tests ────────────────────────────────────────────

    #[test]
    fn available_parameters_includes_body_mass_and_driver_omega() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let params = state.available_parameters();
        assert!(!params.is_empty());
        assert!(
            params.iter().any(|p| matches!(p, SweepParameter::DriverOmega)),
            "should include DriverOmega"
        );
        assert!(
            params.iter().any(|p| matches!(p, SweepParameter::BodyMass(_))),
            "should include at least one BodyMass"
        );
    }

    #[test]
    fn run_parametric_study_produces_results() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Use PeakKineticEnergy — always non-zero for a mechanism with mass
        state.parametric_config = ParametricStudyConfig {
            parameter: SweepParameter::DriverOmega,
            min_value: 1.0,
            max_value: 10.0,
            num_steps: 3,
            metric: ParametricMetric::PeakKineticEnergy,
        };

        state.run_parametric_study();

        let result = state.parametric_result.as_ref().expect("should have results");
        assert_eq!(result.parameter_values.len(), 3);
        assert_eq!(result.metric_values.len(), 3);
        assert!(
            result.metric_values.iter().all(|v| v.is_finite()),
            "all metric values should be finite, got {:?}",
            result.metric_values
        );
    }

    #[test]
    fn parametric_study_body_mass_sweep() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find a non-ground body
        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        // Sweep mass — just verify it produces finite results at each step
        state.parametric_config = ParametricStudyConfig {
            parameter: SweepParameter::BodyMass(body_id),
            min_value: 0.5,
            max_value: 5.0,
            num_steps: 3,
            metric: ParametricMetric::PeakKineticEnergy,
        };

        state.run_parametric_study();

        let result = state.parametric_result.as_ref().expect("should have results");
        assert_eq!(result.parameter_values.len(), 3);
        assert!(
            result.metric_values.iter().all(|v| v.is_finite()),
            "all metric values should be finite, got {:?}",
            result.metric_values
        );
    }

    #[test]
    fn parametric_metric_extract_peak_ke() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let sweep = state.sweep_data.as_ref().expect("should have sweep data");

        let ke = ParametricMetric::PeakKineticEnergy.extract(sweep);
        assert!(ke >= 0.0, "peak KE should be non-negative, got {}", ke);
    }

    // ── Counterbalance tests ──────────────────────────────────────────────

    #[test]
    fn counterbalance_study_produces_results() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Use "crank" body point A and ground point O2
        state.counterbalance_config = CounterbalanceConfig {
            body_a: GROUND_ID.to_string(),
            point_a: "O2".to_string(),
            body_b: "crank".to_string(),
            point_b: "B".to_string(),
            k_min: 10.0,
            k_max: 100.0,
            k_steps: 3,
            free_length_min: 0.01,
            free_length_max: 0.05,
            free_length_steps: 2,
        };

        state.run_counterbalance_study();

        let result = state.counterbalance_result.as_ref().expect("should have results");
        assert_eq!(result.k_values.len(), 3);
        assert_eq!(result.fl_values.len(), 2);
        assert_eq!(result.grid.len(), 3);
        assert_eq!(result.grid[0].len(), 2);
        assert!(!result.angles_deg.is_empty(), "should have sweep angles");
        assert!(!result.baseline_torques.is_empty(), "should have baseline torques");
    }

    // ── Error panel / simulation error surfacing tests ────────────────────

    #[test]
    fn run_simulation_no_blueprint_no_panic() {
        let mut state = AppState::default();
        state.blueprint = None;
        state.run_simulation(1.0);
        // No blueprint means early return with no errors logged.
        assert!(state.error_log.is_empty());
    }

    #[test]
    fn run_simulation_corrupted_mechanism_surfaces_error() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        if let Some(ref mut bp) = state.blueprint {
            bp.bodies.clear();
        }
        state.run_simulation(1.0);
        assert!(
            !state.error_log.is_empty(),
            "Simulation with corrupted mechanism should surface an error"
        );
        assert!(state.show_error_panel, "Error panel should auto-show on failure");
    }

    #[test]
    fn run_simulation_valid_mechanism_succeeds() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.run_simulation(1.0);
        if state.simulation.is_some() {
            assert!(state.error_log.is_empty());
        } else {
            assert!(!state.error_log.is_empty());
        }
    }

    #[test]
    fn rebuild_marks_sweep_dirty() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.sweep_dirty = false; // reset for test
        state.rebuild();
        assert!(state.sweep_dirty, "rebuild() should mark sweep as dirty");
    }

    #[test]
    fn load_sample_produces_sweep_data() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(
            state.sweep_data.is_some(),
            "load_sample should produce sweep data immediately"
        );
        let sweep = state.sweep_data.as_ref().unwrap();
        assert!(
            !sweep.angles_deg.is_empty(),
            "Sweep should have angle data"
        );
        assert!(
            !sweep.joint_reaction_magnitudes.is_empty(),
            "Sweep should have joint reaction data"
        );
    }

    #[test]
    fn gravity_magnitude_zero_disables_gravity_force() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 0.0;
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(
            !mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_))),
            "Gravity should be removed when magnitude is 0"
        );
    }

    #[test]
    fn gravity_magnitude_nondefault_updates_element() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 3.71;
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        let grav = mech.forces().iter().find(|f| matches!(f, ForceElement::Gravity(_)));
        assert!(grav.is_some(), "Gravity element should exist");
        if let Some(ForceElement::Gravity(g)) = grav {
            assert!((g.g_vector[1] - (-3.71)).abs() < 1e-10);
        }
    }

    #[test]
    fn set_link_length_maintains_direction() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa = crank.attachment_points.get("A").unwrap();
        let pb = crank.attachment_points.get("B").unwrap();
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let orig_len = (dx * dx + dy * dy).sqrt();
        let orig_angle = dy.atan2(dx);

        let new_len = orig_len * 1.5;
        state.set_link_length("crank", "A", "B", new_len);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa2 = crank.attachment_points.get("A").unwrap();
        let pb2 = crank.attachment_points.get("B").unwrap();
        let dx2 = pb2[0] - pa2[0];
        let dy2 = pb2[1] - pa2[1];
        let actual_len = (dx2 * dx2 + dy2 * dy2).sqrt();
        let actual_angle = dy2.atan2(dx2);

        assert!(
            (actual_len - new_len).abs() < 1e-10,
            "Length should be {}, got {}", new_len, actual_len
        );
        assert!(
            (actual_angle - orig_angle).abs() < 1e-10,
            "Direction should be preserved: expected {}, got {}", orig_angle, actual_angle
        );
    }

    #[test]
    fn set_link_length_point_a_stays_fixed() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_before = *bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap();

        state.set_link_length("crank", "A", "B", 0.05);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_after = bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap();

        assert!((pa_after[0] - pa_before[0]).abs() < 1e-12);
        assert!((pa_after[1] - pa_before[1]).abs() < 1e-12);
    }
}
