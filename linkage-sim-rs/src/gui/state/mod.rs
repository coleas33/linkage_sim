//! Application state: mechanism, solver results, selection, view transform.

mod display_units;
mod grid;
mod view_transform;
mod load_cases;
mod types;
mod parametric;
mod simulation;
mod blueprint_ops;
mod entity_crud;
mod driver_ops;
mod undo_ops;
mod trajectory_ops;
pub mod file_io;
mod solver_helpers;
mod templates;

// Re-export all public items so external code can use `crate::gui::state::*`.
pub use display_units::{LengthUnit, AngleUnit, DisplayUnits};
pub use grid::GridSettings;
pub use view_transform::ViewTransform;
pub use load_cases::{LoadCase, LoadCaseManager};
pub use types::{
    PendingJointType, PendingCanvasPickKind, EditorTool, ContextMenuTarget, SelectedEntity,
    ValidationWarnings, SolverStatus, ForceResults, PropertyPanelTab,
    AlignmentAxis, AlignmentGuide, KeyframeTrajectory, Trajectory, TrajectoryProfile,
};
pub use parametric::{
    SweepParameter, ParametricMetric, ParametricStudyConfig, ParametricStudyResult,
    CounterbalanceConfig, CounterbalanceResult,
};
pub use simulation::SimulationState;

// Re-export blueprint helper functions used in tests and other modules.
pub(crate) use blueprint_ops::detect_driver_joint_id;

// ── Motion Profile ──────────────────────────────────────────────────────────

/// Selects how the driver angular velocity varies over one sweep cycle.
///
/// `ConstantSpeed` (the default) uses a fixed omega throughout the cycle.
/// `Trapezoidal` accelerates from rest, cruises, then decelerates to rest,
/// producing realistic inertial loads for motor sizing.
/// `SCurve` is a jerk-limited (quintic ease-in-out) profile with zero
/// velocity AND zero acceleration at both endpoints — used in real actuator
/// hardware to avoid mechanical shocks at start/stop.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MotionProfile {
    /// Constant angular velocity (existing default behaviour).
    ConstantSpeed,
    /// Trapezoidal velocity profile: ramp up, cruise, ramp down.
    Trapezoidal {
        /// Fraction of the cycle spent accelerating (0.05 .. 0.45).
        accel_fraction: f64,
        /// Fraction of the cycle spent decelerating (0.05 .. 0.45).
        decel_fraction: f64,
    },
    /// Jerk-limited S-curve profile (quintic ease-in-out).
    ///
    /// v1 implementation uses the pure quintic blending function
    /// `s(τ) = τ³(10 − 15τ + 6τ²)` for trajectory mode; this gives zero
    /// velocity and zero acceleration at both endpoints. The `jerk_fraction`
    /// field is reserved for a future full 7-segment formal version and is
    /// currently unused.
    SCurve {
        /// Reserved for future 7-segment SCurve. Currently unused (v1).
        jerk_fraction: f64,
    },
}

impl Default for MotionProfile {
    fn default() -> Self {
        MotionProfile::ConstantSpeed
    }
}

use eframe::egui;
use nalgebra::DVector;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::analysis::grashof::GrashofResult;

/// Discriminant for the active driver type. Each non-`None` variant
/// carries the rate / initial-value scalars relevant to its driver
/// kind, so per-variant semantics is type-checked rather than relying
/// on convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriverKind {
    /// No driver — the mechanism is statically posed and animation is a no-op.
    None,
    /// Revolute driver: angle-controlled (slider is the crank angle).
    /// `angle` is the slider's *body-frame* current value (rad);
    /// `omega` is the angular velocity (rad/s); `theta_0` is the
    /// driver progress at t=0 (rad).
    Revolute { angle: f64, omega: f64, theta_0: f64 },
    /// Linear driver: stroke-controlled (slider is the actuator length).
    /// `stroke` is the slider's current value (m); `velocity` is the
    /// linear velocity (m/s); `length_0` is the actuator length at
    /// t=0 (m).
    Linear { stroke: f64, velocity: f64, length_0: f64 },
}

impl Default for DriverKind {
    fn default() -> Self {
        DriverKind::None
    }
}
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::{ForceElement, GravityElement};
use crate::gui::samples::{build_sample, SampleMechanism};
use crate::gui::undo::UndoHistory;
use crate::gui::sweep::SweepData;
use crate::io::{
    load_mechanism_unbuilt_from_json, mechanism_to_json,
    BodyJson, MechanismJson,
};
use crate::solver::forward_dynamics::{simulate, ForwardDynamicsConfig};

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
    /// Multiple selected entities for batch operations (Shift+click).
    pub multi_selected: Vec<SelectedEntity>,
    /// View / zoom transform.
    pub view: ViewTransform,
    /// Show the debug overlay (defaults true in debug builds).
    pub show_debug_overlay: bool,
    /// Which sample is currently loaded.
    pub current_sample: Option<SampleMechanism>,
    /// Driver kind discriminant. Each non-`None` variant carries the
    /// rate / initial-value scalars relevant to its driver kind:
    /// `Revolute { angle, omega, theta_0 }` and `Linear { stroke,
    /// velocity, length_0 }`. The accessor methods
    /// `driver_omega()` / `driver_theta_0()` / `driver_stroke()` and
    /// their setters provide ergonomic access for sites that don't
    /// need exhaustive matching; dispatch points (animation step,
    /// solve_at_*, plot labelling) match on the variant directly.
    pub driver_kind: DriverKind,
    /// Display-angle offset α (rad) for the driver body. Applied to all
    /// user-visible crank-angle surfaces (slider, sweep range DragValues,
    /// canvas indicator arc, plot x-axes) so that `visible = θ_driver +
    /// α` matches the orientation of the driver body's primary axis on
    /// canvas. `θ_driver` is the solver's body-frame rotation, which
    /// equals the visible bar angle only when the body's local A→B
    /// vector happens to lie along +X — true for sample-built bodies,
    /// generally false for DXF imports. Recomputed on every rebuild.
    /// Zero when there is no driver, no grounded pivot on the driver,
    /// or the driver body has fewer than two attachment points.
    pub driver_display_offset: f64,
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
    /// Sweep angle range minimum in degrees.
    pub sweep_angle_min_deg: f64,
    /// Sweep angle range maximum in degrees.
    pub sweep_angle_max_deg: f64,
    /// Whether the custom sweep range is enabled (false = full 360°).
    pub sweep_range_enabled: bool,
    /// Sweep stroke range minimum in meters (used when linear driver is active).
    pub sweep_stroke_min: f64,
    /// Sweep stroke range maximum in meters (used when linear driver is active).
    pub sweep_stroke_max: f64,
    /// Timestamp (egui time in seconds) when sweep was last marked dirty (for debounce).
    pub sweep_dirty_since: Option<f64>,
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
    /// Whether to show body/joint labels on the canvas.
    pub show_labels: bool,
    /// Whether to overlay loop-equation labels on the canvas (View ▸ Show
    /// equations). Off by default — purely diagnostic.
    pub show_equation_overlay: bool,
    /// Gravity magnitude in m/s² (0 = disabled, 9.81 = Earth standard).
    pub gravity_magnitude: f64,
    /// Mechanism mounting angle in radians (0 = horizontal).
    pub mounting_angle: f64,
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
    // ── Place Force state ───────────────────────────────────────────────
    /// Two-click force placement state. None when not in PlaceForce mode.
    pub place_force_state: Option<PlaceForceState>,
    // ── Force Zone creation state ────────────────────────────────────────
    /// Drag-to-define force zone state. None when not in CreateForceZone mode.
    pub creating_force_zone: Option<ForceZoneDragState>,
    // ── Draw Body Geometry state ─────────────────────────────────────────
    /// Drag-to-draw body geometry state. None when not in DrawBodyGeometry mode.
    pub drawing_body_geometry: Option<DrawBodyGeometryState>,
    // ── Ground pivot drag state ─────────────────────────────────────────
    /// Ground pivot being dragged: (pivot_name, start_world_pos).
    pub dragging_ground_pivot: Option<(String, [f64; 2])>,
    // ── Force zone application-point drag state ─────────────────────────
    /// Force zone whose application point is being dragged. Holds the
    /// index into the mechanism's force list. The body id is resolved on
    /// drag-end so we don't need to carry it across frames.
    pub dragging_force_zone_app_point: Option<usize>,
    // ── Alignment guides ───────────────────────────────────────────────
    /// Active alignment guide lines shown during drag operations.
    /// Cleared at the start of each frame when no drag is active.
    pub alignment_guides: Vec<AlignmentGuide>,
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
    /// Saved parametric sweep results for comparison overlay.
    /// Each entry is a (label, parameter_values, metric_values) triple.
    pub saved_parametric_sweeps: Vec<(String, Vec<f64>, Vec<f64>)>,
    /// Auto-incrementing counter for naming saved parametric sweeps.
    pub saved_parametric_counter: usize,
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
    // ── Link editor ────────────────────────────────────────────────
    /// Which body is being edited in the Link Editor panel (selected via dropdown).
    pub link_editor_body: Option<String>,
    /// Active tab in the left sidebar (Properties vs. Equations). Persistence
    /// is in-memory only — defaults to `Properties` on startup.
    pub property_panel_tab: PropertyPanelTab,
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
    /// Whether a WASM localStorage autosave was found on startup.
    #[cfg(target_arch = "wasm32")]
    pub wasm_has_recovery: bool,
    // ── Status toast ──────────────────────────────────────────────────
    /// Transient status message shown in the status bar (e.g. "Saved: foo.json").
    pub status_message: Option<String>,
    /// Remaining display time for the status message (seconds).
    pub status_message_time: f64,
    // ── Highlight ──────────────────────────────────────────────────────
    /// Joint ID to visually highlight on the canvas (e.g. from panel hover).
    pub highlight_joint: Option<String>,
    // ── View automation ─────────────────────────────────────────────────
    /// When true, `fit_to_view` is called on the next canvas frame and then
    /// cleared. Set after a mechanism is loaded so the view auto-fits.
    pub pending_fit_to_view: bool,
    // ── Templates ───────────────────────────────────────────────────────
    /// Saved mechanism templates: (name, json_string) pairs.
    pub saved_templates: Vec<(String, String)>,
    /// Whether the "Save as Template" name-entry dialog is open.
    pub show_template_name_dialog: bool,
    /// Text buffer for template name input.
    pub template_name_buf: String,
    // ── Custom samples ──────────────────────────────────────────────────
    /// User's custom samples (promoted templates) shown in the Samples dropdown.
    /// Each entry is (name, json_string). Persisted alongside templates.
    pub custom_samples: Vec<(String, String)>,
    /// Whether the "Save as Sample" name-entry dialog is open.
    pub show_custom_sample_dialog: bool,
    /// Text buffer for custom sample name input.
    pub custom_sample_name_buf: String,
    // ── Tutorial ────────────────────────────────────────────────────
    /// Interactive tutorial overlay state.
    pub tutorial: crate::gui::tutorial::TutorialState,
    /// Nathan Mode: grayscale everything.
    pub nathan_mode: bool,
    // ── Background image overlay ────────────────────────────────────
    /// Optional background image for tracing real-world mechanisms.
    pub background_image: Option<BackgroundImage>,
    /// Whether the Image Settings floating window is open.
    pub show_image_settings: bool,
    // ── DXF overlay ────────────────────────────────────────────────────
    /// Optional DXF overlay for importing CAD geometry.
    pub dxf_overlay: Option<super::dxf_import::DxfOverlay>,
    /// Whether the "pick target link" popup for DXF → Add Geometry is open.
    pub show_dxf_geometry_target_dialog: bool,
    /// DXF entity indices captured when the popup opened. The overlay's
    /// live selection may change before the user picks a link, so we snapshot.
    pub dxf_geometry_pending_indices: Vec<usize>,
    // ── Welcome screen ──────────────────────────────────────────────
    /// When true, the welcome screen is dismissed (user started working).
    pub dismiss_welcome: bool,
    // ── Load path visualization ─────────────────────────────────────
    /// Whether to color-code links by joint reaction force magnitude.
    pub show_load_path: bool,
    // ── Place Mass state ───────────────────────────────────────────
    /// Body selected for point mass placement (phase 1 of PlaceMass tool).
    /// When Some, the tool is in phase 2: click anywhere to place the mass.
    pub place_mass_body: Option<String>,
    /// Point mass being reassigned to a different link. (body_id, index)
    /// When Some, next link click moves the mass to that body.
    pub reassigning_point_mass: Option<(String, usize)>,
    /// Point mass being repositioned via mouse click. (body_id, index)
    /// When Some, next canvas click updates the mass position.
    pub repositioning_point_mass: Option<(String, usize)>,
    /// Body to add a new attachment point to. Set when user clicks "Add Joint Point".
    /// When Some, next canvas click places a new attachment point on this body.
    pub adding_joint_point: Option<String>,
    /// User-specified actuator rated force (N) for margin/safety factor display.
    /// When 0.0, the margin overlay is disabled.
    pub actuator_rated_force: f64,
    // ── Motion profile ────────────────────────────────────────────────
    /// Driver velocity profile for sweep analysis (constant speed vs trapezoidal).
    pub motion_profile: MotionProfile,
    // ── Trajectory mode ──────────────────────────────────────────────
    /// Severity for the active trajectory analysis.
    pub trajectory_severity: crate::solver::inverse_kinematics::Severity,
    /// Active sweep mode (Angle, Stroke, or Trajectory). The Angle/Stroke
    /// variants drive `compute_sweep` via the existing auto-detect path
    /// (which still re-derives mode from `mech.n_linear_drivers()`); the
    /// Trajectory variant is the source of truth for the input panel,
    /// which delegates to `gui::trajectory_panel::draw` when active.
    pub sweep_mode: crate::gui::sweep::SweepMode,
    /// When `Some`, the next canvas left-click populates the indicated field of
    /// the active trajectory `ControlTarget` instead of doing normal selection.
    /// Cleared after consumption.
    pub pending_canvas_pick: Option<PendingCanvasPickKind>,
    /// Most recent click-to-scrub time (seconds) on the trajectory plot.
    /// Persists across recomputes — it's a UX state for the visible cursor,
    /// not derived from sweep_data.
    pub last_trajectory_scrub_t: Option<f64>,
    // ── Motion ribbon (ghost poses along trajectory) ─────────────────
    /// Render N evenly-spaced ghost poses of the mechanism along the
    /// trajectory on the canvas, behind the live pose. Visualises the
    /// swept path without animating. Trajectory mode only.
    pub show_motion_ribbon: bool,
    /// Number of ghost poses to render when `show_motion_ribbon` is
    /// true. Clamped to 2..=20 by the slider; values outside that range
    /// are tolerated by the renderer (it short-circuits if < 2).
    pub motion_ribbon_n_ghosts: usize,
    // ── Trajectory playback ──────────────────────────────────────────
    /// When true, the per-frame update advances `trajectory_playback_t`
    /// and back-solves the canvas pose at that trajectory time. Distinct
    /// from `playing` (which is the constant-omega kinematic animation);
    /// the two are mutually exclusive — entering trajectory playback
    /// forces `playing = false` and vice versa.
    pub trajectory_playback_active: bool,
    /// Current trajectory playback time in seconds. Persists across
    /// pauses so resume picks up where it left off.
    pub trajectory_playback_t: f64,
    /// Playback speed multiplier (1.0 = real-time, 0.5 = half, 2.0 =
    /// double). Clamped to 0.05..=4.0 by the UI.
    pub trajectory_playback_speed: f64,
    /// When true, playback loops back to t=0 at the end; otherwise it
    /// stops at duration.
    pub trajectory_playback_loop: bool,
}

/// Background image overlay for tracing mechanisms from photos/sketches.
pub struct BackgroundImage {
    /// The egui texture handle for the loaded image.
    pub texture: egui::TextureHandle,
    /// World-space position of the image center (meters).
    pub world_offset: [f64; 2],
    /// Scale factor: pixels per meter in world space.
    pub scale_px_per_m: f64,
    /// Opacity (0.0 = transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Original image dimensions in pixels.
    pub size_px: [usize; 2],
}

/// Tracks placement state for the Add Body tool.
#[derive(Debug, Clone)]
pub struct AddBodyState {
    /// Points placed so far: (name, world_position).
    pub points: Vec<(String, [f64; 2])>,
}

/// Tracks the state of a Place Force two-click interaction.
#[derive(Debug, Clone)]
pub struct PlaceForceState {
    /// The force element template (type + default parameters).
    /// Body IDs and point coordinates will be filled in by the clicks.
    pub force_template: ForceElement,
    /// Set after the first click.
    pub start: Option<PlaceForceStart>,
}

/// First click of a Place Force interaction.
#[derive(Debug, Clone)]
pub struct PlaceForceStart {
    /// World coordinates of point A.
    pub world_pos: [f64; 2],
    /// Body ID that point A belongs to.
    pub body_id: String,
    /// Named point (attachment or mount) if snapped, None for raw coords.
    pub point_name: Option<String>,
}

/// State for drag-to-define force zone creation on the canvas.
#[derive(Debug, Clone)]
pub struct ForceZoneDragState {
    /// World coordinates of the drag start corner. Set on mouse press.
    pub start_world: [f64; 2],
}

/// State for draw-body-geometry tool: drag on canvas to define a rectangle
/// in body-local space for the target body.
#[derive(Debug, Clone)]
pub struct DrawBodyGeometryState {
    /// Which body is receiving the geometry.
    pub body_id: String,
    /// World coordinates of the drag start. None while waiting for first click.
    pub start_world: Option<[f64; 2]>,
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
                        mount_points: HashMap::new(),
                        coupler_points: HashMap::new(),
                        point_masses: Vec::new(),
                        label: None,
                        geometry: None,
                    },
                );
                m
            },
            joints: HashMap::new(),
            drivers: HashMap::new(),
            load_cases: Vec::new(),
            forces: Vec::new(),
            sweep_config: None,
            mounting_angle: 0.0,
            linear_drivers: Vec::new(),
            sweep_state: None,
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
            multi_selected: Vec::new(),
            view: ViewTransform::default(),
            show_debug_overlay: cfg!(debug_assertions),
            current_sample: None,
            driver_kind: DriverKind::None,
            driver_display_offset: 0.0,
            playing: false,
            animation_speed_deg_per_sec: 90.0,
            loop_mode: true,
            animation_direction: 1.0,
            driver_joint_id: None,
            pending_driver_reassignment: None,
            undo_history: UndoHistory::new(50),
            sweep_data: None,
            show_plots: true,
            sweep_dirty: false,
            sweep_angle_min_deg: 0.0,
            sweep_angle_max_deg: 360.0,
            sweep_range_enabled: false,
            sweep_stroke_min: 0.0,
            sweep_stroke_max: 0.0,
            sweep_dirty_since: None,
            creating_joint: None,
            validation_warnings: ValidationWarnings::default(),
            display_units: DisplayUnits::default(),
            grid: GridSettings::default(),
            force_results: ForceResults::default(),
            show_forces: true,
            show_dimensions: true,
            show_labels: true,
            show_equation_overlay: false,
            gravity_magnitude: 9.81,
            mounting_angle: 0.0,
            load_cases: LoadCaseManager::default(),
            active_tool: EditorTool::Select,
            context_menu_target: ContextMenuTarget::default(),
            draw_link_start: None,
            add_body_state: None,
            place_force_state: None,
            creating_force_zone: None,
            drawing_body_geometry: None,
            dragging_ground_pivot: None,
            dragging_force_zone_app_point: None,
            alignment_guides: Vec::new(),
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
            saved_parametric_sweeps: Vec::new(),
            saved_parametric_counter: 0,
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
            link_editor_body: None,
            property_panel_tab: PropertyPanelTab::default(),
            show_shortcuts: false,
            autosave_timer: 0.0,
            last_save_path: None,
            dirty: false,
            #[cfg(not(target_arch = "wasm32"))]
            recent_files: Self::load_recent_files(),
            #[cfg(target_arch = "wasm32")]
            recent_files: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            recovery_path: Self::check_autosave_recovery(),
            #[cfg(target_arch = "wasm32")]
            recovery_path: None,
            #[cfg(target_arch = "wasm32")]
            wasm_has_recovery: Self::check_wasm_autosave_recovery(),
            status_message: None,
            status_message_time: 0.0,
            highlight_joint: None,
            pending_fit_to_view: false,
            saved_templates: Self::load_saved_templates(),
            show_template_name_dialog: false,
            template_name_buf: String::new(),
            custom_samples: Self::load_saved_custom_samples(),
            show_custom_sample_dialog: false,
            custom_sample_name_buf: String::new(),
            tutorial: crate::gui::tutorial::TutorialState::default(),
            nathan_mode: false,
            background_image: None,
            show_image_settings: false,
            dxf_overlay: None,
            show_dxf_geometry_target_dialog: false,
            dxf_geometry_pending_indices: Vec::new(),
            dismiss_welcome: false,
            show_load_path: true,
            place_mass_body: None,
            reassigning_point_mass: None,
            repositioning_point_mass: None,
            adding_joint_point: None,
            actuator_rated_force: 0.0,
            motion_profile: MotionProfile::default(),
            trajectory_severity: crate::solver::inverse_kinematics::Severity::Analysis,
            sweep_mode: crate::gui::sweep::SweepMode::Angle,
            pending_canvas_pick: None,
            last_trajectory_scrub_t: None,
            show_motion_ribbon: false,
            motion_ribbon_n_ghosts: 8,
            trajectory_playback_active: false,
            trajectory_playback_t: 0.0,
            trajectory_playback_speed: 1.0,
            trajectory_playback_loop: true,
        };
        state.rebuild();
        state
    }
}

// ── Methods that remain in mod.rs (core solve/animation/sample loading) ──────

impl AppState {
    // ── Driver scalar accessors ────────────────────────────────────────────
    //
    // These read/write the rate / initial-value scalars stored on the
    // active `DriverKind` variant. Callers that want exhaustive
    // matching should pattern-match `driver_kind` directly; these
    // accessors exist to keep flat call sites readable.

    /// Active driver's primary rate parameter.
    /// Revolute → ω (rad/s); Linear → velocity (m/s); None → 0.0.
    pub fn driver_omega(&self) -> f64 {
        match self.driver_kind {
            DriverKind::Revolute { omega, .. } => omega,
            DriverKind::Linear { velocity, .. } => velocity,
            DriverKind::None => 0.0,
        }
    }

    /// Active driver's initial value parameter.
    /// Revolute → θ₀ (rad); Linear → L₀ (m); None → 0.0.
    pub fn driver_theta_0(&self) -> f64 {
        match self.driver_kind {
            DriverKind::Revolute { theta_0, .. } => theta_0,
            DriverKind::Linear { length_0, .. } => length_0,
            DriverKind::None => 0.0,
        }
    }

    /// Active driver's current stroke (Linear) or 0 (Revolute / None).
    pub fn driver_stroke(&self) -> f64 {
        match self.driver_kind {
            DriverKind::Linear { stroke, .. } => stroke,
            _ => 0.0,
        }
    }

    /// Set the active driver's rate parameter. Setter on `None` is a no-op.
    pub fn set_driver_omega(&mut self, v: f64) {
        match &mut self.driver_kind {
            DriverKind::Revolute { omega, .. } => *omega = v,
            DriverKind::Linear { velocity, .. } => *velocity = v,
            DriverKind::None => {}
        }
    }

    /// Set the active driver's initial value parameter. Setter on `None` is a no-op.
    pub fn set_driver_theta_0(&mut self, v: f64) {
        match &mut self.driver_kind {
            DriverKind::Revolute { theta_0, .. } => *theta_0 = v,
            DriverKind::Linear { length_0, .. } => *length_0 = v,
            DriverKind::None => {}
        }
    }

    /// Set the active driver's current stroke. No-op for non-Linear drivers.
    pub fn set_driver_stroke(&mut self, v: f64) {
        if let DriverKind::Linear { stroke, .. } = &mut self.driver_kind {
            *stroke = v;
        }
    }

    /// Convert a color to grayscale when Nathan Mode is active; pass-through otherwise.
    pub fn nc(&self, c: eframe::egui::Color32) -> eframe::egui::Color32 {
        if self.nathan_mode {
            crate::gui::canvas::to_grayscale(c)
        } else {
            c
        }
    }

    /// Reset to an empty mechanism (just ground body), clearing all state.
    pub fn new_empty_mechanism(&mut self) {
        let fresh = AppState::default();
        // Preserve user preferences across reset.
        let recent = std::mem::take(&mut self.recent_files);
        let templates = std::mem::take(&mut self.saved_templates);
        let custom_samples = std::mem::take(&mut self.custom_samples);
        let units = DisplayUnits {
            length: self.display_units.length,
            angle: self.display_units.angle,
        };
        *self = fresh;
        self.recent_files = recent;
        self.saved_templates = templates;
        self.custom_samples = custom_samples;
        self.display_units = units;
        // Clear WASM autosave so the recovery prompt doesn't reappear.
        #[cfg(target_arch = "wasm32")]
        {
            self.wasm_has_recovery = false;
            Self::wasm_clear_autosave();
        }

        // Create a ground-only mechanism so the canvas shows (ready for editing).
        let ground = crate::core::body::make_ground(&[]);
        let mut mech = crate::core::mechanism::Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.build().unwrap();
        self.q = mech.state().make_q();
        self.last_good_q = self.q.clone();
        self.q_at_zero = self.q.clone();
        self.blueprint = crate::io::mechanism_to_json(&mech).ok();
        self.mechanism = Some(mech);
        self.pending_fit_to_view = true;
    }

    /// Compute view transform that fits all body attachment points in the canvas.
    pub fn fit_to_view(&mut self, canvas_width: f32, canvas_height: f32) {
        if canvas_width < 1.0 || canvas_height < 1.0 {
            return;
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        // Include all body attachment + mount points at current pose
        if let Some(ref mech) = self.mechanism {
            let q = &self.q;
            let sim_state = mech.state();
            for (body_id, body) in mech.bodies() {
                for pt in body.attachment_points.values() {
                    let g = sim_state.body_point_global(body_id, pt, q);
                    x_min = x_min.min(g.x); x_max = x_max.max(g.x);
                    y_min = y_min.min(g.y); y_max = y_max.max(g.y);
                }
                for pt in body.mount_points.values() {
                    let g = sim_state.body_point_global(body_id, pt, q);
                    x_min = x_min.min(g.x); x_max = x_max.max(g.x);
                    y_min = y_min.min(g.y); y_max = y_max.max(g.y);
                }
            }
        }

        // Include coupler trace bounds from sweep data (covers full motion range)
        if let Some(ref sweep) = self.sweep_data {
            for trace in sweep.coupler_traces.values() {
                for &[tx, ty] in trace {
                    if tx.is_finite() && ty.is_finite() {
                        x_min = x_min.min(tx); x_max = x_max.max(tx);
                        y_min = y_min.min(ty); y_max = y_max.max(ty);
                    }
                }
            }
        }

        if !x_min.is_finite() || !x_max.is_finite() {
            return;
        }

        let margin = 0.20; // 20% margin on each side
        let w = (x_max - x_min).max(0.001);
        let h = (y_max - y_min).max(0.001);
        let cx = (x_min + x_max) / 2.0;
        let cy = (y_min + y_max) / 2.0;

        let scale_x = canvas_width as f64 / (w * (1.0 + 2.0 * margin));
        let scale_y = canvas_height as f64 / (h * (1.0 + 2.0 * margin));
        let scale = scale_x.min(scale_y) as f32;

        // No lower clamp — let the mechanism be as small as needed to fit
        self.view.scale = scale.clamp(0.1, 100_000.0);
        self.view.offset = [
            canvas_width / 2.0 - (cx as f32) * self.view.scale,
            canvas_height / 2.0 + (cy as f32) * self.view.scale,
        ];
    }

    /// Load a named sample: build mechanism, solve at t=0, and store state.
    pub fn load_sample(&mut self, sample: SampleMechanism) {
        let (mech, q0) = build_sample(sample);

        // Extract driver parameters from the sample before storing mechanism.
        // Samples currently all use revolute drivers (the
        // ChebyshevLambdaActuator sample has a LinearDriver but
        // load_sample is reset to revolute defaults; rebuild() will
        // re-detect from the built mechanism's blueprint and update
        // driver_kind appropriately).
        self.driver_kind = DriverKind::Revolute {
            angle: 0.0,
            omega: 2.0 * PI,
            theta_0: 0.0,
        };

        self.solve_and_update(&mech, &q0, 0.0, 1e-10, 50, Some(q0.clone()));

        self.driver_angle = self.driver_theta_0();
        self.q_at_zero = self.q.clone();

        // Samples always start at zero mounting angle.
        self.mounting_angle = 0.0;

        // Create blueprint from the built mechanism.
        self.blueprint = mechanism_to_json(&mech).ok();
        self.mechanism = Some(mech);
        self.current_sample = Some(sample);

        // Pre-fill sweep range for samples with a natural working stroke,
        // but leave disabled so the full coupler curve is visible on load.
        // User can enable "Limit Sweep Range" to focus on the working stroke.
        self.sweep_range_enabled = false;
        if sample == SampleMechanism::ParallelogramPress {
            self.sweep_angle_min_deg = 150.0;
            self.sweep_angle_max_deg = 210.0;
        } else {
            self.sweep_angle_min_deg = 0.0;
            self.sweep_angle_max_deg = 360.0;
        }

        // Initialize stroke range from the first LinearActuator force element
        // (if any) so the stroke sweep UI has sensible defaults.
        self.sweep_stroke_min = 0.0;
        self.sweep_stroke_max = 0.0;
        if let Some(ref m) = self.mechanism {
            for force in m.forces() {
                if let ForceElement::LinearActuator(act) = force {
                    if act.stroke_min > 0.0 || act.stroke_max > 0.0 {
                        self.sweep_stroke_min = act.stroke_min;
                        self.sweep_stroke_max = act.stroke_max;
                        break;
                    }
                }
            }
        }

        self.selected = None;

        // Detect which joint is currently driven
        self.driver_joint_id = self.mechanism.as_ref().and_then(|m| detect_driver_joint_id(m));
        // Initialize default load case from current driver settings
        self.load_cases = if let Some(ref joint_id) = self.driver_joint_id {
            LoadCaseManager::new_default(joint_id, self.driver_omega(), self.driver_theta_0())
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
        self.recompute_driver_display_offset();
        self.pending_fit_to_view = true;
    }

    /// Solve the position problem for the given driver angle (radians).
    ///
    /// Recompute `driver_display_offset` from the current mechanism.
    ///
    /// Finds the driver body, locates its grounded revolute pivot, and
    /// returns the angle (in the body's local frame) from that grounded
    /// pivot to the driver's farthest other attachment point. This is
    /// the angle by which the visible bar direction leads the body's
    /// internal θ. See `driver_display_offset` field docs.
    pub fn recompute_driver_display_offset(&mut self) {
        self.driver_display_offset = self.compute_driver_display_offset();
    }

    fn compute_driver_display_offset(&self) -> f64 {
        use crate::core::constraint::{Constraint, JointConstraint};
        use crate::core::state::GROUND_ID;

        // The display offset only makes sense for revolute drivers —
        // linear drivers parameterise in stroke (m), not angle (rad),
        // so adding a radian offset to a stroke is meaningless. Return
        // 0 for linear/none drivers; downstream consumers should also
        // gate on `driver_kind` before applying the offset.
        if !matches!(self.driver_kind, DriverKind::Revolute { .. }) {
            return 0.0;
        }
        let Some(mech) = self.mechanism.as_ref() else { return 0.0 };
        let Some((_partner, driver)) = mech.driver_body_pair() else { return 0.0 };
        let Some(body) = mech.bodies().get(driver) else { return 0.0 };

        // Find the grounded revolute joint on the driver body and
        // record the driver's local attachment point for that joint.
        let mut anchor_local: Option<nalgebra::Vector2<f64>> = None;
        for joint in mech.joints() {
            if let JointConstraint::Revolute(rev) = joint {
                if rev.body_i_id() == driver && rev.body_j_id() == GROUND_ID {
                    anchor_local = Some(*rev.point_i_local());
                    break;
                }
                if rev.body_j_id() == driver && rev.body_i_id() == GROUND_ID {
                    anchor_local = Some(*rev.point_j_local());
                    break;
                }
            }
        }
        let Some(anchor) = anchor_local else { return 0.0 };

        // Farthest other attachment point in the driver's local frame.
        let mut best: Option<(f64, nalgebra::Vector2<f64>)> = None;
        for pt in body.attachment_points.values() {
            let dx = pt.x - anchor.x;
            let dy = pt.y - anchor.y;
            let d2 = dx * dx + dy * dy;
            if d2 < 1e-12 {
                continue;
            }
            if best.as_ref().map_or(true, |(prev_d2, _)| d2 > *prev_d2) {
                best = Some((d2, *pt));
            }
        }
        let Some((_, far)) = best else { return 0.0 };

        (far.y - anchor.y).atan2(far.x - anchor.x)
    }

    /// Uses `last_good_q` as the initial guess for Newton-Raphson.
    /// On success, updates both `q` and `last_good_q`.
    /// On failure, keeps `last_good_q` unchanged and reports the failure in
    /// `solver_status`.
    pub fn solve_at_angle(&mut self, angle_rad: f64) {
        let Some(mech) = self.mechanism.take() else {
            return;
        };

        // Guard against omega==0 driving t to infinity. Other solver
        // callsites (blueprint_ops, file_io, undo_ops) use the same
        // pattern. Clamping at UI / load boundaries should keep
        // `driver_omega` above `MIN_DRIVER_OMEGA_ABS`; this is defense
        // in depth.
        let omega = self.driver_omega();
        let t = if omega.abs() > f64::EPSILON {
            (angle_rad - self.driver_theta_0()) / omega
        } else {
            0.0
        };

        let guess = self.last_good_q.clone();
        let converged = self.solve_and_update(&mech, &guess, t, 1e-10, 50, None);

        self.mechanism = Some(mech);

        if converged {
            self.driver_angle = angle_rad;
            self.compute_forces(t);
        }
    }

    /// Try flipping the mechanism's assembly configuration onto the
    /// alternate branch. Useful when adjusting sweep limits makes the
    /// solver jump to a config the user didn't want.
    ///
    /// Strategy: reflect every non-ground non-driver body pose across
    /// the line joining the two farthest-apart ground pivots (or the
    /// x-axis if there are fewer than 2 ground pivots), then re-solve
    /// at the current driver angle with the reflected q as the initial
    /// guess. The driver body is left untouched because its angle is
    /// locked by the driver constraint.
    ///
    /// On convergence to a pose that differs from the current q, the
    /// state is updated and the sweep is marked dirty. On convergence
    /// to the same pose, a status message notes "same branch". On
    /// non-convergence, the state is left unchanged and an error toast
    /// is shown.
    pub fn flip_assembly_branch(&mut self) {
        let Some(mech) = self.mechanism.take() else {
            self.status_message = Some("No mechanism to flip".to_string());
            self.status_message_time = 3.0;
            return;
        };

        // 1. Find reflection axis from ground pivots.
        let ground_pivots: Vec<[f64; 2]> = mech
            .bodies()
            .get(GROUND_ID)
            .map(|g| {
                g.attachment_points
                    .values()
                    .map(|v| [v.x, v.y])
                    .collect()
            })
            .unwrap_or_default();
        let (axis_origin, axis_angle) = if ground_pivots.len() >= 2 {
            // Pick the two farthest-apart ground pivots as the axis.
            let mut best_sq = 0.0_f64;
            let mut best = (ground_pivots[0], ground_pivots[1]);
            for i in 0..ground_pivots.len() {
                for j in (i + 1)..ground_pivots.len() {
                    let dx = ground_pivots[j][0] - ground_pivots[i][0];
                    let dy = ground_pivots[j][1] - ground_pivots[i][1];
                    let d_sq = dx * dx + dy * dy;
                    if d_sq > best_sq {
                        best_sq = d_sq;
                        best = (ground_pivots[i], ground_pivots[j]);
                    }
                }
            }
            let (a, b) = best;
            let angle = (b[1] - a[1]).atan2(b[0] - a[0]);
            (nalgebra::Vector2::new(a[0], a[1]), angle)
        } else {
            // Fallback: reflect across the world x-axis.
            (nalgebra::Vector2::zeros(), 0.0)
        };

        // 2. Identify the driver body so we skip reflecting it (its
        //    angle is locked by the driver constraint).
        let driver_body: Option<String> =
            mech.driver_body_pair().map(|(_, d)| d.to_string());

        // 3. Build a reflected initial guess from the current q.
        let mut q_guess = self.q.clone();
        let mech_state = mech.state();
        let two_alpha = 2.0 * axis_angle;
        let (sin_2a, cos_2a) = two_alpha.sin_cos();

        for bid in mech.body_order() {
            if bid == GROUND_ID || Some(bid) == driver_body.as_ref() {
                continue;
            }
            let (x, y, theta) = mech_state.get_pose(bid, &self.q);
            let dx = x - axis_origin.x;
            let dy = y - axis_origin.y;
            // Reflection of (dx, dy) across a line through origin at
            // angle α: (dx*cos 2α + dy*sin 2α, dx*sin 2α − dy*cos 2α)
            let x_new = axis_origin.x + dx * cos_2a + dy * sin_2a;
            let y_new = axis_origin.y + dx * sin_2a - dy * cos_2a;
            let theta_new = two_alpha - theta;
            mech_state.set_pose(bid, &mut q_guess, x_new, y_new, theta_new);
        }

        // 4. Re-solve at the current driver angle with the reflected
        //    guess. solve_and_update handles status + q bookkeeping.
        let q_before = self.q.clone();
        let t = (self.driver_angle - self.driver_theta_0()) / self.driver_omega();
        let converged = self.solve_and_update(&mech, &q_guess, t, 1e-10, 50, None);

        self.mechanism = Some(mech);

        if !converged {
            self.status_message = Some(
                "Branch flip failed to converge — try a different crank angle first".to_string(),
            );
            self.status_message_time = 4.0;
            return;
        }

        // 5. Decide if we actually moved to a different branch.
        let diff = (&self.q - &q_before).norm();
        if diff < 1e-6 {
            self.status_message = Some(
                "Flip returned the same configuration (mechanism may only have one branch)"
                    .to_string(),
            );
            self.status_message_time = 4.0;
        } else {
            self.mark_sweep_dirty();
            self.compute_forces(t);
            self.status_message = Some("Flipped to alternate assembly branch".to_string());
            self.status_message_time = 3.0;
        }
    }

    /// Solve the position problem for the given actuator stroke (meters).
    ///
    /// For linear drivers, the time mapping is:
    ///   length = length_0 + velocity * t  =>  t = (stroke - length_0) / velocity
    /// Uses `driver_omega` (= velocity) and `driver_theta_0` (= length_0) which
    /// are set by `rebuild()` when a linear driver is present.
    pub fn solve_at_stroke(&mut self, stroke_m: f64) {
        if self.mechanism.is_none() {
            return;
        }

        let velocity = self.driver_omega();
        let t = if velocity.abs() > f64::EPSILON {
            (stroke_m - self.driver_theta_0()) / velocity
        } else {
            0.0
        };

        let guess = self.last_good_q.clone();
        let mech = self.mechanism.take().unwrap();
        let converged = self.solve_and_update(&mech, &guess, t, 1e-10, 50, None);
        self.mechanism = Some(mech);

        if converged {
            self.set_driver_stroke(stroke_m);
            self.compute_forces(t);
        }
    }

    /// Returns true if the current mechanism uses a linear driver (actuator).
    pub fn has_linear_driver(&self) -> bool {
        self.mechanism.as_ref()
            .map(|m| m.n_linear_drivers() > 0)
            .unwrap_or(false)
    }

    /// Returns true if a mechanism has been loaded.
    pub fn has_mechanism(&self) -> bool {
        self.mechanism.is_some()
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

        // Sync gravity (rotated by mounting angle)
        if self.gravity_magnitude > 0.0 {
            let g = self.gravity_magnitude;
            let theta = self.mounting_angle;
            mech.add_force(ForceElement::Gravity(GravityElement {
                g_vector: [-g * theta.sin(), -g * theta.cos()],
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

        match self.driver_kind {
            DriverKind::Revolute { .. } => self.step_animation_revolute(dt),
            DriverKind::Linear { .. } => self.step_animation_linear(dt),
            DriverKind::None => {
                self.playing = false;
                false
            }
        }
    }

    /// Advance animation for a linear driver (stroke in metres).
    ///
    /// Mirrors `step_animation_revolute` but operates on actuator
    /// stroke. Without an explicit sweep range, the stroke pings
    /// between the actuator's `stroke_min` / `stroke_max` (cached on
    /// `self.sweep_stroke_min/max` after rebuild). With the sweep
    /// range enabled, the stored `sweep_angle_min/max_deg` are
    /// reinterpreted as **stroke values in mm** (display frame); we
    /// convert to metres for the bounce check and ping-pong / stop
    /// at the limits exactly like the revolute path.
    fn step_animation_linear(&mut self, dt: f64) -> bool {
        // Use the ANIMATION_SPEED slider (deg/s for revolute) as
        // mm/s here so the same control feels consistent. 1 deg/s
        // → 1 mm/s. The user can dial 0.5..720 like before.
        let step_m = (self.animation_speed_deg_per_sec * 1e-3) * dt * self.animation_direction;
        let mut new_stroke = self.driver_stroke() + step_m;

        let (anim_min, anim_max) = if self.sweep_range_enabled {
            // Display values stored in mm; convert to m for stroke.
            (
                self.sweep_angle_min_deg * 1e-3,
                self.sweep_angle_max_deg * 1e-3,
            )
        } else if self.sweep_stroke_max > self.sweep_stroke_min {
            (self.sweep_stroke_min, self.sweep_stroke_max)
        } else {
            // No stroke limits known — give a reasonable ±10 cm window
            // around the initial length so the actuator doesn't shoot
            // off forever.
            let length_0 = self.driver_theta_0();
            (length_0 - 0.1, length_0 + 0.1)
        };

        if new_stroke >= anim_max {
            new_stroke = anim_max;
            if self.loop_mode {
                self.animation_direction *= -1.0;
            } else {
                self.playing = false;
            }
        } else if new_stroke <= anim_min {
            new_stroke = anim_min;
            if self.loop_mode {
                self.animation_direction *= -1.0;
            } else {
                self.playing = false;
            }
        }

        let prev_converged = self.solver_status.converged;
        self.solve_at_stroke(new_stroke);

        // Match the revolute path's ping-pong-on-failure behaviour.
        if self.loop_mode && !self.solver_status.converged && prev_converged {
            self.animation_direction *= -1.0;
        }
        if !self.loop_mode && !self.solver_status.converged {
            self.playing = false;
        }

        self.playing
    }

    /// Advance animation for a revolute driver (angle in degrees).
    fn step_animation_revolute(&mut self, dt: f64) -> bool {
        let step_deg = self.animation_speed_deg_per_sec * dt * self.animation_direction;
        let mut new_angle_deg = self.driver_angle.to_degrees() + step_deg;

        // Determine effective animation bounds in body-frame θ. The
        // stored sweep_angle_min/max are DISPLAY frame, so subtract
        // the driver display offset before comparing to the body-frame
        // driver_angle. The no-range case stays body-frame [0, 360).
        let (anim_min, anim_max) = if self.sweep_range_enabled {
            let offset_deg = self.driver_display_offset.to_degrees();
            (
                self.sweep_angle_min_deg - offset_deg,
                self.sweep_angle_max_deg - offset_deg,
            )
        } else {
            (0.0, 360.0)
        };

        if self.sweep_range_enabled {
            // Bounce at sweep range limits in both loop and once modes.
            if new_angle_deg >= anim_max {
                new_angle_deg = anim_max;
                if self.loop_mode {
                    self.animation_direction *= -1.0;
                } else {
                    self.playing = false;
                }
            } else if new_angle_deg <= anim_min {
                new_angle_deg = anim_min;
                if self.loop_mode {
                    self.animation_direction *= -1.0;
                } else {
                    self.playing = false;
                }
            }
        } else if self.loop_mode {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
