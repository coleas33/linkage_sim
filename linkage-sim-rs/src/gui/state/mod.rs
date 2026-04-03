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
pub mod file_io;
mod solver_helpers;
mod templates;

// Re-export all public items so external code can use `crate::gui::state::*`.
pub use display_units::{LengthUnit, AngleUnit, DisplayUnits};
pub use grid::GridSettings;
pub use view_transform::ViewTransform;
pub use load_cases::{LoadCase, LoadCaseManager};
pub use types::{
    PendingJointType, EditorTool, ContextMenuTarget, SelectedEntity,
    ValidationWarnings, SolverStatus, ForceResults,
    AlignmentAxis, AlignmentGuide,
};
pub use parametric::{
    SweepParameter, ParametricMetric, ParametricStudyConfig, ParametricStudyResult,
    CounterbalanceConfig, CounterbalanceResult,
};
pub use simulation::SimulationState;

// Re-export blueprint helper functions used in tests and other modules.
pub(crate) use blueprint_ops::detect_driver_joint_id;

use eframe::egui;
use nalgebra::DVector;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::analysis::grashof::GrashofResult;
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
    /// Angular velocity of the driver (rad/s).
    pub driver_omega: f64,
    /// Initial driver angle (rad) at t=0.
    pub driver_theta_0: f64,
    /// Current actuator stroke in meters (for linear driver mode).
    /// When a linear driver is active, this tracks the slider value instead of `driver_angle`.
    pub driver_stroke: f64,
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
    // ── Ground pivot drag state ─────────────────────────────────────────
    /// Ground pivot being dragged: (pivot_name, start_world_pos).
    pub dragging_ground_pivot: Option<(String, [f64; 2])>,
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
            driver_omega: 2.0 * PI,
            driver_theta_0: 0.0,
            driver_stroke: 0.0,
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
            gravity_magnitude: 9.81,
            mounting_angle: 0.0,
            load_cases: LoadCaseManager::default(),
            active_tool: EditorTool::Select,
            context_menu_target: ContextMenuTarget::default(),
            draw_link_start: None,
            add_body_state: None,
            place_force_state: None,
            creating_force_zone: None,
            dragging_ground_pivot: None,
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
            tutorial: crate::gui::tutorial::TutorialState::default(),
            nathan_mode: false,
            background_image: None,
            show_image_settings: false,
            dismiss_welcome: false,
            show_load_path: true,
            place_mass_body: None,
            reassigning_point_mass: None,
            repositioning_point_mass: None,
            adding_joint_point: None,
        };
        state.rebuild();
        state
    }
}

// ── Methods that remain in mod.rs (core solve/animation/sample loading) ──────

impl AppState {
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
        let units = DisplayUnits {
            length: self.display_units.length,
            angle: self.display_units.angle,
        };
        *self = fresh;
        self.recent_files = recent;
        self.saved_templates = templates;
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
        // All samples use revolute drivers: omega=2π, theta_0=0.
        self.driver_omega = 2.0 * PI;
        self.driver_theta_0 = 0.0;
        self.driver_stroke = 0.0;

        self.solve_and_update(&mech, &q0, 0.0, 1e-10, 50, Some(q0.clone()));

        self.driver_angle = self.driver_theta_0;
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
        self.pending_fit_to_view = true;
    }

    /// Solve the position problem for the given driver angle (radians).
    ///
    /// Uses `last_good_q` as the initial guess for Newton-Raphson.
    /// On success, updates both `q` and `last_good_q`.
    /// On failure, keeps `last_good_q` unchanged and reports the failure in
    /// `solver_status`.
    pub fn solve_at_angle(&mut self, angle_rad: f64) {
        let Some(mech) = self.mechanism.take() else {
            return;
        };

        let t = (angle_rad - self.driver_theta_0) / self.driver_omega;

        let guess = self.last_good_q.clone();
        let converged = self.solve_and_update(&mech, &guess, t, 1e-10, 50, None);

        self.mechanism = Some(mech);

        if converged {
            self.driver_angle = angle_rad;
            self.compute_forces(t);
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

        let t = if self.driver_omega.abs() > f64::EPSILON {
            (stroke_m - self.driver_theta_0) / self.driver_omega
        } else {
            0.0
        };

        let guess = self.last_good_q.clone();
        let mech = self.mechanism.take().unwrap();
        let converged = self.solve_and_update(&mech, &guess, t, 1e-10, 50, None);
        self.mechanism = Some(mech);

        if converged {
            self.driver_stroke = stroke_m;
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

        self.step_animation_revolute(dt)
    }

    /// Advance animation for a revolute driver (angle in degrees).
    fn step_animation_revolute(&mut self, dt: f64) -> bool {
        let step_deg = self.animation_speed_deg_per_sec * dt * self.animation_direction;
        let mut new_angle_deg = self.driver_angle.to_degrees() + step_deg;

        // Determine effective animation bounds.
        let (anim_min, anim_max) = if self.sweep_range_enabled {
            (self.sweep_angle_min_deg, self.sweep_angle_max_deg)
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
