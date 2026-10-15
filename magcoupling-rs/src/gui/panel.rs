//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
//! decision M42-1), opening on Size and torque under a welcome strip the session can hide
//! (decision G-5, which reverses M42-2). Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.
//!
//! The panel never touches files or the network: what needs the platform (saving a file,
//! picking one) it queues as a [`PanelRequest`] for the host, which drains them with
//! [`MagcouplingPanel::take_requests`] after each frame and hands a picked design file back
//! through [`MagcouplingPanel::load_design_file`].
//!
//! Session (spec M4 "Session"): undo and redo of every change to the design (the inputs and
//! the sizing state, [`crate::gui::history`]: one step per settled edit), reset all, save and
//! load a design file, and a share link copied to the clipboard ([`crate::gui::session`]). A
//! host may open a design as the session's start ([`MagcouplingPanel::open_share_payload`]:
//! no undo step) and may keep the undo keys for itself
//! ([`MagcouplingPanel::set_keyboard_shortcuts`]).
//!
//! Sizing (spec Addendum A1; guidance 3.5): the Size tab's Auto toggle is Torque → Magnets, the
//! solver's one home. In it the panel shows the design with the free variable at the solved value
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame); the Size-tab slider and the
//! input row of the free variable show that value, locked, and the Key design group says so. The
//! target is the requirement (decision G-2), the Size tab's Required slider.

use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::size_controls::ring_radius_range;
use crate::engine::sizing::{FreeVariable, SizingOutcome};
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, panel_heights};
use crate::gui::format::{format_value, search_needle, with_unit};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::guidance::{ConsequenceCache, FixCache, SIZE_TAB_ALIASES, within_minimum_diameter};
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row, slider};
use crate::gui::inputs::{
    ADVANCED_HEADING, FILTER_HINT, InputCatalogue, InputEntry, InputGroup, InputOrder,
    InputSection, KEY_DESIGN, filter_inputs, optional_seed,
};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
use crate::gui::session::{
    Design, LoadError, LoadedDesign, PUBLIC_BASE_URL, design_to_json, read_design_json,
    read_share_payload, share_link,
};
use crate::gui::size_controls;
use crate::gui::sizing::{
    REQUIREMENT_INPUT, SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, set_by_auto,
    with_own_value,
};
use crate::gui::spreadsheet::Snapshot;
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::gui::xlsx::{XLSX_FILE_NAME, XLSX_MIME, now_unix_s, spreadsheet_bytes};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";

/// The welcome strip above the centre tabs (spec 3.1), until the session hides it.
pub const WELCOME: &str =
    "Set the size -> read the torque -> open any red check to see why and what to change.";

/// The button at the end of the welcome strip that hides it for the rest of the session.
pub const HIDE_WELCOME: &str = "Hide";

/// The hover text of [`HIDE_WELCOME`].
pub const HIDE_WELCOME_HINT: &str =
    "Hide this line until the page is opened again; the design is not changed";

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The session buttons.
pub const UNDO: &str = "Undo";
pub const REDO: &str = "Redo";
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

/// The header's spreadsheet button (decision X-8; the results table has its own, beside the CSV
/// and JSON exports).
pub const EXPORT_SPREADSHEET: &str = "Export spreadsheet";

/// The file name a saved design suggests.
pub const DESIGN_FILE_NAME: &str = "magcoupling-design.json";

/// Undo: Ctrl+Z (Cmd+Z on a Mac).
pub const UNDO_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);

/// Redo: Ctrl+Shift+Z (Cmd+Shift+Z on a Mac), or Ctrl+Y.
pub const REDO_SHORTCUTS: [egui::KeyboardShortcut; 2] = [
    egui::KeyboardShortcut::new(
        egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
        egui::Key::Z,
    ),
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y),
];

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";

/// The label of the sizing target in the spreadsheet's Summary (Torque → Magnets): the
/// requirement's value (decision G-2; the panel shows the requirement's own row).
pub const TARGET_LABEL: &str = "Target hot-low torque";

/// The label of the free-variable picker.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The start of the log line of each sizing outcome (the web smoke test looks for it).
pub const SIZING_LOG_PREFIX: &str = "magcoupling sizing: ";

/// The note under the free variable's row while Auto (Torque → Magnets) sets it.
pub const SIZED_NOTE: &str = "Set by Auto (Size tab)";

/// The banner shown while an assumption differs from its workbook default (spec Addendum A3),
/// followed by the assumptions' names.
pub const ASSUMPTIONS_MODIFIED: &str = "Assumptions modified";

/// The button beside the banner.
pub const RESET_ASSUMPTIONS: &str = "Reset to workbook defaults";

/// The start of an assumption's source line.
pub const SOURCE: &str = "Source";

/// The line at the top of the Assumptions view.
pub const ASSUMPTIONS_NOTE: &str = "The model's assumptions, apart from the design inputs (each \
     also stays in its input group). The equation panel tags them and what they flow into.";

/// What the inputs side shows (decision M43-7): the design inputs, or the model assumptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputsView {
    Design,
    Assumptions,
}

impl InputsView {
    /// Both views, in tab order.
    pub const ALL: [InputsView; 2] = [InputsView::Design, InputsView::Assumptions];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            InputsView::Design => "Design inputs",
            InputsView::Assumptions => "Assumptions",
        }
    }
}

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;

/// Starting width of the dashboard side [points].
const DASHBOARD_WIDTH: f32 = 300.0;

/// The id of the Equation panel docked at the bottom of the centre region: egui keeps the
/// panel's rect under it, and its resize handle's id derives from it.
const EQUATION_PANEL_ID: &str = "magcoupling_equation_panel";

/// What the panel asks of its host: the platform work an egui panel cannot do itself.
#[derive(Clone, Debug, PartialEq)]
pub enum PanelRequest {
    /// Save `contents` to a file the user picks (native) or download it (web).
    SaveFile {
        /// The suggested file name.
        file_name: String,
        /// The media type (the web download's Blob type).
        mime: &'static str,
        /// The file's bytes: UTF-8 text for a design file and the CSV and JSON exports, as
        /// they are for a binary format.
        contents: Vec<u8>,
    },
    /// Let the user pick a design file, then hand its text to
    /// [`MagcouplingPanel::load_design_file`].
    OpenDesign,
}

/// The views of the centre region, one tab row (decision M42-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreView {
    /// Direct envelope dimensions, magnet count and gap, with the live torque and geometry (the
    /// view a new panel opens on: decision G-5, which reverses M42-2).
    SizeAndTorque,
    /// The end view and the side view to scale, with the dimension callouts.
    Geometry,
    /// One of the plots (egui_plot).
    Plot(PlotKind),
    /// The clamp drawing (drawing.py's end and top views) and the clamp table.
    Clamp,
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 9] = [
        CentreView::SizeAndTorque,
        CentreView::Geometry,
        CentreView::Plot(PlotKind::TorqueTemperature),
        CentreView::Plot(PlotKind::GapSweep),
        CentreView::Plot(PlotKind::PoleSweep),
        CentreView::Plot(PlotKind::SlipHeating),
        CentreView::Plot(PlotKind::TorqueAngle),
        CentreView::Clamp,
        CentreView::Results,
    ];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::SizeAndTorque => crate::gui::size_controls::HEADING,
            CentreView::Geometry => "Geometry",
            CentreView::Plot(kind) => kind.label(),
            CentreView::Clamp => "Clamp",
            CentreView::Results => "Results table",
        }
    }
}

/// The calculator panel: design inputs, their results, and the UI that edits the one and
/// shows the other.
///
/// Hostable by any egui app: the standalone app shows it as a full page
/// (`app::MagcouplingApp`), the linkage app in an `egui::Window` (M5). Call
/// [`MagcouplingPanel::ui`] once per frame.
pub struct MagcouplingPanel {
    inputs: DesignInputs,
    /// The sizing mode and free variable (the target is the requirement input, decision G-2).
    sizing: SizingState,
    /// Runs inverse sizing in Torque → Magnets.
    runner: SizingRunner,
    /// Undo and redo of the design (inputs and sizing state).
    history: History<Design>,
    /// The address share links point at.
    share_base: String,
    /// What the last session action did, until the next one.
    status: Option<String>,
    /// The results of the design shown ([`MagcouplingPanel::shown_inputs`]), recomputed by
    /// every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// The main widget of every input row drawn in the last frame (a slider's value box while
    /// it has focus): a text field among them edits the design.
    design_widgets: Vec<egui::Id>,
    /// The Size tab's Required slider's widget as drawn in the last frame. The centre view is
    /// drawn after the solve checks [`MagcouplingPanel::editing`], so the inputs side lists it
    /// in `design_widgets` ahead of that check: typing in its value box holds the solve.
    requirement_widget: Option<egui::Id>,
    /// Whether the panel acts on the undo and redo keys
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]).
    keyboard_shortcuts: bool,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    /// Whether the session has hidden the welcome strip (spec 3.1): a view choice, never in a
    /// design file, a share link or the undo history, and kept by reset all and every load.
    welcome_dismissed: bool,
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
    /// a design file or a share link.
    input_order: InputOrder,
    /// The user-selected lower OD limit, a view preference that does not resize the design.
    minimum_diameter_mm: Option<f64>,
    /// The Size tab's what-moves-what lines (spec 3.3) for the design shown and the minimum OD,
    /// recomputed only when either changes: view state, in no design file, link or undo step.
    consequences: ConsequenceCache,
    /// Every check guide's fixes for the design shown (spec 3.4), recomputed only when it
    /// changes: each fix runs `compute_all` on the design its move reaches.
    fixes: FixCache,
    /// The inputs filter's text: while it holds more than blanks, the design inputs view shows
    /// the matching inputs alone.
    input_filter: String,
    /// The assumptions banner drawn in the last frame: the header sizes from the last frame,
    /// so a change asks for one more frame.
    banner: Option<String>,
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
    /// The Equation panel: the equation open, its trail, the readout hovered in the last frame
    /// (this frame marks its equation's terms) and the input row a leaf term highlights.
    explorer: Explorer,
    /// The input or result traced (decision O-8): a click on an input's label or on a result
    /// sets it; every path it reaches is framed in the selection colour.
    trace: Option<Trace>,
}

impl Default for MagcouplingPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl MagcouplingPanel {
    /// A panel at the default design.
    pub fn new() -> Self {
        // The equation registry, built once per process, at start-up rather than on the first
        // hover.
        registry();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        Self {
            inputs,
            sizing: SizingState::default(),
            runner: SizingRunner::default(),
            history: History::new(Design::default()),
            share_base: PUBLIC_BASE_URL.to_owned(),
            status: None,
            results,
            key_widgets: Vec::new(),
            design_widgets: Vec::new(),
            requirement_widget: None,
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::SizeAndTorque,
            welcome_dismissed: false,
            inputs_view: InputsView::Design,
            input_order: InputOrder::default(),
            minimum_diameter_mm: None,
            consequences: ConsequenceCache::default(),
            fixes: FixCache::default(),
            input_filter: String::new(),
            banner: None,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            explorer: Explorer::default(),
            trace: None,
        }
    }

    /// The platform work queued since the last call (the host saves files), oldest first.
    pub fn take_requests(&mut self) -> Vec<PanelRequest> {
        std::mem::take(&mut self.requests)
    }

    /// The design: the inputs and the sizing state, as a file holds them.
    pub fn design(&self) -> Design {
        Design {
            inputs: self.inputs.clone(),
            sizing: self.sizing,
        }
    }

    /// Replaces the design (an edit like any other: it can be undone).
    fn set_design(&mut self, design: Design) {
        self.inputs = design.inputs;
        self.sizing = design.sizing;
        self.results = compute_all(&self.shown_inputs());
        self.last_error = None;
    }

    /// The design the panel shows: the inputs, or in Torque → Magnets the inputs with the free
    /// variable at the solved value (the best value when the target is out of reach).
    pub fn shown_inputs(&self) -> DesignInputs {
        match self.sizing.mode {
            SizingMode::MagnetsToTorque => self.inputs.clone(),
            SizingMode::TorqueToMagnets => {
                self.runner
                    .shown(&self.inputs, &self.sizing, self.sizing_range())
            }
        }
    }

    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
    }

    /// The range Auto's solve searches (spec guidance 3.5): ring size over the ring radii of the
    /// Size tab's diameter range, its Minimum OD included
    /// ([`crate::engine::size_controls::ring_radius_range`]), or over its slider when the size
    /// controls define no range (an odd magnet count); every other free variable over its slider.
    /// Read from the inputs, never the design shown, so a solve cannot move its own range.
    fn sizing_range(&self) -> SliderRange {
        match self.sizing.variable {
            FreeVariable::RingRadius => ring_radius_range(&self.inputs, self.minimum_diameter_mm)
                .unwrap_or_else(|| FreeVariable::RingRadius.range()),
            variable => variable.range(),
        }
    }

    /// The Size tab's Auto outcome line and its colour: what the solve changed, from the
    /// inputs' own value to the value `shown` (good); else the runner's status line ("Not
    /// reachable: best ...", "No valid value of ... in range" or a refusal: bad; "Solving...":
    /// weak).
    fn auto_outcome(
        &self,
        shown: &DesignInputs,
        visuals: &egui::Visuals,
    ) -> (String, egui::Color32) {
        let range = self.sizing_range();
        let solved = self.runner.is_current(&self.inputs, &self.sizing, range)
            && matches!(
                self.runner.outcome(self.sizing.variable),
                Some(Ok(SizingOutcome::Solved(_)))
            );
        if solved {
            let line = size_controls::change_text(self.sizing.variable, &self.inputs, shown);
            (line, Level::Good.color(visuals))
        } else {
            let status = self.runner.status(&self.inputs, &self.sizing, range);
            let color = status_color(&status, visuals);
            (status, color)
        }
    }

    /// Sets the address share links point at: the page's own address on the web (so a link
    /// made on a local server opens there), [`PUBLIC_BASE_URL`] by default.
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        self.share_base = base.into();
    }

    /// The share link of the design.
    pub fn share_link(&self) -> String {
        share_link(&self.share_base, &self.design())
    }

    /// Loads a design file's text; on refusal the design is unchanged and the panel shows why.
    pub fn load_design_file(&mut self, text: &str) -> Result<(), LoadError> {
        self.load(read_design_json(text), "Design file loaded")
    }

    /// Loads the design a share link's `?m=` value holds; on refusal the design is unchanged
    /// and the panel shows why.
    pub fn load_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load(
            read_share_payload(payload),
            "Design loaded from the share link",
        )
    }

    /// Opens `design` as the session's start: the design is replaced and the undo history
    /// starts there, so the first Undo does not throw it away. Every other replacement of the
    /// design (a loaded file, reset all) is an edit and can be undone.
    pub fn open_design(&mut self, design: Design) {
        self.minimum_diameter_mm = None;
        self.history = History::new(design.clone());
        self.set_design(design);
    }

    /// Opens the design a share link's `?m=` value holds as the session's start
    /// ([`MagcouplingPanel::open_design`]: the web app's link at start-up); on refusal the
    /// design and the history are unchanged and the panel shows why.
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load_with(
            read_share_payload(payload),
            "Design loaded from the share link",
            Self::open_design,
        )
    }

    /// Lets the panel act on Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y (`true`, the default) or leaves
    /// the key events to the host. A host with its own undo turns them off while its own part
    /// has the user's attention, so one key press never undoes both (M5: the linkage app reads
    /// the same keys without consuming them).
    pub fn set_keyboard_shortcuts(&mut self, enabled: bool) {
        self.keyboard_shortcuts = enabled;
    }

    /// Applies a loaded design as an edit that can be undone.
    fn load(
        &mut self,
        loaded: Result<LoadedDesign, LoadError>,
        done: &str,
    ) -> Result<(), LoadError> {
        self.load_with(loaded, done, Self::set_design)
    }

    /// Applies a loaded design with `apply` (an edit, or a new start) and says `done`, then the
    /// load's notice (a sizing target that differed from the requirement, decision G-2); on
    /// refusal changes nothing and shows why.
    fn load_with(
        &mut self,
        loaded: Result<LoadedDesign, LoadError>,
        done: &str,
        apply: fn(&mut Self, Design),
    ) -> Result<(), LoadError> {
        match loaded {
            Ok(LoadedDesign { design, notice }) => {
                apply(self, design);
                self.minimum_diameter_mm = None;
                self.status = Some(match notice {
                    Some(notice) => format!("{done}. {notice}"),
                    None => done.to_owned(),
                });
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                self.status = None;
                Err(error)
            }
        }
    }

    /// Shows what a host's work for a request did: a message in the header, or why it failed.
    pub fn report(&mut self, outcome: Result<String, String>) {
        match outcome {
            Ok(message) => {
                self.status = Some(message);
                self.last_error = None;
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    /// Undoes the last change to the design, if any.
    pub fn undo(&mut self) {
        if let Some(design) = self.history.undo(&self.design()) {
            self.set_design(design);
        }
    }

    /// Redoes the last undone change, if any.
    pub fn redo(&mut self) {
        if let Some(design) = self.history.redo(&self.design()) {
            self.set_design(design);
        }
    }

    /// The design inputs.
    pub fn inputs(&self) -> &DesignInputs {
        &self.inputs
    }

    /// The results of the design shown ([`MagcouplingPanel::shown_inputs`]), as of the last
    /// frame or edit.
    pub fn results(&self) -> &DesignResults {
        &self.results
    }

    /// Back to the default design (inputs and sizing state); it can be undone.
    pub fn reset(&mut self) {
        self.minimum_diameter_mm = None;
        self.set_design(Design::default());
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut marks = self.explorer.marks(&self.inputs, &self.results);
        if let Some(trace) = &self.trace {
            let reached = trace.paths.iter().map(String::as_str);
            marks = marks.with_marked(
                reached.chain([trace.source.as_str()]),
                ui.visuals().selection.stroke.color,
            );
        }
        let mut readouts = Readouts::new(marks);
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui, &readouts));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits. One copy of the design
            // shown per frame, for the results and the centre region's views.
            let shown = self.shown_inputs();
            self.results = compute_all(&shown);
            // The guides' fixes, the Outside diameter's left out where the Minimum OD stops it.
            let fixes = within_minimum_diameter(
                self.fixes.get(&shown, &self.results),
                &shown,
                self.minimum_diameter_mm,
            );
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| {
                            dashboard_ui(ui, &shown, &self.results, &fixes, &mut readouts)
                        });
                });
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        let mut events = readouts.finish();
        // A fix on an input the Size tab names (the Outside diameter, contract C-4) opens that
        // tab, where its slider is, instead of the input's row.
        if events
            .focus
            .is_some_and(|path| SIZE_TAB_ALIASES.iter().any(|&(aliased, _)| aliased == path))
        {
            events.focus = None;
            self.centre = CentreView::SizeAndTorque;
        }
        // A result clicked opens in the Equation panel and traces its inputs, unless an input's
        // trace marks it: reading the equations of the results an input drives keeps that trace
        // (and the table's "Traced only" rows). A result without an equation record keeps the
        // trace too: nothing is known of its inputs.
        if let Some(path) = &events.clicked {
            let inside = self
                .trace
                .as_ref()
                .is_some_and(|t| t.kind == TraceKind::Input && t.paths.contains(path));
            if !inside && let Some(trace) = Trace::of(path) {
                self.trace = Some(trace);
            }
        }
        // After every change of the trace this frame (Clear trace, a label click, a result
        // click), whichever view the centre region shows: the results table's trace filter goes
        // off with an input's trace even while the table is not drawn.
        self.results_table.sync_trace_filter(self.trace.as_ref());
        self.explorer.end_frame(ui.ctx(), events);
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
    /// down (an arrow key auto-repeating: the whole run is one step, decision M41-14), or a
    /// text field of an input row focused (a typed value, a part name). Undo steps wait for it
    /// to end. A focused text field elsewhere (the results search) edits no design, so it holds
    /// nothing back.
    fn editing(&self, ui: &egui::Ui) -> bool {
        let design_text =
            focused_text_field(ui.ctx()).is_some_and(|id| self.design_widgets.contains(&id));
        design_text || ui.input(|i| i.pointer.any_down() || !i.keys_down.is_empty())
    }

    /// In Torque → Magnets, lets the runner solve if the design has settled, asks for a frame
    /// when a solve is pending, and logs each outcome.
    fn run_sizing(&mut self, ui: &egui::Ui) {
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        let now = ui.input(|i| i.time);
        let solves = self.runner.solves;
        let range = self.sizing_range();
        if let Some(wait) =
            self.runner
                .update(&self.inputs, &self.sizing, range, now, self.editing(ui))
        {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_secs_f64(wait));
        }
        if self.runner.solves != solves {
            self.log_sizing();
            // The inputs side was drawn before the solve: one more frame shows its outcome
            // there (an idle page would otherwise keep showing "Solving...").
            ui.ctx().request_repaint();
        }
    }

    /// Logs the outcome of the last solve (the web smoke reads it).
    fn log_sizing(&self) {
        log::info!(
            "{SIZING_LOG_PREFIX}{}",
            self.runner
                .status(&self.inputs, &self.sizing, self.sizing_range())
        );
    }

    /// Turning Auto off (leaving Torque → Magnets) adopts the value shown (decision M41-7),
    /// solving first a change still waiting for its debounce.
    fn set_sizing_mode(&mut self, mode: SizingMode) {
        if mode == self.sizing.mode {
            return;
        }
        if mode == SizingMode::MagnetsToTorque {
            let range = self.sizing_range();
            if !self.runner.is_current(&self.inputs, &self.sizing, range) {
                self.runner.solve_now(&self.inputs, &self.sizing, range);
                self.log_sizing();
            }
            self.inputs = self.runner.shown(&self.inputs, &self.sizing, range);
        }
        self.sizing.mode = mode;
    }

    /// Auto's line at the top of the Key design group (spec guidance 3.5: the solver's home is
    /// the Size tab): which size Auto sets, a button to the Size tab and the solve's status;
    /// then, when the Key design group does not list it, the free variable's row locked at the
    /// value shown (the ring radius). Nothing while Auto is off.
    fn sizing_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::auto_status(self.sizing.variable));
            if ui.button(size_controls::OPEN_SIZE_TAB).clicked() {
                self.centre = CentreView::SizeAndTorque;
                ui.ctx().request_repaint();
            }
        });
        let status = self
            .runner
            .status(&self.inputs, &self.sizing, self.sizing_range());
        let color = status_color(&status, ui.visuals());
        ui.colored_label(color, status);
        // The free variable's row, locked at the value shown, when the Key design group does
        // not list it (the ring radius is in the Coupling group, closed by default).
        let path = self.sizing.variable.path();
        if !KEY_DESIGN.contains(&path) {
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("every free variable is an input");
            let widget = self.input_row_ui(ui, entry, readouts);
            self.design_widgets.push(widget);
        }
        ui.separator();
    }

    /// Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y, unless the host keeps them
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]) or a text field has focus (it undoes its
    /// own typing).
    fn shortcuts(&mut self, ui: &mut egui::Ui) {
        if !self.keyboard_shortcuts || focused_text_field(ui.ctx()).is_some() {
            return;
        }
        // The redo shortcuts first: Ctrl+Z would also match Ctrl+Shift+Z.
        let redo = ui.input_mut(|i| REDO_SHORTCUTS.iter().any(|s| i.consume_shortcut(s)));
        let undo = !redo && ui.input_mut(|i| i.consume_shortcut(&UNDO_SHORTCUT));
        if redo {
            self.redo();
        } else if undo {
            self.undo();
        }
    }

    /// The centre region: the Equation panel docked at its foot when open, then the view tabs
    /// (wrapping when the region is narrow), the end-effect banner when f_end <= 0 (over every
    /// view, decision M42-1) and the view of `shown`, the design shown, its values readouts
    /// (`readouts`), each in its own area above the panel.
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // The Equation panel docks at the bottom of the centre region (decision M43-1). It
        // starts at about 45 % of the region, and its drag stops short of the view's strip.
        if self.explorer.open {
            let (range, start) = panel_heights(ui.available_rect_before_wrap().height());
            egui::TopBottomPanel::bottom(EQUATION_PANEL_ID)
                .resizable(true)
                .default_height(start)
                .height_range(range)
                .show_inside(ui, |ui| {
                    explorer_ui(ui, &mut self.explorer, shown, &self.results);
                });
        }
        // egui's panel only shrinks the region's cursor: a widget laid out after it may still
        // run past its top edge, and it paints over the panel with a clip rect covering it. So
        // the tab row and the banner live in the space above the panel, and the view in the
        // space left under them, each confined to it.
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            if !self.welcome_dismissed {
                self.welcome_ui(ui);
            }
            ui.horizontal_wrapped(|ui| {
                for view in CentreView::ALL {
                    ui.selectable_value(&mut self.centre, view, view.label());
                }
            });
            ui.separator();
            if let Some(banner) = end_effect_banner(&self.results) {
                ui.colored_label(ui.visuals().error_fg_color, banner);
            }
            confined(ui, "magcoupling_centre_view", |ui| {
                self.view_ui(ui, shown, readouts);
            });
        });
    }

    /// The Size tab's torque readouts (spec 3.2): the requirement's slider, an ordinary design
    /// edit (undo, files, links); the hot-low torque with its gap to the requirement, red short
    /// or green margin; the nominal pull-out at the operating temperature. Where the model makes
    /// no prediction a torque reads [`size_controls::NO_PREDICTION`] (or `NO_TORQUE`) and the
    /// gap is left out.
    fn torque_readouts_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts) {
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata");
        let range = entry.meta.range.expect("the requirement has a slider");
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::REQUIRED);
            let mut value = required;
            let response = ui
                .push_id(REQUIREMENT_INPUT, |ui| {
                    ui.add(slider(&mut value, entry.meta, range))
                })
                .inner;
            // Its value box is a text field of the design, as an input row's is; the next
            // frame's inputs side lists it before the solve looks (`requirement_widget`).
            if !self.design_widgets.contains(&response.id) {
                self.design_widgets.push(response.id);
            }
            self.requirement_widget = Some(response.id);
            if response.changed() && value != required {
                self.last_error = self
                    .inputs
                    .set(REQUIREMENT_INPUT, Value::Num(value))
                    .err()
                    .map(|e| e.to_string());
                // The dashboard was drawn before this edit: the readouts below show it now, the
                // dashboard on the next frame.
                self.results = compute_all(&self.shown_inputs());
                ui.ctx().request_repaint();
            }
        });
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::HOT_LOW);
            if let Some(hot_low) =
                torque_readout(ui, &self.results, "metal.torque_hot_low_Nm", readouts)
            {
                let (gap, meets) = size_controls::torque_gap_text(hot_low, required);
                let level = if meets { Level::Good } else { Level::Bad };
                ui.colored_label(level.color(ui.visuals()), gap);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::pullout_label(self.results.metal.op_temp_C));
            torque_readout(ui, &self.results, "model.pullout_Nm", readouts);
        });
    }

    /// The welcome strip (spec 3.1): what to do first, wrapping in a narrow region, and the
    /// button that hides it for the rest of the session.
    fn welcome_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(WELCOME);
            if ui
                .small_button(HIDE_WELCOME)
                .on_hover_text(HIDE_WELCOME_HINT)
                .clicked()
            {
                self.welcome_dismissed = true;
            }
        });
    }

    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`). Each view fits itself to `ui`'s available height.
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        match self.centre {
            CentreView::SizeAndTorque => {
                self.torque_readouts_ui(ui, readouts);
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        "Exploration only: magnets do not fit at this diameter.",
                    );
                }
                let key_wall = self.results.model.hub_wall_past_key_mm;
                if !key_wall.is_finite() || key_wall <= 0.0 {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        "Exploration only: the keyway breaks through the hub.",
                    );
                } else if key_wall < 2.5 {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "Wall past keyway: {key_wall:.2} mm; existing guidance is about 2.5 mm."
                        ),
                    );
                }
                ui.label("Build checks are shown in the dashboard. The geometry drawing fits the view; use the dimensions to compare sizes.");
                geometry_ui(ui, &self.shown_inputs(), &self.results, readouts);
            }
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results, readouts);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results, readouts),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results, readouts),
            CentreView::Results => {
                let action =
                    self.results_table
                        .ui(ui, &self.results, self.trace.as_ref(), readouts);
                match action {
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
                        mime: "text/csv",
                        contents: results_csv(&self.results).into_bytes(),
                    }),
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
                        mime: "application/json",
                        // The design that produced the results: in Torque -> Magnets the
                        // inputs with the free variable at the value shown.
                        contents: results_json(
                            &Design {
                                inputs: shown.clone(),
                                sizing: self.sizing,
                            },
                            &self.results,
                        )
                        .into_bytes(),
                    }),
                    Some(TableAction::ExportXlsx) => self.export_spreadsheet(),
                    None => {}
                }
            }
        }
    }

    /// Queues the spreadsheet of the design shown (decision X-8: both spreadsheet buttons) for
    /// the host to save, stamped with the time now; when it cannot be written, the header says
    /// why instead.
    fn export_spreadsheet(&mut self) {
        let shown = self.shown_inputs();
        let snapshot = Snapshot {
            inputs: &shown,
            results: &self.results,
            sizing: self.sizing,
            sizing_status: (self.sizing.mode == SizingMode::TorqueToMagnets).then(|| {
                self.runner
                    .status(&self.inputs, &self.sizing, self.sizing_range())
            }),
            input_order: self.input_order,
            share_link: self.share_link(),
            exported_unix_s: now_unix_s(),
        };
        match spreadsheet_bytes(&snapshot) {
            Ok(contents) => self.requests.push(PanelRequest::SaveFile {
                file_name: XLSX_FILE_NAME.to_owned(),
                mime: XLSX_MIME,
                contents,
            }),
            Err(error) => self.report(Err(format!("Could not write the spreadsheet: {error}"))),
        }
    }

    /// The header line: heading, the session buttons, then the last refusal or what the last
    /// session action did.
    fn header_ui(&mut self, ui: &mut egui::Ui) {
        let design = self.design();
        ui.horizontal_wrapped(|ui| {
            ui.heading(HEADING);
            ui.separator();
            let can_undo = self.history.can_undo(&design);
            if ui
                .add_enabled(can_undo, egui::Button::new(UNDO))
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                self.undo();
            }
            let can_redo = self.history.can_redo(&design);
            if ui
                .add_enabled(can_redo, egui::Button::new(REDO))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
                .clicked()
            {
                self.redo();
            }
            if ui.button(RESET_ALL).clicked() {
                self.reset();
                self.status = Some("Default design restored".to_owned());
            }
            ui.separator();
            if ui.button(SAVE_DESIGN).clicked() {
                self.requests.push(PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&design).into_bytes(),
                });
            }
            if ui.button(LOAD_DESIGN).clicked() {
                self.requests.push(PanelRequest::OpenDesign);
            }
            if ui
                .button(COPY_SHARE_LINK)
                .on_hover_text("A link that opens this design, sizing state included")
                .clicked()
            {
                let link = self.share_link();
                self.status = Some(format!(
                    "Share link copied to the clipboard ({} characters)",
                    link.len()
                ));
                ui.ctx().copy_text(link);
            }
            if ui
                .button(EXPORT_SPREADSHEET)
                .on_hover_text(
                    "This design and its results as an .xlsx workbook laid out as this page (values, no formulas)",
                )
                .clicked()
            {
                self.export_spreadsheet();
            }
            ui.separator();
            if ui
                .selectable_label(self.explorer.open, EQUATION_PANEL)
                .on_hover_text("Show or hide the equation of the value clicked")
                .clicked()
            {
                if self.explorer.open {
                    self.explorer.close();
                } else {
                    self.explorer.open = true;
                }
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
        let banner = assumptions_banner(&self.inputs);
        if banner != self.banner {
            self.banner.clone_from(&banner);
            ui.ctx().request_repaint();
        }
        if let Some(banner) = banner {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().warn_fg_color, banner);
                if ui
                    .button(RESET_ASSUMPTIONS)
                    .on_hover_text(
                        "Every assumption back to its workbook default; the design inputs stay as they are",
                    )
                    .clicked()
                {
                    assumptions::reset_to_workbook_defaults(&mut self.inputs);
                    self.status = Some("Assumptions reset to the workbook defaults".to_owned());
                    self.last_error = None;
                }
            });
        }
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        } else if let Some(status) = &self.status {
            ui.weak(status);
        }
    }

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen,
    /// with the trace's banner at its foot. A leaf term clicked in the Equation panel shows the
    /// view that holds its row.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        if self.centre == CentreView::SizeAndTorque {
            ui.heading(crate::gui::size_controls::HEADING);
            // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
            // shown (decision M41-7) through the mode switch.
            let asked = size_controls::auto_ui(ui, self.sizing);
            self.sizing.variable = asked.variable;
            self.set_sizing_mode(asked.mode);
            let auto =
                (self.sizing.mode == SizingMode::TorqueToMagnets).then_some(self.sizing.variable);
            let shown = self.shown_inputs();
            if auto.is_some() {
                let (line, color) = self.auto_outcome(&shown, ui.visuals());
                ui.add(egui::Label::new(egui::RichText::new(line).color(color)).wrap());
            }
            let minimum = self.minimum_diameter_mm;
            // The cache keys on the design alone, so it is handed that design's own results:
            // `self.results` can lag the inputs by a frame (the header's assumptions reset edits
            // them after the last compute), and a lagging pull-out would stick in the lines until
            // the next edit. One compute_all (about 17 us) per frame of this tab; the neighbour
            // runs and the fit search rerun only when the design or the minimum changes.
            let shown_results = compute_all(&shown);
            let consequences = self.consequences.get(&shown, &shown_results, minimum);
            let output = size_controls::controls_ui(ui, &shown, minimum, consequences, auto);
            // The lines under the controls were drawn from the design before this frame's edit:
            // one more frame shows the edited design's.
            if output.inputs != shown || output.minimum_diameter_mm != minimum {
                ui.ctx().request_repaint();
            }
            self.design_widgets.extend(output.widgets);
            // The Required slider is drawn after `run_sizing`: last frame's id stands in.
            self.design_widgets.extend(self.requirement_widget);
            self.minimum_diameter_mm = output.minimum_diameter_mm;
            if output.inputs != shown {
                // The controls edit the design shown: the inputs keep their own value of the size
                // Auto sets until Auto turns off (decision M41-7).
                self.inputs = match auto {
                    Some(variable) => with_own_value(variable, output.inputs, &self.inputs),
                    None => output.inputs,
                };
                self.last_error = output.error;
            } else if output.error.is_some() {
                self.last_error = output.error;
            }
            ui.separator();
            egui::CollapsingHeader::new("Detailed inputs")
                .default_open(false)
                .open(
                    self.explorer
                        .focus()
                        .filter(|focus| focus.scroll)
                        .map(|_| true),
                )
                .show(ui, |ui| self.design_inputs_ui(ui, readouts));
            return;
        }
        if let Some(focus) = self.explorer.focus().filter(|f| f.scroll) {
            let assumption = ASSUMPTIONS
                .iter()
                .any(|a| a.paths.contains(&focus.path.as_str()));
            self.inputs_view = if assumption {
                InputsView::Assumptions
            } else {
                InputsView::Design
            };
        }
        ui.horizontal(|ui| {
            for view in InputsView::ALL {
                ui.selectable_value(&mut self.inputs_view, view, view.label());
            }
        });
        // The trace's banner at the foot of the side, laid out bottom up first so the view takes
        // the room above it in the same frame: a trace moves no row above it, so a second click
        // on the label lands on the label again (decision O-8).
        let banner = self.trace.as_ref().map(Trace::banner);
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            if let Some(banner) = banner {
                if ui.small_button(CLEAR_TRACE).clicked() {
                    self.trace = None;
                }
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(banner).color(ui.visuals().selection.stroke.color),
                    )
                    .wrap(),
                );
                ui.separator();
            }
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.separator();
                match self.inputs_view {
                    InputsView::Design => self.design_inputs_ui(ui, readouts),
                    InputsView::Assumptions => self.assumptions_ui(ui, readouts),
                }
            });
        });
    }

    /// The Assumptions view (spec Addendum A3 "Toggle panel"): each assumption with its
    /// changed-from-default dot, the rows of its inputs (value, unit, reset: the inputs' own
    /// rows, so an edit here is an edit like any other), its rationale and its source.
    fn assumptions_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_assumptions_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(ASSUMPTIONS_NOTE).weak()).wrap());
                for state in assumptions::states(&self.inputs) {
                    let assumption = state.assumption;
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        let dot = if state.modified { CHANGED_DOT } else { " " };
                        ui.colored_label(ui.visuals().selection.stroke.color, dot)
                            .on_hover_text("Changed from the workbook default");
                        ui.strong(assumption.label);
                    });
                    for path in assumption.paths {
                        let entry = catalogue
                            .entry(path)
                            .expect("an assumption path is an input (tests/assumptions.rs)");
                        let widget = self.input_row_ui(ui, entry, readouts);
                        self.design_widgets.push(widget);
                    }
                    ui.add(
                        egui::Label::new(egui::RichText::new(assumption.rationale).weak()).wrap(),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("{SOURCE}: {}", assumption.source))
                                .small()
                                .weak(),
                        )
                        .wrap(),
                    );
                }
            });
    }

    /// The design inputs: the order toggle and the filter box, then the Key design group and
    /// every input by group in the order chosen (decision O-1), each workflow group's advanced
    /// sections under its closed Advanced heading; while the filter holds more than blanks, the
    /// matching inputs alone.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // A leaf term clicked in the Equation panel: its group (and its Advanced heading) opens
        // so its row can scroll into view (the Key design group for a Key design input), and
        // the filter is cleared so the row is drawn.
        let focus = self
            .explorer
            .focus()
            .filter(|f| f.scroll)
            .map(|f| f.path.clone());
        if focus.is_some() {
            self.input_filter.clear();
        }
        ui.horizontal(|ui| {
            for order in InputOrder::ALL {
                ui.selectable_value(&mut self.input_order, order, order.label());
            }
        });
        ui.add(
            egui::TextEdit::singleline(&mut self.input_filter)
                .hint_text(FILTER_HINT)
                .desired_width(f32::INFINITY),
        );
        let groups = catalogue.groups_in(self.input_order);
        if !search_needle(&self.input_filter).is_empty() {
            self.filtered_inputs_ui(ui, groups, readouts);
            return;
        }
        let open_key_design = focus
            .as_deref()
            .is_some_and(|path| KEY_DESIGN.contains(&path));
        // The group and the section of the focused row (outside the Key design group).
        let focused = focus
            .as_deref()
            .filter(|_| !open_key_design)
            .and_then(|path| catalogue.section_of(self.input_order, path))
            .map(|(group, section)| (group.name.as_str(), section.advanced));
        // The workbook order keeps its groups' ids (and so their open state) from before the
        // workflow order existed.
        let salt = match self.input_order {
            InputOrder::Workflow => "workflow",
            InputOrder::Workbook => "group",
        };
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .open(open_key_design.then_some(true))
                    .show(ui, |ui| {
                        self.sizing_ui(ui, readouts);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry, readouts);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                        }
                    });
                for group in groups {
                    let (open_group, open_advanced) = match focused {
                        Some((name, advanced)) if name == group.name => (true, advanced),
                        _ => (false, false),
                    };
                    // A trace counts the rows it frames in each group, closed or open.
                    let traced = self.trace.as_ref().map_or(0, |trace| {
                        trace.count_in(
                            group
                                .sections
                                .iter()
                                .flat_map(|s| s.entries.iter())
                                .map(|e| e.path.as_str()),
                        )
                    });
                    let title = if traced > 0 {
                        format!("{} ({traced} traced)", group.label)
                    } else {
                        group.label.to_owned()
                    };
                    egui::CollapsingHeader::new(title)
                        .id_salt((salt, &group.name))
                        .default_open(false)
                        .open(open_group.then_some(true))
                        .show(ui, |ui| {
                            for section in group.plain_sections() {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.has_advanced() {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.advanced_sections() {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
                            }
                        });
                }
            });
    }

    /// One section of a group: its heading ([`InputSection::heading_in`]: none for the group's
    /// own section), then its rows.
    fn section_ui(
        &mut self,
        ui: &mut egui::Ui,
        group: &InputGroup,
        section: &'static InputSection,
        readouts: &Readouts,
    ) {
        if let Some(heading) = section.heading_in(group) {
            ui.add_space(4.0);
            ui.strong(heading);
        }
        for entry in &section.entries {
            let widget = self.input_row_ui(ui, entry, readouts);
            self.design_widgets.push(widget);
        }
    }

    /// The inputs the filter matches, by section in the order chosen, each run under its
    /// group and section, with their count.
    fn filtered_inputs_ui(
        &mut self,
        ui: &mut egui::Ui,
        groups: &'static [InputGroup],
        readouts: &Readouts,
    ) {
        let matches = filter_inputs(groups, &self.input_filter);
        let shown: usize = matches.iter().map(|m| m.entries.len()).sum();
        let total = InputCatalogue::get().all().count();
        ui.weak(format!("{shown} of {total} inputs"));
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_filter_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for run in &matches {
                    ui.add_space(4.0);
                    ui.strong(run.heading());
                    for &entry in &run.entries {
                        let widget = self.input_row_ui(ui, entry, readouts);
                        self.design_widgets.push(widget);
                    }
                }
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget. In Torque →
    /// Magnets the rows of the inputs Auto sets ([`set_by_auto`]: the free variable's, and for
    /// ring size the cap OD and thread that follow it) show the value the panel shows, locked
    /// (the label still traces the input). The row is framed in its term's colour while the equation in view
    /// reads the input (`readouts`).
    fn input_row_ui(
        &mut self,
        ui: &mut egui::Ui,
        entry: &'static InputEntry,
        readouts: &Readouts,
    ) -> egui::Id {
        let locked = self.sizing.mode == SizingMode::TorqueToMagnets
            && set_by_auto(self.sizing.variable).any(|path| path == entry.path);
        // A locked row reads the design shown (one clone, for that row only).
        let current = if locked {
            self.shown_inputs().get(&entry.path)
        } else {
            self.inputs.get(&entry.path)
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let row = ui.scope(|ui| input_row(ui, entry, &current, seed, locked));
        readouts.mark(ui, row.response.rect, &entry.path);
        if self
            .explorer
            .focus()
            .is_some_and(|focus| focus.path == entry.path)
        {
            ui.painter().rect_stroke(
                row.response.rect.expand(3.0),
                3.0,
                egui::Stroke::new(FOCUS_WIDTH, ui.visuals().selection.stroke.color),
                egui::StrokeKind::Outside,
            );
            if self.explorer.take_scroll(&entry.path) {
                row.response.scroll_to_me(Some(egui::Align::Center));
            }
        }
        let output = row.inner;
        if output.label_clicked {
            // A second click on the source's label ends the trace.
            let same = self.trace.as_ref().is_some_and(|t| t.source == entry.path);
            self.trace = if same { None } else { Trace::of(&entry.path) };
            ui.ctx().request_repaint();
        }
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
        }
        if let Some(edit) = output.edit {
            let value = match edit {
                RowEdit::Set(value) => value,
                RowEdit::Reset => entry.default.clone(),
            };
            self.last_error = self
                .inputs
                .set(&entry.path, value)
                .err()
                .map(|e| e.to_string());
            // The header (the assumptions banner) was drawn before this edit.
            ui.ctx().request_repaint();
        }
        output.widget.id
    }

    /// Where an optional input starts when a value is entered: the result it overrides,
    /// clamped into its slider range but not rounded to the step (decision M41-12: rounding
    /// would move the design it is meant to keep). Like a loaded value, an off-grid seed is
    /// kept until the first edit.
    fn seed(&self, entry: &InputEntry) -> Option<f64> {
        let range = entry.meta.range?;
        match self.results.get(optional_seed(&entry.path)?)? {
            Value::Num(x) if x.is_finite() => Some(x.clamp(range.min, range.max)),
            _ => None,
        }
    }
}

/// The assumptions banner (spec Addendum A3) of `inputs`: [`ASSUMPTIONS_MODIFIED`] and the names
/// of the assumptions that differ from their workbook default; `None` while none does. The header
/// shows it, and the spreadsheet's Summary.
pub(crate) fn assumptions_banner(inputs: &DesignInputs) -> Option<String> {
    let modified = assumptions::modified(inputs);
    (!modified.is_empty()).then(|| {
        let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
        format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
    })
}

/// The colour of a sizing status line ([`SizingRunner::status`]): good when solved, weak while
/// solving, bad otherwise (not reachable, no valid value, refused).
fn status_color(status: &str, visuals: &egui::Visuals) -> egui::Color32 {
    if status.starts_with(SOLVED_PREFIX) {
        Level::Good.color(visuals)
    } else if status == SOLVING {
        visuals.weak_text_color()
    } else {
        Level::Bad.color(visuals)
    }
}

/// One torque readout of the Size tab: the value of the result at `path` with its unit, a
/// readout (`readouts`: hover text and equation; a click opens it), or
/// [`size_controls::NO_PREDICTION`] / [`size_controls::NO_TORQUE`] where the model makes none
/// ([`size_controls::predicts`]). Returns the torque shown [N·m], if any.
fn torque_readout(
    ui: &mut egui::Ui,
    results: &DesignResults,
    path: &str,
    readouts: &mut Readouts,
) -> Option<f64> {
    let Some(Value::Num(torque)) = results.get(path) else {
        unreachable!("{path} is a torque result")
    };
    if !size_controls::predicts(results, torque) {
        ui.weak(if end_effect_banner(results).is_some() {
            size_controls::NO_PREDICTION
        } else {
            size_controls::NO_TORQUE
        });
        return None;
    }
    let text = with_unit(format_value(&Value::Num(torque)), "N\u{b7}m");
    let response = ui.add(egui::Label::new(text).sense(egui::Sense::click()));
    readouts.show(ui, response, path, || {
        crate::gui::dashboard::hover_text(path).unwrap_or_default()
    });
    Some(torque)
}

/// Draws `add` in a child of `ui` (salted `id_salt`) confined to the space left in `ui`: the
/// child's max rect and clip rect are that space, so nothing drawn in it paints or takes a
/// click outside it. `ui` then moves past the whole space, never past what the child used, so
/// an overflow the clip hides cannot grow a host `egui::Window`.
fn confined<R>(ui: &mut egui::Ui, id_salt: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let region = ui.available_rect_before_wrap();
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(id_salt)
            .max_rect(region)
            .layout(*ui.layout()),
    );
    child.set_clip_rect(region.intersect(ui.clip_rect()));
    let inner = add(&mut child);
    ui.advance_cursor_after_rect(region);
    inner
}

/// The widget with keyboard focus if it is a text field: a text input, or a slider's value box
/// being typed in (egui's `wants_keyboard_input` is true for any focused widget, a slider rail
/// too).
fn focused_text_field(ctx: &egui::Context) -> Option<egui::Id> {
    ctx.memory(|m| m.focused())
        .filter(|&id| egui::text_edit::TextEditState::load(ctx, id).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::size_controls::{DiameterFit, diameter_breakdown, dimensions};
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::guidance::{
        AUTO_RESETS_LENGTH, AUTO_RESIZES_RING, Consequences, RING_SIZE_STAYS, plain_text_violations,
    };
    use crate::gui::input_ui::{BLANK_TEXT, CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::plain_text::label_for;
    use crate::gui::session::{
        decode_share_payload, design_from_json, encode_share_payload, encode_share_text,
        target_mismatch_notice,
    };
    use crate::gui::size_controls::{
        DIAMETER, GAP, HOT_LOW, LENGTH, MAGNETS, NO_PREDICTION, REQUIRED, predicts, pullout_label,
        torque_gap_text,
    };
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_rect, text_rects, with_poles,
    };
    use crate::headline;

    const FACE_GAP: &str = "metal.face_gap_mm";
    const POLES: &str = "coupling.npole";
    const AXIAL_LENGTH: &str = "coupling.magnets.axial_length_mm";
    const PART_INNER: &str = "coupling.magnets.part_inner";
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";

    /// A screen tall enough for every dashboard row: the check guides (their blocks and every
    /// row's Why line) push the lower rows past [`SCREEN`]'s, and egui paints no label
    /// scrolled out of view.
    const TALL: egui::Vec2 = egui::vec2(1280.0, 3000.0);

    /// A panel and the egui context it is drawn in, frame by frame.
    pub(crate) struct Harness {
        pub(crate) ctx: egui::Context,
        pub(crate) panel: MagcouplingPanel,
        screen: egui::Vec2,
    }

    impl Harness {
        pub(crate) fn new() -> Self {
            Self::on_screen(SCREEN)
        }

        /// A harness on a screen of `size`, on the Geometry tab: most tests drive the whole
        /// inputs side (the Key design group, the groups, the filter, the Assumptions view),
        /// which the Size tab a new panel opens on folds under "Detailed inputs". The first
        /// view's own tests use [`Harness::first_view`].
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
            let mut panel = MagcouplingPanel::new();
            panel.centre = CentreView::Geometry;
            Self::of(panel, size)
        }

        /// A harness on a new panel as it opens (decision G-5): the Size tab under the welcome
        /// strip, on the laptop screen.
        pub(crate) fn first_view() -> Self {
            Self::of(MagcouplingPanel::new(), SCREEN)
        }

        /// A harness drawing `panel` on a screen of `size`, one frame in.
        fn of(panel: MagcouplingPanel, size: egui::Vec2) -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel,
                screen: size,
            };
            harness.frame(Vec::new());
            harness
        }

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            sized_frame(&self.ctx, self.screen, events, |ui| panel.ui(ui))
        }

        /// A frame `dt` seconds after the last one.
        fn frame_after(&mut self, dt: f64, events: Vec<egui::Event>) -> egui::FullOutput {
            let time = self.ctx.input(|i| i.time) + dt;
            let panel = &mut self.panel;
            sized_frame_at(&self.ctx, self.screen, Some(time), events, |ui| {
                panel.ui(ui)
            })
        }

        /// Turns Auto on (Torque -> Magnets) without changing the view shown, and lets the
        /// solve run. The Auto toggle's own clicks are tested on the Size tab.
        fn size(&mut self) -> egui::FullOutput {
            self.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            self.frame(Vec::new());
            self.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            self.frame(Vec::new());
            self.frame(Vec::new())
        }

        /// The Size tab's slider under `label` as drawn in the last frame (the Size tab must be
        /// shown): `size_controls::controls_ui` pushes its sliders first, in this order.
        fn size_widget(&self, label: &str) -> egui::Response {
            use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
            let index = [DIAMETER, LENGTH, GAP, MAGNETS]
                .iter()
                .position(|l| *l == label)
                .unwrap_or_else(|| panic!("no size slider {label}"));
            self.ctx
                .read_response(self.panel.design_widgets[index])
                .expect("the slider has a response")
        }

        /// The solved point of the last solve.
        fn solved(&self) -> crate::engine::sizing::SizingPoint {
            match self.panel.runner.outcome(self.panel.sizing.variable) {
                Some(Ok(SizingOutcome::Solved(point))) => point.clone(),
                other => panic!("not solved: {other:?}"),
            }
        }

        /// The Key design row's main widget as drawn in the last frame.
        fn widget(&self, path: &str) -> egui::Response {
            let (_, id) = self
                .panel
                .key_widgets
                .iter()
                .find(|(p, _)| *p == path)
                .copied()
                .unwrap_or_else(|| panic!("no Key design row {path}"));
            self.ctx
                .read_response(id)
                .expect("the widget has a response")
        }

        /// Gives the Key design row's widget keyboard focus.
        fn focus(&mut self, path: &str) {
            let id = self.widget(path).id;
            self.ctx.memory_mut(|m| m.request_focus(id));
            self.frame(Vec::new());
            assert!(self.ctx.memory(|m| m.has_focus(id)), "{path} has focus");
        }

        /// A click (move, press, release) at `at`.
        fn click(&mut self, at: egui::Pos2) -> egui::FullOutput {
            self.frame(vec![egui::Event::PointerMoved(at)]);
            self.frame(vec![primary_button(at, true)]);
            self.frame(vec![primary_button(at, false)])
        }

        /// Clicks the first drawn text equal to `text`.
        fn click_text(&mut self, text: &str) -> egui::FullOutput {
            let output = self.frame(Vec::new());
            let rect = text_rect(&output, text).unwrap_or_else(|| panic!("no text {text:?}"));
            self.click(rect.center())
        }

        fn number(&self, path: &str) -> f64 {
            match self.panel.inputs.get(path) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{path}: {other:?}"),
            }
        }
    }

    /// The headline of `inputs`, as the panel displays it (value with unit).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
        headline(&compute_all(inputs))
            .into_iter()
            .zip(HEADLINE)
            .map(|((_, value), (_, path))| {
                with_unit(format_value(&value), result_info(path).unwrap().meta.unit)
            })
            .collect()
    }

    fn assert_drew_headline(output: &egui::FullOutput, inputs: &DesignInputs) {
        let texts = drawn_texts(output);
        for want in displayed_headline(inputs) {
            assert!(texts.contains(&want), "missing {want:?} in {texts:?}");
        }
    }

    /// The first headline value (pull-out at the operating temperature) as displayed.
    fn displayed_pullout(inputs: &DesignInputs) -> String {
        displayed_headline(inputs).remove(0)
    }

    fn count(output: &egui::FullOutput, text: &str) -> usize {
        drawn_texts(output).iter().filter(|t| *t == text).count()
    }

    /// Asserts the locked free-variable row draws `value` in its value box: on the line
    /// between the row's `label` and the sized note under the row. The results table draws
    /// the same text elsewhere; the inputs side paints first, so `text_rect` finds the row's
    /// copy when the row draws it.
    fn assert_locked_row_shows(output: &egui::FullOutput, label: &str, value: &str) {
        let rect = |text: &str| text_rect(output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let (label_rect, value_rect, note_rect) = (rect(label), rect(value), rect(SIZED_NOTE));
        assert!(
            label_rect.bottom() <= value_rect.top() && value_rect.bottom() <= note_rect.top(),
            "{value:?} at {value_rect:?} is not in the row between {label_rect:?} and {note_rect:?}"
        );
    }

    /// Whether `text` is drawn on the row of the slider whose rail is `rail`, at or right of its
    /// start: the slider's value box.
    fn on_slider_row(output: &egui::FullOutput, rail: egui::Rect, text: &str) -> bool {
        text_rects(output, text).iter().any(|r| {
            r.left() >= rail.left() && (rail.top()..=rail.bottom()).contains(&r.center().y)
        })
    }

    #[test]
    fn the_geometry_view_follows_the_design_shown() {
        // The harness opens on the Geometry tab (a new panel opens on Size and torque:
        // a_new_panel_opens_on_size_and_torque_under_the_welcome_strip).
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "Face gap 1.400 mm"), "{texts:?}");
        // An arrow key on the face gap moves the callout in the same frame (a live redraw).
        harness.focus(FACE_GAP);
        let output = harness.frame(key_tap(egui::Key::ArrowRight));
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == "Face gap 1.410 mm")
        );
        // In Torque -> Magnets it draws the sized design: the axial stack follows the solved
        // length.
        let output = harness.size();
        let stack = harness.panel.results().metal.axial_stack_mm;
        assert_ne!(
            stack,
            compute_all(&DesignInputs::default()).metal.axial_stack_mm
        );
        let want = format!(
            "Overall length: axial stack {} of 35.00 mm",
            crate::gui::geometry::mm(stack)
        );
        assert!(drawn_texts(&output).contains(&want), "{want}");
        // The Results tab shows the table; the Geometry tab brings the view back.
        harness.click_text(CentreView::Results.label());
        assert_eq!(harness.panel.centre, CentreView::Results);
        let total = crate::gui::results_table::table_entries().len();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with("Face gap"))
        );
        harness.click_text(CentreView::Geometry.label());
        assert_eq!(harness.panel.centre, CentreView::Geometry);
    }

    #[test]
    fn a_new_panel_opens_on_size_and_torque_under_the_welcome_strip() {
        // Decision G-5 (spec 3.1, reversing M42-2): the first view is the Size tab with the
        // welcome strip above the centre tabs; showing them edits nothing, and Explain stays
        // off (M43-2).
        use crate::gui::guidance::{design_status, design_status_text};
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        let panel = MagcouplingPanel::new();
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert!(!panel.welcome_dismissed);
        assert!(!panel.explorer.explain);
        assert_eq!(
            MagcouplingPanel::default().centre,
            CentreView::SizeAndTorque
        );
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(count(&output, HIDE_WELCOME), 1);
        // The inputs side is the Size tab's: its controls, the Key design group folded away
        // under the detailed inputs.
        for label in [DIAMETER, LENGTH, GAP, MAGNETS] {
            assert_eq!(count(&output, label), 1, "{label}");
        }
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        // The dashboard's design status line (spec 3.1) is on the first view, counted live.
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        assert!(status.is_default);
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // The strip is in the centre region, above the tab row.
        let strip = text_rect(&output, WELCOME).expect("the strip");
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
        assert!(
            strip.left() >= INPUTS_WIDTH && strip.right() <= SCREEN.x - DASHBOARD_WIDTH,
            "{strip:?}"
        );
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn hiding_the_welcome_strip_frees_its_row_and_edits_no_design() {
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let tabs_under_strip = text_rect(&output, CentreView::Geometry.label()).unwrap();
        let file = design_to_json(&harness.panel.design());
        let link = harness.panel.share_link();
        harness.click_text(HIDE_WELCOME);
        assert!(harness.panel.welcome_dismissed);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, WELCOME), 0);
        assert_eq!(count(&output, HIDE_WELCOME), 0);
        let tabs = text_rect(&output, CentreView::Geometry.label()).unwrap();
        assert!(
            tabs.top() < tabs_under_strip.top(),
            "the tab row moves up into the strip's room: {tabs:?} vs {tabs_under_strip:?}"
        );
        // A view choice of this session: the view, the design, its file and its link stay as
        // they were, and it is no undo step.
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(design_to_json(&harness.panel.design()), file);
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
        assert!(!harness.panel.history.can_undo(&Design::default()));
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn the_hidden_welcome_strip_stays_hidden_for_the_session() {
        // A loaded file, undo and redo, reset all, a refused file, a share link (as an edit and
        // as the session's start) and every view keep it hidden; only a new panel (the page
        // opened again) shows it.
        let hidden = |harness: &mut Harness, after: &str| {
            let output = harness.frame(Vec::new());
            assert!(harness.panel.welcome_dismissed, "{after}");
            assert_eq!(count(&output, WELCOME), 0, "{after}");
        };
        let mut harness = Harness::first_view();
        harness.click_text(HIDE_WELCOME);
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        hidden(&mut harness, "a loaded design file");
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), Design::default());
        hidden(&mut harness, "undo");
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), gap_design(2.0));
        hidden(&mut harness, "redo");
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
        hidden(&mut harness, "reset all");
        harness.panel.load_design_file("not a design").unwrap_err();
        hidden(&mut harness, "a refused design file");
        harness
            .panel
            .load_share_payload(&encode_share_payload(&gap_design(1.5)))
            .unwrap();
        hidden(&mut harness, "a share link");
        harness
            .panel
            .open_share_payload(&encode_share_payload(&gap_design(1.6)))
            .unwrap();
        hidden(&mut harness, "a share link opened as the session's start");
        for view in CentreView::ALL {
            harness.panel.centre = view;
            hidden(&mut harness, view.label());
        }
        assert!(!MagcouplingPanel::new().welcome_dismissed, "a new session");
    }

    #[test]
    fn a_share_link_opened_at_start_up_opens_on_the_first_view() {
        // The web app opens a link's design on a new panel (MagcouplingApp::open_share_payload):
        // the Size tab and the welcome strip, as for the default design.
        let mut panel = MagcouplingPanel::new();
        panel
            .open_share_payload(&encode_share_payload(&gap_design(2.0)))
            .unwrap();
        let mut harness = Harness::of(panel, SCREEN);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(harness.panel.design(), gap_design(2.0));
    }

    #[test]
    fn idle_frames_on_the_first_view_change_no_input() {
        // The Size tab's controls (the Required slider among them), like every input row, write
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are.
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.coupling.inner_back_apothem_mm += 0.01234;
        design.inputs.coupling.magnets.axial_length_mm = Some(12.7123);
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        design.inputs.metal.required_min_Nm = 2.2345;
        let mut harness = Harness::first_view();
        harness.panel.open_design(design.clone());
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_focused_input_opens_the_detailed_inputs_on_the_first_view() {
        // On the Size tab the input groups sit under the closed "Detailed inputs": a leaf term
        // or a dashboard fix (spec 3.4) naming an input opens them and frames its row, for a
        // Key design input, an assumption and an input under an Advanced heading alike. A tall
        // screen draws every row.
        for path in [FACE_GAP, "coupling.c_end", "coupling.mu0"] {
            let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(1280.0, 3000.0));
            harness.panel.explorer.focus_input(path);
            let mut output = harness.frame(Vec::new());
            for _ in 0..10 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque, "{path}");
            let label = label_for(path);
            let focus = focus_rects(&output);
            let rows: Vec<egui::Rect> = text_rects(&output, label)
                .into_iter()
                .filter(|r| r.left() < INPUTS_WIDTH)
                .collect();
            assert!(
                rows.iter()
                    .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
                "{path}: {rows:?} {focus:?}"
            );
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
        }
    }

    #[test]
    fn the_welcome_strip_wraps_in_a_narrow_window() {
        // The M4-1 review's ~930 px window leaves the centre region about 300 points: the strip
        // wraps inside it, left of the dashboard, above the tab row.
        let width = 930.0;
        let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(width, 1024.0));
        // The wrapped rows settle on the second frame (egui wraps from the last frame's widths).
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let strip = text_rect(&output, WELCOME).expect("the strip");
        let hide = text_rect(&output, HIDE_WELCOME).expect("its button");
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        assert!(
            strip.left() >= INPUTS_WIDTH && strip.right() <= width - DASHBOARD_WIDTH,
            "{strip:?}"
        );
        assert!(hide.right() <= width - DASHBOARD_WIDTH, "{hide:?}");
        assert!(
            strip.height() >= 2.0 * hide.height(),
            "wrapped: {strip:?} {hide:?}"
        );
        assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
    }

    #[test]
    fn the_welcome_strip_s_texts_are_plain_language() {
        // Spec section 6, test 3: no cell references, correction ids or workbook jargon.
        for text in [WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT] {
            assert_eq!(
                crate::gui::guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text}"
            );
        }
    }

    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
        use crate::gui::plots::{
            HIGH_CASE, POINT_RADIUS, PULL_OUT, PULL_OUT_POINT, PULL_OUT_SMALLEST_APOTHEM,
        };
        use crate::gui::test_support::flat_shapes;
        let mut harness = Harness::new();
        for kind in PlotKind::ALL {
            let legend = match kind {
                PlotKind::SlipHeating => HIGH_CASE,
                PlotKind::TorqueAngle => PULL_OUT_POINT,
                PlotKind::PoleSweep => PULL_OUT_SMALLEST_APOTHEM,
                _ => PULL_OUT,
            };
            let output = harness.click_text(kind.label());
            assert_eq!(harness.panel.centre, CentreView::Plot(kind));
            let output = [output, harness.frame(Vec::new())];
            assert!(
                output
                    .iter()
                    .any(|o| drawn_texts(o).iter().any(|t| t == legend)),
                "{kind:?} draws its legend"
            );
        }
        // The series come from this frame's results: in the edit's frame the design's marker
        // is painted at the new pull-out, where that frame's plot transform puts it.
        harness.click_text(PlotKind::TorqueTemperature.label());
        harness.focus(FACE_GAP);
        let before = harness.panel.results().model.pullout_Nm;
        let output = harness.frame(key_tap(egui::Key::ArrowRight));
        let after = harness.panel.results().model.pullout_Nm;
        assert_ne!(after, before);
        let memory = egui_plot::PlotMemory::load(&harness.ctx, PlotKind::TorqueTemperature.id())
            .expect("the plot ran this frame");
        let op = harness.panel.inputs().coupling.op_temp_C;
        let at = |torque: f64| {
            memory
                .transform()
                .position_from_point(&egui_plot::PlotPoint::new(op, torque))
        };
        assert!((at(after) - at(before)).length() > 0.5, "the edit moves it");
        let markers: Vec<egui::Pos2> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == POINT_RADIUS + 1.0 => Some(c.center),
                _ => None,
            })
            .collect();
        assert!(
            markers.iter().any(|c| (*c - at(after)).length() < 1e-3),
            "{markers:?} vs {:?}",
            at(after)
        );
        assert!(markers.iter().all(|c| (*c - at(before)).length() > 0.5));
    }

    #[test]
    fn the_clamp_tab_draws_the_recommended_clamp_and_follows_the_design() {
        let mut harness = Harness::new();
        harness.click_text(CentreView::Clamp.label());
        assert_eq!(harness.panel.centre, CentreView::Clamp);
        let output = harness.frame(Vec::new());
        let title = "One-piece slotted clamp, \u{d8}10 keyed shaft: ISO 4762 M4 x 14, class 12.9, 5.1 N\u{b7}m, 3 mm key";
        assert_eq!(count(&output, title), 1);
        // A boss too small for any screw: drawing.py's message instead of the drawing.
        harness.panel.inputs.clamps.boss_od_mm = 12.0;
        harness.panel.inputs.clamps.clamp_length_mm = 3.0;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, crate::gui::clamp_drawing::NO_SCREW_FITS), 1);
        assert_eq!(count(&output, title), 0);
    }

    #[test]
    fn a_new_panel_holds_the_default_design_and_its_corrected_results() {
        let panel = MagcouplingPanel::new();
        assert_eq!(panel.inputs(), &DesignInputs::default());
        assert_eq!(panel.results(), &compute_all(&DesignInputs::default()));
        assert_eq!(MagcouplingPanel::default().inputs(), panel.inputs());
    }

    #[test]
    fn every_headline_number_and_key_design_input_is_drawn_with_its_label() {
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        for (_, path) in HEADLINE {
            let label = label_for(path);
            assert!(texts.iter().any(|t| t == label), "missing {label:?}");
        }
        let mut renamed = 0;
        for entry in &InputCatalogue::get().key_design {
            let label = label_for(&entry.path);
            assert!(texts.iter().any(|t| t == label), "missing input {label:?}");
            // A renamed input draws its display label in place of its catalogue label.
            if label != entry.meta.label {
                renamed += 1;
                assert!(
                    !texts.iter().any(|t| t == entry.meta.label),
                    "{} also drew {:?}",
                    entry.path,
                    entry.meta.label
                );
            }
        }
        assert!(renamed > 0, "the Key design group holds renamed inputs");
        // The corrected headline (E2: M4 x 14), not the workbook's (M4 x 12).
        assert!(texts.iter().any(|t| t.contains("M4 x 14")), "{texts:?}");
        // Every Key design row drew its widget.
        assert_eq!(harness.panel.key_widgets.len(), 10);
    }

    #[test]
    fn idle_frames_change_no_input() {
        // Decision M41-2 (`SliderClamping::Edits`): a slider writes only on an edit, so idle
        // frames keep every value as it is, values off their step grid included: a face gap
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, on a screen tall enough to draw every row), in the workbook order and in
        // the workflow order's filtered view (every row, the advanced ones included).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 20000.0));
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        harness.panel.open_design(design.clone());
        // The workbook order: every row is in a group, none under an Advanced heading.
        harness.panel.input_order = InputOrder::Workbook;
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
        for group in InputCatalogue::get().groups.iter().rev() {
            harness.click_text(group.label);
        }
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            count(&output, "Vacuum permeability"),
            2,
            "both mu0 rows drawn"
        );
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(
            harness.panel.history.undo_len(),
            0,
            "no row reported an edit"
        );
        assert!(!harness.panel.history.can_undo(&design));
        // The workflow order, filtered by "." (in every path): all 172 rows, both vacuum
        // permeabilities (under Calibration and model's Advanced heading when unfiltered).
        harness.panel.input_order = InputOrder::Workflow;
        harness.panel.input_filter = ".".to_owned();
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        let total = InputCatalogue::get().all().count();
        assert_eq!(count(&output, &format!("{total} of {total} inputs")), 1);
        assert_eq!(count(&output, "Vacuum permeability"), 2);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn an_input_outside_its_slider_range_is_kept_until_edited_and_flagged() {
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 7.5; // range 0.3 to 5.0
        let mut output = harness.frame(Vec::new());
        for _ in 0..2 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 7.5);
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 7.5;
        assert_eq!(harness.panel.results(), &compute_all(&inputs));
        assert_eq!(count(&output, OUTSIDE_RANGE_NOTE), 1);
        // Decision M41-2: an edit brings it back into the range.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_eq!(count(&harness.frame(Vec::new()), OUTSIDE_RANGE_NOTE), 0);
    }

    #[test]
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::on_screen(TALL);
        harness.focus(FACE_GAP);
        let output = harness.frame(key_tap(egui::Key::ArrowRight));

        // One step (0.01 mm) up from 1.4 mm, rounded to the step's decimals (decision M41-1).
        assert_eq!(harness.number(FACE_GAP), 1.41);
        let mut expected = DesignInputs::default();
        expected.metal.face_gap_mm = 1.41;
        assert_eq!(
            harness.panel.inputs(),
            &expected,
            "only the face gap changed"
        );

        // The same frame shows the recomputed headline, and it differs from the default's.
        assert_eq!(harness.panel.results(), &compute_all(&expected));
        assert_drew_headline(&output, &expected);
        assert_ne!(
            displayed_pullout(&expected),
            displayed_pullout(&DesignInputs::default())
        );
    }

    #[test]
    fn stepping_back_to_the_default_lands_on_it_exactly() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..7 {
            harness.frame(key_tap(egui::Key::ArrowRight));
        }
        assert_eq!(harness.number(FACE_GAP), 1.47);
        for _ in 0..7 {
            harness.frame(key_tap(egui::Key::ArrowLeft));
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
    }

    #[test]
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::on_screen(TALL);
        let rail = harness.widget(FACE_GAP).rect;
        let output = harness.click(rail.right_center() - egui::vec2(1.0, 0.0));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_drew_headline(&output, harness.panel.inputs());
        assert_ne!(
            displayed_pullout(harness.panel.inputs()),
            displayed_pullout(&DesignInputs::default())
        );
    }

    #[test]
    fn a_typed_value_snaps_to_the_step_and_is_clamped_into_the_slider_range() {
        let mut harness = Harness::new();
        // The value box beside the slider shows the value with its unit; a click edits it,
        // and the typed text applies on Enter (decision M41-2).
        let type_in = |harness: &mut Harness, shown: &str, text: &str| {
            harness.click_text(shown);
            harness.frame([select_all(), vec![egui::Event::Text(text.to_owned())]].concat());
            harness.frame(key_tap(egui::Key::Enter));
        };
        type_in(&mut harness, "1.40 mm", "2.344");
        assert_eq!(harness.number(FACE_GAP), 2.34);
        type_in(&mut harness, "2.34 mm", "9");
        assert_eq!(harness.number(FACE_GAP), 5.0);
        type_in(&mut harness, "5.00 mm", "-1");
        assert_eq!(harness.number(FACE_GAP), 0.3);
        // Text that is no number changes nothing (Review Focus 2).
        type_in(&mut harness, "0.30 mm", "wide");
        assert_eq!(harness.number(FACE_GAP), 0.3);
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn the_pole_count_steps_by_two_and_stays_even() {
        let mut harness = Harness::new();
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs.coupling.npole, 12);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs.coupling.npole, 8);
        let rail = harness.widget(POLES).rect;
        harness.click(rail.center());
        let npole = harness.panel.inputs.coupling.npole;
        assert!(npole % 2 == 0 && (4..=40).contains(&npole), "{npole}");
    }

    #[test]
    fn the_slider_range_ends_are_hard_stops_for_arrow_keys() {
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 40;
        harness.panel.inputs.metal.face_gap_mm = 0.3;
        harness.frame(Vec::new());
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs.coupling.npole, 40);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 0.3);
    }

    #[test]
    fn a_changed_input_shows_the_dot_and_its_reset_restores_the_default() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 0);
        assert_eq!(count(&output, RESET_LABEL), 0);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 1);
        assert_eq!(count(&output, RESET_LABEL), 1);
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
    }

    #[test]
    fn the_axial_length_override_starts_blank_and_enters_at_the_ring_length() {
        let mut harness = Harness::new();
        assert_eq!(harness.panel.inputs.coupling.magnets.axial_length_mm, None);
        // The checkbox enters a value: the inner ring's length in use, so nothing moves.
        harness.focus(AXIAL_LENGTH);
        harness.frame(key_tap(egui::Key::Space));
        assert_eq!(
            harness.panel.inputs.coupling.magnets.axial_length_mm,
            Some(12.7)
        );
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
        // Then the slider moves both rings' length, and the torque with it (at the default
        // library part, which the manual lengths never move).
        harness.frame(Vec::new());
        harness.focus(AXIAL_LENGTH);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(
            harness.panel.inputs.coupling.magnets.axial_length_mm,
            Some(12.71)
        );
        assert_ne!(
            displayed_pullout(harness.panel.inputs()),
            displayed_pullout(&DesignInputs::default())
        );
        // Reset leaves it blank again.
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs.coupling.magnets.axial_length_mm, None);
    }

    #[test]
    fn entering_the_measured_drag_starts_at_the_model_s_drag() {
        // Decision M41-12: the measured drag enters at the model's equivalent mean drag torque,
        // unrounded (its step's 0.0131 would move the slip loss again), so the drag in use stays
        // put; the thermal summary leaves the not-measured high estimate for the measured branch.
        let num = |results: &DesignResults, path: &str| match results.get(path) {
            Some(Value::Num(x)) => x,
            other => panic!("{path}: {other:?}"),
        };
        let default = compute_all(&DesignInputs::default());
        let model_drag = num(&default, "temperature.slip_loss.drag_Nm");
        assert_eq!(
            default.get("metal.slip_loss_W"),
            Some(Value::Text("not measured".to_owned()))
        );
        let mut harness = Harness::new();
        harness.focus(MEASURED_DRAG);
        harness.frame(key_tap(egui::Key::Space));
        assert_eq!(
            harness.panel.inputs.metal.measured_drag_Nm,
            Some(model_drag),
            "the seed, not rounded to the slider's step"
        );
        let results = harness.panel.results();
        assert_eq!(num(results, "temperature.slip_loss.drag_Nm"), model_drag);
        let estimate = num(results, "temperature.summary.steady_estimate_C");
        let high = num(results, "temperature.summary.steady_high_C");
        assert_eq!(
            estimate,
            num(&default, "temperature.summary.steady_estimate_C")
        );
        assert_eq!(high, estimate, "measured: the high case is the estimate");
        assert_eq!(format_value(&Value::Num(high)), "74.17");
        let not_measured = num(&default, "temperature.summary.steady_high_C");
        assert_eq!(format_value(&Value::Num(not_measured)), "92.51");
        let loss = num(results, "metal.slip_loss_W");
        assert_eq!(format_value(&Value::Num(loss)), "2.751");
    }

    #[test]
    fn a_selector_switches_the_branch() {
        let mut harness = Harness::new();
        assert_eq!(
            harness.panel.results().get("model.cup_ring_check"),
            Some(Value::Text("Too thin".to_owned()))
        );
        harness.click_text("steel circuit");
        harness.click_text("no back iron");
        assert_eq!(harness.panel.inputs.coupling.backiron, 0);
        assert_eq!(
            harness.panel.results().get("model.cup_ring_check"),
            Some(Value::Text("No back iron".to_owned()))
        );
        let mut expected = DesignInputs::default();
        expected.coupling.backiron = 0;
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn a_text_input_edits_the_part_name_and_says_what_it_resolves_to() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Library part"),
            2,
            "both parts are library parts"
        );
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        let part = harness.panel.inputs.coupling.magnets.part_inner.clone();
        assert!(part == "B842SHX" || part == "XB842SH", "{part}");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Library part"), 1);
        assert_eq!(
            count(
                &output,
                "Not a library part: the manual dimensions are used"
            ),
            1
        );
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.part_inner = part;
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for order in InputOrder::ALL {
            for group in catalogue.groups_in(order) {
                // Tall enough for the longest group (temperature) to fit without scrolling.
                let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
                harness.panel.input_order = order;
                harness.click_text(group.label);
                // The header opens over a few frames (its animation).
                let mut output = harness.frame(Vec::new());
                for _ in 0..10 {
                    output = harness.frame(Vec::new());
                }
                if group.sections.iter().any(|s| s.advanced) {
                    // The advanced sections stay closed until their heading is clicked.
                    let texts = drawn_texts(&output);
                    for section in group.sections.iter().filter(|s| s.advanced) {
                        assert!(
                            !texts.iter().any(|t| t == section.label),
                            "{}: {:?} open by default",
                            group.name,
                            section.label
                        );
                    }
                    harness.click_text(ADVANCED_HEADING);
                    for _ in 0..10 {
                        output = harness.frame(Vec::new());
                    }
                }
                let texts = drawn_texts(&output);
                for section in &group.sections {
                    if section.id != group.name {
                        assert!(
                            texts.iter().any(|t| t == section.label),
                            "{}: no heading {:?}",
                            group.name,
                            section.label
                        );
                    }
                    for entry in &section.entries {
                        assert!(
                            texts.iter().any(|t| t == label_for(&entry.path)),
                            "{}: missing {:?}",
                            group.name,
                            label_for(&entry.path)
                        );
                    }
                }
                assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            }
        }
    }

    #[test]
    fn the_order_toggle_switches_the_view_and_changes_no_input() {
        // Decision O-1: the workflow order by default, the workbook's package groups one click
        // away; a view of this session only, so the design, its share link and the undo history
        // stay as they are.
        let mut harness = Harness::new();
        let link = harness.panel.share_link();
        let workflow: Vec<&str> = InputCatalogue::get()
            .workflow
            .iter()
            .map(|g| g.label)
            .collect();
        let workbook = [
            "Coupling",
            "Metal design",
            "Temperature design",
            "Shaft clamps",
        ];
        let drawn = |output: &egui::FullOutput, labels: &[&str]| {
            labels.iter().all(|label| count(output, label) == 1)
        };
        let none = |output: &egui::FullOutput, labels: &[&str]| {
            labels.iter().all(|label| count(output, label) == 0)
        };
        assert_eq!(harness.panel.input_order, InputOrder::Workflow);
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workflow) && none(&output, &workbook));
        harness.click_text(InputOrder::Workbook.label());
        assert_eq!(harness.panel.input_order, InputOrder::Workbook);
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workbook) && none(&output, &workflow[..4]));
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.share_link(), link);
        assert!(!harness.panel.history.can_undo(&Design::default()));
        harness.click_text(InputOrder::Workflow.label());
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workflow) && none(&output, &workbook));
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design() {
        let mut harness = Harness::new();
        let total = InputCatalogue::get().all().count();
        harness.click_text(FILTER_HINT);
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        assert_eq!(
            count(
                &output,
                "Requirements and operating conditions / Drive and gearbox"
            ),
            1
        );
        for label in [
            "Gearbox ratio",
            "Gearbox efficiency",
            "Gearbox input torque rating",
        ] {
            assert_eq!(count(&output, label), 1, "{label}");
        }
        // The Key design group and the other groups give way to the matches.
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        assert_eq!(count(&output, "Magnets and rings"), 0);
        // A matched row is a row like any other: an arrow key nudges its value.
        let ratio = InputCatalogue::get().entry("coupling.gear_ratio").unwrap();
        let before = harness.number("coupling.gear_ratio");
        let slider = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(slider));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        let step = ratio.meta.range.unwrap().step;
        assert!((harness.number("coupling.gear_ratio") - (before + step)).abs() < 1e-9);
        // Typing in the filter is no edit of the design: the nudge is the one undo step.
        harness.frame(Vec::new());
        assert_eq!(harness.panel.history.undo_len(), 1);
        // The workbook order heads the run by its package group.
        harness.click_text(InputOrder::Workbook.label());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Coupling"), 1);
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        // No match says so; a blank filter brings the groups back.
        harness.panel.input_filter = "no such input".to_owned();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} inputs")), 1);
        harness.panel.input_filter = "   ".to_owned();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 1);
    }

    #[test]
    fn a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter() {
        // The vacuum permeability is no assumption and sits under Calibration and model's
        // Advanced heading: a leaf term naming it opens both, past a filter it does not match.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.panel.input_filter = "gearbox".to_owned();
        harness.frame(Vec::new());
        harness.panel.explorer.focus_input("coupling.mu0");
        let mut output = harness.frame(Vec::new());
        for _ in 0..10 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.input_filter, "");
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
        let label = InputCatalogue::get()
            .entry("coupling.mu0")
            .unwrap()
            .meta
            .label;
        let rows: Vec<egui::Rect> = text_rects(&output, label)
            .into_iter()
            .filter(|r| r.left() < INPUTS_WIDTH)
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "both mu0 rows, under the open Advanced heading"
        );
        let focus = focus_rects(&output);
        assert!(
            rows.iter()
                .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
            "{rows:?} {focus:?}"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers() {
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        // The pull-out's marker: E3 changes it at the defaults, E7 and E8 have probes on it.
        assert!(count(&output, "E3 E7 E8") >= 1);
        assert_eq!(count(&output, "Inside the space claim"), 1);
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(END_EFFECT_BANNER))
        );
    }

    #[test]
    fn an_out_of_range_end_effect_shows_the_banner() {
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        let output = harness.frame(Vec::new());
        let banner: Vec<String> = drawn_texts(&output)
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        // Over the dashboard and over the centre region, whichever view it shows (decision
        // M42-1).
        let text = "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text, text]);
    }

    #[test]
    fn a_design_past_the_space_claim_shows_its_overshoot() {
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let output = harness.frame(Vec::new());
        let claim = drawn_texts(&output)
            .into_iter()
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the badge text");
        assert!(claim.contains("overall length"), "{claim}");
    }

    #[test]
    fn the_results_table_lists_results_and_filters_by_the_search() {
        let entries = crate::gui::results_table::table_entries();
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.click_text(CentreView::Results.label());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The headline's rows are on screen (the pull-out shows its cell); the first and the
        // last rows of the schema are in closed groups.
        assert_eq!(count(&output, "Calculator!C93"), 1);
        assert_eq!(count(&output, &first), 0, "{first}");
        assert_eq!(count(&output, &last), 0, "{last}");
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text(last.to_lowercase())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("1 of {total} results")), 1);
        assert_eq!(count(&output, &last), 1);
        assert_eq!(count(&output, &first), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings() {
        use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
        use crate::gui::results_table::ResultOrder;
        let groups = result_groups();
        let heading = |g: usize| format!("{} ({})", groups[g].label, groups[g].rows.len());
        // Tall enough to draw the headline and the whole torque chain.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        for g in 0..groups.len() {
            assert_eq!(count(&output, &heading(g)), 1, "{}", heading(g));
        }
        assert_eq!(count(&output, OTHER_RESULTS), 1);
        // A heading shows the worst level of its group's checks after its text: the defaults
        // fail the hot minimum on the headline and the cup ring check in the closed coupling
        // model group, so both draw a red badge; the mass has no check and draws none.
        let visuals = harness.ctx.style().visuals.clone();
        let badges_after = |output: &egui::FullOutput, text: &str, level: Level| {
            let text = text_rect(output, text).unwrap();
            crate::gui::test_support::flat_shapes(output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(&visuals)
                            && c.center.x > text.right()
                            && c.center.x < text.right() + 30.0
                            && (c.center.y - text.center().y).abs() < text.height())
                })
                .count()
        };
        let model = groups
            .iter()
            .position(|g| g.other && g.id == "model")
            .unwrap();
        let mass = groups
            .iter()
            .position(|g| !g.other && g.id == "mass")
            .unwrap();
        assert_eq!(badges_after(&output, &heading(0), Level::Bad), 1);
        assert_eq!(badges_after(&output, &heading(model), Level::Bad), 1);
        for level in [Level::Good, Level::Caution, Level::Bad] {
            assert_eq!(badges_after(&output, &heading(mass), level), 0);
        }
        // The torque chain starts closed: its end-effect factor (Calculator!C92) is not drawn
        // until its heading is clicked, and is gone again after a second click.
        assert_eq!(groups[1].id, "torque");
        assert_eq!(count(&output, "Calculator!C92"), 0);
        harness.click_text(&heading(1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1);
        harness.click_text(&heading(1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 0);
        // The engine order: the schema's first row first, no headings.
        harness.click_text(ResultOrder::Engine.label());
        let output = harness.frame(Vec::new());
        let first = crate::gui::results_table::table_entries()[0]
            .info
            .cell
            .clone()
            .unwrap();
        assert_eq!(count(&output, &first), 1);
        assert_eq!(count(&output, &heading(0)), 0);
        assert_eq!(count(&output, OTHER_RESULTS), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_heading_click_while_searching_leaves_the_group_as_it_was() {
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::{CLEAR_TO_CLOSE, SEARCH_HINT, search, table_entries};
        use crate::gui::test_support::select_all;
        // A search opens every group it matches, so a click on a heading would change nothing
        // on screen: it is ignored, and once the search is cleared the torque chain is closed,
        // as it started (the click did not toggle it behind the search).
        let entries = table_entries();
        let torque = &result_groups()[1];
        assert_eq!(torque.id, "torque");
        let matches = search(entries, "f_end");
        let shown = torque.rows.iter().filter(|i| matches.contains(i)).count();
        let heading = format!("{} ({shown})", torque.label);
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        tooltips_at_once(&harness);
        harness.click_text(CentreView::Results.label());
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("f_end".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1, "the search opens it");
        let output = harness.click_text(&heading);
        let at = text_rect(&output, &heading).unwrap().center();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1, "still open");
        // Its hover text says why (egui shows no tooltip until the pointer moves after a click).
        harness.frame_after(
            0.5,
            vec![egui::Event::PointerMoved(at + egui::vec2(2.0, 0.0))],
        );
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CLEAR_TO_CLOSE), 1);
        // Clear the search: the torque chain is closed again.
        harness.click_text("f_end");
        harness.frame(select_all());
        harness.frame(key_tap(egui::Key::Backspace));
        let output = harness.frame(Vec::new());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert_eq!(count(&output, &format!("{} (49)", torque.label)), 1);
        assert_eq!(count(&output, "Calculator!C92"), 0);
    }

    #[test]
    fn the_failing_filter_shows_exactly_the_failing_checks_with_their_badges() {
        use crate::gui::dashboard::{CHECKS, failing_checks};
        use crate::gui::results_table::{
            FAILING_ONLY, NO_RESULT, NOTHING_FAILS, SEARCH_HINT, table_entries,
        };
        // Manual magnets: the defaults' four red checks and two amber rating checks.
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.magnets.part_inner.clear();
        harness.panel.inputs.coupling.magnets.part_outer.clear();
        harness.click_text(CentreView::Results.label());
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results());
        assert_eq!(failing.len(), 6);
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("6 of {total} results")), 1);
        // In the table, between the inputs and the dashboard: each failing check's label, and
        // no passing check's (two checks may share a label: "Verdict").
        let centre =
            |r: &egui::Rect| r.left() > INPUTS_WIDTH && r.right() < SCREEN.x - DASHBOARD_WIDTH;
        let label = |path: &str| result_info(path).unwrap().meta.label;
        for check in CHECKS {
            let drawn = text_rects(&output, label(check))
                .iter()
                .filter(|r| centre(r))
                .count();
            let want = failing
                .iter()
                .filter(|(path, _)| label(path) == label(check))
                .count();
            assert_eq!(drawn, want, "{check}");
        }
        // Four red badges and two amber in the table, in the dashboard's colours: every badge
        // drawn but the dashboard's own.
        let visuals = harness.ctx.style().visuals.clone();
        let lines = crate::gui::dashboard::dashboard_lines(harness.panel.results());
        let table_badges = |level: Level| {
            let drawn = crate::gui::test_support::flat_shapes(&output)
                .into_iter()
                .filter(|shape| matches!(shape, egui::Shape::Circle(c) if c.fill == level.color(&visuals)))
                .count();
            drawn
                - lines
                    .iter()
                    .filter(|line| line.level == Some(level))
                    .count()
        };
        assert_eq!(
            (table_badges(Level::Bad), table_badges(Level::Caution)),
            (4, 2)
        );
        // Off again: the groups come back.
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert_eq!(count(&output, crate::gui::result_groups::OTHER_RESULTS), 1);
        // On, with a search no failing check matches: the table says the search matches
        // nothing, not that no check fails.
        harness.click_text(FAILING_ONLY);
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NO_RESULT), 1);
        assert_eq!(count(&output, NOTHING_FAILS), 0);
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
        use crate::gui::dashboard::{CHECKS, dashboard_lines, failing_checks};
        use crate::gui::results_table::{FAILING_ONLY, table_entries};
        use crate::gui::test_support::flat_shapes;
        // Short magnets (f_end < 0): the dashboard greys the rows computed from the pull-out,
        // without badges. In the table's headline (open) each check row carries exactly the
        // dashboard's badge, so the hot minimum (red by its text) and the screw sized from the
        // invalid pull-out (green by its text) carry none, and the failing filter leaves the hot
        // minimum out.
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        let visuals = harness.ctx.style().visuals.clone();
        // Between the inputs and the dashboard's badges (left of its first label).
        let pullout = label_for("model.pullout_Nm");
        let dashboard_left = text_rects(&output, pullout)
            .into_iter()
            .map(|r| r.left())
            .fold(f32::MIN, f32::max)
            - 30.0;
        let centre = |x: f32| x > INPUTS_WIDTH && x < dashboard_left;
        let badges_on = |output: &egui::FullOutput, row: egui::Rect| -> Vec<Level> {
            let shapes = flat_shapes(output);
            [Level::Good, Level::Caution, Level::Bad]
                .into_iter()
                .filter(|level| {
                    shapes.iter().any(|shape| {
                        matches!(shape, egui::Shape::Circle(c)
                            if c.fill == level.color(&visuals)
                                && centre(c.center.x)
                                && (c.center.y - row.center().y).abs() < row.height())
                    })
                })
                .collect()
        };
        let lines = dashboard_lines(harness.panel.results());
        let checks: Vec<_> = lines
            .iter()
            .filter(|line| CHECKS.contains(&line.path))
            .collect();
        assert_eq!(checks.len(), 7);
        for line in checks {
            let row = text_rects(&output, line.label)
                .into_iter()
                .find(|r| centre(r.center().x))
                .unwrap_or_else(|| panic!("{} in the table", line.path));
            let want: Vec<Level> = line.level.into_iter().collect();
            assert_eq!(badges_on(&output, row), want, "{}", line.path);
        }
        // The failing filter: the hot minimum is not listed; the end-effect check is, first.
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results());
        assert!(
            failing
                .iter()
                .all(|(path, _)| *path != "metal.hot_min_check")
        );
        assert_eq!(failing[0], ("model.end_effect_check", Level::Bad));
        let total = table_entries().len();
        let shown = format!("{} of {total} results", failing.len());
        assert_eq!(count(&output, &shown), 1);
        let drawn = |path: &str| {
            text_rects(&output, result_info(path).unwrap().meta.label)
                .into_iter()
                .filter(|r| centre(r.center().x))
                .count()
        };
        assert_eq!(drawn("metal.hot_min_check"), 0);
        assert_eq!(drawn("model.end_effect_check"), 1);
        // Seven red badges and four amber in the table.
        let in_table = |level: Level| {
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(&visuals) && centre(c.center.x))
                })
                .count()
        };
        assert_eq!((in_table(Level::Bad), in_table(Level::Caution)), (7, 4));
    }

    /// The mark rects of the trace (the selection colour) painted in a frame.
    fn trace_marks(harness: &Harness, output: &egui::FullOutput) -> Vec<egui::Rect> {
        mark_rects(
            output,
            Some(harness.ctx.style().visuals.selection.stroke.color),
        )
    }

    #[test]
    fn clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::trace::TraceKind;
        let mut harness = Harness::new();
        let gap = label_for(FACE_GAP);
        harness.click_text(gap);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, FACE_GAP)
        );
        assert_eq!(trace.paths, registry().downstream(FACE_GAP));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        let marks = trace_marks(&harness, &output);
        // On the dashboard (right of its first label's badge): one frame per row the gap
        // drives, none on the others.
        let pullout_label = label_for("model.pullout_Nm");
        let dashboard_left = text_rect(&output, pullout_label).unwrap().left() - 30.0;
        let on_dashboard = marks
            .iter()
            .filter(|m| m.center().x > dashboard_left)
            .count();
        let driven = DASHBOARD
            .iter()
            .filter(|(path, _)| trace.marks(path))
            .count();
        assert!(driven > 0);
        assert_eq!(on_dashboard, driven);
        // On the inputs side: the gap's own Key design row alone.
        let row = text_rects(&output, gap)
            .into_iter()
            .find(|r| r.left() < INPUTS_WIDTH)
            .unwrap();
        let on_inputs: Vec<&egui::Rect> = marks
            .iter()
            .filter(|m| m.center().x < INPUTS_WIDTH)
            .collect();
        assert_eq!(on_inputs.len(), 1);
        assert!(on_inputs[0].contains_rect(row));
        // A second click on the label ends the trace and its frames.
        harness.click_text(gap);
        assert_eq!(harness.panel.trace, None);
        let output = harness.frame(Vec::new());
        assert!(trace_marks(&harness, &output).is_empty());
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group() {
        use crate::gui::trace::{CLEAR_TRACE, TraceKind};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Result, "model.pullout_Nm")
        );
        assert_eq!(
            &trace.paths,
            registry().upstream_inputs("model.pullout_Nm").unwrap()
        );
        assert!(harness.panel.explorer.open, "the click still opens it");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        // Each Key design row is framed exactly when the pull-out reads its input.
        let marks = trace_marks(&harness, &output);
        for entry in &InputCatalogue::get().key_design {
            let row = text_rects(&output, label_for(&entry.path))
                .into_iter()
                .find(|r| r.left() < INPUTS_WIDTH)
                .unwrap();
            let framed = marks.iter().any(|m| m.contains_rect(row));
            assert_eq!(framed, trace.marks(&entry.path), "{}", entry.path);
        }
        // Each closed group says how many of its rows the trace frames.
        for group in &InputCatalogue::get().workflow {
            let traced = group
                .sections
                .iter()
                .flat_map(|s| s.entries.iter())
                .filter(|e| trace.marks(&e.path))
                .count();
            let title = if traced > 0 {
                format!("{} ({traced} traced)", group.label)
            } else {
                group.label.to_owned()
            };
            assert_eq!(count(&output, &title), 1, "{title}");
        }
        // Clear ends it.
        harness.click_text(CLEAR_TRACE);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_traced_filter_shows_the_results_the_trace_marks() {
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::{TRACED_ONLY, table_entries};
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        // Without a trace the filter cannot be ticked.
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // Trace the face gap and show only the results it drives.
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        let trace = harness.panel.trace.clone().unwrap();
        let traced = entries.iter().filter(|e| trace.marks(&e.path)).count();
        assert!(traced > 0 && traced < total);
        assert_eq!(count(&output, &format!("{traced} of {total} results")), 1);
        // Each heading counts the rows the trace marks in it.
        let headline = result_groups()[0]
            .rows
            .iter()
            .filter(|&&i| trace.marks(&entries[i].path))
            .count();
        assert_eq!(
            count(
                &output,
                &format!("Headline ({headline}, {headline} traced)")
            ),
            1
        );
        // Without the trace the filter lets every row through again.
        harness.panel.trace = None;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
    }

    #[test]
    fn an_input_whose_results_have_no_equation_record_traces_none_and_says_so() {
        use crate::gui::results_table::{NO_RESULT, NOTHING_TRACED, TRACED_ONLY, table_entries};
        // The drive torque sets the required floor and the verdict, which have no equation
        // record: the trace reaches no result, the banner says why, and "Traced only" says the
        // trace marks nothing rather than that the search matches nothing.
        let total = table_entries().len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(InputCatalogue::get().workflow[0].label);
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let drive = "coupling.drive_torque_Nm";
        harness.click_text(InputCatalogue::get().entry(drive).unwrap().meta.label);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(trace.source, drive);
        assert!(trace.paths.is_empty());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        assert!(trace.banner().contains("no explained result reads it"));
        harness.click_text(CentreView::Results.label());
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NOTHING_TRACED), 1);
        assert_eq!(count(&output, NO_RESULT), 0);
        // Clear trace turns the filter off with it: every row again, and a later trace does not
        // filter the table until the filter is ticked again.
        harness.click_text(crate::gui::trace::CLEAR_TRACE);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.trace, None);
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        harness.click_text(label_for(FACE_GAP));
        let output = harness.frame(Vec::new());
        assert!(harness.panel.trace.is_some());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
    }

    #[test]
    fn a_trace_filter_left_ticked_off_the_table_does_not_filter_a_later_trace() {
        use crate::gui::results_table::{TRACED_ONLY, table_entries};
        use crate::gui::trace::CLEAR_TRACE;
        // "Traced only" goes off with the trace even when the table is not on screen: Clear
        // trace and a click on another input's label, both with the Geometry view showing, must
        // not leave the box ticked for the new trace (it would filter the table unasked).
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
        let gap_rows = entries.iter().filter(|e| gap.marks(&e.path)).count();
        let output = harness.frame(Vec::new());
        assert!(gap_rows > 0 && gap_rows < total);
        assert_eq!(count(&output, &format!("{gap_rows} of {total} results")), 1);
        // Off the table: end the trace and trace another input.
        harness.click_text(CentreView::Geometry.label());
        harness.click_text(CLEAR_TRACE);
        assert_eq!(harness.panel.trace, None);
        harness.click_text(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
        assert_eq!(poles.source, POLES);
        let pole_rows = entries.iter().filter(|e| poles.marks(&e.path)).count();
        assert!(pole_rows > 0 && pole_rows < total);
        // Back on the table: every row, the box unticked.
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert_eq!(
            count(&output, &format!("{pole_rows} of {total} results")),
            0
        );
        // Ticking it again filters by the new trace, as the checkbox always did.
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, &format!("{pole_rows} of {total} results")),
            1
        );
    }

    #[test]
    fn a_result_traced_off_the_table_also_turns_the_trace_filter_off() {
        use crate::gui::results_table::{TRACED_ONLY, table_entries};
        use crate::gui::trace::TraceKind;
        // A result clicked on the dashboard replaces an input's trace with its own (which marks
        // inputs, and cannot filter the table): the box goes off with it, so an input traced
        // next, with no Clear trace between, does not find it ticked.
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
        harness.click_text(CentreView::Geometry.label());
        let outside = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .find(|line| registry().equation_for(line.path).is_some() && !gap.marks(line.path))
            .expect("a dashboard result the face gap does not drive");
        let output = harness.frame(Vec::new());
        // The rightmost: the dashboard's.
        let value = text_rects(&output, &outside.value)
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .unwrap();
        harness.click(value.center());
        let replaced = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (replaced.kind, replaced.source.as_str()),
            (TraceKind::Result, outside.path)
        );
        harness.click_text(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (poles.kind, poles.source.as_str()),
            (TraceKind::Input, POLES)
        );
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        let pole_rows = entries.iter().filter(|e| poles.marks(&e.path)).count();
        assert!(pole_rows > 0 && pole_rows < total);
        assert_eq!(
            count(&output, &format!("{pole_rows} of {total} results")),
            0
        );
    }

    #[test]
    fn reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows() {
        use crate::gui::results_table::{ResultOrder, TRACED_ONLY, table_entries};
        use crate::gui::trace::TraceKind;
        // An input traced and "Traced only" ticked: a click on a traced row (the table's main
        // gesture, to read its equation) opens its equation and keeps the input's trace, so the
        // table keeps its rows; a click on a result without an equation record keeps it too.
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let trace = harness.panel.trace.clone().unwrap();
        let traced = entries.iter().filter(|e| trace.marks(&e.path)).count();
        let shown = format!("{traced} of {total} results");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &shown), 1);
        // The pull-out's row in the table (between the inputs and the dashboard).
        let centre =
            |r: &egui::Rect| r.left() > INPUTS_WIDTH && r.right() < 1280.0 - DASHBOARD_WIDTH;
        let pullout = result_info("model.pullout_Nm").unwrap().meta.label;
        let row = text_rects(&output, pullout)
            .into_iter()
            .find(centre)
            .expect("the pull-out's row");
        assert!(trace.marks("model.pullout_Nm"));
        harness.click(row.center());
        let output = harness.frame(Vec::new());
        assert!(harness.panel.explorer.open, "the click opens its equation");
        assert_eq!(
            harness.panel.trace.as_ref(),
            Some(&trace),
            "the input's trace stays"
        );
        assert_eq!(count(&output, &shown), 1, "the table keeps its rows");
        // Every row again, in the engine's order: a result without an equation record keeps
        // the trace (its inputs are unknown).
        harness.click_text(TRACED_ONLY);
        harness.click_text(ResultOrder::Engine.label());
        let output = harness.frame(Vec::new());
        let recordless = entries
            .iter()
            .find(|e| registry().equation_for(&e.path).is_none())
            .unwrap();
        let row = text_rects(&output, recordless.info.meta.label)
            .into_iter()
            .find(centre)
            .unwrap_or_else(|| panic!("{} on screen", recordless.path));
        harness.click(row.center());
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        // A result outside the trace, with a record (on the dashboard): its trace replaces the
        // input's.
        let outside = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .find(|line| registry().equation_for(line.path).is_some() && !trace.marks(line.path))
            .expect("a dashboard result the face gap does not drive");
        let output = harness.frame(Vec::new());
        // The rightmost: the dashboard's.
        let value = text_rects(&output, &outside.value)
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .unwrap();
        harness.click(value.center());
        let replaced = harness.panel.trace.clone().unwrap();
        assert_eq!(
            (replaced.kind, replaced.source.as_str()),
            (TraceKind::Result, outside.path)
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_search_that_hides_the_traced_rows_says_so_and_the_headings_count_what_it_shows() {
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::{
            NO_RESULT, NOTHING_TRACED, SEARCH_HINT, TRACED_ONLY, search, table_entries,
        };
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        let trace = harness.panel.trace.clone().expect("a trace");
        // A search for one row the trace marks: the headline's heading counts the traced rows
        // it shows, not every row of the group the trace marks.
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("calculator!c93".to_owned())]);
        let output = harness.frame(Vec::new());
        let headline = result_groups()[0]
            .rows
            .iter()
            .filter(|&&i| trace.marks(&entries[i].path))
            .count();
        assert!(headline > 1);
        assert_eq!(count(&output, "Headline (1, 1 traced)"), 1);
        assert_eq!(
            count(&output, &format!("Headline (1, {headline} traced)")),
            0
        );
        // "Traced only" with a search that matches results the trace does not mark: the search
        // hides the traced rows, so the table says the search matches nothing, not that the
        // trace marks nothing (its 146 rows are there without the search).
        harness.click_text(TRACED_ONLY);
        harness.click_text("calculator!c93");
        harness.frame(select_all());
        harness.frame(key_tap(egui::Key::Backspace));
        harness.frame(vec![egui::Event::Text("bondline".to_owned())]);
        let output = harness.frame(Vec::new());
        let bondline = search(entries, "bondline");
        assert!(!bondline.is_empty());
        assert!(bondline.iter().all(|&i| !trace.marks(&entries[i].path)));
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NO_RESULT), 1);
        assert_eq!(count(&output, NOTHING_TRACED), 0);
    }

    #[test]
    fn a_trace_moves_no_input_row_so_a_second_click_on_the_label_ends_it() {
        use crate::gui::trace::CLEAR_TRACE;
        // The banner sits at the foot of the inputs side: a trace moves no row above it, so the
        // second click of "a second click on the label ends the trace" lands on the label again,
        // not on whatever a banner on top would have pushed under the pointer.
        let mut harness = Harness::new();
        let gap = label_for(FACE_GAP);
        let on_inputs_side = |output: &egui::FullOutput, text: &str| {
            text_rects(output, text)
                .into_iter()
                .find(|r| r.left() < INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no {text:?} on the inputs side"))
        };
        let toggle = InputOrder::Workbook.label();
        let output = harness.frame(Vec::new());
        let (label, toggle_at) = (
            on_inputs_side(&output, gap),
            on_inputs_side(&output, toggle),
        );
        harness.click(label.center());
        let trace = harness.panel.trace.clone().expect("a trace");
        // The strip sizes from the last frame: let it settle.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        assert_eq!(on_inputs_side(&output, gap), label);
        assert_eq!(on_inputs_side(&output, toggle), toggle_at);
        assert!(on_inputs_side(&output, CLEAR_TRACE).top() > label.bottom());
        // The second click, at the same point.
        harness.click(label.center());
        assert_eq!(harness.panel.trace, None);
        let output = harness.frame(Vec::new());
        assert_eq!(on_inputs_side(&output, gap), label);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_locked_free_variable_s_label_still_traces_it() {
        use crate::gui::trace::TraceKind;
        // In Torque -> Magnets the free variable's row is locked (its value widgets disabled);
        // its label still traces the input, as every row's does.
        let mut harness = Harness::new();
        harness.size();
        assert!(!harness.widget(AXIAL_LENGTH).enabled(), "locked");
        let (design, undo) = (harness.panel.design(), harness.panel.history.undo_len());
        let label = InputCatalogue::get()
            .entry(AXIAL_LENGTH)
            .unwrap()
            .meta
            .label;
        harness.click_text(label);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, AXIAL_LENGTH)
        );
        // A second click ends it; neither click edits the design or makes an undo step.
        harness.click_text(label);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.history.undo_len(), undo);
        assert!(!harness.widget(AXIAL_LENGTH).enabled(), "still locked");
    }

    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
        // The M4-1 review's ~930 px window: the inputs (320) and the dashboard (300) leave the
        // table about 294 points. The row's value must end left of the dashboard.
        let mut harness = Harness::on_screen(egui::vec2(930.0, 1024.0));
        // The wrapped tab row settles on the second frame (egui wraps from the last frame's
        // widths).
        harness.frame(Vec::new());
        harness.click_text(CentreView::Results.label());
        assert_eq!(harness.panel.centre, CentreView::Results);
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("calculator!c93".to_owned())]);
        let output = harness.frame(Vec::new());
        let value = displayed_pullout(&DesignInputs::default());
        let in_table: Vec<egui::Rect> = text_rects(&output, &value)
            .into_iter()
            .filter(|r| r.right() <= 930.0 - 300.0)
            .collect();
        assert_eq!(in_table.len(), 1, "the row's value is on screen: {value}");
        // And the row's label (truncated to its column, but there).
        let label = crate::gui::results_table::table_entries()
            .iter()
            .find(|e| e.info.cell.as_deref() == Some("Calculator!C93"))
            .expect("the pull-out's row")
            .info
            .meta
            .label;
        let labels: Vec<egui::Rect> = text_rects(&output, label)
            .into_iter()
            .filter(|r| r.left() >= 0.0 && r.right() <= 930.0 - 300.0)
            .collect();
        assert_eq!(labels.len(), 1, "the row's label is on screen: {label}");
        // It fills its column, at least 120 points, and ends left of the value.
        let min = crate::gui::results_table::LABEL_MIN_WIDTH;
        assert!(labels[0].width() >= min - 1.0, "{:?}", labels[0]);
        assert!(labels[0].right() <= in_table[0].left());
    }

    #[test]
    fn the_export_buttons_queue_the_files_for_the_host() {
        use crate::gui::results_table::{EXPORT_CSV, EXPORT_JSON};
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.click_text(CentreView::Results.label());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(EXPORT_CSV);
        harness.click_text(EXPORT_JSON);
        let results = harness.panel.results().clone();
        let requests = harness.panel.take_requests();
        assert_eq!(
            requests,
            vec![
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.csv".to_owned(),
                    mime: "text/csv",
                    contents: results_csv(&results).into_bytes(),
                },
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.json".to_owned(),
                    mime: "application/json",
                    contents: results_json(&harness.panel.design(), &results).into_bytes(),
                },
            ]
        );
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    /// The spreadsheets the panel queued since the last call, each read back sheet by sheet;
    /// every request must be a spreadsheet's.
    fn queued_spreadsheets(harness: &mut Harness) -> Vec<Vec<(String, Vec<Vec<calamine::Data>>)>> {
        use crate::gui::test_support::read_xlsx;
        use crate::gui::xlsx::{XLSX_FILE_NAME, XLSX_MIME};
        harness
            .panel
            .take_requests()
            .into_iter()
            .map(|request| match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    assert_eq!(file_name, XLSX_FILE_NAME);
                    assert_eq!(mime, XLSX_MIME);
                    read_xlsx(&contents)
                }
                PanelRequest::OpenDesign => panic!("not a spreadsheet"),
            })
            .collect()
    }

    /// The row of a read-back sheet whose column `column` holds the text `wanted`.
    fn read_row<'r>(
        rows: &'r [Vec<calamine::Data>],
        column: usize,
        wanted: &str,
    ) -> &'r Vec<calamine::Data> {
        rows.iter()
            .find(|row| row.get(column) == Some(&calamine::Data::String(wanted.to_owned())))
            .unwrap_or_else(|| panic!("no row {wanted}"))
    }

    #[test]
    fn both_spreadsheet_buttons_queue_the_workbook_of_the_design_shown() {
        use crate::gui::results_table::EXPORT_XLSX;
        use crate::gui::spreadsheet::{
            ASSUMPTIONS_SHEET, INPUT_COLUMNS, INPUTS_SHEET, RESULTS_SHEET, SHARE_LINK,
            SUMMARY_SHEET,
        };
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The header's button, then the results table's.
        harness.click_text(EXPORT_SPREADSHEET);
        harness.click_text(CentreView::Results.label());
        harness.click_text(EXPORT_XLSX);
        let books = queued_spreadsheets(&mut harness);
        assert_eq!(books.len(), 2);
        let path = INPUT_COLUMNS.iter().position(|c| *c == "Path").unwrap();
        for book in &books {
            let names: Vec<&str> = book.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(
                names,
                [
                    SUMMARY_SHEET,
                    INPUTS_SHEET,
                    RESULTS_SHEET,
                    ASSUMPTIONS_SHEET
                ]
            );
            // The face gap as edited, a number; the link that reopens this design.
            let gap = read_row(&book[1].1, path, FACE_GAP);
            assert_eq!(gap[1], calamine::Data::Float(1.41));
            let link = read_row(&book[0].1, 0, SHARE_LINK);
            assert_eq!(link[1], calamine::Data::String(harness.panel.share_link()));
        }
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    #[test]
    fn the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets() {
        use crate::gui::spreadsheet::{INPUT_COLUMNS, SIZING_MODE, SIZING_OUTCOME};
        let column = |name: &str| INPUT_COLUMNS.iter().position(|c| *c == name).unwrap();
        let mut harness = Harness::new();
        // The workbook order: the Inputs sheet's first heading is the first package group's.
        harness.click_text(InputOrder::Workbook.label());
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
        let first_group = InputCatalogue::get().groups[0].label;
        assert_eq!(
            books[0][1].1[1][0],
            calamine::Data::String(first_group.to_owned())
        );
        // Torque -> Magnets: the solved length, noted, and the sizing on the Summary.
        harness.size();
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
        let (summary, inputs) = (&books[0][0].1, &books[0][1].1);
        assert_eq!(
            read_row(summary, 0, SIZING_MODE)[1],
            calamine::Data::String(SizingMode::TorqueToMagnets.label().to_owned())
        );
        let outcome = read_row(summary, 0, SIZING_OUTCOME);
        assert!(
            matches!(&outcome[1], calamine::Data::String(text) if text.starts_with(SOLVED_PREFIX)),
            "{:?}",
            outcome[1]
        );
        let length = read_row(inputs, column("Path"), AXIAL_LENGTH);
        let shown = harness
            .panel
            .shown_inputs()
            .coupling
            .magnets
            .axial_length_mm;
        assert_eq!(length[1], calamine::Data::Float(shown.expect("sized")));
        assert_eq!(
            length[column("Note")],
            calamine::Data::String(SIZED_NOTE.to_owned())
        );
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::on_screen(TALL);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_ne!(harness.panel.inputs(), &DesignInputs::default());

        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
        assert_drew_headline(&harness.frame(Vec::new()), &DesignInputs::default());
    }

    #[test]
    fn reset_clears_a_refused_edit() {
        let mut panel = MagcouplingPanel::new();
        panel.last_error = Some("refused".to_owned());
        panel.inputs.coupling.npole = 20;
        panel.reset();
        assert_eq!(panel.last_error, None);
        assert_eq!(panel.inputs(), &DesignInputs::default());
        assert_eq!(panel.results(), &compute_all(&DesignInputs::default()));
    }

    /// A keyboard shortcut tapped: pressed and released in one frame.
    fn shortcut_tap(shortcut: egui::KeyboardShortcut) -> Vec<egui::Event> {
        [true, false]
            .map(|pressed| egui::Event::Key {
                key: shortcut.logical_key,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: shortcut.modifiers,
            })
            .to_vec()
    }

    /// The design with the face gap at `gap`.
    fn gap_design(gap: f64) -> Design {
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = gap;
        design
    }

    #[test]
    fn undo_reverses_a_change_and_redo_brings_it_back() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.number(FACE_GAP), 1.41);
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(
            harness.panel.results(),
            &compute_all(&gap_design(1.41).inputs)
        );
    }

    #[test]
    fn each_arrow_nudge_is_one_undo_step() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..3 {
            harness.frame(key_tap(egui::Key::ArrowRight));
        }
        assert_eq!(harness.panel.history.undo_len(), 3);
        harness.panel.undo();
        assert_eq!(harness.number(FACE_GAP), 1.42);
    }

    #[test]
    fn a_held_arrow_key_is_one_undo_step() {
        // A held arrow key auto-repeats (about 30 presses a second; egui reads every press after
        // the first as a repeat). The run is one edit until the key is released (decision
        // M41-14), so holding a key cannot flood the undo levels.
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..20 {
            harness.frame(vec![key_event(egui::Key::ArrowRight, true)]);
        }
        assert_eq!(harness.number(FACE_GAP), 1.6);
        assert_eq!(harness.panel.history.undo_len(), 0, "still held");
        harness.frame(vec![key_event(egui::Key::ArrowRight, false)]);
        assert_eq!(harness.panel.history.undo_len(), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_drag_is_one_undo_step() {
        let mut harness = Harness::new();
        let rail = harness.widget(FACE_GAP).rect;
        let start = rail.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(start)]);
        harness.frame(vec![primary_button(start, true)]);
        for step in 1..=5 {
            let at = start + egui::vec2(10.0 * step as f32, 0.0);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
        }
        let end = start + egui::vec2(50.0, 0.0);
        harness.frame(vec![primary_button(end, false)]);
        harness.frame(Vec::new());
        let dragged = harness.number(FACE_GAP);
        assert!(dragged > 1.4, "{dragged}");
        assert_eq!(harness.panel.history.undo_len(), 1, "the whole drag");
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_keyboard_shortcuts_undo_and_redo() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        assert_eq!(harness.panel.design(), Design::default());
        harness.frame(shortcut_tap(REDO_SHORTCUTS[0]));
        assert_eq!(harness.panel.design(), gap_design(1.41));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        harness.frame(shortcut_tap(REDO_SHORTCUTS[1]));
        assert_eq!(harness.panel.design(), gap_design(1.41));
    }

    #[test]
    fn a_host_can_keep_ctrl_z_for_itself() {
        // M5 hosts the panel in an egui::Window of the linkage app, which undoes its own model
        // on Ctrl+Z. With the panel's shortcuts off the key is the host's alone: the panel does
        // not undo, and the event is still there for the host after the panel's frame.
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
            let mut host_saw_undo = false;
            let input = egui::RawInput {
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 1200.0])
                    .show(ctx, |ui| panel.ui(ui));
                host_saw_undo = ctx.input_mut(|i| i.consume_shortcut(&UNDO_SHORTCUT));
            });
            host_saw_undo
        };
        frame(&mut panel, Vec::new());
        panel.set_keyboard_shortcuts(false);
        assert!(
            frame(&mut panel, shortcut_tap(UNDO_SHORTCUT)),
            "left to the host"
        );
        assert_eq!(panel.design(), gap_design(2.0), "the panel did not undo");
        // On (the default), the panel undoes and takes the event.
        panel.set_keyboard_shortcuts(true);
        assert!(!frame(&mut panel, shortcut_tap(UNDO_SHORTCUT)));
        assert_eq!(panel.design(), Design::default());
    }

    #[test]
    fn ctrl_z_in_a_text_field_is_left_to_the_field() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        // The design's undo did not run: the face gap step is still there, nothing to redo.
        assert_eq!(harness.number(FACE_GAP), 1.41);
        assert!(!harness.panel.history.can_redo(&harness.panel.design()));
    }

    #[test]
    fn typing_a_part_name_is_one_undo_step_once_the_field_loses_focus() {
        let mut harness = Harness::new();
        harness.focus(PART_INNER);
        for letter in ["A", "B", "C"] {
            harness.frame(vec![egui::Event::Text(letter.to_owned())]);
        }
        assert_eq!(harness.panel.history.undo_len(), 0, "still typing");
        harness.ctx.memory_mut(|m| m.stop_text_input());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.history.undo_len(), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_slider_s_value_box_being_typed_in_is_an_edit_of_the_design() {
        // egui gives a Slider's response its value box's id while the box has focus, so the
        // row's widget id names the focused text field: typing a value is an edit in progress.
        let mut harness = Harness::new();
        harness.click_text("1.40 mm");
        let focused = focused_text_field(&harness.ctx).expect("the value box has focus");
        assert!(harness.panel.design_widgets.contains(&focused));
        harness.frame([select_all(), vec![egui::Event::Text("2".to_owned())]].concat());
        assert_eq!(harness.panel.history.undo_len(), 0, "still typing");
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        assert_eq!(harness.number(FACE_GAP), 2.0);
        assert_eq!(harness.panel.history.undo_len(), 1);
    }

    #[test]
    fn a_focused_results_search_holds_back_no_undo_step() {
        // The search box is a text field, but it edits no design: a change while it has focus
        // (a design file the host's picker delivers) is an undo step at once.
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "typing in the search"
        );
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.5)))
            .unwrap();
        harness.frame(Vec::new());
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "still in the search"
        );
        assert_eq!(harness.panel.history.undo_len(), 1);
    }

    #[test]
    fn only_the_rows_drawn_this_frame_count_as_design_fields() {
        let mut harness = Harness::new();
        let key_rows = InputCatalogue::get().key_design.len();
        assert_eq!(harness.panel.design_widgets.len(), key_rows);
        // Every group closed: no row is drawn, so none counts (no stale ids, no growth).
        harness.click_text(KEY_DESIGN_HEADING);
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        assert!(harness.panel.design_widgets.is_empty());
    }

    #[test]
    fn reset_all_can_be_undone() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), gap_design(1.41));
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(SAVE_DESIGN);
        harness.click_text(LOAD_DESIGN);
        assert_eq!(
            harness.panel.take_requests(),
            vec![
                PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&gap_design(1.41)).into_bytes(),
                },
                PanelRequest::OpenDesign,
            ]
        );
    }

    #[test]
    fn a_loaded_design_file_replaces_the_design_and_can_be_undone() {
        let mut harness = Harness::new();
        let mut loaded = gap_design(2.5);
        loaded.inputs.metal.required_min_Nm = 3.0;
        harness
            .panel
            .load_design_file(&design_to_json(&loaded))
            .unwrap();
        // The header grows to the status line on the next frame.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), loaded);
        assert_eq!(harness.panel.results(), &compute_all(&loaded.inputs));
        assert_eq!(count(&output, "Design file loaded"), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_refused_design_file_changes_nothing_and_says_why() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"coupling.backiron": 7}}"#;
        let error = harness.panel.load_design_file(text).unwrap_err();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(count(&output, &error.to_string()), 1);
        assert_eq!(
            error.to_string(),
            "inputs refused: coupling.backiron: 7 is not one of the choices"
        );
    }

    #[test]
    fn copy_share_link_puts_the_design_s_link_on_the_clipboard() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness
            .panel
            .set_share_base("http://localhost:8080/tools/magcoupler/");
        let output = harness.click_text(COPY_SHARE_LINK);
        let copied: Vec<&String> = output
            .platform_output
            .commands
            .iter()
            .filter_map(|c| match c {
                egui::OutputCommand::CopyText(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(copied, [&harness.panel.share_link()]);
        let payload = copied[0]
            .strip_prefix("http://localhost:8080/tools/magcoupler/?m=")
            .expect("the page's own address");
        // The link opens the same design in another panel.
        let mut other = MagcouplingPanel::new();
        other.load_share_payload(payload).unwrap();
        assert_eq!(other.design(), gap_design(1.41));
    }

    #[test]
    fn the_host_reports_what_a_save_did() {
        let mut harness = Harness::new();
        harness
            .panel
            .report(Ok("Saved C:/designs/a.json".to_owned()));
        harness.frame(Vec::new());
        assert_eq!(
            count(&harness.frame(Vec::new()), "Saved C:/designs/a.json"),
            1
        );
        harness
            .panel
            .report(Err("Could not save: disk full".to_owned()));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Could not save: disk full"), 1);
        assert_eq!(count(&output, "Saved C:/designs/a.json"), 0);
    }

    #[test]
    fn a_broken_share_link_changes_nothing_and_says_why() {
        let mut panel = MagcouplingPanel::new();
        let error = panel.load_share_payload("not-a-design").unwrap_err();
        assert!(matches!(error, LoadError::Link(_)), "{error}");
        assert_eq!(panel.design(), Design::default());
        assert_eq!(panel.last_error, Some(error.to_string()));
    }

    #[test]
    fn a_share_link_opened_at_start_up_is_not_an_undo_step() {
        // The web app opens a ?m= link as the session's start: the first Undo must not throw
        // the shared design away.
        let shared = gap_design(2.0);
        let mut harness = Harness::new();
        harness
            .panel
            .open_share_payload(&encode_share_payload(&shared))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), shared);
        assert!(!harness.panel.history.can_undo(&shared));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        assert_eq!(harness.panel.design(), shared);
        // An edit after it undoes back to the shared design, not to the defaults.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.design(), gap_design(2.01));
        harness.panel.undo();
        assert_eq!(harness.panel.design(), shared);
        // A refused link changes neither the design nor the history.
        let error = harness
            .panel
            .open_share_payload("not-a-design")
            .unwrap_err();
        assert_eq!(harness.panel.last_error, Some(error.to_string()));
        assert_eq!(harness.panel.design(), shared);
        assert!(harness.panel.history.can_redo(&shared));
    }

    #[test]
    fn every_text_the_panel_shows_has_glyphs_in_the_default_fonts() {
        // egui draws an empty box for a character its default fonts lack (U+2192, the
        // spec's arrow, is one): every drawn text, in each state, and every hover text.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut texts = drawn_texts(&harness.frame(Vec::new()));
        texts.extend(drawn_texts(&harness.size()));
        // Every view of the centre region, and the geometry callouts' hover texts.
        for view in CentreView::ALL {
            texts.extend(drawn_texts(&harness.click_text(view.label())));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        // The Equation panel open on a chain of temperature symbols (ϑ is drawn as θ).
        harness
            .panel
            .explorer
            .open_path("temperature.summary.governing_limit_C");
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.explorer.open = false;
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results());
        texts.extend(
            geometry
                .callouts()
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path)),
        );
        // Every group open, in both orders; every workflow heading; the filtered view's runs.
        for order in [InputOrder::Workflow, InputOrder::Workbook] {
            harness.click_text(order.label());
            // Bottom up: opening a group moves only the groups below it.
            for group in InputCatalogue::get().groups_in(order).iter().rev() {
                harness.click_text(group.label);
            }
            for _ in 0..10 {
                harness.frame(Vec::new());
            }
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        for group in &crate::gui::inputs::WORKFLOW {
            texts.push(group.label.to_owned());
            texts.extend(group.sections.iter().map(|s| s.label.to_owned()));
        }
        texts.push(ADVANCED_HEADING.to_owned());
        texts.push(FILTER_HINT.to_owned());
        texts.push(crate::gui::results_table::NOTHING_FAILS.to_owned());
        texts.push(crate::gui::results_table::NO_RESULT.to_owned());
        texts.push(crate::gui::results_table::TRACED_ONLY.to_owned());
        texts.push(crate::gui::input_ui::TRACE_HINT.to_owned());
        texts.push(crate::gui::trace::CLEAR_TRACE.to_owned());
        texts.push(crate::gui::results_table::NOTHING_TRACED.to_owned());
        for path in [FACE_GAP, "model.pullout_Nm", "coupling.drive_torque_Nm"] {
            texts.push(crate::gui::trace::Trace::of(path).unwrap().banner());
        }
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
        texts.extend([WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT].map(str::to_owned));
        // The load's notice of a sizing target that differed from the requirement (G-2).
        texts.push(target_mismatch_notice(3.25, 2.5));
        harness.panel.input_filter = "a".to_owned();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.input_filter.clear();
        // The Assumptions view: every rationale and source.
        harness.click_text(InputsView::Assumptions.label());
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.click_text(InputsView::Design.label());
        harness.panel.inputs = short_magnets();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        // Its c_end of 0.5 is a modified assumption: the banner (the header grows a frame later).
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        assert!(texts.iter().any(|t| t.starts_with(ASSUMPTIONS_MODIFIED)));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        // The material warnings the dashboard can show.
        texts.extend(
            crate::engine::warnings::WARNING_RULES
                .iter()
                .map(|rule| rule.text.to_owned()),
        );
        let results = harness.panel.results().clone();
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
                .into_iter()
                .map(|line| line.tooltip()),
        );
        for entry in crate::gui::results_table::table_entries() {
            let value = results.get(&entry.path).unwrap_or(Value::None);
            texts.push(crate::gui::results_table::row_tooltip(entry, &value));
        }
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the panel");
        }
    }

    #[test]
    fn size_and_torque_has_direct_dimension_and_gap_controls() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Size and torque"), 1);
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        for label in [
            "Outside diameter",
            "Overall length",
            "Magnet face gap",
            "Magnets per ring",
        ] {
            assert_eq!(count(&output, label), 1, "{label}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn manual_size_has_an_adjustable_minimum_od() {
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 8;
        harness.frame(Vec::new());
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Minimum OD"), 1);
        let before = harness.panel.design();
        let before_results = harness.panel.results.clone();
        let undo_count = harness.panel.history.undo_len();
        harness.click_text("40.3 mm");
        harness.frame([select_all(), vec![egui::Event::Text("37".to_owned())]].concat());
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        assert_eq!(harness.panel.minimum_diameter_mm, Some(37.0));
        assert_eq!(harness.panel.design(), before);
        assert_eq!(harness.panel.results(), &before_results);
        assert_eq!(harness.panel.history.undo_len(), undo_count);

        harness.click_text("42.8 mm");
        harness.frame([select_all(), vec![egui::Event::Text("38".to_owned())]].concat());
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        let smaller = harness.panel.design();
        assert!(smaller.inputs.coupling.inner_back_apothem_mm < 9.0);
        assert!((smaller.inputs.metal.cap_od_mm - 38.0).abs() < 1e-9);
        assert_eq!(harness.panel.results(), &compute_all(&smaller.inputs));
        assert_ne!(
            harness.panel.results.model.pullout_Nm,
            before_results.model.pullout_Nm
        );
        for _ in 0..3 {
            harness.frame(Vec::new());
            assert_eq!(harness.panel.design(), smaller);
        }
        harness.panel.undo();
        assert_eq!(harness.panel.design(), before);
        assert_eq!(harness.panel.minimum_diameter_mm, Some(37.0));
        harness.panel.redo();
        assert_eq!(harness.panel.design(), smaller);
    }

    #[test]
    fn manual_minimum_rejects_nonfinite_typed_values_without_changing_design() {
        let mut harness = Harness::new();
        harness.click_text("Size and torque");
        let before = harness.panel.design();
        harness.click_text("40.4 mm");
        harness.frame([select_all(), vec![egui::Event::Text("NaN".to_owned())]].concat());
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), before);
        assert!(
            harness
                .panel
                .minimum_diameter_mm
                .is_none_or(|value| value.is_finite())
        );
    }

    #[test]
    fn loaded_smaller_designs_keep_their_size_and_expose_a_stable_minimum() {
        let mut inputs = DesignInputs::default();
        inputs.coupling.npole = 8;
        let smaller =
            crate::engine::size_controls::resize_diameter_with_minimum(&inputs, 38.0, 37.0)
                .unwrap();
        let design = Design {
            inputs: smaller,
            ..Design::default()
        };
        let text = design_to_json(&design);
        let mut harness = Harness::new();
        harness.panel.minimum_diameter_mm = Some(45.0);
        harness.panel.load_design_file(&text).unwrap();
        harness.click_text("Size and torque");
        for _ in 0..3 {
            harness.frame(Vec::new());
            assert_eq!(harness.panel.design(), design);
        }
        assert!((harness.panel.minimum_diameter_mm.unwrap() - 38.0).abs() < 1e-9);
        let widget = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(widget));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        assert!((harness.panel.inputs.metal.cap_od_mm - 38.1).abs() < 1e-9);
        assert!((harness.panel.minimum_diameter_mm.unwrap() - 38.0).abs() < 1e-9);
        assert_eq!(
            widget, harness.panel.design_widgets[0],
            "diameter widget stays stable"
        );
        assert!(
            harness.ctx.memory(|m| m.has_focus(widget)),
            "diameter keeps keyboard focus"
        );
        harness.frame(key_tap(egui::Key::ArrowLeft));
        harness.frame(Vec::new());
        assert!(
            (harness.panel.inputs.metal.cap_od_mm - 38.0).abs() < 1e-9,
            "diameter={}, minimum={:?}, error={:?}",
            harness.panel.inputs.metal.cap_od_mm,
            harness.panel.minimum_diameter_mm,
            harness.panel.last_error
        );
        // The existing share schema carries the actual smaller dimensions.
        let payload = crate::gui::session::encode_share_payload(&harness.panel.design());
        let mut reopened = MagcouplingPanel::new();
        reopened.open_share_payload(&payload).unwrap();
        assert_eq!(reopened.design(), harness.panel.design());
        harness.panel.reset();
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn smaller_exploration_shows_magnet_fit_and_keyway_breakthrough() {
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 8;
        harness.panel.inputs = crate::engine::size_controls::resize_diameter_with_minimum(
            &harness.panel.inputs,
            35.0,
            34.0,
        )
        .unwrap();
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(
                &output,
                "Exploration only: magnets do not fit at this diameter."
            ),
            1
        );
        assert_eq!(
            count(
                &output,
                "Exploration only: the keyway breaks through the hub."
            ),
            1
        );
    }

    #[test]
    fn manual_size_edits_update_torque_and_are_one_undo_step() {
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 8;
        harness.frame(Vec::new());
        harness.click_text("Size and torque");
        harness.frame(Vec::new());
        let before = harness.panel.inputs.clone();
        let before_torque = harness.panel.results.model.pullout_Nm;
        let diameter_id = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(diameter_id));
        harness.frame(key_tap(egui::Key::ArrowLeft));
        harness.frame(Vec::new());
        let resized = harness.panel.inputs.clone();
        assert!(resized.coupling.inner_back_apothem_mm < before.coupling.inner_back_apothem_mm);
        assert!(resized.metal.cap_od_mm < before.metal.cap_od_mm);
        assert_eq!(resized.coupling.npole, 8);
        assert_eq!(resized.metal.face_gap_mm, before.metal.face_gap_mm);
        assert_eq!(harness.panel.results(), &compute_all(&resized));
        assert_ne!(harness.panel.results.model.pullout_Nm, before_torque);
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &before);
        harness.panel.redo();
        assert_eq!(harness.panel.inputs(), &resized);
    }

    #[test]
    fn manual_size_gap_is_the_existing_gap_and_idle_frames_do_not_edit() {
        let mut harness = Harness::new();
        harness.click_text("Size and torque");
        harness.frame(Vec::new());
        let before = harness.panel.inputs.clone();
        for _ in 0..4 {
            harness.frame(Vec::new());
            assert_eq!(harness.panel.inputs(), &before);
        }
        let gap_id = harness.panel.design_widgets[2];
        harness.ctx.memory_mut(|m| m.request_focus(gap_id));
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        assert!(harness.panel.inputs.metal.face_gap_mm > before.metal.face_gap_mm);
        assert_eq!(harness.panel.inputs.coupling, before.coupling);
        assert_eq!(
            harness.panel.results(),
            &compute_all(harness.panel.inputs())
        );
    }

    #[test]
    fn manual_size_does_not_present_out_of_range_torque_as_a_prediction() {
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        harness.frame(Vec::new());
        assert!(end_effect_banner(harness.panel.results()).is_some());
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        // Hot-low and the pull-out read no prediction, and the gap is not shown (spec 3.2, 5).
        assert_eq!(count(&output, NO_PREDICTION), 2);
        assert_eq!(beside(&output, HOT_LOW)[0].0, NO_PREDICTION);
        let (gap, _) = torque_gap_text(
            harness.panel.results().metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        assert_eq!(count(&output, &gap), 0, "{gap}");
    }

    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls.
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
        let mut harness = Harness::on_screen(screen);
        harness.panel.inputs = inputs;
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
        harness
    }

    /// The what-moves-what lines of the design the panel shows, as the Size tab computes them:
    /// that design, its own results, the panel's Minimum OD.
    fn shown_consequences(harness: &Harness) -> Consequences {
        let shown = harness.panel.shown_inputs();
        Consequences::of(
            &shown,
            &compute_all(&shown),
            harness.panel.minimum_diameter_mm,
        )
    }

    /// The rows egui laid the first drawn text equal to `needle` out on: more than one once it
    /// wrapped.
    fn rows(output: &egui::FullOutput, needle: &str) -> usize {
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Text(text) if text.galley.text() == needle => {
                    Some(text.galley.rows.len())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no {needle:?}"))
    }

    /// The texts drawn on the row of the label `label` and right of it, nearest first, with
    /// their rects: what a readout row shows beside its label (texts further right on the same
    /// row, the dashboard's, come after).
    fn beside(output: &egui::FullOutput, label: &str) -> Vec<(String, egui::Rect)> {
        let at = text_rect(output, label).unwrap_or_else(|| panic!("no {label:?}"));
        let mut row: Vec<(String, egui::Rect)> = crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Text(text) => {
                    let mut rect = text.galley.rect.translate(text.pos.to_vec2());
                    // In a wrapped row a label's galley starts at the row's left edge and is
                    // indented to where the text goes: its first glyph marks the text's left.
                    if let Some(glyph) = text.galley.rows.first().and_then(|r| r.glyphs.first()) {
                        rect.min.x = rect.min.x.max(text.pos.x + glyph.pos.x);
                    }
                    let on_row = (rect.center().y - at.center().y).abs() < at.height() / 2.0;
                    (on_row && rect.left() >= at.right() - 0.5)
                        .then(|| (text.galley.text().to_owned(), rect))
                }
                _ => None,
            })
            .collect();
        row.sort_by(|a, b| a.1.left().total_cmp(&b.1.left()));
        row
    }

    #[test]
    fn size_tab_reads_the_requirement_the_hot_low_gap_and_the_nominal_pull_out() {
        // Spec 3.2 at the default design, whose hot-low torque is below the requirement: the gap
        // reads short, in red, as the dashboard's hot minimum check fails.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let results = harness.panel.results().clone();
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, HOT_LOW, label.as_str()] {
            assert_eq!(count(&output, text), 1, "{text}");
        }
        assert_eq!(
            count(&output, "Expected torque"),
            0,
            "replaced by the nominal pull-out"
        );
        let shown = |path: &str| with_unit(format_value(&results.get(path).unwrap()), "N\u{b7}m");
        let required = harness.panel.inputs().metal.required_min_Nm;
        let (gap, meets) = torque_gap_text(results.metal.torque_hot_low_Nm, required);
        assert!(!meets && gap.starts_with("short "), "{gap}");
        let hot_low_row: Vec<String> = beside(&output, HOT_LOW)
            .into_iter()
            .take(2)
            .map(|(text, _)| text)
            .collect();
        assert_eq!(hot_low_row, [shown("metal.torque_hot_low_Nm"), gap.clone()]);
        assert_eq!(beside(&output, &label)[0].0, shown("model.pullout_Nm"));
        assert!(
            beside(&output, REQUIRED)[0].0.ends_with(" N\u{b7}m"),
            "the Required slider's value box"
        );
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(
            crate::gui::dashboard::check_level(&results, "metal.hot_min_check"),
            Some(Level::Bad),
            "the gap agrees with the hot minimum check"
        );
        // The label follows the live operating temperature.
        harness.panel.inputs.coupling.op_temp_C = 37.5;
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.results().metal.op_temp_C, 37.5);
        assert_eq!(count(&output, &pullout_label(37.5)), 1);
        assert_eq!(count(&output, &label), 0);
    }

    #[test]
    fn size_tab_required_slider_is_an_ordinary_design_edit() {
        // Spec 3.2 and G-2: the requirement's own slider is an edit like any other (one undo
        // step, carried by files and links), typed values clamped into its range (decision
        // M41-2); a value that is no number changes nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let before = harness.panel.design();
        let undo = harness.panel.history.undo_len();
        let type_in = |harness: &mut Harness, text: &str| {
            let output = harness.frame(Vec::new());
            let (_, value_box) = beside(&output, REQUIRED).remove(0);
            harness.click(value_box.center());
            harness.frame([select_all(), vec![egui::Event::Text(text.to_owned())]].concat());
            harness.frame(key_tap(egui::Key::Enter));
            harness.frame(Vec::new())
        };
        let output = type_in(&mut harness, "2");
        let mut lowered = before.inputs.clone();
        lowered.metal.required_min_Nm = 2.0;
        assert_eq!(
            harness.panel.inputs(),
            &lowered,
            "only the requirement changed"
        );
        assert_eq!(harness.panel.results(), &compute_all(&lowered));
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), undo + 1);
        // The gap now reads a margin, in green, from the design's own hot-low torque.
        let hot_low = harness.panel.results().metal.torque_hot_low_Nm;
        let (gap, meets) = torque_gap_text(hot_low, 2.0);
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Good.color(&harness.ctx.style().visuals))
        );
        // Files and links carry it.
        let design = harness.panel.design();
        assert_eq!(
            decode_share_payload(&encode_share_payload(&design)).unwrap(),
            design
        );
        assert_eq!(design_from_json(&design_to_json(&design)).unwrap(), design);
        // Past the slider's range: clamped to its end.
        let range = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .unwrap()
            .meta
            .range
            .unwrap();
        type_in(&mut harness, &format!("{}", range.max * 5.0));
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, range.max);
        assert_eq!(harness.panel.history.undo_len(), undo + 2);
        // Not a number: nothing changes.
        type_in(&mut harness, "NaN");
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, range.max);
        assert_eq!(harness.panel.history.undo_len(), undo + 2);
        // One undo per edit.
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &lowered);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), before);
    }

    #[test]
    fn typing_in_the_size_tab_s_required_box_holds_the_solve() {
        // Torque -> Magnets with the Size tab open: the Required slider's value box is a text
        // field of the design from the frame it takes focus, although the centre view is drawn
        // after the solve runs: a solve pending when it is clicked waits for the typing to end,
        // as it does for an input row's box. (A typed value applies on Enter alone,
        // `update_while_editing(false)`.) A file in that mode opened with the tab showing gets
        // here, as Auto turned on there does.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let mut design = Design::default();
        design.sizing.mode = SizingMode::TorqueToMagnets;
        harness.panel.open_design(design);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.runner.solves, 1);
        let output = harness.frame(Vec::new());
        let (_, value_box) = beside(&output, REQUIRED).remove(0);
        harness.click(value_box.center());
        let focused = focused_text_field(&harness.ctx).expect("the value box has focus");
        assert!(harness.panel.design_widgets.contains(&focused));
        // An edit of the design inside the debounce (as the results-search test makes one),
        // then the box is typed in.
        harness.panel.inputs.coupling.op_temp_C += 1.0;
        harness.frame([select_all(), vec![egui::Event::Text("3".to_owned())]].concat());
        for _ in 0..3 {
            harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        }
        assert_eq!(focused_text_field(&harness.ctx), Some(focused));
        assert_eq!(
            harness.panel.runner.solves, 1,
            "not while the box is typed in"
        );
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 3.0);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        assert!((harness.solved().torque_hot_low_Nm - 3.0).abs() < 1e-6);
    }

    #[test]
    fn size_tab_idle_frames_keep_an_off_step_or_out_of_range_requirement() {
        // Spec 5 and the invariant of `idle_frames_change_no_input`: the Required slider writes
        // only on an edit (decision M41-2), so a file's requirement off its step or past its
        // range is kept, and drawing the what-moves-what lines edits nothing.
        let range = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .unwrap()
            .meta
            .range
            .unwrap();
        for required in [2.345, range.max + 2.0] {
            let mut design = Design::default();
            design.inputs.metal.required_min_Nm = required;
            design.inputs.coupling.npole = 8;
            let mut harness = Harness::new();
            harness.panel.open_design(design.clone());
            harness.click_text(CentreView::SizeAndTorque.label());
            for _ in 0..10 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.design(), design, "{required}");
            assert_eq!(harness.panel.last_error, None, "{required}");
            assert_eq!(
                harness.panel.history.undo_len(),
                0,
                "{required}: no edit reported"
            );
        }
    }

    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
        // Spec 3.3 at the default design: the Set-by line under Outside diameter, the length
        // split under Overall length, and the magnet count's lines under Magnets per ring, the
        // ring-size line first, each under the one before.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let consequences = shown_consequences(&harness);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let diameter = consequences
            .diameter_line()
            .expect("the default design has a breakdown");
        let length = consequences.length_line().expect("and a length split");
        assert_eq!(count(&output, &diameter), 1);
        assert_eq!(count(&output, &length), 1);
        assert!(rect(DIAMETER).bottom() <= rect(diameter.as_str()).top());
        assert!(rect(diameter.as_str()).bottom() <= rect(LENGTH).top());
        assert!(rect(LENGTH).bottom() <= rect(length.as_str()).top());
        assert!(rect(length.as_str()).bottom() <= rect(GAP).top());
        let lines = consequences.magnet_lines(10, None);
        assert_eq!(lines.first().map(String::as_str), Some(RING_SIZE_STAYS));
        let mut above = rect(MAGNETS);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
            let at = rect(line.as_str());
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            above = at;
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "drawing them edits nothing"
        );
    }

    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
        // Spec 6 test 10, the owner's case of 2026-10-13: from 10 to 8 magnets the coupling did
        // not shrink and nothing said why. At 8 magnets the Size tab says the ring keeps its
        // size, and its Set-by line shows the cap as the largest part: the outside diameter is
        // the cap's, so fewer magnets cannot lower it.
        let mut harness = size_tab(SCREEN, with_poles(8));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, RING_SIZE_STAYS), 1);
        let shown = harness.panel.shown_inputs();
        let parts =
            diameter_breakdown(&shown).expect("the default ring at 8 magnets has a breakdown");
        assert!(
            parts.cap_mm > parts.cup_mm && parts.cap_mm > parts.boss_mm,
            "{parts:?}"
        );
        assert_eq!(
            parts.cap_mm,
            dimensions(&shown).expect("size controls").diameter_mm,
            "the cap is the outside diameter"
        );
        let consequences = shown_consequences(&harness);
        assert_eq!(consequences.diameter, Some(parts));
        let set_by = consequences.diameter_line().expect("a Set-by line");
        assert!(set_by.starts_with("Set by: cap "), "{set_by}");
        assert_eq!(count(&output, &set_by), 1);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        assert!(
            rect(DIAMETER).bottom() <= rect(set_by.as_str()).top(),
            "under Outside diameter"
        );
        assert!(
            rect(set_by.as_str()).bottom() <= rect(LENGTH).top(),
            "above Overall length"
        );
        assert!(
            rect(MAGNETS).bottom() <= rect(RING_SIZE_STAYS).top(),
            "under Magnets per ring"
        );
        // T5/T6 amendment: with Minimum OD unset the slider floor, not the coupling, stops the
        // fit search. The line says how far it could go and that Minimum OD unlocks it.
        let fit = consequences
            .magnet_lines(8, None)
            .into_iter()
            .find(|line| line.starts_with("Fits down to "))
            .expect("a fit line at the slider's floor");
        assert!(fit.ends_with("set Minimum OD lower to reach it."), "{fit}");
        assert_eq!(count(&output, &fit), 1, "{fit}");
    }

    #[test]
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Physics review 3, N4: the line under Magnets per ring says what a new count does to
        // the size Auto sets: the ring keeps its radius with Auto off, Auto resizes the ring
        // or resets the length when it sets them, and with Auto on the count (its slider
        // locked) no line speaks of changing it. Only the state's own line is drawn, under
        // Magnets per ring, and the neighbour lines are drawn with each.
        let firsts = [RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH];
        let states = [
            (None, Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::RingRadius), Some(AUTO_RESIZES_RING)),
            (Some(FreeVariable::AxialLength), Some(AUTO_RESETS_LENGTH)),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            if let Some(variable) = auto {
                harness.panel.sizing.variable = variable;
                harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            }
            harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            for line in firsts {
                let wanted = usize::from(Some(line) == first);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let shown = harness.panel.shown_inputs();
            let lines = shown_consequences(&harness).magnet_lines(shown.coupling.npole, auto);
            assert_eq!(
                lines.first().map(String::as_str) == first,
                first.is_some(),
                "{auto:?}"
            );
            let rect =
                |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
            let mut above = rect(MAGNETS);
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{auto:?}: {line}");
                let at = rect(line.as_str());
                assert!(above.bottom() <= at.top() + 0.5, "{auto:?}: {line:?}");
                above = at;
            }
        }
    }

    #[test]
    fn size_tab_what_moves_what_lines_follow_each_edit() {
        // Spec 6 test 5: the lines are recomputed when the design changes (the cache keys on
        // it), from the magnet count's own slider and from any other input.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
        // The controls' widgets in drawing order: diameter, length, gap, magnets, Minimum OD.
        let magnets = harness.panel.design_widgets[3];
        harness.ctx.memory_mut(|m| m.request_focus(magnets));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowLeft));
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        let at_8 = shown_consequences(&harness).magnet_lines(8, None);
        assert_ne!(at_8, at_10);
        for line in &at_8 {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_10.iter().filter(|line| !at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // Another input: a wider face gap moves every neighbour's pull-out and cup.
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        let output = harness.frame(Vec::new());
        let wider = shown_consequences(&harness);
        let lines = wider.magnet_lines(8, None);
        assert_ne!(lines, at_8);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_8.iter().filter(|line| !lines.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        for line in wider.diameter_line().into_iter().chain(wider.length_line()) {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
    }

    #[test]
    fn size_tab_smallest_diameter_line_searches_with_the_minimum_od() {
        // Spec 3.3 and 3.5: the fit search covers the Size tab's diameter range, the Minimum OD
        // preference included. At 8 magnets the blocks fit at the slider's floor, so the line
        // takes its minimum form; a lower Minimum OD lets the search find the smallest fit.
        let mut harness = size_tab(SCREEN, with_poles(8));
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        let floor = shown_consequences(&harness);
        assert!(
            matches!(floor.fit, Some(DiameterFit::AtMinimum(_))),
            "{:?}",
            floor.fit
        );
        let undo = harness.panel.history.undo_len();
        harness.panel.minimum_diameter_mm = Some(34.0);
        let output = harness.frame(Vec::new());
        let lowered = shown_consequences(&harness);
        assert!(
            matches!(lowered.fit, Some(DiameterFit::At(_))),
            "{:?}",
            lowered.fit
        );
        let (old, new) = (floor.magnet_lines(8, None), lowered.magnet_lines(8, None));
        assert_ne!(old, new);
        for line in &new {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in old.iter().filter(|line| !new.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // A view preference: the design and its history stay as they were.
        assert_eq!(harness.panel.inputs(), &with_poles(8));
        assert_eq!(harness.panel.history.undo_len(), undo);
    }

    #[test]
    fn size_tab_magnet_slider_ends_show_only_the_neighbour_in_range() {
        // Spec 5: at 4 or 40 magnets per ring the line for 2 or 42 is left out.
        for (npole, inside, outside) in [(4, 6, 2), (40, 38, 42)] {
            let mut harness = size_tab(SCREEN, with_poles(npole));
            let output = harness.frame(Vec::new());
            let consequences = shown_consequences(&harness);
            let neighbours: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(neighbours, [inside], "{npole}");
            let lines = consequences.magnet_lines(npole, None);
            assert!(lines.contains(&RING_SIZE_STAYS.to_owned()), "{lines:?}");
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            let texts = drawn_texts(&output);
            assert!(
                !texts
                    .iter()
                    .any(|t| t.starts_with(&format!("At {outside}:"))),
                "{npole}: {texts:?}"
            );
        }
    }

    #[test]
    fn size_tab_narrow_window_wraps_the_new_lines_inside_their_sides() {
        // Spec 5 "Narrow windows", at the M4-1 review's ~930 px window and the owner's 8 magnets:
        // the what-moves-what lines wrap inside the inputs side (its 320 points), one under the
        // next, and the torque readouts stay in the centre region, left of the dashboard.
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
        let output = harness.frame(Vec::new());
        let side = |id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("{id} is drawn"))
                .rect
        };
        let inputs_left = side("magcoupling_inputs").left();
        let inputs_right = inputs_left + INPUTS_WIDTH;
        let dashboard_left = side("magcoupling_dashboard").left();
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let consequences = shown_consequences(&harness);
        let magnet_lines = consequences.magnet_lines(8, None);
        let mut above = rect(MAGNETS);
        for line in &magnet_lines {
            let at = rect(line.as_str());
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} overlaps the line above it"
            );
            above = at;
        }
        let mut lines = magnet_lines.clone();
        lines.extend(consequences.diameter_line());
        lines.extend(consequences.length_line());
        for line in &lines {
            let at = rect(line.as_str());
            assert!(
                at.left() >= inputs_left - 1.0 && at.right() <= inputs_right + 1.0,
                "{line:?} at {at:?} runs out of the inputs side ({inputs_left} to {inputs_right})"
            );
        }
        assert!(
            lines.iter().any(|line| rows(&output, line) > 1),
            "no line wrapped: the test no longer exercises wrapping"
        );
        let results = harness.panel.results().clone();
        let (gap, _) = torque_gap_text(
            results.metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, HOT_LOW, gap.as_str(), label.as_str()] {
            let at = rect(text);
            assert!(
                at.left() >= inputs_right - 1.0 && at.right() <= dashboard_left + 1.0,
                "{text:?} at {at:?} runs out of the centre ({inputs_right} to {dashboard_left})"
            );
        }
    }

    #[test]
    fn size_tab_new_texts_have_glyphs_and_plain_words() {
        // Spec 6 tests 3 and 11 for the Size tab, in each state its new texts take: the
        // default, a met requirement, both ends of the magnet slider, the owner's 8 magnets, no
        // prediction (short magnets) and a ring too small for its magnets. Every drawn text has
        // glyphs in the default fonts; the new ones name no workbook cell, correction or sheet.
        let mut met = DesignInputs::default();
        met.metal.required_min_Nm = 2.0;
        let too_small =
            crate::engine::size_controls::resize_diameter_with_minimum(&with_poles(8), 35.0, 34.0)
                .unwrap();
        let states = [
            (DesignInputs::default(), None),
            (met, None),
            (with_poles(4), None),
            (with_poles(8), None),
            (with_poles(40), None),
            (short_magnets(), None),
            (too_small, Some(34.0)),
        ];
        let mut harness = size_tab(egui::vec2(1280.0, 4000.0), DesignInputs::default());
        let mut drawn = Vec::new();
        let mut new_texts = vec![REQUIRED.to_owned(), HOT_LOW.to_owned()];
        for (inputs, minimum) in states {
            harness.panel.inputs = inputs;
            harness.panel.minimum_diameter_mm = minimum;
            // The second frame draws with any minimum the first one pinned.
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let shown = harness.panel.shown_inputs();
            let results = compute_all(&shown);
            let consequences = shown_consequences(&harness);
            let mut texts = consequences.magnet_lines(shown.coupling.npole, None);
            if dimensions(&shown).is_some() {
                texts.extend(consequences.diameter_line());
                texts.extend(consequences.length_line());
            }
            texts.push(pullout_label(results.metal.op_temp_C));
            let hot_low = results.metal.torque_hot_low_Nm;
            texts.push(if predicts(&results, hot_low) {
                torque_gap_text(hot_low, shown.metal.required_min_Nm).0
            } else {
                NO_PREDICTION.to_owned()
            });
            let all = drawn_texts(&output);
            for text in &texts {
                assert!(all.contains(text), "{text:?} is not drawn");
            }
            new_texts.extend(texts);
            drawn.extend(all);
        }
        for text in &drawn {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the Size tab");
        }
        for text in &new_texts {
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }

    #[test]
    fn torque_to_magnets_shows_the_solved_design() {
        let mut harness = Harness::on_screen(TALL);
        let output = harness.size();
        let point = harness.solved();
        // The axial length (the default free variable) meets the 2.5 N\u{b7}m target.
        assert!((point.torque_hot_low_Nm - 2.5).abs() < 1e-6);
        let status = format!(
            "Solved at {} mm (hot-low torque 2.500 N\u{b7}m)",
            format_value(&Value::Num(point.value))
        );
        assert_eq!(count(&output, &status), 1, "{:?}", drawn_texts(&output));
        // The panel shows the sized design; the inputs keep their own (blank) length.
        assert_eq!(harness.panel.shown_inputs(), point.inputs);
        assert_eq!(harness.panel.results(), &compute_all(&point.inputs));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_drew_headline(&output, &point.inputs);
    }

    #[test]
    fn the_solve_waits_for_the_debounce_and_never_runs_per_frame() {
        let repaint_delay = |output: &egui::FullOutput| {
            output.viewport_output[&egui::ViewportId::ROOT].repaint_delay
        };
        // An idle web page draws no frame unless asked: a pending solve asks for one by the
        // end of its debounce (`<=`: egui's own hover animation asks for one sooner at first).
        let asks_within_the_debounce = |output: &egui::FullOutput| {
            let delay = repaint_delay(output);
            let debounce = std::time::Duration::from_secs_f64(DEBOUNCE_S);
            assert!(delay <= debounce, "repaint after {delay:?}");
        };
        let mut harness = Harness::new();
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        for _ in 0..5 {
            asks_within_the_debounce(&harness.frame_after(0.02, Vec::new()));
        }
        assert_eq!(harness.panel.runner.solves, 0, "still waiting");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SOLVING), 1);
        asks_within_the_debounce(&output);
        let solved = harness.frame_after(DEBOUNCE_S, Vec::new());
        assert_eq!(harness.panel.runner.solves, 1);
        // The solve asks for one more frame, so the inputs side shows its outcome.
        assert_eq!(repaint_delay(&solved), std::time::Duration::ZERO);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SOLVING), 0);
        for _ in 0..10 {
            harness.frame_after(0.1, Vec::new());
        }
        assert_eq!(harness.panel.runner.solves, 1, "idle frames never solve");
        let idle = harness.frame_after(0.1, Vec::new());
        assert_eq!(
            repaint_delay(&idle),
            std::time::Duration::MAX,
            "nothing pending"
        );
    }

    #[test]
    fn a_drag_in_progress_defers_the_solve() {
        let mut harness = Harness::new();
        harness.size();
        assert_eq!(harness.panel.runner.solves, 1);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        // A pointer held down (a drag) over empty space of the centre region.
        let blank = egui::pos2(700.0, 1000.0);
        harness.frame(vec![egui::Event::PointerMoved(blank)]);
        harness.frame(vec![primary_button(blank, true)]);
        for _ in 0..5 {
            harness.frame_after(0.2, Vec::new());
        }
        assert_eq!(
            harness.panel.runner.solves, 1,
            "not while the button is down"
        );
        harness.frame_after(0.01, vec![primary_button(blank, false)]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_row_shows_the_solved_value_locked() {
        let mut harness = Harness::new();
        // Magnets -> Torque: the axial length override and the measured drag are blank.
        assert_eq!(count(&harness.frame(Vec::new()), BLANK_TEXT), 2);
        let output = harness.size();
        let point = harness.solved();
        assert_eq!(count(&output, SIZED_NOTE), 1);
        assert!(!harness.widget(AXIAL_LENGTH).enabled(), "locked");
        // Its value box shows the solved length, not the blank of the inputs.
        assert_eq!(count(&output, BLANK_TEXT), 1, "only the measured drag");
        let label = InputCatalogue::get()
            .entry(AXIAL_LENGTH)
            .unwrap()
            .meta
            .label;
        assert_locked_row_shows(&output, label, &format!("{:.2} mm", point.value));
        // Arrow keys on it change nothing.
        let id = harness.widget(AXIAL_LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(id));
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn leaving_torque_to_magnets_while_solving_writes_the_current_solution() {
        // Decision M41-7: the value written into the inputs is this design's solution, even
        // when the user leaves before the debounce ran out ("Solving...").
        let mut harness = Harness::new();
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        assert_eq!(count(&harness.frame(Vec::new()), SOLVING), 1);
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.runner.solves, 2, "solved on leaving");
        let mut required = DesignInputs::default();
        required.metal.required_min_Nm = 3.0;
        let Ok(SizingOutcome::Solved(point)) =
            crate::engine::sizing::solve(&required, FreeVariable::AxialLength, 3.0)
        else {
            panic!("3.0 N·m is reachable by the axial length")
        };
        assert_eq!(harness.panel.inputs(), &point.inputs);
        assert!((point.torque_hot_low_Nm - 3.0).abs() < 1e-6);
    }

    #[test]
    fn typing_in_the_results_search_does_not_defer_the_solve() {
        let mut harness = Harness::new();
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        harness.click_text(CentreView::Results.label());
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "typing in the search"
        );
        assert_eq!(harness.panel.runner.solves, 2, "the search edits no design");
        // A text field of the design does defer it.
        harness.panel.inputs.metal.required_min_Nm = 3.5;
        harness.focus(PART_INNER);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(
            harness.panel.runner.solves, 2,
            "not while a part name is typed"
        );
    }

    #[test]
    fn the_json_export_in_torque_to_magnets_holds_the_inputs_that_produced_the_results() {
        use crate::gui::results_table::EXPORT_JSON;
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(CentreView::Results.label());
        harness.click_text(EXPORT_JSON);
        let requests = harness.panel.take_requests();
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
            panic!("one export: {requests:?}")
        };
        let text = std::str::from_utf8(contents).expect("the JSON export is UTF-8");
        let json: serde_json::Value = serde_json::from_str(text).unwrap();
        let design = design_from_json(&json["design"].to_string()).unwrap();
        assert_eq!(design.inputs, harness.panel.shown_inputs());
        assert_eq!(&compute_all(&design.inputs), harness.panel.results());
        assert_ne!(
            &design.inputs,
            harness.panel.inputs(),
            "the solved length, not the blank override"
        );
        assert_eq!(design.sizing, *harness.panel.sizing());
    }

    #[test]
    fn leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step() {
        use crate::gui::size_controls::AUTO;
        let mut harness = Harness::new();
        harness.size();
        let point = harness.solved();
        let steps = harness.panel.history.undo_len();
        // Auto off on the Size tab; opening the tab is no edit.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.history.undo_len(), steps);
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.inputs(), &point.inputs, "decision M41-7");
        assert_eq!(harness.panel.results(), &compute_all(&point.inputs));
        assert_eq!(harness.panel.history.undo_len(), steps + 1);
        harness.panel.undo();
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value_and_its_overshoot() {
        let mut harness = Harness::on_screen(TALL);
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Not reachable: best 9.937 N\u{b7}m at 50.80 mm"),
            1
        );
        assert_eq!(
            harness
                .panel
                .shown_inputs()
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8),
            "decision M41-8: the best value"
        );
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with("Exceeds the space claim:"))
        );
    }

    #[test]
    fn the_free_variable_picker_switches_the_variable() {
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness.size();
        // The picker is the Size tab's Auto row; the Key design group then shows the result.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::RingRadius));
        assert_eq!(harness.panel.sizing().variable, FreeVariable::RingRadius);
        harness.click_text(CentreView::Geometry.label());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let point = harness.solved();
        // The ring radius is not in the Key design group: its locked row shows there anyway,
        // with the Coupling group closed.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SIZED_NOTE), 1);
        let label = label_for(FreeVariable::RingRadius.path());
        assert_eq!(count(&output, label), 1);
        // Its value box shows the solved radius, not the inputs' own.
        assert_locked_row_shows(&output, label, &format!("{:.2} mm", point.value));
        assert_eq!(
            harness.panel.shown_inputs().coupling.inner_back_apothem_mm,
            point.value
        );
        assert_eq!(
            harness
                .panel
                .shown_inputs()
                .coupling
                .magnets
                .axial_length_mm,
            None,
            "the axial length is no longer sized"
        );
    }

    #[test]
    fn a_share_link_carries_the_sizing_state() {
        let mut harness = Harness::new();
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        let link = harness.panel.share_link();
        let payload = link.split("?m=").nth(1).unwrap();
        let mut other = Harness::new();
        other.panel.load_share_payload(payload).unwrap();
        assert_eq!(other.panel.design(), harness.panel.design());
        other.frame(Vec::new());
        other.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(other.panel.shown_inputs(), harness.panel.shown_inputs());
        assert_eq!(other.panel.results(), harness.panel.results());
    }

    #[test]
    fn a_file_whose_sizing_target_differs_loads_its_requirement_and_says_so() {
        // A file written before decision G-2: its Torque -> Magnets target (3.25) was its own
        // number beside the requirement (2.5). The requirement is used, the solve meets it,
        // and the header says so (a wrapping line).
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"metal.required_min_Nm": 2.5},
            "sizing": {"mode": "torque_to_magnets", "target_torque_Nm": 3.25}}"#;
        let notice = target_mismatch_notice(3.25, 2.5);
        let mut harness = Harness::new();
        harness.panel.load_design_file(text).unwrap();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let status = format!("Design file loaded. {notice}");
        assert_eq!(count(&output, &status), 1);
        let wrap = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Text(t) if t.galley.text() == status => {
                    Some(t.galley.job.wrap.max_width)
                }
                _ => None,
            });
        assert!(wrap.is_some_and(f32::is_finite), "the line wraps: {wrap:?}");
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!((harness.solved().torque_hot_low_Nm - 2.5).abs() < 1e-6);
        // The same text as a share link, opened at start-up or loaded: the same notice.
        let payload = encode_share_text(text);
        for open in [true, false] {
            let mut other = Harness::new();
            if open {
                other.panel.open_share_payload(&payload).unwrap();
            } else {
                other.panel.load_share_payload(&payload).unwrap();
            }
            other.frame(Vec::new());
            let output = other.frame(Vec::new());
            assert_eq!(
                count(
                    &output,
                    &format!("Design loaded from the share link. {notice}")
                ),
                1,
                "open: {open}"
            );
            assert_eq!(other.panel.design(), harness.panel.design());
        }
    }

    #[test]
    fn idle_frames_in_torque_to_magnets_keep_the_requirement_as_loaded() {
        // A guard test: it passes before decision G-2 too, by design. The target row is the
        // requirement's slider (SliderClamping::Edits, decision M41-2): idle frames keep a
        // file's value as it is, off the 0.01 grid and past the slider's 10 N·m, and record no
        // undo step.
        let mut design = Design::default();
        design.inputs.metal.required_min_Nm = 12.345;
        design.sizing.mode = SizingMode::TorqueToMagnets;
        let mut harness = Harness::new();
        harness.panel.open_design(design.clone());
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.history.undo_len(), 0);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.runner.solves, 1, "solved once, then idle");
    }

    #[test]
    fn auto_locks_the_matching_size_slider_and_says_why() {
        use crate::gui::size_controls::{
            AUTO, DIAMETER, GAP, LENGTH, MAGNETS, SET_BY_AUTO, change_text,
        };
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        harness.click_text(CentreView::SizeAndTorque.label());
        // Off: every size slider is the user's.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SET_BY_AUTO), 0);
        for label in [DIAMETER, LENGTH, GAP, MAGNETS] {
            assert!(harness.size_widget(label).enabled(), "{label}");
        }
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let point = harness.solved();
        // Magnet length, the default choice, locks Overall length alone.
        assert!(!harness.size_widget(LENGTH).enabled(), "locked");
        for label in [DIAMETER, GAP, MAGNETS] {
            assert!(harness.size_widget(label).enabled(), "{label}");
        }
        assert_eq!(
            count(&output, SET_BY_AUTO),
            1,
            "the line under the locked slider"
        );
        // The locked slider shows the solved design's length, as the slider writes it.
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(&point.inputs)
            .expect("a sized design has size controls")
            .overall_length_mm;
        assert!(
            on_slider_row(&output, rail, &format!("{length:.1} mm")),
            "{length}"
        );
        // The outcome line names the change, from the inputs' own length to the one shown.
        let line = change_text(
            FreeVariable::AxialLength,
            harness.panel.inputs(),
            &point.inputs,
        );
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, &line),
            Some(Level::Good.color(&harness.ctx.style().visuals))
        );
        // Its hover says why it does not move.
        harness.frame_after(0.2, vec![egui::Event::PointerMoved(rail.center())]);
        assert_eq!(
            count(&harness.frame(Vec::new()), SET_BY_AUTO),
            2,
            "the line and the hover text"
        );
        // Neither arrow keys nor a drag move it.
        let before = (harness.panel.design(), harness.panel.shown_inputs());
        let id = harness.size_widget(LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(id));
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(vec![primary_button(rail.center(), true)]);
        harness.frame(vec![egui::Event::PointerMoved(rail.right_center())]);
        harness.frame(vec![primary_button(rail.right_center(), false)]);
        harness.frame(Vec::new());
        assert_eq!(
            (harness.panel.design(), harness.panel.shown_inputs()),
            before
        );
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "M41-7: the inputs keep their own (blank) length"
        );
        // Another size: its slider locks, Overall length is the user's again, nothing is adopted.
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::MagnetsPerRing));
        assert_eq!(
            harness.panel.sizing().variable,
            FreeVariable::MagnetsPerRing
        );
        harness.frame(Vec::new());
        assert!(!harness.size_widget(MAGNETS).enabled());
        assert!(harness.size_widget(LENGTH).enabled());
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn auto_solves_again_after_the_debounce_when_another_size_changes() {
        use crate::engine::sizing::solve_in;
        use crate::gui::size_controls::{AUTO, GAP, change_text};
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 1);
        let steps = harness.panel.history.undo_len();
        // A nudge of the gap on the Size tab edits the inputs, not the solved length.
        let gap = harness.size_widget(GAP).id;
        harness.ctx.memory_mut(|m| m.request_focus(gap));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        let inputs = harness.panel.inputs().clone();
        assert!(inputs.metal.face_gap_mm > DesignInputs::default().metal.face_gap_mm);
        assert_eq!(
            inputs.coupling.magnets.axial_length_mm, None,
            "the inputs keep their own (blank) length"
        );
        assert_eq!(harness.panel.runner.solves, 1, "waiting for the debounce");
        assert_eq!(count(&output, SOLVING), 1);
        // The slider keeps keyboard focus; it is not a text field, so it holds no solve.
        assert_eq!(focused_text_field(&harness.ctx), None);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        let range = FreeVariable::AxialLength.range();
        let required = inputs.metal.required_min_Nm;
        let Ok(SizingOutcome::Solved(want)) =
            solve_in(&inputs, FreeVariable::AxialLength, required, range)
        else {
            panic!("the requirement is reachable at the nudged gap")
        };
        assert_eq!(harness.panel.shown_inputs(), want.inputs);
        assert_eq!(harness.panel.results(), &compute_all(&want.inputs));
        let line = change_text(FreeVariable::AxialLength, &inputs, &want.inputs);
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        // The nudge is one undo step; undoing it keeps Auto on.
        assert_eq!(harness.panel.history.undo_len(), steps + 1);
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
    }

    #[test]
    fn idle_frames_with_auto_on_change_nothing_and_solve_once() {
        use crate::gui::size_controls::AUTO;
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let design = harness.panel.design();
        let shown = harness.panel.shown_inputs();
        let steps = harness.panel.history.undo_len();
        for _ in 0..6 {
            harness.frame_after(0.1, Vec::new());
            assert_eq!(harness.panel.design(), design);
            assert_eq!(harness.panel.shown_inputs(), shown);
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the inputs keep their own length"
        );
        assert_eq!(
            harness.panel.minimum_diameter_mm, None,
            "no view state moves either"
        );
        assert_eq!(harness.panel.history.undo_len(), steps);
        assert_eq!(harness.panel.runner.solves, 1, "idle frames never solve");
    }

    #[test]
    fn auto_out_of_reach_shows_the_best_value_in_red() {
        use crate::gui::size_controls::{AUTO, LENGTH};
        use crate::gui::sizing::NOT_REACHABLE_PREFIX;
        let mut harness = Harness::new();
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) = harness
            .panel
            .runner
            .outcome(FreeVariable::AxialLength)
            .cloned()
        else {
            panic!("50 N·m is out of reach of the axial length")
        };
        let line = format!(
            "{NOT_REACHABLE_PREFIX}: best {} at {}",
            with_unit(
                format_value(&Value::Num(best.torque_hot_low_Nm)),
                "N\u{b7}m"
            ),
            with_unit(format_value(&Value::Num(best.value)), "mm")
        );
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, &line),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        // The locked slider sits at the best value (decision M41-8); the inputs keep their own.
        assert_eq!(harness.panel.shown_inputs(), best.inputs);
        assert_eq!(
            harness.panel.inputs().coupling.magnets.axial_length_mm,
            None
        );
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(&best.inputs)
            .expect("the best design has size controls")
            .overall_length_mm;
        assert!(
            on_slider_row(&output, rail, &format!("{length:.1} mm")),
            "{length}"
        );
    }

    #[test]
    fn auto_with_no_valid_ring_size_says_so_and_keeps_the_inputs_value() {
        use crate::gui::size_controls::{AUTO, DIAMETER};
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        harness.panel.sizing.variable = FreeVariable::RingRadius;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        // 2 mm magnets: f_end <= 0 at every ring size, so no layout counts.
        let line = "No valid value of ring size in range";
        assert_eq!(count(&output, line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, line),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(
            harness.panel.shown_inputs(),
            short_magnets(),
            "nothing is sized"
        );
        assert!(!harness.size_widget(DIAMETER).enabled());
        let rail = harness.size_widget(DIAMETER).rect;
        let diameter = dimensions(&short_magnets())
            .expect("2 mm magnets still have size controls")
            .diameter_mm;
        assert!(
            on_slider_row(&output, rail, &format!("{diameter:.1} mm")),
            "the slider keeps the inputs' value: {diameter}"
        );
    }

    #[test]
    fn auto_refuses_a_requirement_that_is_not_positive() {
        use crate::gui::size_controls::{AUTO, LENGTH};
        use crate::gui::sizing::REQUIREMENT_NOT_POSITIVE;
        let mut harness = Harness::new();
        harness.panel.inputs.metal.required_min_Nm = 0.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, REQUIREMENT_NOT_POSITIVE),
            1,
            "{:?}",
            drawn_texts(&output)
        );
        assert_eq!(
            crate::gui::test_support::text_color(&output, REQUIREMENT_NOT_POSITIVE),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(&harness.panel.shown_inputs(), harness.panel.inputs());
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(harness.panel.inputs())
            .expect("the default design has size controls")
            .overall_length_mm;
        assert!(
            on_slider_row(&output, rail, &format!("{length:.1} mm")),
            "{length}"
        );
    }

    #[test]
    fn auto_ring_size_searches_the_size_tab_range_below_the_old_floor() {
        use crate::engine::size_controls::ring_radius_range;
        use crate::engine::sizing::{is_valid, solve, solve_in};
        use crate::gui::size_controls::{AUTO, DIAMETER, LENGTH, change_text};
        use crate::gui::sizing::{auto_choice, sized};
        const RING: FreeVariable = FreeVariable::RingRadius;
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 8;
        harness.click_text(CentreView::SizeAndTorque.label());
        // The requirement: the hot-low torque of an 8.5 mm ring of 8 magnets, a valid layout.
        let at = RING.apply(harness.panel.inputs(), 8.5);
        let at_results = compute_all(&at);
        assert!(
            is_valid(&at, &at_results),
            "an 8.5 mm ring of 8 magnets fits"
        );
        harness.panel.inputs.metal.required_min_Nm = at_results.metal.torque_hot_low_Nm;
        // Ring size, picked in the Auto row, then Auto on.
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(RING));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(
            (harness.panel.sizing().mode, harness.panel.sizing().variable),
            (SizingMode::TorqueToMagnets, RING)
        );
        let inputs = harness.panel.inputs().clone();
        let required = inputs.metal.required_min_Nm;
        // At the default Minimum OD the search starts at the radius slider's floor, where it
        // meets the requirement.
        let floor = ring_radius_range(&inputs, None).expect("8 magnets define a size range");
        let at_floor = harness.solved();
        assert_eq!(
            Ok(SizingOutcome::Solved(at_floor.clone())),
            solve_in(&inputs, RING, required, floor)
        );
        assert!(at_floor.value >= RING.range().min, "{}", at_floor.value);
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        // A lowered Minimum OD widens the search: one more solve, after the debounce.
        harness.panel.minimum_diameter_mm = Some(37.0);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 1, "waiting for the debounce");
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        let range =
            ring_radius_range(&inputs, Some(37.0)).expect("a bounded minimum defines a range");
        assert!(range.min < 8.5, "{range:?}");
        let point = harness.solved();
        assert_eq!(
            Ok(SizingOutcome::Solved(point.clone())),
            solve_in(&inputs, RING, required, range)
        );
        assert!(
            point.value < RING.range().min && point.value >= range.min && point.value <= 8.5 + 1e-6,
            "{}",
            point.value
        );
        assert!(
            is_valid(&point.inputs, &compute_all(&point.inputs)),
            "fit and keyway wall still apply"
        );
        assert_eq!(
            harness.panel.inputs(),
            &inputs,
            "the inputs keep their own ring radius"
        );
        // T7 amendment (Review Focus 2): the design shown maps the solved radius as the Outside
        // diameter slider does, so the cap and its thread follow the cup and the locked slider
        // shows the solved outside diameter.
        let shown = harness.panel.shown_inputs();
        assert_eq!(shown, sized(RING, &inputs, point.value));
        assert_eq!(shown.coupling.inner_back_apothem_mm, point.value);
        let parts = |design: &DesignInputs| {
            diameter_breakdown(design).expect("a ring of 8 magnets has a diameter breakdown")
        };
        let (own, solved) = (parts(&inputs), parts(&shown));
        assert!(
            ((solved.cap_mm - solved.cup_mm) - (own.cap_mm - own.cup_mm)).abs() < 1e-9,
            "the cap keeps its offset from the cup: {own:?} {solved:?}"
        );
        let diameter = dimensions(&shown).expect("size controls").diameter_mm;
        assert!(
            diameter < dimensions(&inputs).unwrap().diameter_mm,
            "a smaller ring is a smaller coupling"
        );
        assert!(
            !harness.size_widget(DIAMETER).enabled(),
            "ring size locks Outside diameter"
        );
        assert!(harness.size_widget(LENGTH).enabled());
        let rail = harness.size_widget(DIAMETER).rect;
        assert!(
            on_slider_row(&output, rail, &format!("{diameter:.1} mm")),
            "{diameter}"
        );
        let line = change_text(RING, &inputs, &shown);
        assert!(line.starts_with(DIAMETER), "{line}");
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        // Idle frames move neither the design nor the Minimum OD, and solve nothing.
        let (design, steps) = (harness.panel.design(), harness.panel.history.undo_len());
        for _ in 0..3 {
            harness.frame_after(0.1, Vec::new());
            assert_eq!(harness.panel.design(), design);
        }
        assert_eq!(harness.panel.minimum_diameter_mm, Some(37.0));
        assert_eq!(harness.panel.history.undo_len(), steps);
        assert_eq!(harness.panel.runner.solves, 2);
        // An odd magnet count defines no size range: ring size then searches its own slider.
        harness.click_text(CentreView::Geometry.label());
        harness.panel.inputs.coupling.npole = 7;
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let odd = harness.panel.inputs().clone();
        assert_eq!(ring_radius_range(&odd, Some(37.0)), None);
        assert_eq!(
            harness.panel.runner.outcome(RING),
            Some(&solve(&odd, RING, odd.metal.required_min_Nm))
        );
    }

    #[test]
    fn auto_ring_size_out_of_reach_names_the_outside_diameter_shown() {
        // T7 amendment (Review Focus 2): the outcome line names Outside diameter, the value the
        // locked slider below it reads, never the internal ring radius.
        use crate::gui::format::fixed;
        use crate::gui::size_controls::{AUTO, DIAMETER};
        use crate::gui::sizing::{NOT_REACHABLE_PREFIX, auto_choice};
        let mut harness = Harness::new();
        harness.panel.inputs = with_poles(8);
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::RingRadius));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) = harness
            .panel
            .runner
            .outcome(FreeVariable::RingRadius)
            .cloned()
        else {
            panic!("50 N·m is out of reach of the ring size")
        };
        let shown = harness.panel.shown_inputs();
        assert_eq!(shown.coupling.inner_back_apothem_mm, best.value);
        let diameter = dimensions(&shown).expect("size controls").diameter_mm;
        let line = format!(
            "{NOT_REACHABLE_PREFIX}: best {} at {DIAMETER} {} mm",
            with_unit(
                format_value(&Value::Num(best.torque_hot_low_Nm)),
                "N\u{b7}m"
            ),
            fixed(diameter, 1)
        );
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, &line),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        let radius = with_unit(format_value(&Value::Num(best.value)), "mm");
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(NOT_REACHABLE_PREFIX) && t.contains(&radius)),
            "no outcome line quotes the radius {radius}"
        );
        // The longer outcome line wraps; one more frame settles the rows below it.
        let output = harness.frame(Vec::new());
        let rail = harness.size_widget(DIAMETER).rect;
        assert!(
            on_slider_row(&output, rail, &format!("{diameter:.1} mm")),
            "the locked slider reads the same diameter, {diameter}"
        );
    }

    #[test]
    fn ring_size_auto_locks_the_cap_rows_at_the_design_shown() {
        // The ring-size design shown moves the cap OD and its thread with the cup. While Auto is
        // on, their rows in the detailed inputs read the design shown and are locked like the
        // ring radius's, so no row shows or takes a value the results do not use.
        use crate::gui::inputs::step_decimals;
        use crate::gui::size_controls::AUTO;
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        harness.panel.inputs = with_poles(8);
        harness.panel.inputs.metal.required_min_Nm = 1.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::RingRadius));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.solved();
        let own = harness.panel.inputs().clone();
        let shown = harness.panel.shown_inputs();
        // The cap's rows, and two cap rows Auto does not set, on the inputs side of another tab.
        harness.click_text(CentreView::Geometry.label());
        harness.panel.input_filter = "metal.cap_".to_owned();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, SIZED_NOTE),
            2,
            "the cap OD and thread rows only: {:?}",
            drawn_texts(&output)
        );
        let number = |design: &DesignInputs, path: &str| match design.get(path) {
            Some(Value::Num(value)) => value,
            other => panic!("{path} is a number: {other:?}"),
        };
        // The value box of `path`'s row, showing `value`: the first such text below its label.
        let value_box = |path: &str, value: f64| {
            let entry = InputCatalogue::get().entry(path).unwrap();
            let decimals = step_decimals(entry.meta.range.expect("a slider").step);
            let text = format!("{value:.decimals$} mm");
            let label = text_rect(&output, entry.meta.label).expect("the row is drawn");
            text_rects(&output, &text)
                .into_iter()
                .filter(|r| r.top() >= label.bottom())
                .min_by(|a, b| a.top().total_cmp(&b.top()))
                .unwrap_or_else(|| panic!("{path} shows {text:?}: {:?}", drawn_texts(&output)))
        };
        let notes = text_rects(&output, SIZED_NOTE);
        let mut locked = Vec::new();
        for path in ["metal.cap_od_mm", "metal.cap_thread_dia_mm"] {
            let value = number(&shown, path);
            assert_ne!(value, number(&own, path), "{path} follows the cup");
            let rect = value_box(path, value);
            assert!(
                notes
                    .iter()
                    .any(|n| n.top() >= rect.bottom() && n.top() - rect.bottom() < 40.0),
                "{path}'s row is noted as set by Auto"
            );
            locked.push(rect.center());
        }
        let free = value_box("metal.cap_axial_mm", number(&own, "metal.cap_axial_mm")).center();
        // A drag on a value box: an edit wherever the row takes one.
        let mut drag = |at: egui::Pos2| {
            let to = at + egui::vec2(60.0, 0.0);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            harness.frame(vec![primary_button(at, true)]);
            harness.frame(vec![egui::Event::PointerMoved(to)]);
            harness.frame(vec![primary_button(to, false)]);
            harness.panel.inputs().clone()
        };
        // A locked row takes no edit: the inputs keep their own cap.
        for at in locked {
            assert_eq!(drag(at), own);
        }
        // The same drag edits a cap row Auto does not set.
        let edited = drag(free);
        assert_ne!(
            edited.metal.cap_axial_mm, own.metal.cap_axial_mm,
            "the drag edits an unlocked row"
        );
    }

    #[test]
    fn a_size_tab_edit_under_ring_size_auto_keeps_the_inputs_cap() {
        // The ring-size design shown moves the cap OD and thread with the cup. A Size-tab edit
        // of another size is made on that design: the inputs take the edit and keep their own
        // ring radius, cap and thread (decision M41-7), so no offset drifts edit by edit.
        use crate::gui::size_controls::{AUTO, GAP};
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        harness.panel.inputs = with_poles(8);
        harness.panel.inputs.metal.required_min_Nm = 2.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::RingRadius));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let own = harness.panel.inputs().clone();
        let shown = harness.panel.shown_inputs();
        assert_ne!(
            shown.metal.cap_od_mm, own.metal.cap_od_mm,
            "the cap follows the cup"
        );
        let gap = harness.size_widget(GAP).id;
        harness.ctx.memory_mut(|m| m.request_focus(gap));
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        let mut want = own.clone();
        want.metal.face_gap_mm = harness.panel.inputs().metal.face_gap_mm;
        assert!(
            want.metal.face_gap_mm > own.metal.face_gap_mm,
            "the gap moved"
        );
        assert_eq!(harness.panel.inputs(), &want);
    }

    #[test]
    fn a_diameter_edit_under_magnet_count_auto_maps_under_the_count_shown() {
        // Critic item 2.7, the behaviour chosen: with magnet-count Auto on, the Size tab edits
        // the design shown, whose count is the solved one. The typed outside diameter maps to a
        // ring radius under that count, so the design shown reads the diameter typed; the
        // inputs take that ring radius and keep their own count (decision M41-7). The solve then
        // runs again for the new ring.
        use crate::gui::size_controls::{AUTO, DIAMETER, SET_BY_AUTO};
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        // The hot-low torque of 8 magnets per ring: met below the inputs' own 10.
        let required = compute_all(&with_poles(8)).metal.torque_hot_low_Nm;
        harness.panel.inputs.metal.required_min_Nm = required;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::MagnetsPerRing));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let own = harness.panel.inputs().clone();
        let solved = harness.solved();
        assert!((solved.torque_hot_low_Nm - required).abs() < 1e-6);
        assert_ne!(
            solved.value as i64, own.coupling.npole,
            "the test needs a solved count other than the inputs' own"
        );
        let before = dimensions(&harness.panel.shown_inputs())
            .expect("size controls")
            .diameter_mm;
        let slider = harness.size_widget(DIAMETER).id;
        harness.ctx.memory_mut(|m| m.request_focus(slider));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        let inputs = harness.panel.inputs().clone();
        assert_eq!(
            inputs.coupling.npole, own.coupling.npole,
            "the inputs' own count"
        );
        assert_ne!(
            inputs.coupling.inner_back_apothem_mm, own.coupling.inner_back_apothem_mm,
            "the edit moved the ring"
        );
        // Until the solve runs again the design shown is the inputs at the solved count, and it
        // reads the diameter the slider was moved to.
        let shown = harness.panel.shown_inputs();
        assert_eq!(
            shown,
            FreeVariable::MagnetsPerRing.apply(&inputs, solved.value)
        );
        let typed = dimensions(&shown).expect("size controls").diameter_mm;
        assert!(typed > before, "{typed} after {before}");
        let rail = harness.size_widget(DIAMETER).rect;
        assert!(
            on_slider_row(&output, rail, &format!("{typed:.1} mm")),
            "{typed}"
        );
        assert_eq!(
            count(&output, SET_BY_AUTO),
            1,
            "Magnets per ring stays locked"
        );
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        assert_eq!(harness.panel.inputs(), &inputs, "the solve edits no input");
    }

    #[test]
    fn an_inverse_mode_design_file_opens_with_auto_on() {
        use crate::gui::size_controls::{LENGTH, SET_BY_AUTO, change_text};
        let design = Design {
            inputs: DesignInputs::default(),
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::AxialLength,
            },
        };
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness
            .panel
            .load_design_file(&design_to_json(&design))
            .unwrap();
        // Opening the Size tab switches no mode.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.design(), design);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the inputs keep their own length"
        );
        assert!(!harness.size_widget(LENGTH).enabled(), "locked");
        assert_eq!(count(&output, SET_BY_AUTO), 1);
        let shown = harness.panel.shown_inputs();
        let line = change_text(FreeVariable::AxialLength, harness.panel.inputs(), &shown);
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        // A share link of the same design opens the same way, as the session's start.
        let mut linked = Harness::new();
        linked
            .panel
            .open_share_payload(&encode_share_payload(&design))
            .unwrap();
        linked.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(linked.panel.design(), design);
    }

    #[test]
    fn the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab() {
        use crate::gui::size_controls::{AUTO_STATUS_START, OPEN_SIZE_TAB, auto_status};
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        let output = harness.frame(Vec::new());
        // The solver's controls live on the Size tab: no mode switch, picker or target here.
        let gone = [
            SizingMode::MagnetsToTorque.label(),
            SizingMode::TorqueToMagnets.label(),
            FREE_VARIABLE_LABEL,
            TARGET_LABEL,
        ];
        for text in gone {
            assert_eq!(count(&output, text), 0, "{text}");
        }
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(AUTO_STATUS_START)),
            "Auto is off"
        );
        // Auto on: one line naming the size it sets, and the button.
        let output = harness.size();
        assert_eq!(count(&output, &auto_status(FreeVariable::AxialLength)), 1);
        assert_eq!(count(&output, OPEN_SIZE_TAB), 1);
        for text in gone {
            assert_eq!(count(&output, text), 0, "{text}");
        }
        let design = harness.panel.design();
        harness.click_text(OPEN_SIZE_TAB);
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.design(), design, "the button edits nothing");
    }

    #[test]
    fn the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution() {
        use crate::gui::size_controls::{AUTO, LENGTH};
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness.size();
        let solved = harness.panel.shown_inputs();
        // Opening the Size tab switches nothing.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.shown_inputs(), solved);
        assert!(!harness.size_widget(LENGTH).enabled());
        // Auto off: the inputs adopt the length shown (decision M41-7); the slider is the user's.
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.inputs(), &solved);
        assert_eq!(harness.panel.results(), &compute_all(&solved));
        harness.frame(Vec::new());
        assert!(harness.size_widget(LENGTH).enabled());
    }

    #[test]
    fn the_auto_texts_have_glyphs_and_wrap_on_the_inputs_side() {
        use crate::gui::size_controls::{
            AUTO, AUTO_HOVER, AUTO_SUFFIX, LENGTH, OPEN_SIZE_TAB, SET_BY_AUTO, auto_status,
            change_text,
        };
        use crate::gui::sizing::{
            NOT_REACHABLE_PREFIX, REQUIREMENT_NOT_POSITIVE, auto_choice, sized,
        };
        // The M4-1 review's ~930 px window; the inputs side keeps its 320 points.
        let mut harness = Harness::on_screen(egui::vec2(930.0, 1024.0));
        tooltips_at_once(&harness);
        // The longest outcome line: a requirement out of reach.
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        let outcome = texts
            .iter()
            .find(|t| t.starts_with(NOT_REACHABLE_PREFIX))
            .cloned()
            .expect("the outcome line");
        // The centre region starts at the Size tab's tab (the inputs side's heading of the same
        // text paints first, further left).
        let centre_left = text_rects(&output, CentreView::SizeAndTorque.label())
            .iter()
            .map(|r| r.left())
            .fold(f32::MIN, f32::max);
        let row = [
            AUTO,
            AUTO_SUFFIX,
            SET_BY_AUTO,
            auto_choice(FreeVariable::AxialLength),
            outcome.as_str(),
        ];
        for text in row {
            let rects = text_rects(&output, text);
            assert!(!rects.is_empty(), "{text:?} is drawn");
            for rect in rects {
                assert!(
                    rect.right() <= centre_left,
                    "{text:?} at {rect:?} runs past the inputs side ({centre_left})"
                );
            }
        }
        let mut all = texts;
        // The locked slider's hover text.
        let rail = harness.size_widget(LENGTH).rect;
        harness.frame_after(0.2, vec![egui::Event::PointerMoved(rail.center())]);
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Key design group's line, on another tab.
        harness.click_text(CentreView::Geometry.label());
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        assert!(all.contains(&auto_status(FreeVariable::AxialLength)));
        // Every Auto text, each choice's.
        all.extend([AUTO_HOVER, OPEN_SIZE_TAB, REQUIREMENT_NOT_POSITIVE].map(str::to_owned));
        let from = DesignInputs::default();
        for variable in FreeVariable::ALL {
            all.push(auto_status(variable));
            let to = sized(variable, &from, variable.range().max);
            all.push(change_text(variable, &from, &to));
        }
        for text in &all {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the Auto texts");
        }
    }

    /// Shows tooltips at once (no delay, the pointer need not rest), as the geometry view's
    /// hover test does.
    fn tooltips_at_once(harness: &Harness) {
        harness.ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
    }

    /// The rects of the term marks painted in a frame, in `color` (any colour for `None`).
    fn mark_rects(output: &egui::FullOutput, color: Option<egui::Color32>) -> Vec<egui::Rect> {
        use crate::gui::readouts::MARK_WIDTH;
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r)
                    if r.stroke.width == MARK_WIDTH
                        && color.is_none_or(|c| r.stroke.color == c) =>
                {
                    Some(r.rect)
                }
                _ => None,
            })
            .collect()
    }

    /// The rects of the frames a leaf term draws around its input row in a frame.
    fn focus_rects(output: &egui::FullOutput) -> Vec<egui::Rect> {
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.stroke.width == FOCUS_WIDTH => Some(r.rect),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_size_tab_s_gap_and_count_labels_show_the_overlay_help() {
        use crate::gui::plain_text::help_for;
        // F19's second help site: the Size tab's gap and magnet-count labels hover their
        // input's plain_text::help_for. It passes before HELP has rows; it bites once help is
        // filled and a label stops showing it.
        for (label, path) in [(GAP, FACE_GAP), (MAGNETS, POLES)] {
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            tooltips_at_once(&harness);
            let output = harness.frame(Vec::new());
            let at = text_rect(&output, label)
                .unwrap_or_else(|| panic!("no {label:?}"))
                .center();
            // egui shows a tooltip only after time has passed with the pointer on the widget.
            harness.frame_after(0.5, vec![egui::Event::PointerMoved(at)]);
            let output = harness.frame_after(0.5, Vec::new());
            let help = help_for(path);
            assert!(!help.is_empty(), "{path}");
            assert!(
                drawn_texts(&output).iter().any(|t| t == help),
                "{path}: {help:?}"
            );
        }
    }

    #[test]
    fn hovering_a_dashboard_value_shows_its_equation() {
        use crate::gui::readouts::OPEN_HINT;
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        let output = harness.frame(Vec::new());
        let pullout = displayed_pullout(&DesignInputs::default());
        let at = text_rect(&output, &pullout).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.explorer.hovered(), Some("model.pullout_Nm"));
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with(&format!("{}\n", label_for("model.pullout_Nm")))),
            "the hover text: {texts:?}"
        );
        // T_pull = T_2D f_end f_cal, typeset run by run, and the hint.
        for run in ["T", "pull", " = ", "2D", "end", "cal", OPEN_HINT] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
    }

    #[test]
    fn hovering_a_value_marks_its_terms_where_their_values_are_shown() {
        // D_cup = 2 (r_corner + t_wall): hovering the cup OD on the dashboard frames the Key
        // design row of the wall in its term's colour, from the next frame, and its tooltip
        // draws t_wall in that colour; moving the pointer away clears every mark.
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        // The dashboard settles its width on the second frame, and the check guides above the
        // cup OD wrap to it: measure where the row rests.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let cup_od = harness.panel.results().model.cup_od_mm;
        let shown = with_unit(format_value(&Value::Num(cup_od)), "mm");
        let at = text_rect(&output, &shown).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let wall = crate::gui::typeset::TermColors::of(eq)
            .get("metal.cup_wall_corner_mm")
            .unwrap();
        let label = InputCatalogue::get()
            .entry("metal.cup_wall_corner_mm")
            .unwrap()
            .meta
            .label;
        let label_rect = text_rect(&output, label).unwrap();
        let marks = mark_rects(&output, Some(wall));
        assert!(
            marks.iter().any(|r| r.contains_rect(label_rect)),
            "{label_rect:?} not inside a mark: {marks:?}"
        );
        // The tooltip and the marks colour the term alike: t_wall's subscript run.
        assert_eq!(
            crate::gui::test_support::text_color(&output, "wall"),
            Some(wall),
            "the tooltip's t_wall: {:?}",
            drawn_texts(&output)
        );
        harness.frame(vec![egui::Event::PointerMoved(egui::pos2(
            5.0,
            SCREEN.y - 5.0,
        ))]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.explorer.hovered(), None);
        assert!(mark_rects(&output, None).is_empty());
    }

    #[test]
    fn the_equation_panel_starts_closed_and_its_header_button_toggles_it() {
        use crate::gui::explorer::{CLOSE, EMPTY_TEXT};
        let mut harness = Harness::new();
        assert!(!harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 0);
        harness.click_text(EQUATION_PANEL);
        assert!(harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 1);
        harness.click_text(CLOSE);
        assert!(!harness.panel.explorer.open);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns() {
        use crate::gui::explorer::TERMS;
        let mut harness = Harness::new();
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, TERMS), 1);
        // The term list: T_2D, f_end and f_cal by their labels.
        for label in ["2D pull-out torque (infinite length)", "End-effect factor"] {
            assert!(
                drawn_texts(&output).iter().any(|t| t == label),
                "no {label:?}"
            );
        }
        harness.click_text("End-effect factor");
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.pullout_Nm", "model.f_end"]
        );
        let output = harness.frame(Vec::new());
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == "End-effect coefficient"),
            "f_end's terms"
        );
        // The first crumb goes back.
        harness.click_text("T_pull");
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_used_by_list_goes_up_the_chain() {
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        // The "used by" row wraps: it settles on its second frame.
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        harness.click_text("T_pull");
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.f_end", "model.pullout_Nm"]
        );
    }

    #[test]
    fn a_leaf_term_opens_its_group_and_highlights_its_input_row() {
        // T_ripple,in = T_pull / (i_g η_g): the gearbox ratio is in the Coupling group, closed
        // by default.
        let mut harness = Harness::new();
        harness
            .panel
            .explorer
            .open_path("model.gearbox_input_ripple_Nm");
        harness.frame(Vec::new());
        harness.click_text("Gearbox ratio");
        assert_eq!(
            harness.panel.explorer.focus().map(|f| f.path.as_str()),
            Some("coupling.gear_ratio")
        );
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.gearbox_input_ripple_Nm"]
        );
        let mut output = harness.frame(Vec::new());
        for _ in 0..10 {
            output = harness.frame(Vec::new());
        }
        let focus = focus_rects(&output);
        // The inputs side paints first: the first "Gearbox ratio" is the row's.
        let label = text_rect(&output, "Gearbox ratio").unwrap();
        assert!(label.left() < INPUTS_WIDTH, "the row on the inputs side");
        assert!(
            focus.iter().any(|r| r.contains_rect(label)),
            "{label:?} not inside {focus:?}"
        );
        assert!(
            !harness.panel.explorer.focus().unwrap().scroll,
            "scrolled once"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_equation_panel_draws_in_a_tiny_window() {
        // A window smaller than the panel's starting height, beside every view of the centre
        // region (the view gets a strip a few points tall, or none): nothing panics, every
        // frame.
        for size in [
            egui::vec2(1.0, 1.0),
            egui::vec2(200.0, 150.0),
            egui::vec2(640.0, 240.0),
            egui::vec2(1280.0, 240.0),
            egui::vec2(1280.0, 300.0),
        ] {
            let mut harness = Harness::on_screen(size);
            harness
                .panel
                .explorer
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.explorer.explain = true;
            for view in CentreView::ALL {
                harness.panel.centre = view;
                for _ in 0..3 {
                    harness.frame(Vec::new());
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }

    /// The Equation panel's rect as egui stored it in the last frame. Its top edge is the
    /// panel's; its bottom can run past the centre region's when the panel's contents are
    /// taller than the panel (egui clips them).
    fn equation_panel_rect(ctx: &egui::Context) -> egui::Rect {
        egui::containers::panel::PanelState::load(ctx, egui::Id::new(EQUATION_PANEL_ID))
            .expect("the Equation panel is open")
            .rect
    }

    /// The header's rect in the last frame: the centre region starts under it.
    fn header_rect(ctx: &egui::Context) -> egui::Rect {
        egui::containers::panel::PanelState::load(ctx, egui::Id::new("magcoupling_header"))
            .expect("the header is drawn")
            .rect
    }

    /// Where what the centre region painted after the Equation panel in the frame `output`
    /// (the tab row, the end-effect banner, the view) shows: each shape's bounding rect within
    /// its clip rect, in paint order, the empty ones left out. Paint order splits the frame:
    /// egui paints the panel's separator line with the centre region's painter after the
    /// panel's contents, so the centre's own shapes are those after that line.
    fn painted_after_the_equation_panel(
        ctx: &egui::Context,
        output: &egui::FullOutput,
    ) -> Vec<egui::Rect> {
        let panel = equation_panel_rect(ctx);
        let top = panel.top();
        // Every shape with its clip rect, nested shapes flattened, in paint order.
        fn walk(shape: &egui::Shape, clip: egui::Rect, out: &mut Vec<(egui::Shape, egui::Rect)>) {
            match shape {
                egui::Shape::Vec(nested) => nested.iter().for_each(|s| walk(s, clip, out)),
                other => out.push((other.clone(), clip)),
            }
        }
        let mut shapes = Vec::new();
        for clipped in &output.shapes {
            walk(&clipped.shape, clipped.clip_rect, &mut shapes);
        }
        let separator = shapes
            .iter()
            .position(|(shape, _)| match shape {
                egui::Shape::LineSegment { points: [a, b], .. } => {
                    a.y == b.y
                        && (top..=top + 1.0).contains(&a.y)
                        && a.x.min(b.x) <= panel.left() + 0.5
                        && a.x.max(b.x) >= panel.right() - 0.5
                }
                _ => false,
            })
            .unwrap_or_else(|| panic!("no separator line at the panel's top {top}"));
        shapes[separator + 1..]
            .iter()
            .map(|(shape, clip)| shape.visual_bounding_rect().intersect(*clip))
            .filter(egui::Rect::is_positive)
            .collect()
    }

    /// What the centre region drew after the Equation panel in the frame `output` that paints
    /// into the panel or takes a click inside it, described; `None` when nothing does. Every
    /// shape after the panel ([`painted_after_the_equation_panel`]), where it shows, and every
    /// widget registered after the panel's resize handle (which egui registers after the
    /// panel's contents) that senses a click or a drag, where it takes them (its interact
    /// rect), must stay out of the panel's rect. Anything else drawn later (a host window's
    /// resize corner) lies beside it, and a frame's hover-only widget, which egui registers
    /// over its contents, takes no click.
    fn equation_panel_overlap(ctx: &egui::Context, output: &egui::FullOutput) -> Option<String> {
        let panel = equation_panel_rect(ctx);
        let top = panel.top();
        // How far `rect` reaches into the panel [points], when it does.
        let depth = |rect: egui::Rect| {
            let inside = rect.intersect(panel);
            (inside.width() > 0.0 && inside.height() > 0.01).then(|| inside.bottom() - top)
        };
        let painted: Vec<(f32, egui::Rect)> = painted_after_the_equation_panel(ctx, output)
            .into_iter()
            .filter_map(|seen| depth(seen).map(|d| (d, seen)))
            .collect();
        let resize = egui::Id::new(EQUATION_PANEL_ID).with("__resize");
        let widgets: Vec<egui::WidgetRect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .find(|(_, widgets)| widgets.iter().any(|w| w.id == resize))
                .map(|(_, widgets)| widgets.to_vec())
                .expect("the panel's resize handle is registered")
        });
        let handle = widgets.iter().position(|w| w.id == resize).unwrap();
        let clickable: Vec<(f32, egui::Rect)> = widgets[handle + 1..]
            .iter()
            .filter(|w| w.sense.interactive())
            .filter_map(|w| depth(w.interact_rect).map(|d| (d, w.interact_rect)))
            .collect();
        if painted.is_empty() && clickable.is_empty() {
            return None;
        }
        let deepest = |hits: &[(f32, egui::Rect)]| hits.iter().map(|(d, _)| *d).fold(0.0, f32::max);
        Some(format!(
            "{} shapes paint up to {:.1} points and {} widgets take clicks up to {:.1} points              into the Equation panel (top {top}): {painted:?} {clickable:?}",
            painted.len(),
            deepest(&painted),
            clickable.len(),
            deepest(&clickable),
        ))
    }

    /// How far above the Equation panel's top edge the lowest thing the centre region painted
    /// after the panel in the frame `output` ends [points]: under a row for a view that fills
    /// the space left (none of it lost to a region cut too short).
    fn gap_above_the_equation_panel(ctx: &egui::Context, output: &egui::FullOutput) -> f32 {
        let panel = equation_panel_rect(ctx);
        let lowest = painted_after_the_equation_panel(ctx, output)
            .into_iter()
            .filter(|seen| seen.x_range().intersects(panel.x_range()))
            .map(|seen| seen.bottom())
            .fold(f32::NEG_INFINITY, f32::max);
        panel.top() - lowest
    }

    /// Drags the Equation panel's top edge to `y` with frames of `frame` on `ctx`: a press 2
    /// points inside the panel's top (its resize handle, registered after the panel's
    /// contents, takes it there), a move, the release, then the pointer leaves (no tooltip
    /// over the frames that follow) and two idle frames let the panel settle.
    fn drag_equation_panel(
        ctx: &egui::Context,
        frame: &mut dyn FnMut(Vec<egui::Event>) -> egui::FullOutput,
        y: f32,
    ) {
        let panel = equation_panel_rect(ctx);
        let grab = egui::pos2(panel.center().x, panel.top() + 2.0);
        let to = egui::pos2(grab.x, y);
        frame(vec![egui::Event::PointerMoved(grab)]);
        frame(vec![primary_button(grab, true)]);
        frame(vec![egui::Event::PointerMoved(to)]);
        frame(vec![egui::Event::PointerMoved(to)]);
        frame(vec![primary_button(to, false)]);
        frame(vec![egui::Event::PointerGone]);
        frame(Vec::new());
        frame(Vec::new());
    }

    #[test]
    fn no_centre_view_paints_or_takes_clicks_in_the_equation_panel() {
        // A short page (1280 x 620, about a laptop browser's) and the laptop screen, with the
        // panel at its starting height and dragged as far up as it goes, and a page shorter
        // than the panel's floor plus the view's strip: every view stays above the panel's top
        // edge. Before the fix the callout list ran about 35 points into the panel on the short
        // page, and with the panel dragged tall the geometry, plot, results and clamp views
        // overflowed by up to 130.
        let short = egui::vec2(1280.0, 620.0);
        let mut failures = Vec::new();
        for (size, dragged) in [
            (short, false),
            (short, true),
            (SCREEN, false),
            (SCREEN, true),
            (egui::vec2(1280.0, 300.0), true),
        ] {
            let mut harness = Harness::on_screen(size);
            harness.panel.explorer.open_path("model.pullout_Nm");
            harness.frame(Vec::new());
            if dragged {
                let ctx = harness.ctx.clone();
                drag_equation_panel(&ctx, &mut |events| harness.frame(events), 0.0);
            }
            for view in CentreView::ALL {
                harness.panel.centre = view;
                let mut output = harness.frame(Vec::new());
                for _ in 0..2 {
                    output = harness.frame(Vec::new());
                }
                let case = format!("{size:?}, dragged up {dragged}, {}", view.label());
                if let Some(overlap) = equation_panel_overlap(&harness.ctx, &output) {
                    failures.push(format!("{case}: {overlap}"));
                }
                // The scrolling views fill the space left, down to the panel: their last text
                // shown ends less than a row (with its spacing) above it, or is cut by its
                // edge.
                if matches!(view, CentreView::Clamp | CentreView::Results) {
                    let gap = gap_above_the_equation_panel(&harness.ctx, &output);
                    if gap > 20.0 {
                        failures.push(format!("{case}: ends {gap} points above the panel"));
                    }
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
        assert!(
            failures.is_empty(),
            "{}",
            failures.join(
                "
"
            )
        );
    }

    #[test]
    fn inside_an_egui_window_no_centre_view_reaches_into_the_equation_panel() {
        // The linkage app's Tools window (M5) at its starting contents size, 1100 x 700: the
        // panel at its starting height, then dragged as far up as it goes. Every view stays
        // above the panel and the window keeps its size.
        const TITLE: &str = "Magnetic coupling";
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
            let input = egui::RawInput {
                events,
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
                ..Default::default()
            };
            ctx.run(input, |ctx| {
                egui::Window::new(TITLE)
                    .default_size([1100.0, 700.0])
                    .show(ctx, |ui| panel.ui(ui));
            })
        };
        // A window sizes itself on its first frame and paints on the next.
        frame(&mut panel, Vec::new());
        frame(&mut panel, Vec::new());
        panel.explorer.open_path("model.pullout_Nm");
        frame(&mut panel, Vec::new());
        frame(&mut panel, Vec::new());
        let window = || ctx.memory(|m| m.area_rect(egui::Id::new(TITLE)));
        let size = window().expect("the window is shown").size();
        let mut failures = Vec::new();
        for dragged in [false, true] {
            if dragged {
                drag_equation_panel(&ctx, &mut |events| frame(&mut panel, events), 0.0);
            }
            for view in CentreView::ALL {
                panel.centre = view;
                let mut output = frame(&mut panel, Vec::new());
                for _ in 0..2 {
                    output = frame(&mut panel, Vec::new());
                }
                if let Some(overlap) = equation_panel_overlap(&ctx, &output) {
                    failures.push(format!("dragged up {dragged}, {}: {overlap}", view.label()));
                }
                assert_eq!(window().unwrap().size(), size, "{}", view.label());
            }
        }
        assert!(
            failures.is_empty(),
            "{}",
            failures.join(
                "
"
            )
        );
    }

    #[test]
    fn the_equation_panel_starts_at_about_45_percent_and_its_drag_stops_short_of_the_view() {
        use crate::gui::explorer::VIEW_STRIP;
        // On the short page the panel opens at about 45 % of the centre region (measured from
        // the header's foot, so with the central panel's margin), not at a fixed 320 points.
        let size = egui::vec2(1280.0, 620.0);
        let mut harness = Harness::on_screen(size);
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        let header = header_rect(&harness.ctx);
        let opened = equation_panel_rect(&harness.ctx);
        let share = opened.height() / (opened.bottom() - header.bottom());
        assert!(
            (0.40..=0.50).contains(&share),
            "the panel opens {} points tall, {share:.3} of the centre region",
            opened.height()
        );
        // Dragged to the top of the screen, it stops with the view's strip above it.
        let ctx = harness.ctx.clone();
        drag_equation_panel(&ctx, &mut |events| harness.frame(events), 0.0);
        let dragged = equation_panel_rect(&harness.ctx);
        assert!(dragged.height() > opened.height(), "{dragged:?}");
        let strip = dragged.top() - header.bottom();
        assert!(
            (VIEW_STRIP..=VIEW_STRIP + 20.0).contains(&strip),
            "the centre keeps {strip} points above the panel"
        );
        // It stays resizable: dragged back down, its top edge follows the pointer.
        let y = size.y - 200.0;
        drag_equation_panel(&ctx, &mut |events| harness.frame(events), y);
        let lowered = equation_panel_rect(&harness.ctx);
        assert!((lowered.top() - y).abs() <= 1.0, "{lowered:?}");
    }

    #[test]
    fn a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window() {
        use crate::gui::explorer::{ELIDED, MAX_TRAIL, SHOWN_CRUMBS, plain_symbol};
        // A walk longer than the trail, down distinct equations.
        let paths: Vec<&str> = registry()
            .equations()
            .iter()
            .map(|eq| eq.target.as_str())
            .take(MAX_TRAIL + 5)
            .collect();
        let last = plain_symbol(
            &registry()
                .equation_for(paths[MAX_TRAIL + 4])
                .unwrap()
                .symbol,
        );
        // A laptop screen, a short one, and a narrow one: at 640 x 240 the side panels (320 +
        // 300 points) leave the centre region about 20 points, so the panel draws nothing to
        // read there and must only not panic or edit.
        let narrow = egui::vec2(640.0, 240.0);
        for size in [SCREEN, egui::vec2(1280.0, 240.0), narrow] {
            let mut harness = Harness::on_screen(size);
            harness.panel.explorer.open_path(paths[0]);
            for path in &paths[1..] {
                harness.panel.explorer.drill(path);
            }
            assert_eq!(harness.panel.explorer.trail().len(), MAX_TRAIL);
            let mut output = harness.frame(Vec::new());
            for _ in 0..3 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            if size != narrow {
                // The last crumbs after one ELIDED, each after its ">".
                assert!(drawn_texts(&output).contains(&last), "{size:?}");
                assert_eq!(count(&output, ELIDED), 1);
                assert_eq!(count(&output, ">"), SHOWN_CRUMBS);
                crate::gui::test_support::assert_glyphs(&harness.ctx, ELIDED, "the crumbs");
            }
        }
    }

    #[test]
    fn a_harmonic_set_outside_the_choices_lists_only_its_selector() {
        use crate::gui::explorer::term_label;
        // A design file's max_harmonic of 4 is no choice: every harmonic sum is NaN (decision
        // D3) and τ's term list keeps N_h, every harmonic left out.
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.max_harmonic = 4;
        let expected = harness.panel.inputs.clone();
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(matches!(
            harness.panel.results().get("model.tau_Pa"),
            Some(Value::Num(x)) if x.is_nan()
        ));
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t == term_label("coupling.max_harmonic"))
        );
        for n in [1, 3, 5, 7, 9, 11] {
            let path = format!("model.tau{n}_Pa");
            let label = term_label(&path);
            assert!(!texts.iter().any(|t| t == label), "{label}");
        }
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn an_undefined_value_shows_its_equation() {
        // A positive Hcj coefficient typed for manual magnets (no rating): the magnets reach no
        // hot limit, so the torque there takes the record's nan arm, drawn "undefined".
        let mut harness = Harness::new();
        let inputs = &mut harness.panel.inputs;
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.005;
        let expected = inputs.clone();
        harness
            .panel
            .explorer
            .open_path("temperature.demag.torque_at_limit_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(matches!(
            harness
                .panel
                .results()
                .get("temperature.demag.torque_at_limit_Nm"),
            Some(Value::Num(x)) if x.is_nan()
        ));
        assert_eq!(count(&output, "undefined"), 1);
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn a_result_without_an_equation_shows_its_value_and_cell() {
        use crate::engine::explain::TermKind;
        use crate::gui::explorer::NO_EQUATION;
        // Most results without an equation record have a workbook cell, a few are Rust-only:
        // each case must exist, so a change of the data cannot make the test vacuous.
        for has_cell in [true, false] {
            let entry = crate::gui::results_table::table_entries()
                .iter()
                .find(|e| {
                    registry().term_kind(&e.path) == Some(TermKind::CellOnly)
                        && e.info.cell.is_some() == has_cell
                })
                .unwrap_or_else(|| panic!("no result without an equation (cell: {has_cell})"));
            let mut harness = Harness::new();
            let shut = harness.frame(Vec::new());
            harness.panel.explorer.open_path(&entry.path);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, NO_EQUATION), 1);
            // The panel adds a line to what the views draw: the cell, or "Rust-only result".
            let cell = entry.info.cell.as_deref().unwrap_or("Rust-only result");
            assert_eq!(count(&output, cell), count(&shut, cell) + 1, "{cell}");
            // And the label with the value and its unit.
            let value = harness.panel.results().get(&entry.path).unwrap();
            let line = format!(
                "{} = {}",
                entry.info.meta.label,
                with_unit(format_value(&value), entry.info.meta.unit)
            );
            assert_eq!(count(&output, &line), 1, "{line:?}");
        }
    }

    #[test]
    fn the_open_equation_shows_its_label_cell_value_and_corrections() {
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        // Decision M43-14: the corrections the equation embodies, upstream of the pull-out.
        let value_line = format!("T_pull = {}", displayed_pullout(&DesignInputs::default()));
        for line in [
            "Pull-out torque at operating temperature (Calculator!C93)",
            value_line.as_str(),
            "Embodies corrections: E7, E8, E3",
        ] {
            assert_eq!(count(&output, line), 1, "{line:?}");
        }
    }

    #[test]
    fn the_assumptions_banner_appears_on_change_and_clears_on_reset() {
        let mut harness = Harness::new();
        let banner = |output: &egui::FullOutput| {
            drawn_texts(output)
                .into_iter()
                .filter(|t| t.starts_with(ASSUMPTIONS_MODIFIED))
                .collect::<Vec<_>>()
        };
        assert!(banner(&harness.frame(Vec::new())).is_empty());
        // The thermal conductance is an assumption and a Key design row.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus("temperature.thermal.conductance_W_K");
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The header sizes from the last frame: the banner line shows from the next one.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            banner(&output),
            [format!("{ASSUMPTIONS_MODIFIED}: Thermal conductance")]
        );
        harness.click_text(RESET_ASSUMPTIONS);
        let output = harness.frame(Vec::new());
        assert!(banner(&output).is_empty());
        let defaults = DesignInputs::default();
        assert_eq!(
            harness.panel.inputs().temperature.thermal.conductance_W_K,
            defaults.temperature.thermal.conductance_W_K
        );
        // The design input stays as it was.
        assert_eq!(harness.number(FACE_GAP), 1.41);
        assert!(!assumptions::any_modified(harness.panel.inputs()));
        // The reset is an edit: undo brings the assumption back.
        harness.panel.undo();
        assert!(assumptions::any_modified(harness.panel.inputs()));
    }

    #[test]
    fn the_assumptions_view_lists_each_assumption_with_its_rationale_and_source() {
        // Tall enough for all fourteen without scrolling.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
        harness.click_text(InputsView::Assumptions.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        for assumption in ASSUMPTIONS {
            for want in [
                assumption.label.to_owned(),
                assumption.rationale.to_owned(),
                format!("{SOURCE}: {}", assumption.source),
            ] {
                assert!(texts.contains(&want), "missing {want:?}");
            }
        }
        // Idle frames rewrite no assumption (the rows are the inputs' own).
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        harness.click_text(InputsView::Design.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
    }

    #[test]
    fn an_assumption_term_is_styled_in_the_equation_panel() {
        use crate::gui::explorer::{ASSUMPTION_TAG, DEPENDS_ON_MODIFIED};
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, ASSUMPTION_TAG), 1, "c_end is an assumption");
        harness.panel.inputs.coupling.c_end = 0.2;
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, &format!("{CHANGED_DOT} changed; {ASSUMPTION_TAG}")),
            1
        );
        assert_eq!(
            count(
                &output,
                &format!("{DEPENDS_ON_MODIFIED}: End-effect coefficient")
            ),
            1
        );
    }

    #[test]
    fn a_leaf_assumption_term_shows_its_row_in_the_assumptions_view() {
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        harness.frame(Vec::new());
        harness.click_text("End-effect coefficient");
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let focus = focus_rects(&output);
        // The assumption's heading and its row's label are both "End-effect coefficient";
        // the row's label sits inside the frame.
        let labels = crate::gui::test_support::text_rects(&output, "End-effect coefficient");
        assert!(
            labels
                .iter()
                .any(|l| l.left() < INPUTS_WIDTH && focus.iter().any(|r| r.contains_rect(*l))),
            "{labels:?} {focus:?}"
        );
    }

    #[test]
    fn a_sum_s_harmonic_set_marks_its_row_and_its_h_finds_it() {
        // σ = Σ_{n ∈ H} σ_n reads the harmonic set's selector: with σ open, the set's row in
        // the Assumptions view is framed in the selector's colour, as every term's row is; a
        // click on the H under the Σ shows that row and frames it, as a click on any leaf term
        // does. A tall screen draws every assumption's row.
        use crate::engine::explain::markup::IndexSet;
        let selector = IndexSet::Harmonics.selector();
        let label = InputCatalogue::get().entry(selector).unwrap().meta.label;
        let eq = registry().equation_for("model.tau_Pa").unwrap();
        let color = crate::gui::typeset::TermColors::of(eq)
            .get(selector)
            .expect("the set has a colour");
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.click_text(InputsView::Assumptions.label());
        let output = harness.frame(Vec::new());
        let row = text_rects(&output, label)
            .into_iter()
            .find(|r| r.left() < INPUTS_WIDTH)
            .expect("the set's row on the inputs side");
        let marks = mark_rects(&output, Some(color));
        assert!(
            marks.iter().any(|m| m.contains_rect(row)),
            "{row:?} not inside {marks:?}"
        );
        // Back on the design inputs, the H under the Σ in the Equation panel.
        harness.click_text(InputsView::Design.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
        let output = harness.frame(Vec::new());
        let h = text_rects(&output, "H")
            .into_iter()
            .find(|r| r.left() > INPUTS_WIDTH && r.right() < SCREEN.x - DASHBOARD_WIDTH)
            .expect("the H under the Σ");
        harness.click(h.center());
        assert_eq!(
            harness.panel.explorer.focus().map(|f| f.path.as_str()),
            Some(selector)
        );
        assert_eq!(harness.panel.explorer.trail(), ["model.tau_Pa"]);
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let row = text_rects(&output, label)
            .into_iter()
            .find(|r| r.left() < INPUTS_WIDTH)
            .expect("the set's row on the inputs side");
        let focus = focus_rects(&output);
        assert!(
            focus.iter().any(|r| r.contains_rect(row)),
            "{row:?} not inside {focus:?}"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    /// The screen rect of the first drawn text equal to `needle` and the rect it is clipped
    /// to.
    fn clipped_text(output: &egui::FullOutput, needle: &str) -> Option<(egui::Rect, egui::Rect)> {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) if text.galley.text() == needle => Some((
                    text.galley.rect.translate(text.pos.to_vec2()),
                    clipped.clip_rect,
                )),
                _ => None,
            })
    }

    #[test]
    fn the_term_list_scrolls_sideways_in_a_narrow_centre_region() {
        // At 1000 points the centre region is about 380 wide: σ's term list (swatch, symbol,
        // value, label, tag) is wider, so its last column, the tag, starts cut off or out of
        // view. The list scrolls sideways (as the equation above it does), so the tag can be
        // read in full.
        use crate::gui::explorer::ASSUMPTION_TAG;
        let mut harness = Harness::on_screen(egui::vec2(1000.0, 800.0));
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(
            clipped_text(&output, ASSUMPTION_TAG).is_none_or(|(r, clip)| r.right() > clip.right()),
            "the tag fits without scrolling: the test would prove nothing"
        );
        // The pointer on the start of the set's value, in view at the left of the list.
        let at = text_rect(&output, "5: 1, 3, 5 (workbook)")
            .expect("the set's value in the term list")
            .left_center()
            + egui::vec2(4.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(-5000.0, 0.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        // egui spreads a wheel step over several frames.
        let mut output = harness.frame(Vec::new());
        for _ in 0..20 {
            output = harness.frame(Vec::new());
        }
        let (tag, clip) = clipped_text(&output, ASSUMPTION_TAG).expect("the tag is drawn");
        assert!(clip.contains_rect(tag), "{tag:?} cut by {clip:?}");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn closing_the_equation_panel_clears_the_input_row_s_frame() {
        // Decision M43-12: a leaf term frames its row while its equation is shown; closing
        // the panel, by its Close button or the header's toggle, takes the frame away.
        use crate::gui::explorer::CLOSE;
        for close in [CLOSE, EQUATION_PANEL] {
            let mut harness = Harness::new();
            harness
                .panel
                .explorer
                .open_path("model.gearbox_input_ripple_Nm");
            harness.frame(Vec::new());
            harness.click_text("Gearbox ratio");
            let mut output = harness.frame(Vec::new());
            for _ in 0..3 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(focus_rects(&output).len(), 1, "{close}: the row is framed");
            harness.click_text(close);
            assert!(!harness.panel.explorer.open, "{close}");
            assert_eq!(harness.panel.explorer.focus(), None, "{close}");
            // The inputs side paints before the dock: the frame goes on the next frame.
            let output = harness.frame(Vec::new());
            assert!(focus_rects(&output).is_empty(), "{close}");
            // Shown again, the panel keeps its equation but frames no row.
            harness.click_text(EQUATION_PANEL);
            assert!(harness.panel.explorer.open);
            assert_eq!(
                harness.panel.explorer.trail(),
                ["model.gearbox_input_ripple_Nm"]
            );
            let output = harness.frame(Vec::new());
            assert!(focus_rects(&output).is_empty(), "{close}: shown again");
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }

    /// Every painted shape that is a diagram's: the frames of the term marks and the panel's
    /// own shapes aside, a diagram paints filled circles of radius 3.5 (its marked points).
    fn diagram_points(output: &egui::FullOutput) -> usize {
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter(|s| matches!(s, egui::Shape::Circle(c) if c.radius == 3.5))
            .count()
    }

    #[test]
    fn the_explain_toggle_shows_the_open_equation_s_note_and_follows_it() {
        use crate::engine::explain::notes::note;
        use crate::gui::explorer::{EXPLAIN, NO_NOTE};
        let mut harness = Harness::on_screen(TALL);
        // The dashboard's Why links name notes too: the Equation panel's note is one more.
        let pullout = note("pullout_angle").unwrap();
        let end = note("end_effect").unwrap();
        let output = harness.frame(Vec::new());
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
        assert!(
            pullout_links >= 1 && end_links >= 1,
            "the dashboard links both"
        );
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, pullout.title),
            pullout_links,
            "hidden by default"
        );
        harness.click_text(EXPLAIN);
        assert!(harness.panel.explorer.explain);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), pullout_links + 1);
        assert_eq!(count(&output, pullout.sentences[0]), 1);
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end (its term row is below the note now): the note follows the equation.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, end.title), end_links + 1);
        assert_eq!(count(&output, pullout.title), pullout_links);
        // An equation no reviewed note explains.
        harness.panel.explorer.open_path("mass.total_g");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 1);
        // Off again: no note.
        harness.click_text(EXPLAIN);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 0);
    }

    #[test]
    fn start_here_opens_the_matching_equations_in_turn() {
        use crate::engine::explain::notes::{START_HERE, note};
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        let mut harness = Harness::new();
        harness.click_text(EQUATION_PANEL);
        harness.click_text(START_HERE_BUTTON);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[0].1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &start_here_text(0)), 1);
        assert_eq!(count(&output, note(START_HERE[0].0).unwrap().title), 1);
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[2].1));
        harness.click_text(PREVIOUS);
        assert_eq!(harness.panel.explorer.start_here(), Some(1));
        harness.click_text(STOP);
        assert_eq!(harness.panel.explorer.start_here(), None);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn every_start_here_step_shows_with_glyphs_in_the_default_fonts() {
        // The bar names each step's note (one title holds ∝, decision M43-6); it sits above
        // the panel's scroll area, so it is drawn whatever the panel's height.
        use crate::engine::explain::notes::{START_HERE, note};
        use crate::gui::explorer::start_here_text;
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::new();
        for (step, &(id, _)) in START_HERE.iter().enumerate() {
            harness.panel.explorer.start(step);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            for text in drawn_texts(&output) {
                crate::gui::test_support::assert_glyphs(&harness.ctx, &text, id);
            }
            let bar = start_here_text(step);
            assert_eq!(count(&output, &bar), 1, "{id}: {bar:?}");
            assert!(
                bar.ends_with(&glyph_safe(note(id).unwrap().title)),
                "{id}: {bar:?}"
            );
        }
    }

    #[test]
    fn every_teaching_note_shows_with_glyphs_in_the_default_fonts() {
        use crate::engine::explain::notes::NOTES;
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        for note in NOTES {
            harness.panel.explorer.open_note(note.id);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let texts = drawn_texts(&output);
            let title = crate::gui::typeset::glyph_safe(note.title);
            assert!(texts.contains(&title), "{}", note.id);
            for text in &texts {
                crate::gui::test_support::assert_glyphs(&harness.ctx, text, note.id);
            }
        }
    }

    #[test]
    fn the_part_picker_sets_a_library_part_or_custom_dimensions() {
        use crate::gui::pickers::{CUSTOM, part_label};
        let mut harness = Harness::new();
        let b842sh = part_label(crate::engine::library::lookup("B842SH").unwrap());
        // The inner part's picker (the Key design row, drawn first).
        harness.click_text(&b842sh);
        harness.click_text(CUSTOM);
        assert_eq!(harness.panel.inputs().coupling.magnets.part_inner, "");
        let output = harness.frame(Vec::new());
        assert!(count(&output, CUSTOM) >= 1);
        // Back to a library part, from the same picker.
        harness.click_text(CUSTOM);
        let b842 = part_label(crate::engine::library::lookup("B842").unwrap());
        harness.click_text(&b842);
        assert_eq!(harness.panel.inputs().coupling.magnets.part_inner, "B842");
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.part_inner = "B842".to_owned();
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn the_outer_ring_s_part_picker_sets_the_outer_part_only() {
        use crate::engine::library::lookup;
        use crate::gui::pickers::{CUSTOM, part_label};
        let mut harness = Harness::new();
        // Both rings read B842SH: the Key design draws the inner ring's picker, then the
        // outer ring's.
        let b842sh = part_label(lookup("B842SH").unwrap());
        let output = harness.frame(Vec::new());
        let pickers = text_rects(&output, &b842sh);
        assert_eq!(pickers.len(), 2, "one picker per ring");
        assert!(pickers[0].bottom() <= pickers[1].top(), "{pickers:?}");
        harness.click(pickers[1].center());
        harness.click_text(CUSTOM);
        let magnets = &harness.panel.inputs().coupling.magnets;
        assert_eq!(magnets.part_outer, "");
        assert_eq!(magnets.part_inner, "B842SH");
        // Back to a library part, from the outer ring's picker (the inner ring's is not blank,
        // so the blank choice is drawn once).
        harness.click_text(CUSTOM);
        harness.click_text(&part_label(lookup("B861").unwrap()));
        let magnets = &harness.panel.inputs().coupling.magnets;
        assert_eq!(magnets.part_outer, "B861");
        assert_eq!(magnets.part_inner, "B842SH");
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.part_outer = "B861".to_owned();
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn the_grade_pickers_set_a_grade_by_its_id_or_none_for_either_ring() {
        use crate::engine::grades::{GRADES, grade};
        use crate::gui::pickers::{BLANK_GRADE, grade_label};
        // A grade whose id is not its display name: the input holds the id.
        let by_id = GRADES
            .iter()
            .find(|g| g.id != g.name)
            .expect("a grade named other than its id");
        let other = GRADES
            .iter()
            .find(|g| g.id != by_id.id && g.id != "Y30")
            .unwrap();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        // The drop-down scrolls past 200 points; its 17 grades all in view, the one picked
        // among them (the named grades come last) is drawn.
        harness
            .ctx
            .style_mut(|style| style.spacing.combo_height = 2000.0);
        // The outer ring starts on a grade of its own, so the blank choice is drawn once.
        harness.panel.inputs.coupling.magnets.grade_outer = "Y30".to_owned();
        // Both grade rows are in the Magnets and rings group.
        harness.click_text("Magnets and rings");
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let magnets = |harness: &Harness| harness.panel.inputs().coupling.magnets.clone();
        // The inner ring's picker: a grade, then back to blank.
        harness.click_text(BLANK_GRADE);
        harness.click_text(&grade_label(by_id));
        assert_eq!(magnets(&harness).grade_inner, by_id.id);
        assert_eq!(magnets(&harness).grade_outer, "Y30");
        harness.click_text(&grade_label(by_id));
        harness.click_text(BLANK_GRADE);
        assert_eq!(magnets(&harness).grade_inner, "");
        assert_eq!(magnets(&harness).grade_outer, "Y30");
        // The outer ring's picker, from its own grade to another.
        harness.click_text(&grade_label(grade("Y30").unwrap()));
        harness.click_text(&grade_label(other));
        assert_eq!(magnets(&harness).grade_outer, other.id);
        assert_eq!(magnets(&harness).grade_inner, "");
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.grade_outer = other.id.to_owned();
        assert_eq!(harness.panel.inputs(), &expected);
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn a_material_choice_sums_up_its_properties_and_fires_its_warnings() {
        use crate::gui::dashboard::WARNINGS_HEADING;
        use crate::gui::pickers::{material_of, material_summary};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text("Materials");
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let steel = material_of("materials.parts.back_iron", 1).unwrap();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &material_summary(steel)), 1);
        assert_eq!(count(&output, WARNINGS_HEADING), 0);
        harness.click_text("4140 annealed");
        harness.click_text("304 stainless (non-magnetic)");
        assert_eq!(harness.panel.inputs().materials.parts.back_iron, 7);
        let output = harness.frame(Vec::new());
        let stainless = material_of("materials.parts.back_iron", 7).unwrap();
        assert_eq!(count(&output, &material_summary(stainless)), 1);
        assert_eq!(count(&output, WARNINGS_HEADING), 1);
    }

    #[test]
    fn a_material_code_outside_the_choices_is_shown_and_kept() {
        // A design file's back iron code 99 is no choice: the drop-down says so, the row sums
        // up no material, and idle frames write nothing back.
        use crate::gui::pickers::{MATERIAL_PICKERS, material_of, material_summary};
        let summaries: Vec<String> = MATERIAL_PICKERS
            .iter()
            .flat_map(|(path, choices)| {
                choices
                    .iter()
                    .filter_map(|&(code, _)| material_of(path, code))
            })
            .map(material_summary)
            .collect();
        let drawn_summaries = |output: &egui::FullOutput| {
            drawn_texts(output)
                .iter()
                .filter(|t| summaries.contains(t))
                .count()
        };
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text("Materials");
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let output = harness.frame(Vec::new());
        assert_eq!(drawn_summaries(&output), MATERIAL_PICKERS.len());
        harness.panel.inputs.materials.parts.back_iron = 99;
        let mut expected = DesignInputs::default();
        expected.materials.parts.back_iron = 99;
        let mut output = harness.frame(Vec::new());
        for _ in 0..2 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(count(&output, "99 (not a choice)"), 1);
        assert_eq!(drawn_summaries(&output), MATERIAL_PICKERS.len() - 1);
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn a_warning_shows_in_its_colour_and_links_to_its_note() {
        use crate::engine::warnings::WARNING_RULES;
        use crate::gui::dashboard::WHY;
        let mut harness = Harness::new();
        harness.panel.inputs.materials.parts.back_iron = 7;
        let output = harness.frame(Vec::new());
        let rule = &WARNING_RULES[0];
        assert_eq!(
            crate::gui::test_support::text_color(&output, rule.text),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        let caution = &WARNING_RULES[5];
        assert_eq!(
            crate::gui::test_support::text_color(&output, caution.text),
            Some(Level::Caution.color(&harness.ctx.style().visuals))
        );
        let note = crate::engine::explain::notes::note(rule.note_id).unwrap();
        harness.click_text(&format!(
            "{WHY}: {}",
            crate::gui::typeset::glyph_safe(note.title)
        ));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.note(), Some(rule.note_id));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, note.sentences[0]), 1);
        assert!(
            diagram_points(&output) == 0,
            "the flux-path diagram marks no point"
        );
        // Its one equation is a link: the circuit in effect, opened like a value.
        harness.click_text("circuit");
        assert_eq!(harness.panel.explorer.note(), None);
        assert_eq!(
            harness.panel.explorer.trail(),
            ["materials.circuit_backiron"]
        );
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app. Past its first frames: a value
        // opened, a term followed, a leaf assumption term's row shown (the Assumptions view
        // and its frame), then idle frames. The window keeps its size, the Equation panel
        // draws its lists, egui reports no ID clash (the text it paints in a debug build, as
        // the tests are) and the design is unchanged.
        use crate::gui::explorer::{TERMS, USED_BY};
        use crate::gui::guidance::{design_status, design_status_text};
        const TITLE: &str = "Magnetic coupling";
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        // The Assumptions view is the inputs side of every view but Size and torque (which
        // folds the inputs under "Detailed inputs"; a new panel opens there).
        panel.centre = CentreView::Geometry;
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
            let input = egui::RawInput {
                events,
                ..Default::default()
            };
            let output = ctx.run(input, |ctx| {
                egui::Window::new(TITLE)
                    .default_size([1100.0, 1200.0])
                    .show(ctx, |ui| panel.ui(ui));
            });
            let clashes: Vec<String> = drawn_texts(&output)
                .into_iter()
                .filter(|t| t.contains("use of") && t.contains(" ID "))
                .collect();
            assert!(clashes.is_empty(), "{clashes:?}");
            output
        };
        let click = |panel: &mut MagcouplingPanel, output: &egui::FullOutput, text: &str| {
            let at = text_rect(output, text)
                .unwrap_or_else(|| panic!("no text {text:?}"))
                .center();
            frame(panel, vec![egui::Event::PointerMoved(at)]);
            frame(panel, vec![primary_button(at, true)]);
            frame(panel, vec![primary_button(at, false)]);
            frame(panel, Vec::new())
        };
        // A window sizes itself on its first frame and paints on the next; the check guides
        // wrap to the dashboard's width as it settles, which a third frame takes in.
        frame(&mut panel, Vec::new());
        frame(&mut panel, Vec::new());
        let output = frame(&mut panel, Vec::new());
        // The dashboard's top in view (its lower rows scroll, under the check guides): the
        // status line and the first headline value.
        let defaults = DesignInputs::default();
        let status = design_status_text(&design_status(&defaults, &compute_all(&defaults)));
        assert_eq!(count(&output, &status), 1);
        assert!(count(&output, &displayed_pullout(&defaults)) >= 1);
        let window = || ctx.memory(|m| m.area_rect(egui::Id::new(TITLE)));
        let size = window().expect("the window is shown").size();
        let output = click(
            &mut panel,
            &output,
            &displayed_pullout(&DesignInputs::default()),
        );
        assert_eq!(panel.explorer.trail(), ["model.pullout_Nm"]);
        let output = click(&mut panel, &output, "End-effect factor");
        assert_eq!(panel.explorer.trail(), ["model.pullout_Nm", "model.f_end"]);
        click(&mut panel, &output, "End-effect coefficient");
        assert_eq!(
            panel.explorer.focus().map(|f| f.path.as_str()),
            Some("coupling.c_end")
        );
        let mut output = frame(&mut panel, Vec::new());
        for _ in 0..5 {
            output = frame(&mut panel, Vec::new());
        }
        assert_eq!(panel.inputs_view, InputsView::Assumptions);
        assert_eq!(focus_rects(&output).len(), 1, "the row is framed");
        assert_eq!(window().map(|r| r.size()), Some(size), "the window's size");
        for heading in [TERMS, USED_BY] {
            assert_eq!(count(&output, heading), 1, "{heading}");
        }
        assert!(assumptions::modified(panel.inputs()).is_empty());
        assert_eq!(panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::guidance::{all_fixes, fix_text};
        // Every fix of the workbook example's red checks, from the Geometry view (the input's
        // group or the Assumptions view opens) and from Size and torque (its Detailed inputs
        // open). Tall enough for every dashboard row.
        let mut harness = Harness::on_screen(TALL);
        // The dashboard settles its width on the second frame, and the guides above a link wrap
        // to it: click where the link rests.
        harness.frame(Vec::new());
        let defaults = DesignInputs::default();
        let fixes: std::collections::BTreeSet<(&str, String)> =
            all_fixes(&defaults, &compute_all(&defaults))
                .into_iter()
                .flatten()
                .map(|(path, dir)| (path, fix_text(path, dir)))
                .collect();
        assert!(!fixes.is_empty(), "the red checks have fixes");
        for view in [CentreView::Geometry, CentreView::SizeAndTorque] {
            for (path, text) in &fixes {
                // No frame left from the last fix.
                harness.panel.explorer.close();
                harness.panel.centre = view;
                harness.click_text(text);
                assert_eq!(
                    harness.panel.explorer.focus().map(|f| f.path.as_str()),
                    Some(*path),
                    "{view:?}"
                );
                assert!(
                    !harness.panel.explorer.open,
                    "the Equation panel stays shut"
                );
                assert_eq!(harness.panel.centre, view, "the view stays");
                let mut output = harness.frame(Vec::new());
                for _ in 0..10 {
                    output = harness.frame(Vec::new());
                }
                let label = label_for(path);
                let rows: Vec<egui::Rect> = text_rects(&output, label)
                    .into_iter()
                    .filter(|r| r.left() < INPUTS_WIDTH)
                    .collect();
                let focus = focus_rects(&output);
                assert!(
                    rows.iter()
                        .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
                    "{view:?} {path}: {rows:?} {focus:?}"
                );
            }
        }
        // Following a fix edits nothing: no input, no undo step.
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing() {
        use crate::gui::guidance::{Direction, all_fixes, fix_text};
        use crate::gui::size_controls::DIAMETER;
        // Contract C-4: the ring radius's fix bears the Size tab's name, and its link opens
        // that tab, where the Outside diameter slider is, instead of a row. At 12 poles the
        // blocks do not fit, and the fit's first fix raises the Outside diameter.
        let inputs = with_poles(12);
        let ring = "coupling.inner_back_apothem_mm";
        assert!(
            all_fixes(&inputs, &compute_all(&inputs))
                .iter()
                .flatten()
                .any(|&fix| fix == (ring, Direction::Raise))
        );
        let text = fix_text(ring, Direction::Raise);
        assert_eq!(text, format!("Raise {DIAMETER}"));
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs = inputs.clone();
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        let undo = harness.panel.history.undo_len();
        assert_eq!(harness.panel.centre, CentreView::Geometry);
        harness.click_text(&text);
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.explorer.focus(), None, "no row is asked for");
        assert!(!harness.panel.explorer.open);
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &inputs);
        assert_eq!(harness.panel.history.undo_len(), undo);
    }

    #[test]
    fn a_raised_minimum_od_takes_lower_outside_diameter_off_the_dashboard() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::guidance::{Direction, all_fixes, fix_text};
        // The space claim of a 12 mm ring (the cup sets the diameter, past its claim) offers
        // "Lower Outside diameter"; with the Minimum OD at the diameter the slider cannot go
        // down, so the link goes and the claim's other fixes stay.
        let mut inputs = crate::gui::test_support::passing_design();
        inputs.coupling.inner_back_apothem_mm = 12.0;
        let ring = "coupling.inner_back_apothem_mm";
        let all = all_fixes(&inputs, &compute_all(&inputs));
        let lower = fix_text(ring, Direction::Lower);
        let others: Vec<String> = all
            .iter()
            .flatten()
            .filter(|&&fix| fix != (ring, Direction::Lower))
            .map(|&(path, dir)| fix_text(path, dir))
            .collect();
        assert!(all.iter().flatten().any(|&f| f == (ring, Direction::Lower)));
        assert!(!others.is_empty(), "the claim has another fix");
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs = inputs.clone();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &lower), 1);
        harness.panel.minimum_diameter_mm = Some(dimensions(&inputs).unwrap().diameter_mm);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &lower), 0);
        for other in &others {
            assert_eq!(count(&output, other), 1, "{other}");
        }
        assert_eq!(harness.panel.inputs(), &inputs);
    }

    #[test]
    fn a_check_s_why_link_opens_its_note_in_the_equation_panel() {
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
        let notes: Vec<&'static str> = CHECK_GUIDES
            .iter()
            .filter_map(|g| match g.learn {
                Learn::Note(id) => Some(id),
                _ => None,
            })
            .collect();
        assert!(!notes.is_empty(), "a guide links a reviewed note");
        let mut harness = Harness::on_screen(TALL);
        // Click where the links rest (the dashboard settles its width on the second frame).
        harness.frame(Vec::new());
        for id in notes {
            let note = crate::gui::explorer::reviewed_note(id).expect("a guide's note is reviewed");
            harness.panel.explorer.close();
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
            assert!(harness.panel.explorer.open, "{id}");
            assert_eq!(harness.panel.explorer.note(), Some(id));
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, note.sentences[0]), 1, "{id}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_status_line_counts_the_checks_at_default_and_after_a_fix() {
        use crate::gui::guidance::{design_status, design_status_text};
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        // The workbook example fails its hot minimum, its clearance and its cup wall.
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (3, 0, 8, true)
        );
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // A fix: the cup wall at 2.4 mm (its rule asks for 2.0 mm and up).
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        assert_eq!((status.bad, status.is_default), (2, false));
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // Reset all: the example again.
        harness.click_text(RESET_ALL);
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        assert_eq!((status.bad, status.is_default), (3, true));
        assert_eq!(count(&output, &design_status_text(&status)), 1);
    }

    #[test]
    fn idle_frames_compute_the_fixes_once_and_an_edit_again() {
        // Each fix runs compute_all on the design its move reaches: the panel's cache computes
        // them once per design shown.
        let mut harness = Harness::new();
        let computed = harness.panel.fixes.computed;
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.fixes.computed, computed);
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.fixes.computed, computed + 1);
    }

    #[test]
    fn a_narrow_window_wraps_the_check_guides_inside_the_dashboard() {
        // The M4-1 review's ~930 px window, tall enough for every row: every guide line of the
        // workbook example's red checks and the status line wrap inside the dashboard, never past
        // its right margin (a line past it would widen the dashboard into the centre region).
        use crate::gui::guidance::{
            CHECK_GUIDES, all_fixes, design_status, design_status_text, fix_text, level,
        };
        let width = 930.0;
        let mut harness = Harness::on_screen(egui::vec2(width, 3000.0));
        // The dashboard settles its width on the second frame.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let inputs = DesignInputs::default();
        let results = harness.panel.results().clone();
        let fixes = all_fixes(&inputs, &results);
        let mut texts = vec![design_status_text(&design_status(&inputs, &results))];
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
            if level(guide, &inputs, &results) == Level::Bad {
                texts.extend((guide.problem)(&inputs, &results));
                texts.push(guide.why.to_owned());
                texts.extend(fixes.iter().map(|&(path, dir)| fix_text(path, dir)));
            }
        }
        // Left of its first label's badge; right, the side panel's margin inside the frame's
        // (8 points each), give or take egui's rounding of a widget's edges (a line that wraps
        // to the full width can end one rounding step past it).
        let pullout = label_for("model.pullout_Nm");
        let dashboard = text_rect(&output, pullout).unwrap().left() - 30.0
            ..=width - 16.0 + egui::emath::GUI_ROUNDING;
        for text in &texts {
            let rects = text_rects(&output, text);
            assert!(!rects.is_empty(), "{text:?} is drawn");
            for rect in rects {
                assert!(
                    dashboard.contains(&rect.left()) && dashboard.contains(&rect.right()),
                    "{text:?} at {rect:?}, the dashboard {dashboard:?}"
                );
            }
        }
    }
}
