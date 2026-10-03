//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: the geometry view first, the results table
//! last; decisions M42-1 and M42-2). Every result is recomputed with every approved correction
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
//! Sizing (spec Addendum A1): the mode switch tops the Key design group. In Torque → Magnets
//! the panel shows the design with the free variable at the solved value
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::format::search_needle;
use crate::gui::geometry_view::geometry_ui;
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
    Design, LoadError, PUBLIC_BASE_URL, decode_share_payload, design_from_json, design_to_json,
    share_link,
};
use crate::gui::sizing::{
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The session buttons.
pub const UNDO: &str = "Undo";
pub const REDO: &str = "Redo";
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

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

/// The label of the target torque slider (Torque → Magnets).
pub const TARGET_LABEL: &str = "Target hot-low torque";

/// The label of the free-variable picker.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The start of the log line of each sizing outcome (the web smoke test looks for it).
pub const SIZING_LOG_PREFIX: &str = "magcoupling sizing: ";

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";

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

/// What the panel asks of its host: the platform work an egui panel cannot do itself.
#[derive(Clone, Debug, PartialEq)]
pub enum PanelRequest {
    /// Save `contents` to a file the user picks (native) or download it (web).
    SaveFile {
        /// The suggested file name.
        file_name: String,
        /// The media type (the web download's Blob type).
        mime: &'static str,
        contents: String,
    },
    /// Let the user pick a design file, then hand its text to
    /// [`MagcouplingPanel::load_design_file`].
    OpenDesign,
}

/// The views of the centre region, one tab row (decision M42-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreView {
    /// The end view and the side view to scale, with the dimension callouts (the default view:
    /// decision M42-2).
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
    pub const ALL: [CentreView; 8] = [
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
    /// The sizing mode, free variable and target torque.
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
    /// Whether the panel acts on the undo and redo keys
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]).
    keyboard_shortcuts: bool,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
    /// a design file or a share link.
    input_order: InputOrder,
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
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Geometry,
            inputs_view: InputsView::Design,
            input_order: InputOrder::default(),
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
            SizingMode::TorqueToMagnets => self.runner.shown(&self.inputs, &self.sizing),
        }
    }

    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
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
        self.load(design_from_json(text), "Design file loaded")
    }

    /// Loads the design a share link's `?m=` value holds; on refusal the design is unchanged
    /// and the panel shows why.
    pub fn load_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load(
            decode_share_payload(payload),
            "Design loaded from the share link",
        )
    }

    /// Opens `design` as the session's start: the design is replaced and the undo history
    /// starts there, so the first Undo does not throw it away. Every other replacement of the
    /// design (a loaded file, reset all) is an edit and can be undone.
    pub fn open_design(&mut self, design: Design) {
        self.history = History::new(design.clone());
        self.set_design(design);
    }

    /// Opens the design a share link's `?m=` value holds as the session's start
    /// ([`MagcouplingPanel::open_design`]: the web app's link at start-up); on refusal the
    /// design and the history are unchanged and the panel shows why.
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load_with(
            decode_share_payload(payload),
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
    fn load(&mut self, design: Result<Design, LoadError>, done: &str) -> Result<(), LoadError> {
        self.load_with(design, done, Self::set_design)
    }

    /// Applies a loaded design with `apply` (an edit, or a new start) and says `done`; on
    /// refusal changes nothing and shows why.
    fn load_with(
        &mut self,
        design: Result<Design, LoadError>,
        done: &str,
        apply: fn(&mut Self, Design),
    ) -> Result<(), LoadError> {
        match design {
            Ok(design) => {
                apply(self, design);
                self.status = Some(done.to_owned());
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
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results, &mut readouts));
                });
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        let events = readouts.finish();
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
        if let Some(wait) = self
            .runner
            .update(&self.inputs, &self.sizing, now, self.editing(ui))
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
            self.runner.status(&self.inputs, &self.sizing)
        );
    }

    /// The sizing controls at the top of the Key design group: the mode switch, then in
    /// Torque → Magnets the free variable, the target torque and the outcome.
    fn sizing_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let mut mode = self.sizing.mode;
        ui.horizontal(|ui| {
            for choice in SizingMode::ALL {
                ui.selectable_value(&mut mode, choice, choice.label());
            }
        });
        if mode != self.sizing.mode {
            if mode == SizingMode::MagnetsToTorque {
                // Decision M41-7: leaving inverse sizing keeps the value it shows, solved for
                // this design: a change still waiting for its debounce is solved first.
                if !self.runner.is_current(&self.inputs, &self.sizing) {
                    self.runner.solve_now(&self.inputs, &self.sizing);
                    self.log_sizing();
                }
                self.inputs = self.runner.shown(&self.inputs, &self.sizing);
            }
            self.sizing.mode = mode;
        }
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        egui::ComboBox::from_label(FREE_VARIABLE_LABEL)
            .selected_text(variable_label(self.sizing.variable))
            .show_ui(ui, |ui| {
                for variable in FreeVariable::ALL {
                    ui.selectable_value(
                        &mut self.sizing.variable,
                        variable,
                        variable_label(variable),
                    );
                }
            });
        let entry = InputCatalogue::get()
            .entry(TARGET_RANGE_INPUT)
            .expect("the target's metadata input exists");
        let range = entry.meta.range.expect("the hot minimum has a slider");
        ui.horizontal(|ui| {
            ui.label(TARGET_LABEL).on_hover_text(
                "The hot-low torque with production variation (metal.torque_hot_low_Nm) the free variable must reach",
            );
            let mut target = self.sizing.target_Nm;
            let response = ui.add(slider(&mut target, entry.meta, range));
            // Its value box is a text field of the design, as an input row's is.
            self.design_widgets.push(response.id);
            if response.changed() && target != self.sizing.target_Nm {
                self.sizing.target_Nm = target;
            }
        });
        let status = self.runner.status(&self.inputs, &self.sizing);
        let color = if status.starts_with(SOLVED_PREFIX) {
            Level::Good.color(ui.visuals())
        } else if status == SOLVING {
            ui.visuals().weak_text_color()
        } else {
            Level::Bad.color(ui.visuals())
        };
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

    /// The centre region: the view tabs (wrapping when the region is narrow), the end-effect
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown, its values readouts (`readouts`).
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // The Equation panel docks at the bottom of the centre region (decision M43-1).
        if self.explorer.open {
            egui::TopBottomPanel::bottom("magcoupling_equation_panel")
                .resizable(true)
                .default_height(PANEL_HEIGHT)
                .show_inside(ui, |ui| {
                    explorer_ui(ui, &mut self.explorer, shown, &self.results);
                });
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
        match self.centre {
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
                        contents: results_csv(&self.results),
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
                        ),
                    }),
                    None => {}
                }
            }
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
                    contents: design_to_json(&design),
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
        let modified = assumptions::modified(&self.inputs);
        let banner = (!modified.is_empty()).then(|| {
            let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
            format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
        });
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

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen.
    /// A leaf term clicked in the Equation panel shows the view that holds its row.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
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
        if let Some(banner) = self.trace.as_ref().map(Trace::banner) {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().selection.stroke.color, banner);
                if ui.small_button(CLEAR_TRACE).clicked() {
                    self.trace = None;
                }
            });
        }
        ui.separator();
        match self.inputs_view {
            InputsView::Design => self.design_inputs_ui(ui, readouts),
            InputsView::Assumptions => self.assumptions_ui(ui, readouts),
        }
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
                            for section in group.sections.iter().filter(|s| !s.advanced) {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.sections.iter().any(|s| s.advanced) {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.sections.iter().filter(|s| s.advanced)
                                        {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
                            }
                        });
                }
            });
    }

    /// One section of a group: its heading (none for the group's own section), then its rows.
    fn section_ui(
        &mut self,
        ui: &mut egui::Ui,
        group: &InputGroup,
        section: &'static InputSection,
        readouts: &Readouts,
    ) {
        if section.id != group.name {
            ui.add_space(4.0);
            ui.strong(section.label);
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
    /// Magnets the free variable's row shows the value the panel shows, locked. The row is
    /// framed in its term's colour while the equation in view reads the input (`readouts`).
    fn input_row_ui(
        &mut self,
        ui: &mut egui::Ui,
        entry: &'static InputEntry,
        readouts: &Readouts,
    ) -> egui::Id {
        let locked = self.sizing.mode == SizingMode::TorqueToMagnets
            && entry.path == self.sizing.variable.path();
        // The locked row reads the design shown (one clone, for that row only).
        let current = if locked {
            self.shown_inputs().get(&entry.path)
        } else {
            self.inputs.get(&entry.path)
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let row = ui.add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed));
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
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{BLANK_TEXT, CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::session::encode_share_payload;
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_rect, text_rects,
    };
    use crate::headline;

    const FACE_GAP: &str = "metal.face_gap_mm";
    const POLES: &str = "coupling.npole";
    const AXIAL_LENGTH: &str = "coupling.magnets.axial_length_mm";
    const PART_INNER: &str = "coupling.magnets.part_inner";
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";

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

        /// A harness on a screen of `size`.
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
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

        /// Switches to Torque \u{2192} Magnets and lets the solve run.
        fn size(&mut self) -> egui::FullOutput {
            self.click_text(SizingMode::TorqueToMagnets.label());
            self.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            self.frame(Vec::new());
            self.frame(Vec::new())
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

    #[test]
    fn the_geometry_view_is_the_default_and_follows_the_design_shown() {
        let mut harness = Harness::new();
        assert_eq!(harness.panel.centre, CentreView::Geometry);
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
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        for (_, path) in HEADLINE {
            let label = result_info(path).unwrap().meta.label;
            assert!(texts.iter().any(|t| t == label), "missing {label:?}");
        }
        for entry in &InputCatalogue::get().key_design {
            assert!(
                texts.iter().any(|t| t == entry.meta.label),
                "missing input {:?}",
                entry.meta.label
            );
        }
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
        let mut harness = Harness::new();
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
        let mut harness = Harness::new();
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
                            texts.iter().any(|t| t == entry.meta.label),
                            "{}: missing {:?}",
                            group.name,
                            entry.meta.label
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
        let mut harness = Harness::new();
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
        let mut harness = Harness::new();
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
        let gap = InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label;
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
        let pullout_label = result_info("model.pullout_Nm").unwrap().meta.label;
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
            let row = text_rects(&output, entry.meta.label)
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
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(POLES).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(POLES).unwrap().meta.label);
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
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
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
                    contents: results_csv(&results),
                },
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.json".to_owned(),
                    mime: "application/json",
                    contents: results_json(&harness.panel.design(), &results),
                },
            ]
        );
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
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
                    contents: design_to_json(&gap_design(1.41)),
                },
                PanelRequest::OpenDesign,
            ]
        );
    }

    #[test]
    fn a_loaded_design_file_replaces_the_design_and_can_be_undone() {
        let mut harness = Harness::new();
        let mut loaded = gap_design(2.5);
        loaded.sizing.target_Nm = 3.0;
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
            .set_share_base("http://localhost:8080/magcoupling/");
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
            .strip_prefix("http://localhost:8080/magcoupling/?m=")
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
    fn torque_to_magnets_shows_the_solved_design() {
        let mut harness = Harness::new();
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
        harness.click_text(SizingMode::TorqueToMagnets.label());
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
        harness.panel.sizing.target_Nm = 3.0;
        assert_eq!(count(&harness.frame(Vec::new()), SOLVING), 1);
        harness.click_text(SizingMode::MagnetsToTorque.label());
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.runner.solves, 2, "solved on leaving");
        let Ok(SizingOutcome::Solved(point)) =
            crate::engine::sizing::solve(&DesignInputs::default(), FreeVariable::AxialLength, 3.0)
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
        harness.panel.sizing.target_Nm = 3.0;
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
        harness.panel.sizing.target_Nm = 3.5;
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
        let json: serde_json::Value = serde_json::from_str(contents).unwrap();
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
        let mut harness = Harness::new();
        harness.size();
        let point = harness.solved();
        let steps = harness.panel.history.undo_len();
        harness.click_text(SizingMode::MagnetsToTorque.label());
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
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 50.0;
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Not reachable (best 9.937 N\u{b7}m at 50.80 mm)"),
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
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(variable_label(FreeVariable::AxialLength));
        harness.click_text(variable_label(FreeVariable::RingRadius));
        assert_eq!(harness.panel.sizing().variable, FreeVariable::RingRadius);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let point = harness.solved();
        // The ring radius is not in the Key design group: its locked row shows there anyway,
        // with the Coupling group closed.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SIZED_NOTE), 1);
        let label = InputCatalogue::get()
            .entry(FreeVariable::RingRadius.path())
            .unwrap()
            .meta
            .label;
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
        harness.panel.sizing.target_Nm = 3.0;
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
                .any(|t| t.starts_with("Pull-out torque at operating temperature\n")),
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
        // A window smaller than the panel's starting height: nothing panics, every frame.
        for size in [
            egui::vec2(1.0, 1.0),
            egui::vec2(200.0, 150.0),
            egui::vec2(640.0, 240.0),
        ] {
            let mut harness = Harness::on_screen(size);
            harness
                .panel
                .explorer
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.explorer.explain = true;
            for _ in 0..3 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
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
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let pullout = note("pullout_angle").unwrap();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), 0, "hidden by default");
        harness.click_text(EXPLAIN);
        assert!(harness.panel.explorer.explain);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), 1);
        assert_eq!(count(&output, pullout.sentences[0]), 1);
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end (its term row is below the note now): the note follows the equation.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let end = note("end_effect").unwrap();
        assert_eq!(count(&output, end.title), 1);
        assert_eq!(count(&output, pullout.title), 0);
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
        const TITLE: &str = "Magnetic coupling";
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
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
        // A window sizes itself on its first frame and paints on the next.
        frame(&mut panel, Vec::new());
        let output = frame(&mut panel, Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
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
}
