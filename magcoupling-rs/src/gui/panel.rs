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

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
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
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 2] = [CentreView::Geometry, CentreView::Results];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::Geometry => "Geometry",
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
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
}

impl Default for MagcouplingPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl MagcouplingPanel {
    /// A panel at the default design.
    pub fn new() -> Self {
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
            results_table: ResultsTable::default(),
            requests: Vec::new(),
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
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
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
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui, &shown));
        });
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
    fn sizing_ui(&mut self, ui: &mut egui::Ui) {
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
            let widget = self.input_row_ui(ui, entry);
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
    /// design shown.
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs) {
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
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
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
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        } else if let Some(status) = &self.status {
            ui.weak(status);
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.sizing_ui(ui);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                        }
                    });
                for group in &catalogue.groups {
                    egui::CollapsingHeader::new(group.label)
                        .id_salt(("group", &group.name))
                        .default_open(false)
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.prefix != group.name {
                                    ui.add_space(4.0);
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    let widget = self.input_row_ui(ui, entry);
                                    self.design_widgets.push(widget);
                                }
                            }
                        });
                }
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget. In Torque →
    /// Magnets the free variable's row shows the value the panel shows, locked.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
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
        let output = ui
            .add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed))
            .inner;
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
        sized_frame, sized_frame_at, text_rect,
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
        // group open, on a screen tall enough to draw every row).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        harness.panel.open_design(design.clone());
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
        for group in &catalogue.groups {
            // Tall enough for the longest group (temperature) to fit without scrolling.
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
            harness.click_text(group.label);
            // The header opens over a few frames (its animation).
            let mut output = harness.frame(Vec::new());
            for _ in 0..10 {
                output = harness.frame(Vec::new());
            }
            let texts = drawn_texts(&output);
            for entry in group.sections.iter().flat_map(|s| s.entries.iter()) {
                assert!(
                    texts.iter().any(|t| t == entry.meta.label),
                    "{}: missing {:?}",
                    group.name,
                    entry.meta.label
                );
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
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
        // The first rows are on screen (they show their cells), the last is far below.
        assert_eq!(count(&output, &first), 1, "{first}");
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
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results());
        texts.extend(
            geometry
                .callouts()
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path)),
        );
        for group in &InputCatalogue::get().groups {
            harness.click_text(group.label);
        }
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.inputs = short_magnets();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        let results = harness.panel.results().clone();
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
                .into_iter()
                .map(|line| line.tooltip),
        );
        for entry in crate::gui::results_table::table_entries() {
            let value = results.get(&entry.path).unwrap_or(Value::None);
            texts.push(crate::gui::results_table::row_tooltip(entry, &value));
        }
        let font = egui::FontId::proportional(14.0);
        for text in &texts {
            for c in text.chars().filter(|c| !c.is_whitespace()) {
                assert!(
                    harness.ctx.fonts(|f| f.has_glyph(&font, c)),
                    "no glyph for {c:?} (U+{:04X}) in {text:?}",
                    c as u32
                );
            }
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

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        let mut output = None;
        for _ in 0..2 {
            // A window sizes itself on its first frame and paints on the next.
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 1200.0])
                    .show(ctx, |ui| panel.ui(ui));
            }));
        }
        assert_drew_headline(&output.expect("two frames ran"), &DesignInputs::default());
    }
}
