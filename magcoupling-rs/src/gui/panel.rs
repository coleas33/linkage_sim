//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the check list over the Explain
//! panel on the right ([`crate::gui::checks`], [`crate::gui::explain`]: UX overhaul spec 4.5) and
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
//! decision M42-1), opening on Size and torque with the four tiles over the drawing (decision
//! G-5, which reverses M42-2). Every result is recomputed with every approved correction
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

use crate::engine::api::workbook_defaults;
use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::clamps;
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::sizing::{FreeVariable, SizingOutcome};
use crate::gui::checks::{CHECKS_SHARE, checks_ui};
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, end_effect_banner_ui};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
use crate::gui::format::search_needle;
use crate::gui::geometry::wall_cautions;
use crate::gui::geometry_view::geometry_ui;
use crate::gui::guidance::{
    ConsequenceCache, Direction, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    within_minimum_diameter,
};
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, hint_line, input_row, slider};
use crate::gui::inputs::{
    ADVANCED_HEADING, FILTER_HINT, InputCatalogue, InputEntry, InputGroup, InputOrder,
    InputSection, KEY_DESIGN, filter_inputs, optional_seed,
};
use crate::gui::look;
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
    REQUIREMENT_INPUT, SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, held_by_auto,
    sizing_range, with_own_value,
};
use crate::gui::spreadsheet::Snapshot;
use crate::gui::tiles::tiles_ui;
use crate::gui::torque_card::{KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::gui::xlsx::{XLSX_FILE_NAME, XLSX_MIME, now_unix_s, spreadsheet_bytes};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";

/// The top bar's menu of the actions on the whole design (UX overhaul spec 4.2): the reset,
/// the example design, the file's save and load, and the details toggle (decision C-12).
pub const DESIGN_MENU: &str = "Design";

/// The Design menu's item that restores the baseline on a panel whose baseline is the
/// spreadsheet's design (a host's, or a test's, panel built with `MagcouplingPanel::new`).
pub const RESET_ALL: &str = "Reset all";

/// The same item on a panel whose baseline is another design: the app's, which opens on the
/// starting design (UX overhaul spec 4.2 and 7.1).
pub const RESET_TO_START: &str = "Reset to the starting design";

/// The status line after [`RESET_ALL`].
const DEFAULT_RESTORED: &str = "Default design restored";

/// The status line after [`RESET_TO_START`]: the item's name for the design.
const STARTING_RESTORED: &str = "Starting design restored";

/// The Design menu's item that opens the spreadsheet's design (UX overhaul spec 7.1), shown on
/// a panel whose baseline is another design: there the reset no longer reaches it.
pub const OPEN_EXAMPLE: &str = "Open the example design";

/// The status line after [`OPEN_EXAMPLE`].
pub const EXAMPLE_OPENED: &str = "Example design opened";

/// The session buttons.
pub const UNDO: &str = "Undo";
pub const REDO: &str = "Redo";
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";

/// The top bar's button that copies the link of the design shown (UX overhaul spec 4.2: the
/// action of the button it replaces, "Copy share link").
pub const SHARE: &str = "Share";

/// The header's toggle of the provenance lines (spec S2-5), off when a panel opens: with it on,
/// the hover texts add the internal name, the original spreadsheet cell and note and the
/// corrections that change a value; the dashboard shows the correction ids, the results table
/// the cells and the ids, the Equation panel the cell and corrections lines and each term's
/// internal name, and the Assumptions view each source.
pub const SHOW_DETAILS: &str = "Show spreadsheet details";

/// The hover text of [`SHOW_DETAILS`].
pub const SHOW_DETAILS_HINT: &str = "Show where each value comes from in the original spreadsheet: its cell, its internal name, the corrections that change it and each assumption's source";

/// The top bar's spreadsheet button (decision X-8; UX overhaul spec 4.2 shortens its name from
/// "Export spreadsheet"; the results table has its own, beside the CSV and JSON exports).
pub const EXPORT: &str = "Export";

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

/// The label of the free variable's row in the spreadsheet's Summary.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The start of the log line of each sizing outcome (the web smoke test looks for it).
pub const SIZING_LOG_PREFIX: &str = "magcoupling sizing: ";

/// The note under the free variable's row while Auto (Torque → Magnets) sets it.
pub const SIZED_NOTE: &str = "Set by Auto (Size tab)";

/// The banner shown while an assumption differs from its workbook default (spec Addendum A3),
/// followed by the assumptions' names.
pub const ASSUMPTIONS_MODIFIED: &str = "Assumptions modified";

/// The button beside the banner.
pub const RESET_ASSUMPTIONS: &str = "Reset assumptions to defaults";

/// The hover text of [`RESET_ASSUMPTIONS`].
pub const RESET_ASSUMPTIONS_HINT: &str =
    "Every assumption back to its default; the design inputs stay as they are";

/// The status line after [`RESET_ASSUMPTIONS`].
pub const ASSUMPTIONS_RESET: &str = "Assumptions reset to their defaults";

/// The start of an assumption's source line.
pub const SOURCE: &str = "Source";

/// The line at the top of the Assumptions view.
pub const ASSUMPTIONS_NOTE: &str = "The model's assumptions, apart from the design inputs (each \
     also stays in its input group). The equation panel tags them and what they flow into.";

/// The start of the Size tab's exploration lines: a design the checks rule out (its magnets do
/// not fit, or the keyway breaks through the hub) is still drawn, for exploring. Each line ends
/// in the torque card's side words ([`SIDE_DOES_NOT_FIT`], [`KEYWAY_BREAKS_THROUGH`]), so a
/// change of those words reaches both (slice 3 spec 4.2).
const EXPLORATION_ONLY: &str = "Exploration only: ";

/// The Size column's fold of every design input (UX overhaul spec 4.3), over its fold of the
/// model's assumptions, which bears the Assumptions view's name.
pub const DETAILED_INPUTS: &str = "Detailed inputs";
pub const ASSUMPTIONS_FOLD: &str = InputsView::Assumptions.label();

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

/// The results of the design the Explain panel last asked its [`ConsequenceCache`] about, kept
/// with that design (slice 3 spec 3.5): a Size control's entry and the Hot-low tile's ask once
/// a frame, mostly about one design, so its `compute_all` runs once per design, not once per
/// ask.
#[derive(Default)]
struct ResultsCache {
    last: Option<(DesignInputs, DesignResults)>,
    /// How many times results were computed (the tests check that idle frames compute none).
    computed: usize,
}

impl ResultsCache {
    /// `compute_all(inputs)`, computed only when `inputs` differs from the last call's.
    fn get(&mut self, inputs: &DesignInputs) -> &DesignResults {
        let current = self.last.as_ref().is_some_and(|(last, _)| last == inputs);
        if !current {
            self.computed += 1;
            self.last = Some((inputs.clone(), compute_all(inputs)));
        }
        &self.last.as_ref().expect("filled above").1
    }
}

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

    /// The views the tab row offers, in tab order: every view but the clamp drawing while the
    /// shaft-clamp section is paused (`clamps_paused`).
    pub fn tabs(clamps_paused: bool) -> impl Iterator<Item = CentreView> {
        Self::ALL
            .into_iter()
            .filter(move |&view| !(clamps_paused && view == CentreView::Clamp))
    }

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
    /// The session's baseline (UX overhaul spec 7.1, C-21 and C-22): what "the default" means
    /// on screen (the changed dot, a row's reset, a tooltip's Default line, Reset all, the
    /// Equation panel's changed tags, the status line's name). Set when the panel is built and
    /// never moved: not by an undo, a file, a link or the example design. In no design file,
    /// link or undo step.
    baseline: DesignInputs,
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
    /// Whether the panel acts on the undo and redo keys
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]).
    keyboard_shortcuts: bool,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    /// Whether the header's Show spreadsheet details is on (spec S2-5): a view choice, never in
    /// a design file, a share link or the undo history, and kept by reset all and every load.
    show_details: bool,
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
    /// a design file or a share link.
    input_order: InputOrder,
    /// The user-selected lower OD limit, a view preference that does not resize the design.
    minimum_diameter_mm: Option<f64>,
    /// The Size tab's what-moves-what lines (spec 3.3) and its torque card (slice 3) for the
    /// design shown, the minimum OD and the shaft-clamp pause, recomputed only when one of the
    /// three changes: view state, in no design file, link or undo step. Asked only by the
    /// Explain panel, for an entry that reads them (`checks_and_explain_ui`).
    consequences: ConsequenceCache,
    /// The results the Explain panel hands `consequences`, kept with the design they were
    /// computed for: a cache like it, in no design file, link or undo step.
    size_tab_results: ResultsCache,
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
    /// sets it; every path it reaches is framed in the panel's mark colour.
    trace: Option<Trace>,
    /// Whether the shaft-clamp section is paused ([`clamps::PAUSED`], which every panel starts
    /// from): the views leave it out. A field, not the constant, so a test draws either state.
    clamps_paused: bool,
    /// What the Explain panel shows (UX overhaul spec 5.1): the thing clicked last, or nothing
    /// (the orientation). View state: in no design file, link or undo step, and kept by reset
    /// all and every load.
    selection: Selection,
    /// The selection the Explain panel drew last frame: when [`Self::selection`] differs, the
    /// new entry opens at its top (spec 5.1), not at the last entry's scroll offset.
    explain_drawn: Selection,
    /// Whether a text field had the keyboard as the last frame ended: Esc is then the field's,
    /// not the Explain panel's.
    typing: bool,
}

impl Default for MagcouplingPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl MagcouplingPanel {
    /// A panel at the default design (the spreadsheet's values), which is also its baseline.
    pub fn new() -> Self {
        Self::with_baseline(DesignInputs::default())
    }

    /// A panel that opens on `baseline` and keeps it as its baseline for the session (UX
    /// overhaul spec 7.1): the standalone app hands it the starting design.
    pub fn with_baseline(baseline: DesignInputs) -> Self {
        // The equation registry, built once per process, at start-up rather than on the first
        // hover.
        registry();
        let inputs = baseline.clone();
        let results = compute_all(&inputs);
        let design = Design {
            inputs: baseline.clone(),
            sizing: SizingState::default(),
        };
        Self {
            inputs,
            baseline,
            sizing: SizingState::default(),
            runner: SizingRunner::default(),
            history: History::new(design),
            share_base: PUBLIC_BASE_URL.to_owned(),
            status: None,
            results,
            key_widgets: Vec::new(),
            design_widgets: Vec::new(),
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::SizeAndTorque,
            show_details: false,
            inputs_view: InputsView::Design,
            input_order: InputOrder::default(),
            minimum_diameter_mm: None,
            consequences: ConsequenceCache::default(),
            size_tab_results: ResultsCache::default(),
            fixes: FixCache::default(),
            input_filter: String::new(),
            banner: None,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            explorer: Explorer::default(),
            trace: None,
            clamps_paused: clamps::PAUSED,
            selection: Selection::default(),
            explain_drawn: Selection::default(),
            typing: false,
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

    /// The variable Auto sets while it is on (Torque → Magnets), `None` while it is off: the
    /// size the Size tab's controls lock and the torque card holds (slice 3 spec 6.1).
    fn auto(&self) -> Option<FreeVariable> {
        (self.sizing.mode == SizingMode::TorqueToMagnets).then_some(self.sizing.variable)
    }

    /// The range Auto's solve searches ([`crate::gui::sizing::sizing_range`], spec guidance 3.5),
    /// read from the inputs, never the design shown, so a solve cannot move its own range.
    fn sizing_range(&self) -> SliderRange {
        sizing_range(&self.inputs, self.sizing.variable, self.minimum_diameter_mm)
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

    /// The address share links point at ([`MagcouplingPanel::set_share_base`]).
    pub fn share_base(&self) -> &str {
        &self.share_base
    }

    /// The share link of the design.
    pub fn share_link(&self) -> String {
        share_link(&self.share_base, &self.design())
    }

    /// The design as of the last settled edit, undo, redo or load: what the undo history holds
    /// as current. An edit in progress (a drag, a held key, a value being typed) is not in it
    /// until it settles. The web app writes the page's address from it (UX overhaul spec 7.2).
    pub fn settled_design(&self) -> &Design {
        self.history.committed()
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
        self.history = History::new(design.clone());
        self.replace_design(design);
    }

    /// Replaces the design with one that is not an undo or redo step: the Minimum OD an earlier
    /// design lowered does not carry over (undo and redo keep it, through [`Self::set_design`]).
    fn replace_design(&mut self, design: Design) {
        self.minimum_diameter_mm = None;
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
        self.load_with(loaded, done, Self::replace_design)
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

    /// Back to the baseline (its inputs, with the default sizing state); it can be undone.
    pub fn reset(&mut self) {
        self.replace_design(Design {
            inputs: self.baseline.clone(),
            sizing: SizingState::default(),
        });
    }

    /// Opens the example design (the spreadsheet's values, with the default sizing state): an
    /// edit like a loaded file, so it can be undone. The baseline stays where it is.
    pub fn open_example_design(&mut self) {
        self.replace_design(Design::default());
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut marks = self.explorer.marks(&self.inputs, &self.results);
        if let Some(trace) = &self.trace {
            let reached = trace.paths.iter().map(String::as_str);
            marks = marks.with_marked(
                reached.chain([trace.source.as_str()]),
                look::mark_color(ui.visuals()),
            );
        }
        // The header's toggle as the frame begins: every view of this frame reads it here.
        let mut readouts = Readouts::new(marks, self.show_details);
        // Esc returns the Explain panel to its orientation (UX overhaul spec 5.1), unless the
        // key is another's: a text field that had the keyboard as the last frame ended (egui
        // takes the keyboard from it at this key before the frame is drawn), or an open list.
        if !self.typing
            && !egui::Popup::is_any_open(ui.ctx())
            && ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            self.selection = Selection::Nothing;
        }
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
                self.fixes.get(&shown, &self.results, self.clamps_paused),
                &shown,
                self.minimum_diameter_mm,
            );
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| {
                    self.checks_and_explain_ui(ui, &shown, &fixes, &mut readouts)
                });
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        let mut events = readouts.finish();
        // A note asked for anywhere (a Why link, a warning's link, an entry's lesson link)
        // shows in the Explain panel (UX overhaul spec 5.2, decision C-11).
        if let Some(id) = events.note.take() {
            self.show_note(ui.ctx(), id);
        }
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
        // After every change of the trace this frame (Clear trace, an entry's trace link, a
        // result click), whichever view the centre region shows: the results table's trace filter goes
        // off with an input's trace even while the table is not drawn.
        self.results_table.sync_trace_filter(self.trace.as_ref());
        self.explorer.end_frame(ui.ctx(), events);
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
        self.typing = focused_text_field(ui.ctx()).is_some();
    }

    /// The right column (UX overhaul spec 4.5): the check list, at its own height up to
    /// [`CHECKS_SHARE`] of the column and scrolling beyond it, then the Explain panel in the
    /// rest, scrolling on its own. A click on a check's row selects it, and the entry under the
    /// list shows it in the same frame.
    fn checks_and_explain_ui(
        &mut self,
        ui: &mut egui::Ui,
        shown: &DesignInputs,
        fixes: &[Vec<(&'static str, Direction)>],
        readouts: &mut Readouts,
    ) {
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_checks_scroll")
            .max_height(CHECKS_SHARE * ui.available_height())
            .show(ui, |ui| {
                let clicked = checks_ui(
                    ui,
                    shown,
                    &self.baseline,
                    &self.results,
                    self.clamps_paused,
                    self.selection.check(),
                    readouts,
                );
                if let Some(check) = clicked {
                    self.selection = Selection::Check(check);
                }
            });
        ui.separator();
        let auto = self.auto();
        // The Size tab's what-moves-what lines and torque card for the design shown (slice 3
        // spec 3.5), asked only for an entry that reads them, with one key: the design, the
        // panel's Minimum OD and its shaft-clamp pause (spec 3.2). The cache is handed that
        // design's own results, not `self.results`, which can lag the inputs by a frame (the
        // header's assumptions reset edits them after the last compute). `size_tab_results`
        // keeps them with their design, so they are computed once per design, not once per
        // ask: none on an idle frame, one on an edit's.
        let consequences = self.selection.reads_consequences().then(|| {
            let results = self.size_tab_results.get(shown);
            self.consequences
                .get(shown, results, self.minimum_diameter_mm, self.clamps_paused)
        });
        let mut explain_scroll = egui::ScrollArea::vertical()
            .id_salt("magcoupling_explain_scroll")
            .auto_shrink([false, false]);
        if self.selection != self.explain_drawn {
            explain_scroll = explain_scroll.vertical_scroll_offset(0.0);
            self.explain_drawn = self.selection;
        }
        let asked = explain_scroll
            .show(ui, |ui| {
                // A note's change entry names an input Auto or a part pick holds as plain
                // text, on the design shown.
                let sizing = self.sizing;
                let held = |path: &str| held_reason(&sizing, shown, path);
                let shown = crate::gui::explain::Shown {
                    inputs: shown,
                    results: &self.results,
                    fixes,
                    clamps_paused: self.clamps_paused,
                    consequences,
                    auto,
                    held: &held,
                    minimum_diameter_mm: self.minimum_diameter_mm,
                };
                explain_ui(ui, self.selection, &shown, readouts)
            })
            .inner;
        match asked {
            Some(Asked::Select(selection)) => self.selection = selection,
            Some(Asked::Equation(path)) => self.explorer.open_path(path),
            Some(Asked::StartHere) => {
                if let Some(id) = self.explorer.start(0) {
                    self.show_note(ui.ctx(), id);
                }
            }
            Some(Asked::Trace(path)) => self.toggle_trace(path),
            None => {}
        }
        if asked.is_some() {
            ui.ctx().request_repaint();
        }
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
        let show_details = readouts.show_details();
        if self.explorer.open {
            let (range, start) = panel_heights(ui.available_rect_before_wrap().height());
            let note = egui::TopBottomPanel::bottom(EQUATION_PANEL_ID)
                .resizable(true)
                .default_height(start)
                .height_range(range)
                .show_inside(ui, |ui| {
                    explorer_ui(
                        ui,
                        &mut self.explorer,
                        shown,
                        &self.results,
                        &self.baseline,
                        show_details,
                        self.clamps_paused,
                    )
                })
                .inner;
            // A lesson link or a Start here step: its note in the Explain panel (spec 5.5).
            if let Some(id) = note {
                self.show_note(ui.ctx(), id);
            }
        }
        // egui's panel only shrinks the region's cursor: a widget laid out after it may still
        // run past its top edge, and it paints over the panel with a clip rect covering it. So
        // the tab row and the banner live in the space above the panel, and the view in the
        // space left under them, each confined to it.
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            ui.horizontal_wrapped(|ui| {
                for view in CentreView::tabs(self.clamps_paused) {
                    if look::tab_ui(ui, self.centre == view, view.label()).clicked() {
                        self.centre = view;
                    }
                }
            });
            ui.separator();
            end_effect_banner_ui(ui, &self.results, readouts);
            confined(ui, "magcoupling_centre_view", |ui| {
                self.view_ui(ui, shown, readouts);
            });
        });
    }

    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`). Each view fits itself to `ui`'s available height. The opening
    /// tab (UX overhaul spec 4.4): the four tiles, a click on one selecting it for the Explain
    /// panel; the "Exploration only" lines and the keyway's advisory, only when true; then the
    /// drawing and its callout list in all the height left (the plan's decision D-19).
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        match self.centre {
            CentreView::SizeAndTorque => {
                let selected = self.selection.tile();
                if let Some(tile) = tiles_ui(ui, shown, &self.results, selected, readouts) {
                    // The Explain panel was drawn before this click: the next frame shows it.
                    self.selection = Selection::Result(tile.path());
                    ui.ctx().request_repaint();
                }
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter."),
                    );
                }
                let key_wall = self.results.model.hub_wall_past_key_mm;
                if !key_wall.is_finite() || key_wall <= 0.0 {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}."),
                    );
                }
                for line in wall_cautions(shown, &self.results) {
                    ui.colored_label(ui.visuals().warn_fg_color, line);
                }
                geometry_ui(ui, shown, &self.results, self.clamps_paused, readouts);
            }
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results, self.clamps_paused, readouts);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results, readouts),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results, readouts),
            CentreView::Results => {
                let action = self.results_table.ui(
                    ui,
                    &self.results,
                    self.trace.as_ref(),
                    self.clamps_paused,
                    readouts,
                );
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
            look::mark_ui(ui);
            ui.heading(HEADING);
            ui.separator();
            let can_undo = self.history.can_undo(&design);
            if ui
                .add_enabled(can_undo, bar_button(UNDO))
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                self.undo();
            }
            let can_redo = self.history.can_redo(&design);
            if ui
                .add_enabled(can_redo, bar_button(REDO))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
                .clicked()
            {
                self.redo();
            }
            ui.separator();
            egui::containers::menu::MenuButton::from_button(bar_button(DESIGN_MENU))
                .ui(ui, |ui| self.design_menu_ui(ui, &design));
            if ui
                .add(bar_button(SHARE))
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
                .add(bar_button(EXPORT))
                .on_hover_text(
                    "This design and its results as an .xlsx workbook laid out as this page (values, no formulas)",
                )
                .clicked()
            {
                self.export_spreadsheet();
            }
            ui.separator();
            if ui.add(bar_button(LEARN)).clicked() {
                self.selection = Selection::NoteList;
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
        let banner = assumptions_banner(&self.inputs, self.clamps_paused);
        if banner != self.banner {
            self.banner.clone_from(&banner);
            ui.ctx().request_repaint();
        }
        if let Some(banner) = banner {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().warn_fg_color, banner);
                if ui
                    .button(RESET_ASSUMPTIONS)
                    .on_hover_text(RESET_ASSUMPTIONS_HINT)
                    .clicked()
                {
                    assumptions::reset_to_workbook_defaults(&mut self.inputs);
                    self.status = Some(ASSUMPTIONS_RESET.to_owned());
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

    /// The Design menu (UX overhaul spec 4.2): the reset to the baseline, the example design
    /// where the baseline is another, the file's save and load of `design`, and the details
    /// toggle (decision C-12). Running an item, by click or from the keyboard, closes the menu
    /// (egui closes it on a pointer click only, so each item calls `ui.close()`).
    fn design_menu_ui(&mut self, ui: &mut egui::Ui, design: &Design) {
        let reset_label = self.reset_label();
        if ui.button(reset_label).clicked() {
            self.reset();
            let restored = if reset_label == RESET_ALL {
                DEFAULT_RESTORED
            } else {
                STARTING_RESTORED
            };
            self.status = Some(restored.to_owned());
            ui.close();
        }
        // Where the baseline is the spreadsheet's design, the reset already opens it.
        if self.baseline != *workbook_defaults()
            && ui
                .button(OPEN_EXAMPLE)
                .on_hover_text(EXAMPLE_DESIGN_HOVER)
                .clicked()
        {
            self.open_example_design();
            self.status = Some(EXAMPLE_OPENED.to_owned());
            ui.close();
        }
        ui.separator();
        if ui.button(SAVE_DESIGN).clicked() {
            self.requests.push(PanelRequest::SaveFile {
                file_name: DESIGN_FILE_NAME.to_owned(),
                mime: "application/json",
                contents: design_to_json(design).into_bytes(),
            });
            ui.close();
        }
        if ui.button(LOAD_DESIGN).clicked() {
            self.requests.push(PanelRequest::OpenDesign);
            ui.close();
        }
        ui.separator();
        // The frame's readouts were set up before the header: the next frame draws every
        // view with the new setting.
        if ui
            .checkbox(&mut self.show_details, SHOW_DETAILS)
            .on_hover_text(SHOW_DETAILS_HINT)
            .changed()
        {
            ui.ctx().request_repaint();
            ui.close();
        }
    }

    /// Shows the teaching note `id` in the Explain panel, drawn on the next frame (decision
    /// C-11): the one place a note asked for anywhere becomes the selection.
    fn show_note(&mut self, ctx: &egui::Context, id: &'static str) {
        self.selection = Selection::Note(id);
        ctx.request_repaint();
    }

    /// The Design menu's reset: [`RESET_TO_START`] where the baseline is not the spreadsheet's
    /// design, [`RESET_ALL`] where it is (there the starting design and the example design are
    /// one).
    fn reset_label(&self) -> &'static str {
        if self.baseline == *workbook_defaults() {
            RESET_ALL
        } else {
            RESET_TO_START
        }
    }

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen,
    /// with the trace's banner at its foot. A leaf term clicked in the Equation panel shows the
    /// view that holds its row.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        if self.centre == CentreView::SizeAndTorque {
            // The column scrolls in a short window: its overflow would stretch the panel past
            // the window's foot, and the centre region and its drawing with it.
            egui::ScrollArea::vertical()
                .id_salt("magcoupling_size_column")
                .auto_shrink([false, true])
                .show(ui, |ui| self.size_column_ui(ui, readouts));
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
        // the room above it in the same frame: a trace moves no row above it (decision O-8).
        let clamps_paused = self.clamps_paused;
        let banner = self.trace.as_ref().map(|trace| trace.banner(clamps_paused));
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            if let Some(banner) = banner {
                if ui.small_button(CLEAR_TRACE).clicked() {
                    self.trace = None;
                }
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(banner).color(look::mark_color(ui.visuals())),
                    )
                    .wrap(),
                );
                ui.separator();
            }
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.separator();
                match self.inputs_view {
                    InputsView::Design => self.design_inputs_ui(ui, readouts),
                    InputsView::Assumptions => {
                        egui::ScrollArea::vertical()
                            .id_salt("magcoupling_assumptions_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| self.assumptions_ui(ui, readouts));
                    }
                }
            });
        });
    }

    /// The Size column of the opening tab (UX overhaul spec 4.3): its heading, the Required
    /// torque row, Auto and its outcome, the size controls, then two folds, both closed until a
    /// fix link or a term asks for a row in one: the detailed inputs and the assumptions. A
    /// control touched selects its row, whose lines show in the Explain panel.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        ui.heading(crate::gui::size_controls::HEADING);
        let (requirement, required_touched) = self.requirement_ui(ui);
        // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
        // shown (decision M41-7) through the mode switch.
        let asked = size_controls::auto_ui(ui, self.sizing);
        self.sizing.variable = asked.variable;
        self.set_sizing_mode(asked.mode);
        let auto = self.auto();
        let shown = self.shown_inputs();
        if auto.is_some() {
            let (line, color) = self.auto_outcome(&shown, ui.visuals());
            ui.add(egui::Label::new(egui::RichText::new(line).color(color)).wrap());
        }
        let minimum = self.minimum_diameter_mm;
        let output = size_controls::controls_ui(ui, &shown, minimum, auto);
        let touched = output
            .touched
            .or(required_touched.then_some(REQUIREMENT_INPUT));
        if let Some(key) = touched {
            self.selection = Selection::Input(key);
        }
        // The top bar was drawn before this frame's edit: one more frame shows its state.
        if output.inputs != shown || output.minimum_diameter_mm != minimum {
            ui.ctx().request_repaint();
        }
        self.design_widgets.extend(output.widgets);
        // After the controls' own: the tests find those by their place in the list.
        self.design_widgets.push(requirement);
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
        // A row asked for (a fix link, a term) opens the fold that holds it: an assumption's
        // row is in the Assumptions fold, every other in the detailed inputs.
        let assumption = self
            .explorer
            .focus()
            .filter(|focus| focus.scroll)
            .map(|focus| {
                ASSUMPTIONS
                    .iter()
                    .any(|a| a.paths.contains(&focus.path.as_str()))
            });
        egui::CollapsingHeader::new(DETAILED_INPUTS)
            .default_open(false)
            .open((assumption == Some(false)).then_some(true))
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
        egui::CollapsingHeader::new(ASSUMPTIONS_FOLD)
            .default_open(false)
            .open((assumption == Some(true)).then_some(true))
            .show(ui, |ui| self.assumptions_ui(ui, readouts));
    }

    /// The Required torque row that leads the Size column (UX overhaul spec 4.3; it was the
    /// first of the Size tab's readouts): its label, quieter than the value in the slider's box
    /// ([`look::quiet_text`]), then the requirement's slider, an ordinary design edit (undo,
    /// files, links). Its Why link is its Explain entry's lesson. Returns the slider's widget
    /// (its value box is a text field of the design, as an input row's is) and whether a person
    /// touched the row ([`size_controls::touched`]).
    fn requirement_ui(&mut self, ui: &mut egui::Ui) -> (egui::Id, bool) {
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata");
        let range = entry.meta.range.expect("the requirement has a slider");
        let required = self.inputs.metal.required_min_Nm;
        let row = size_controls::size_row(REQUIREMENT_INPUT).expect("the requirement has its row");
        let clicked = size_controls::row_label(ui, row, Some(look::quiet_text(ui.visuals())));
        let mut value = required;
        let response = ui
            .push_id(REQUIREMENT_INPUT, |ui| {
                ui.add(slider(&mut value, entry.meta, range))
            })
            .inner;
        if response.changed() && value != required {
            self.last_error = self
                .inputs
                .set(REQUIREMENT_INPUT, Value::Num(value))
                .err()
                .map(|e| e.to_string());
            ui.ctx().request_repaint();
        }
        (response.id, clicked || size_controls::touched(&response))
    }

    /// The Assumptions view (spec Addendum A3 "Toggle panel"): each assumption with its
    /// changed-from-default dot, the rows of its inputs (value, unit, reset: the inputs' own
    /// rows, so an edit here is an edit like any other), its rationale and, with the
    /// spreadsheet details (spec S2-5), its source. No scroll area of its own: the Size
    /// column's fold scrolls with the column, the inputs side wraps it in one.
    fn assumptions_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        let clamps_paused = self.clamps_paused;
        ui.add(egui::Label::new(egui::RichText::new(ASSUMPTIONS_NOTE).weak()).wrap());
        for state in assumptions::states(&self.inputs) {
            let assumption = state.assumption;
            // The paused shaft-clamp section's assumptions are left out with it.
            if !assumption.shown(clamps_paused) {
                continue;
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let dot = if state.modified { CHANGED_DOT } else { " " };
                ui.colored_label(look::mark_color(ui.visuals()), dot)
                    .on_hover_text(crate::gui::input_ui::CHANGED_FROM_DEFAULT);
                ui.strong(assumption.label);
            });
            for path in assumption.paths {
                let entry = catalogue
                    .entry(path)
                    .expect("an assumption path is an input (tests/assumptions.rs)");
                let widget = self.input_row_ui(ui, entry, readouts);
                self.design_widgets.push(widget);
            }
            ui.add(egui::Label::new(egui::RichText::new(assumption.rationale).weak()).wrap());
            if readouts.show_details() {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!("{SOURCE}: {}", assumption.source))
                            .small()
                            .weak(),
                    )
                    .wrap(),
                );
            }
        }
    }

    /// The design inputs: the order toggle and the filter box, then the Key design group (each
    /// row with its hint line, slice 2 spec 3.5) and every input by group in the order chosen
    /// (decision O-1), each workflow group's advanced sections under its closed Advanced heading;
    /// while the filter holds more than blanks, the matching inputs alone.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // Every input but those the paused shaft-clamp section hides.
        let catalogue = InputCatalogue::shown(self.clamps_paused);
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
                            hint_line(ui, &entry.path);
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
        let total = InputCatalogue::shown(self.clamps_paused).all().count();
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
    /// Magnets the rows of the inputs Auto sets ([`held_by_auto`]: the free variable's, and for
    /// ring size the cap OD and thread that follow it) show the value the panel shows, locked
    /// (the label still selects the input). A click on a row's label selects its input (spec
    /// 5.2). The row is framed in its term's colour while the equation in view
    /// reads the input (`readouts`).
    fn input_row_ui(
        &mut self,
        ui: &mut egui::Ui,
        entry: &'static InputEntry,
        readouts: &Readouts,
    ) -> egui::Id {
        let locked = held_by_auto(&self.sizing, &entry.path);
        // A locked row reads the design shown (one clone, for that row only).
        let current = if locked {
            self.shown_inputs().get(&entry.path)
        } else {
            self.inputs.get(&entry.path)
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let show_details = readouts.show_details();
        // "The default" on screen is the baseline's value (UX overhaul spec 7.1, C-22).
        let default = self.baseline.get(&entry.path).unwrap_or(Value::None);
        let row =
            ui.scope(|ui| input_row(ui, entry, &current, &default, seed, locked, show_details));
        readouts.mark(ui, row.response.rect, &entry.path);
        if self
            .explorer
            .focus()
            .is_some_and(|focus| focus.path == entry.path)
        {
            ui.painter().rect_stroke(
                row.response.rect.expand(3.0),
                3.0,
                egui::Stroke::new(FOCUS_WIDTH, look::mark_color(ui.visuals())),
                egui::StrokeKind::Outside,
            );
            if self.explorer.take_scroll(&entry.path) {
                row.response.scroll_to_me(Some(egui::Align::Center));
            }
        }
        let output = row.inner;
        // Spec 5.2: touching an input (its label's click, its widget's click, drag or focus, an
        // edit or a reset) selects it, as a Size control's touch does; its entry holds the
        // trace's link. A locked row's widget is disabled: only its label touches it.
        if output.label_clicked || output.edit.is_some() || size_controls::touched(&output.widget) {
            self.selection = Selection::Input(entry.path.as_str());
            ui.ctx().request_repaint();
        }
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
        }
        if let Some(edit) = output.edit {
            let value = match edit {
                RowEdit::Set(value) => value,
                RowEdit::Reset => default,
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

    /// Traces the input at `path`, or ends the trace if `path` is its source: a second click
    /// ends it (decision O-8).
    fn toggle_trace(&mut self, path: &str) {
        let same = self.trace.as_ref().is_some_and(|t| t.source == path);
        self.trace = if same { None } else { Trace::of(path) };
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
/// of the assumptions that differ from their workbook default, those the paused shaft-clamp
/// section hides left out while `clamps_paused`; `None` while none does. The header shows it
/// (with the pause), and the spreadsheet's Summary (without: the exports keep every input).
pub(crate) fn assumptions_banner(inputs: &DesignInputs, clamps_paused: bool) -> Option<String> {
    let modified: Vec<_> = assumptions::modified(inputs)
        .into_iter()
        .filter(|assumption| assumption.shown(clamps_paused))
        .collect();
    (!modified.is_empty()).then(|| {
        let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
        format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
    })
}

/// A button of the top bar: its words alone at rest, its box while the pointer is on it (stage 1
/// review S-3; UX overhaul spec 4.1's bar of words).
fn bar_button(text: &str) -> egui::Button<'_> {
    egui::Button::new(text).frame_when_inactive(false)
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
    use crate::gui::dashboard::{
        END_EFFECT_BANNER, STORED_3D_LABEL, end_effect_banner, result_info,
    };
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::guidance::{
        Consequences, FACE_GAP, RING_SIZE_STAYS, SIZE_TAB_NOTE, TORQUE_BAND_NOTE,
        plain_text_violations,
    };
    use crate::gui::input_ui::{BLANK_TEXT, CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::inputs::WORKFLOW;
    use crate::gui::plain_text::{
        DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, hint_for, label_for,
    };
    use crate::gui::session::{
        decode_share_payload, design_from_json, encode_share_payload, encode_share_text,
        target_mismatch_notice,
    };
    use crate::gui::size_controls::{
        DIAMETER, GAP, HOT_LOW, LENGTH, MAGNETS, MINIMUM, NO_PREDICTION, REQUIRED, predicts,
        pullout_label, torque_gap_text,
    };
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, flat_shapes, hot_day,
        hub_overstressed, key_event, key_tap, last_characters, only_the_clamp_fails,
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
        shown_whole, sized_frame, sized_frame_at, smoke_inputs, text_center, text_clip_rect,
        text_color, text_rect, text_rects, text_shapes, tooltips_at_once, with_poles,
    };
    use crate::gui::torque_card::{CAPTION, CARD_HEADING, Control, held_line};
    use crate::headline;

    const POLES: &str = "coupling.npole";
    const AXIAL_LENGTH: &str = "coupling.magnets.axial_length_mm";
    const PART_INNER: &str = "coupling.magnets.part_inner";
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";
    const GEARBOX_CHECK: &str = "gearbox.cold_high_check";
    const HOT_MIN_CHECK: &str = "metal.hot_min_check";
    const SPACE_CLAIM_CHECK: &str = "housing.space_claim_check";

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
            Self::with_baseline(DesignInputs::default(), size)
        }

        /// [`Harness::on_screen`] for a panel whose baseline is `baseline` (UX overhaul spec
        /// 7.1): the panel opens on it.
        fn with_baseline(baseline: DesignInputs, size: egui::Vec2) -> Self {
            let mut panel = MagcouplingPanel::with_baseline(baseline);
            panel.centre = CentreView::Geometry;
            Self::of(panel, size)
        }

        /// A harness on a new panel as it opens (decision G-5): the Size tab, the Explain panel
        /// on its orientation, on the laptop screen.
        pub(crate) fn first_view() -> Self {
            Self::of(MagcouplingPanel::new(), SCREEN)
        }

        /// A harness drawing `panel` on a screen of `size`, one frame in. Its panel draws the
        /// whole shaft-clamp section, as every panel test did before the pause (ruling 11): a
        /// test of the paused panel sets `harness.panel.clamps_paused` itself. The explorer's
        /// Start here walk still reads `clamps::PAUSED` itself, so even this harness walks the
        /// paused Start here; its hover and Used by links follow the panel's `clamps_paused`.
        fn of(mut panel: MagcouplingPanel, size: egui::Vec2) -> Self {
            panel.clamps_paused = false;
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

        /// Shows the Results tab, one frame in: its Headline group lists every headline number,
        /// each a readout (its hover, its click), as the right column did before it became the
        /// check list (UX overhaul spec 4.5).
        fn on_results(mut self) -> Self {
            self.panel.centre = CentreView::Results;
            self.frame(Vec::new());
            self
        }

        /// Selects `check`, as a click on its row of the check list does: the Explain panel
        /// shows its entry in the frame this returns.
        fn select(&mut self, check: &'static str) -> egui::FullOutput {
            self.panel.selection = Selection::Check(check);
            self.frame(Vec::new())
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

        /// Traces the input labelled `label` as a person does (UX overhaul spec 5.2): a click on
        /// its label selects it, then a click on its Explain entry's trace link.
        fn trace_input(&mut self, label: &str) -> egui::FullOutput {
            self.click_text(label);
            self.click_text(crate::gui::explain::TRACE_LINK)
        }

        /// Opens the top bar's Design menu and clicks its item `item` (UX overhaul spec 4.2).
        fn menu(&mut self, item: &str) -> egui::FullOutput {
            self.click_text(DESIGN_MENU);
            self.click_text(item)
        }

        fn number(&self, path: &str) -> f64 {
            match self.panel.inputs.get(path) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{path}: {other:?}"),
            }
        }
    }

    /// The left edge of the right column in `output`, less its margin: where the check list's
    /// heading starts.
    fn right_column_left(output: &egui::FullOutput) -> f32 {
        text_rect(output, crate::gui::checks::HEADING)
            .expect("the check list's heading")
            .left()
            - 10.0
    }

    /// The headline of `inputs`, as the Results tab's Headline group displays it (value with
    /// unit; the cup wall's failing verdict with its input's label, ruling R9).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
        headline(&compute_all(inputs))
            .into_iter()
            .zip(HEADLINE)
            .map(|((_, value), (_, path))| {
                with_unit(
                    crate::gui::dashboard::shown_text(path, &value),
                    result_info(path).unwrap().meta.unit,
                )
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
        // a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing).
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

    // ------------------------------------------------------------------ the right column

    /// Whether the one copy of `text` drawn in `output` is in view on `screen`: inside its
    /// clip rect and inside the screen.
    fn in_view(output: &egui::FullOutput, text: &str, screen: egui::Vec2) -> bool {
        let rects = text_rects(output, text);
        assert_eq!(rects.len(), 1, "{text:?}: {rects:?}");
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, screen);
        text_clip_rect(output, text).contains_rect(rects[0]) && screen.contains_rect(rects[0])
    }

    #[test]
    fn the_right_column_is_the_check_list_over_the_explain_panel_on_its_orientation() {
        // UX overhaul spec 4.5, 5.4 and 8.2 test 10: at 1280 x 800 a new panel draws the check
        // list (its status line and a row per check, in the guides' order) and under it the
        // Explain panel, which shows the orientation on the first frame. Register row A-18: the
        // line that says two of the checks read fixed 3D fields is in view with no scroll.
        use crate::gui::explorer::START_HERE_BUTTON;
        use crate::gui::guidance::counted_guides;
        use crate::gui::{checks, explain};
        let size = egui::vec2(1280.0, 800.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut harness = Harness::of(MagcouplingPanel::new(), size);
        let first = harness.frame(Vec::new());
        assert_eq!(count(&first, explain::ORIENTATION), 1, "on the first frame");
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing);
        let mut column = vec![checks::HEADING];
        column.extend(counted_guides(false).map(|guide| guide.label));
        column.extend([STORED_3D_LABEL, explain::HEADING, explain::ORIENTATION]);
        column.extend(explain::STEPS);
        column.push(START_HERE_BUTTON);
        let left = right_column_left(&output);
        let mut last = f32::NEG_INFINITY;
        for text in column {
            // The Hot-low torque row's label is the Size tab's readout label too.
            let rect = text_rects(&output, text)
                .into_iter()
                .find(|rect| rect.left() > left)
                .unwrap_or_else(|| panic!("{text:?} in the right column"));
            // In view inside its own scroll area's clip (the list's or the Explain panel's).
            assert!(
                shown_whole(&output, text, screen)
                    .iter()
                    .any(|whole| whole.left() > left),
                "{text:?} is in view at {rect:?}"
            );
            assert!(rect.top() > last, "{text:?} is under the text before it");
            last = rect.top();
        }
        assert!(in_view(&output, STORED_3D_LABEL, size), "row A-18");
        // The list never takes more than its share of the column: the Explain panel's heading
        // starts above the middle of the screen.
        let explain = text_rect(&output, explain::HEADING).unwrap();
        assert!(explain.top() < size.y / 2.0, "{explain:?}");
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn each_explain_entry_opens_at_its_top_whatever_the_last_one_was_scrolled_to() {
        // Spec 5.1 and goal G-8 (review CR-1): an entry scrolled to its foot, then another red
        // check's row clicked: the new entry's title and "What it is" are in view, not left
        // above the view at the last entry's scroll offset.
        use crate::gui::explain;
        use crate::gui::guidance::guide;
        let size = egui::vec2(1024.0, 700.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut harness = Harness::on_screen(size);
        harness.panel.open_example_design();
        harness.frame(Vec::new());
        harness.click_text(guide(GEARBOX_CHECK).unwrap().label);
        let output = harness.frame(Vec::new());
        let inside = text_center(&output, explain::WHAT_IT_IS) + egui::vec2(0.0, 60.0);
        harness.frame(vec![egui::Event::PointerMoved(inside)]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -5000.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        let mut output = harness.frame(Vec::new());
        for _ in 0..20 {
            output = harness.frame(Vec::new());
        }
        assert!(
            shown_whole(&output, explain::WHAT_IT_IS, screen).is_empty(),
            "the gearbox entry is scrolled past its top"
        );
        harness.frame(vec![egui::Event::PointerGone]);
        harness.click_text(guide(HOT_MIN_CHECK).unwrap().label);
        assert_eq!(harness.panel.selection, Selection::Check(HOT_MIN_CHECK));
        let mut output = harness.frame(Vec::new());
        for _ in 0..3 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            shown_whole(&output, explain::WHAT_IT_IS, screen).len(),
            1,
            "the hot-low entry opens at its top"
        );
    }

    #[test]
    fn a_click_on_a_check_shows_its_entry_and_esc_or_about_returns_the_orientation() {
        // Spec 5.1, 5.2 and 8.2 test 10, and goal G-8: one click on a failing check shows its
        // problem, its reason and its fixes. A hover changes nothing (decision C-2).
        use crate::gui::explain;
        use crate::gui::guidance::guide;
        let mut harness = Harness::first_view();
        let gearbox = guide(GEARBOX_CHECK).unwrap();
        let cup_wall = guide("materials.cup_wall_check").unwrap();
        harness.click_text(gearbox.label);
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        let output = harness.frame(Vec::new());
        let panel = &harness.panel;
        let problem = (gearbox.problem)(panel.inputs(), panel.results()).expect("it fails");
        for text in [gearbox.why, problem.as_str(), explain::ABOUT] {
            assert_eq!(count(&output, text), 1, "{text:?}");
        }
        assert_eq!(count(&output, explain::ORIENTATION), 0);
        // Hovering another check's row leaves the entry as it is.
        let at = text_rect(&output, cup_wall.label).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        assert_eq!(count(&output, gearbox.why), 1);
        // The last click owns the panel.
        harness.click(at);
        assert_eq!(
            harness.panel.selection,
            Selection::Check("materials.cup_wall_check")
        );
        // Esc returns the orientation; so does the link at the entry's head.
        let output = harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert_eq!(count(&output, explain::ORIENTATION), 1);
        harness.click_text(gearbox.label);
        harness.click_text(explain::ABOUT);
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn esc_in_a_text_field_is_the_field_s_and_keeps_the_selection() {
        let mut harness = Harness::new();
        harness.select(GEARBOX_CHECK);
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        // Spec 5.2: typing in the field touches its input, which it selects.
        assert_eq!(harness.panel.selection, Selection::Input(PART_INNER));
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Input(PART_INNER));
        // The field let go of the keyboard: the next Esc is the panel's.
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
    }

    #[test]
    fn an_amber_check_s_entry_is_open_at_its_one_click() {
        // Before the Explain panel an amber check's block was shut under a header: two clicks.
        use crate::gui::guidance::guide;
        let mut hot = DesignInputs::default();
        hot_day(&mut hot);
        let temperature = guide("temperature.summary.verdict").unwrap();
        let mut harness = Harness::first_view();
        harness.panel.inputs = hot;
        harness.frame(Vec::new());
        harness.click_text(temperature.label);
        let output = harness.frame(Vec::new());
        let panel = &harness.panel;
        let problem = (temperature.problem)(panel.inputs(), panel.results()).expect("amber");
        assert_eq!(count(&output, &problem), 1);
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Caution.color(&visuals))
        );
    }

    #[test]
    fn the_selection_is_a_view_choice_in_no_file_link_or_undo_step() {
        // Spec 5.1: selecting edits nothing, and reset all, a load and an undo keep it.
        let mut harness = Harness::new();
        let (file, link) = (
            design_to_json(&harness.panel.design()),
            harness.panel.share_link(),
        );
        harness.select(GEARBOX_CHECK);
        harness.frame(Vec::new());
        assert_eq!(design_to_json(&harness.panel.design()), file);
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        // Spec 5.2: the nudge touches the gap, which it selects; the undo, reset all and the
        // load keep that selection.
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.panel.undo();
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.panel.reset();
        harness.panel.load_design_file(&file).unwrap();
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
    }

    #[test]
    fn an_entry_s_equation_link_opens_the_equation_panel_and_the_selection_stays() {
        // Decisions C-11 and C-25: the Equation link opens the docked panel at the entry's
        // result, with no trace; a selection made afterwards leaves the open equation where it
        // is. The orientation's link starts the Start here walk.
        use crate::gui::explain;
        use crate::gui::explorer::START_HERE_BUTTON;
        use crate::gui::guidance::guide;
        let mut harness = Harness::first_view();
        assert!(!harness.panel.explorer.open);
        harness.click_text(guide(GEARBOX_CHECK).unwrap().label);
        harness.click_text(explain::EQUATION);
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.trail(), [GEARBOX_CHECK]);
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        assert_eq!(harness.panel.trace, None);
        harness.click_text(guide(SPACE_CLAIM_CHECK).unwrap().label);
        assert_eq!(harness.panel.selection, Selection::Check(SPACE_CLAIM_CHECK));
        assert_eq!(harness.panel.explorer.trail(), [GEARBOX_CHECK]);
        // Closing the Equation panel leaves the selection as it is.
        harness.panel.explorer.close();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Check(SPACE_CLAIM_CHECK));
        harness.frame(key_tap(egui::Key::Escape));
        harness.click_text(START_HERE_BUTTON);
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.start_here(), Some(0));
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_check_list_and_the_explain_panel_compute_nothing_on_an_idle_frame() {
        // Spec 8.2, test 14: with nothing selected, with a failing check's entry open, with
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), with
        // a note's entry (its diagram), with the note list (plan review COV-1) and with a
        // Detailed input's entry (decision D-59), a frame in which nothing changed runs no fix
        // search and no results of its own.
        use crate::engine::explain::notes::START_HERE;
        let mut harness = Harness::first_view();
        let hot_low = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let checks = [HOT_MIN_CHECK, GEARBOX_CHECK].map(|check| Some(Selection::Check(check)));
        let note = Some(Selection::Note(START_HERE[0].0));
        for check in [
            None,
            checks[0],
            checks[1],
            Some(hot_low),
            note,
            Some(Selection::NoteList),
            Some(Selection::Input("coupling.op_temp_C")),
        ] {
            match check {
                Some(Selection::Check(path)) => {
                    harness.select(path);
                }
                Some(selection) => harness.panel.selection = selection,
                None => {}
            }
            harness.frame(Vec::new());
            let fixes = harness.panel.fixes.computed;
            let results = harness.panel.size_tab_results.computed;
            let consequences = harness.panel.consequences.computed;
            for _ in 0..5 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.fixes.computed, fixes, "{check:?}");
            assert_eq!(
                harness.panel.size_tab_results.computed, results,
                "{check:?}"
            );
            assert_eq!(
                harness.panel.consequences.computed, consequences,
                "{check:?}"
            );
        }
    }

    #[test]
    fn a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing() {
        // Decision G-5 (spec 3.1, reversing M42-2) and UX overhaul spec 4.4: the first view is
        // the Size tab, its four tiles over the drawing; showing them edits nothing, and the
        // Explain panel shows no note (M43-2). The welcome strip is gone: the Explain panel's
        // orientation says what to do first.
        use crate::gui::guidance::{design_status, design_status_text};
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        use crate::gui::tiles::Tile;
        let panel = MagcouplingPanel::new();
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert_eq!(panel.selection, Selection::Nothing);
        assert_eq!(
            MagcouplingPanel::default().centre,
            CentreView::SizeAndTorque
        );
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, crate::gui::explain::ORIENTATION), 1);
        // The inputs side is the Size tab's: the Required torque, then its controls, the Key
        // design group folded away under the detailed inputs.
        for label in [REQUIRED, DIAMETER, LENGTH, GAP, MAGNETS] {
            assert_eq!(count(&output, label), 1, "{label}");
        }
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        // The check list's design status line (spec 3.1) is on the first view, counted live.
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
        assert!(status.is_default);
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // The tiles are in the centre region, under the tab row and over the drawing, in their
        // order from the left.
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        let drawing = drawing_rect(&output);
        let mut left = INPUTS_WIDTH;
        for tile in Tile::ALL {
            let label = tile.label(harness.panel.results());
            let at = centre_rect(&output, &label);
            assert!(
                tabs.bottom() <= at.top() && at.bottom() <= drawing.top(),
                "{label}: {at:?} between {tabs:?} and {drawing:?}"
            );
            assert!(at.left() >= left, "{label}: {at:?} right of {left}");
            left = at.left();
        }
        // Under the drawing its callout list, as on the Geometry tab, until a click on a
        // dimension tag shows its callout in the Explain panel (decision D-19).
        let list = centre_rect(&output, "Face gap 1.400 mm");
        assert!(drawing.bottom() <= list.top(), "{drawing:?} over {list:?}");
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_top_bar_is_undo_redo_design_share_export_and_learn_and_the_menu_holds_the_rest() {
        // UX overhaul spec 4.2: the reset, the example design, the file's save and load and
        // the details toggle in the Design menu; Share and Export on the bar; no Equation panel
        // button (decision C-11).
        use crate::gui::explorer::EQUATION_PANEL;
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        let output = harness.frame(Vec::new());
        let bar = text_rect(&output, HEADING).expect("the title");
        let in_bar = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text).is_some_and(|r| (r.center().y - bar.center().y).abs() < 20.0)
        };
        for text in [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT, LEARN] {
            assert!(in_bar(&output, text), "{text} on the bar");
        }
        let menu = [
            RESET_TO_START,
            OPEN_EXAMPLE,
            SAVE_DESIGN,
            LOAD_DESIGN,
            SHOW_DETAILS,
        ];
        for text in menu.iter().chain(&[EQUATION_PANEL, RESET_ALL]) {
            assert_eq!(count(&output, text), 0, "{text} not on the first view");
        }
        // Left to right: Undo, Redo, Design, Share, Export, Learn.
        let lefts: Vec<f32> = [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT, LEARN]
            .iter()
            .map(|t| text_rect(&output, t).unwrap().left())
            .collect();
        assert!(lefts.windows(2).all(|w| w[0] < w[1]), "{lefts:?}");
        // The menu, under its button, top to bottom.
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        let tops: Vec<f32> = menu
            .iter()
            .map(|t| text_rect(&output, t).unwrap_or_else(|| panic!("{t}")).top())
            .collect();
        assert!(tops.windows(2).all(|w| w[0] < w[1]), "{tops:?}");
        assert!(tops[0] > bar.bottom(), "under the bar");
        // Opening the menu edits nothing.
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_top_bar_s_buttons_show_their_box_only_under_the_pointer() {
        // Stage 1 review S-3: a word at rest, a box while hovered.
        use crate::gui::test_support::flat_shapes;
        let boxed = |output: &egui::FullOutput, text: &str| {
            let at = text_rect(output, text).unwrap();
            flat_shapes(output).iter().any(|shape| match shape {
                // A button's box, not the bar's own fill behind it.
                egui::Shape::Rect(r) => {
                    r.rect.contains_rect(at) && r.rect.width() < 200.0 && r.fill.a() > 0
                }
                _ => false,
            })
        };
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        for text in [DESIGN_MENU, SHARE, EXPORT] {
            assert!(!boxed(&output, text), "{text} at rest");
        }
        let at = text_rect(&output, SHARE).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert!(boxed(&output, SHARE), "Share hovered");
        assert!(!boxed(&output, EXPORT), "Export not hovered");
    }

    #[test]
    fn learn_lists_the_notes_in_the_explain_panel_and_a_title_shows_its_note() {
        // UX overhaul spec 4.2: Learn, on the top bar, lists the shown notes by title in the
        // Explain panel; a title shows the note there; Esc returns the orientation (5.1).
        use crate::gui::explain::{ORIENTATION, shown_notes};
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(TALL);
        harness.click_text(LEARN);
        assert_eq!(harness.panel.selection, Selection::NoteList);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let left = right_column_left(&output);
        for note in shown_notes(harness.panel.clamps_paused) {
            let title = glyph_safe(note.title);
            assert!(
                text_rects(&output, &title).iter().any(|r| r.left() > left),
                "{title} in the Explain panel"
            );
        }
        let note = shown_notes(harness.panel.clamps_paused).nth(3).unwrap();
        harness.click_text(&glyph_safe(note.title));
        assert_eq!(harness.panel.selection, Selection::Note(note.id));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, ORIENTATION), 0);
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert!(!harness.panel.explorer.open, "Learn opens no equation");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_details_toggle_starts_off_and_a_click_turns_it_on_and_off() {
        let mut harness = Harness::new();
        assert!(!harness.panel.show_details);
        assert!(!MagcouplingPanel::default().show_details);
        tooltips_at_once(&harness.ctx);
        // In the Design menu (decision C-12), not on the bar.
        assert_eq!(count(&harness.frame(Vec::new()), SHOW_DETAILS), 0);
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SHOW_DETAILS), 1);
        // Its hover says what it shows.
        // egui holds a tooltip back while the click that opened the menu is newer than the
        // pointer's last move: the move comes a second later.
        let at = text_rect(&output, SHOW_DETAILS).unwrap().center();
        harness.frame_after(1.0, vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SHOW_DETAILS_HINT), 1);
        harness.click_text(SHOW_DETAILS);
        assert!(harness.panel.show_details);
        // The click closed the menu.
        assert_eq!(count(&harness.frame(Vec::new()), SHOW_DETAILS), 0);
        harness.menu(SHOW_DETAILS);
        assert!(!harness.panel.show_details);
    }

    #[test]
    fn the_toggle_and_the_texts_it_adds_are_in_plain_words() {
        use crate::gui::corrections::{MARKER_HEADING, MARKER_SCOPE};
        use crate::gui::dashboard::MARKER_LEGEND;
        use crate::gui::guidance::{EXAMPLE_DESIGN, EXAMPLE_DESIGN_HOVER, plain_text_violations};
        use crate::gui::inputs::ORIGINAL_NOTE;
        // Spec 3.4: what this slice writes passes the guard. The ids and cells the toggle
        // reveals are the data it gates, not texts of this slice.
        for text in [
            SHOW_DETAILS,
            SHOW_DETAILS_HINT,
            MARKER_LEGEND,
            MARKER_HEADING,
            MARKER_SCOPE,
            ORIGINAL_NOTE,
            EXAMPLE_DESIGN,
            EXAMPLE_DESIGN_HOVER,
        ] {
            assert_eq!(plain_text_violations(text), Vec::<&str>::new(), "{text:?}");
        }
    }

    #[test]
    fn the_details_toggle_is_a_view_choice_in_no_file_link_or_undo_step() {
        // Spec S2-5: GUI-only state, like the welcome strip's: the design, its file, its link
        // and the undo history are untouched, and a load, undo, reset all or a share link
        // opened as the session's start keep it; only a new panel starts with it off.
        let kept = |harness: &mut Harness, after: &str| {
            harness.frame(Vec::new());
            assert!(harness.panel.show_details, "{after}");
        };
        let mut harness = Harness::new();
        let file = design_to_json(&harness.panel.design());
        let link = harness.panel.share_link();
        harness.menu(SHOW_DETAILS);
        assert!(harness.panel.show_details);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(design_to_json(&harness.panel.design()), file);
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
        assert!(!harness.panel.history.can_undo(&Design::default()));
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        kept(&mut harness, "a loaded design file");
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), Design::default());
        kept(&mut harness, "undo");
        harness.menu(RESET_ALL);
        kept(&mut harness, "reset all");
        harness
            .panel
            .open_share_payload(&encode_share_payload(&gap_design(1.6)))
            .unwrap();
        kept(&mut harness, "a share link opened as the session's start");
        assert!(!MagcouplingPanel::new().show_details, "a new session");
    }

    #[test]
    fn a_value_s_hover_names_its_path_cell_and_corrections_only_with_the_spreadsheet_details() {
        use crate::gui::corrections::MARKER_HEADING;
        use crate::gui::dashboard::{dashboard_lines, hover_text};
        // The pull-out on the Size tab: its readout's hover, and its dashboard row's text (the
        // Results tab and the spreadsheet's Summary read the row).
        let path = "model.pullout_Nm";
        let cell = result_info(path)
            .unwrap()
            .cell
            .clone()
            .expect("the pull-out has a cell");
        let shown = displayed_pullout(&DesignInputs::default());
        for show_details in [false, true] {
            let mut harness = Harness::first_view();
            harness.panel.show_details = show_details;
            tooltips_at_once(&harness.ctx);
            let output = harness.frame(Vec::new());
            let row = dashboard_lines(harness.panel.results())
                .into_iter()
                .find(|line| line.path == path)
                .expect("a dashboard row")
                .tooltip(show_details);
            let readout = hover_text(path, show_details).expect("a result");
            for want in [&row, &readout] {
                assert_eq!(want.lines().any(|l| l == path), show_details, "{want}");
                assert_eq!(want.lines().any(|l| l == cell), show_details, "{want}");
                assert_eq!(want.contains(MARKER_HEADING), show_details, "{want}");
            }
            // The Size tab's readout in the centre.
            let rects = text_rects(&output, &shown);
            assert_eq!(rects.len(), 1, "{rects:?}");
            // At the text's right end (a wrapped label's rect starts at the row's left).
            let at = egui::pos2(rects[0].right() - 4.0, rects[0].center().y);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert!(drawn_texts(&output).contains(&readout), "{readout:?}");
        }
    }

    #[test]
    fn an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details() {
        use crate::gui::inputs::input_tooltip;
        let entry = InputCatalogue::get().entry(FACE_GAP).unwrap();
        let cell = entry.meta.cell.expect("the face gap has a cell");
        let label = crate::gui::plain_text::label_for(FACE_GAP);
        for show_details in [false, true] {
            // The Key design group's face-gap row, its label's hover.
            let mut harness = Harness::new();
            harness.panel.show_details = show_details;
            tooltips_at_once(&harness.ctx);
            let output = harness.frame(Vec::new());
            let rect = text_rect(&output, label).unwrap_or_else(|| panic!("no {label:?}"));
            harness.frame(vec![egui::Event::PointerMoved(rect.center())]);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let want = input_tooltip(entry, &entry.default, show_details);
            assert!(drawn_texts(&output).contains(&want), "{want:?}");
            assert_eq!(want.lines().any(|line| line == FACE_GAP), show_details);
            assert_eq!(want.lines().any(|line| line == cell), show_details);
        }
    }

    #[test]
    fn a_share_link_opened_at_start_up_opens_on_the_first_view() {
        // The web app opens a link's design on a new panel (MagcouplingApp::open_share_payload):
        // the Size tab and the Explain panel's orientation, as for the default design.
        let mut panel = MagcouplingPanel::new();
        panel
            .open_share_payload(&encode_share_payload(&gap_design(2.0)))
            .unwrap();
        let mut harness = Harness::of(panel, SCREEN);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, crate::gui::explain::ORIENTATION), 1);
        assert_eq!(harness.panel.design(), gap_design(2.0));
    }

    #[test]
    fn idle_frames_on_the_first_view_change_no_input() {
        // The Size tab's controls (the Required slider among them), like every input row, write
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are. Slice 3 test 11: so do they while the Explain panel
        // draws the torque card (the Hot-low tile's entry), at the shipped clamp pause and
        // without it.
        use crate::gui::tiles::Tile;
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.coupling.inner_back_apothem_mm += 0.01234;
        design.inputs.coupling.magnets.axial_length_mm = Some(12.7123);
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        design.inputs.metal.required_min_Nm = 2.2345;
        for (paused, card) in [(false, false), (true, false), (true, true)] {
            let mut harness = Harness::first_view();
            harness.panel.clamps_paused = paused;
            harness.panel.open_design(design.clone());
            if card {
                harness.panel.selection = Selection::Result(Tile::HotLow.path());
            }
            let mut output = harness.frame(Vec::new());
            for _ in 0..15 {
                output = harness.frame(Vec::new());
            }
            let case = format!("paused {paused}, the card {card}");
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
            assert_eq!(
                count(&output, CAPTION),
                usize::from(card),
                "{case}: the card's caption shows exactly in the Hot-low tile's entry"
            );
            assert_eq!(harness.panel.design(), design, "{case}");
            assert_eq!(harness.panel.last_error, None, "{case}");
            assert_eq!(harness.panel.history.undo_len(), 0, "{case}");
        }
    }

    #[test]
    fn a_focused_input_opens_the_detailed_inputs_on_the_first_view() {
        // On the Size tab the input groups sit under the closed "Detailed inputs": a leaf term
        // or a dashboard fix (spec 3.4) naming an input opens them and frames its row, for a
        // Key design input, an assumption and an input under an Advanced heading alike. A tall
        // screen draws every row.
        let mut assumptions = 0;
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
            // An assumption's row is the Assumptions fold's (UX overhaul spec 4.3), which
            // opens with its rationale; any other row's fold is the detailed inputs.
            let assumption = ASSUMPTIONS.iter().find(|a| a.paths.contains(&path));
            assert_eq!(
                assumption.map(|a| count(&output, a.rationale)),
                assumption.map(|_| 1),
                "{path}"
            );
            assumptions += usize::from(assumption.is_some());
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
        }
        assert_eq!(assumptions, 1, "one of the paths is an assumption");
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
        let mut harness = Harness::on_screen(TALL).on_results();
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        // The Results tab's Headline group draws each under its catalogue label.
        for (_, path) in HEADLINE {
            let label = result_info(path).unwrap().meta.label;
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
        // The workflow order, filtered by "." (in every path): all 187 rows, both vacuum
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
        // Slice 3 test 11: the Size tab with the torque card drawn (the Hot-low tile's Explain
        // entry) and then without it, at the shipped clamp pause: idle frames still change no
        // input.
        harness.panel.clamps_paused = true;
        harness.click_text(CentreView::SizeAndTorque.label());
        for open in [true, false] {
            harness.panel.selection = if open {
                Selection::Result(crate::gui::tiles::Tile::HotLow.path())
            } else {
                Selection::Nothing
            };
            let mut output = harness.frame(Vec::new());
            for _ in 0..15 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(count(&output, CAPTION), usize::from(open), "open {open}");
            assert_eq!(harness.panel.design(), design, "open {open}");
            assert_eq!(harness.panel.last_error, None, "open {open}");
            assert_eq!(harness.panel.history.undo_len(), 0, "open {open}");
        }
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
        let mut harness = Harness::on_screen(TALL).on_results();
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
        let mut harness = Harness::on_screen(TALL).on_results();
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
    fn the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline() {
        // UX overhaul spec 7.1 (C-22): on screen "the default" is the baseline's value.
        use crate::gui::inputs::input_tooltip;
        // The passing test design: its face gap is 1.8 mm, the spreadsheet's 1.4 mm.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        tooltips_at_once(&harness.ctx);
        // At the baseline no row carries the dot, though it differs from the spreadsheet's
        // values in two rows of the Key design group (the face gap and the cup wall).
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 0);
        assert_eq!(count(&output, RESET_LABEL), 0);
        // The label's hover gives the baseline's value as the default.
        let entry = InputCatalogue::get().entry(FACE_GAP).unwrap();
        let tooltip = input_tooltip(entry, &Value::Num(1.8), false);
        assert!(tooltip.ends_with("Default: 1.800"), "{tooltip}");
        let label = crate::gui::plain_text::label_for(FACE_GAP);
        let at = text_center(&output, label);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(drawn_texts(&output).contains(&tooltip), "{tooltip:?}");
        // One step off the baseline: the dot and the reset, whose hover names the baseline's
        // value; the reset returns it, not the spreadsheet's.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 1);
        assert_eq!(count(&output, RESET_LABEL), 1);
        let at = text_center(&output, RESET_LABEL);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Back to the default: 1.800"), 1);
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs(), &baseline);
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
        // At the spreadsheet's values both rows carry the dot: they differ from the baseline.
        harness.panel.inputs = DesignInputs::default();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 2);
        assert_eq!(count(&output, RESET_LABEL), 2);
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
        // stay as they are. On a tall screen: the Key design group's hint lines push the
        // groups' headings below a laptop screen's fold, and egui paints no label scrolled out.
        let mut harness = Harness::on_screen(TALL);
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
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
        assert_eq!(
            count(
                &output,
                "Requirements and operating conditions / Drive and gearbox"
            ),
            1
        );
        // BL-064: the gearbox shaft's fitting temperature, under the shaft group.
        assert_eq!(
            count(&output, "Shaft and hub joint / Fitting temperatures"),
            1
        );
        for label in [
            "Gearbox ratio",
            "Gearbox efficiency",
            "Gearbox input torque rating",
            "Gearbox shaft temperature at fitting",
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
        // On the inputs side: the check list has a "Shaft joints" row of its own.
        let headings = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .filter(|rect| rect.left() < INPUTS_WIDTH)
                .count()
        };
        assert_eq!(headings("Coupling"), 1);
        assert_eq!(headings("Shaft joints"), 1);
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
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
    fn the_right_column_shows_the_stored_3d_line_and_the_correction_ids() {
        use crate::gui::dashboard::MARKER_LEGEND;
        let mut harness = Harness::on_screen(TALL);
        // The check list closes with the stored-3D line, whatever is selected; the entry of a
        // check that reads the stored fields repeats it under its numbers.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        let output = harness.select("temperature.summary.verdict");
        assert_eq!(count(&output, STORED_3D_LABEL), 2);
        let output = harness.select(SPACE_CLAIM_CHECK);
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        assert_eq!(count(&output, "Inside the space claim"), 1);
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(END_EFFECT_BANNER))
        );
        // The correction ids and their legend only with the spreadsheet details (spec S2-5):
        // the hot-low torque's, among the numbers behind the hot minimum.
        let marker = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .find(|line| line.path == "metal.torque_hot_low_Nm")
            .expect("the hot-low torque's row")
            .marker;
        assert!(
            !marker.is_empty(),
            "the hot-low torque carries correction ids"
        );
        harness.panel.show_details = true;
        let output = harness.select(HOT_MIN_CHECK);
        assert!(count(&output, &marker) >= 1);
        assert_eq!(count(&output, MARKER_LEGEND), 1);
        harness.panel.show_details = false;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &marker), 0);
        assert_eq!(count(&output, MARKER_LEGEND), 0);
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
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
        // Over the centre region, whichever view it shows (decision M42-1). The check list
        // marks the End-effect model check as failing; it does not repeat the banner.
        let text = "End-effect model out of range (end-effect factor -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text]);
    }

    #[test]
    fn a_design_past_the_space_claim_shows_its_overshoot() {
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let output = harness.select(SPACE_CLAIM_CHECK);
        let claim = drawn_texts(&output)
            .into_iter()
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the verdict, among the numbers of the check's entry");
        assert!(claim.contains("overall length"), "{claim}");
    }

    #[test]
    fn the_results_table_lists_results_and_filters_by_the_search() {
        let entries = crate::gui::results_table::table_entries();
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        // The cell column, drawn with the spreadsheet details (spec S2-5).
        harness.panel.show_details = true;
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
        // Tall enough to draw the headline and the whole torque chain; the cells are drawn with
        // the spreadsheet details (spec S2-5).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.panel.show_details = true;
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
        let matches = search(entries, "f_end", false);
        let shown = torque.rows.iter().filter(|i| matches.contains(i)).count();
        let heading = format!("{} ({shown})", torque.label);
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        // The cell column, drawn with the spreadsheet details (spec S2-5).
        harness.panel.show_details = true;
        tooltips_at_once(&harness.ctx);
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
        // Manual magnets: the defaults' five red checks, two amber rating checks and the amber
        // daily-cycle screen at the entered bondline (BL-063).
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.magnets.part_inner.clear();
        harness.panel.inputs.coupling.magnets.part_outer.clear();
        harness.click_text(CentreView::Results.label());
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results(), false);
        assert_eq!(failing.len(), 8);
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("8 of {total} results")), 1);
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
        // Five red badges and three amber in the table, in the check list's shapes (decision
        // C-4): every disc and triangle drawn but the check list's own, one per failing check.
        let visuals = harness.ctx.style().visuals.clone();
        let panel = &harness.panel;
        let fail_marks = crate::gui::guidance::counted_guides(false)
            .filter(|guide| {
                crate::gui::guidance::level(guide, panel.inputs(), panel.results()) == Level::Bad
            })
            .count();
        let caution_marks = crate::gui::guidance::counted_guides(false)
            .filter(|guide| {
                crate::gui::guidance::level(guide, panel.inputs(), panel.results())
                    == Level::Caution
            })
            .count();
        let table_badges = |level: Level| {
            let drawn = crate::gui::test_support::flat_shapes(&output)
                .into_iter()
                .filter(|shape| mark_at(shape, &visuals).is_some_and(|(_, l)| l == level))
                .count();
            drawn
                - match level {
                    Level::Bad => fail_marks,
                    Level::Caution => caution_marks,
                    Level::Good => 0,
                }
        };
        assert_eq!(
            (table_badges(Level::Bad), table_badges(Level::Caution)),
            (5, 3)
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
        harness.frame(vec![egui::Event::Text("ripple".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NO_RESULT), 1);
        assert_eq!(count(&output, NOTHING_FAILS), 0);
    }

    /// The badge levels drawn on the line of `row`, a rect of the results table's text: the
    /// badge circles whose centre `in_table` places in the table, by x (the dashboard draws its
    /// own beside it).
    fn table_badges_on(
        output: &egui::FullOutput,
        visuals: &egui::Visuals,
        row: egui::Rect,
        in_table: impl Fn(f32) -> bool,
    ) -> Vec<Level> {
        let shapes = crate::gui::test_support::flat_shapes(output);
        [Level::Good, Level::Caution, Level::Bad]
            .into_iter()
            .filter(|level| {
                shapes.iter().any(|shape| {
                    mark_at(shape, visuals).is_some_and(|(at, l)| {
                        l == *level
                            && in_table(at.x)
                            && (at.y - row.center().y).abs() < row.height()
                    })
                })
            })
            .collect()
    }

    /// Where shape sits and the level it stands for, if it is a state's mark
    /// (checks::state_mark, decision C-4): a filled disc fails, a triangle asks for a look,
    /// a ring passes.
    fn mark_at(shape: &egui::Shape, visuals: &egui::Visuals) -> Option<(egui::Pos2, Level)> {
        match shape {
            egui::Shape::Circle(c) if c.fill == Level::Bad.color(visuals) => {
                Some((c.center, Level::Bad))
            }
            egui::Shape::Circle(c)
                if c.fill == egui::Color32::TRANSPARENT
                    && c.stroke.color == Level::Good.color(visuals) =>
            {
                Some((c.center, Level::Good))
            }
            egui::Shape::Path(p)
                if p.closed && p.points.len() == 3 && p.fill == Level::Caution.color(visuals) =>
            {
                let sum = p
                    .points
                    .iter()
                    .fold(egui::Vec2::ZERO, |s, q| s + q.to_vec2());
                Some(((sum / 3.0).to_pos2(), Level::Caution))
            }
            _ => None,
        }
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
        use crate::gui::dashboard::{CHECKS, dashboard_lines, failing_checks};
        use crate::gui::results_table::{FAILING_ONLY, table_entries};
        use crate::gui::test_support::flat_shapes;
        // Short magnets (f_end < 0): the rows computed from the pull-out are greyed, without
        // badges. In the table's headline (open) each check row carries exactly its dashboard
        // line's badge, so the hot minimum (red by its text), the screw sized from the
        // invalid pull-out, and the shaft joints and gearbox checks of a cold-high below zero
        // (all three green by their texts) carry none, and the failing filter leaves the hot
        // minimum out.
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        let visuals = harness.ctx.style().visuals.clone();
        // Between the inputs and the check list.
        let checks_left = right_column_left(&output);
        let centre = |x: f32| x > INPUTS_WIDTH && x < checks_left;
        let lines = dashboard_lines(harness.panel.results());
        let checks: Vec<_> = lines
            .iter()
            .filter(|line| CHECKS.contains(&line.path))
            .collect();
        assert_eq!(checks.len(), 9);
        for line in checks {
            let row = text_rects(&output, line.label)
                .into_iter()
                .find(|r| centre(r.center().x))
                .unwrap_or_else(|| panic!("{} in the table", line.path));
            let want: Vec<Level> = line.level.into_iter().collect();
            assert_eq!(
                table_badges_on(&output, &visuals, row, centre),
                want,
                "{}",
                line.path
            );
        }
        // The failing filter: the hot minimum is not listed; the end-effect check is, first.
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results(), false);
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
        // Seven red badges and four amber in the table, in their shapes.
        let in_table = |level: Level| {
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| {
                    mark_at(shape, &visuals).is_some_and(|(at, l)| l == level && centre(at.x))
                })
                .count()
        };
        assert_eq!((in_table(Level::Bad), in_table(Level::Caution)), (7, 4));
    }

    /// The mark rects of the trace (the panel's mark colour) painted in a frame.
    fn trace_marks(harness: &Harness, output: &egui::FullOutput) -> Vec<egui::Rect> {
        mark_rects(output, Some(look::mark_color(&harness.ctx.style().visuals)))
    }

    #[test]
    fn the_panel_s_marks_are_the_host_s_link_colour() {
        // The changed dot and the trace's banner and frames are drawn in `look::mark_color`,
        // the host's link colour, whatever colour the host gives the name of a selected choice
        // (the standalone app's theme: the text colour, on its selection fill).
        let mut harness = Harness::new();
        let visuals = harness.ctx.style().visuals.clone();
        let blue = visuals.hyperlink_color;
        assert_ne!(blue, visuals.selection.stroke.color);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        assert_eq!(text_color(&output, CHANGED_DOT), Some(blue));
        harness.trace_input(label_for(FACE_GAP));
        let banner = harness.panel.trace.clone().expect("a trace").banner(false);
        let output = harness.frame(Vec::new());
        assert_eq!(text_color(&output, &banner), Some(blue));
        assert!(!mark_rects(&output, Some(blue)).is_empty());
    }

    #[test]
    fn tracing_an_input_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::guidance::counted_guides;
        use crate::gui::trace::TraceKind;
        // Spec 5.2: a click on the label selects the input and traces nothing; its entry's link
        // traces it.
        let mut harness = Harness::new();
        let gap = label_for(FACE_GAP);
        harness.click_text(gap);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, FACE_GAP)
        );
        assert_eq!(trace.paths, registry().downstream(FACE_GAP));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner(false)), 1);
        let marks = trace_marks(&harness, &output);
        // In the check list: one frame per check the gap drives, none on the others.
        let checks_left = right_column_left(&output);
        let on_checks = marks.iter().filter(|m| m.center().x > checks_left).count();
        let driven = counted_guides(false)
            .filter(|guide| trace.marks(guide.check))
            .count();
        assert!(driven > 0);
        assert_eq!(on_checks, driven);
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
        // A second click on the link ends the trace and its frames.
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
        let output = harness.frame(Vec::new());
        assert!(trace_marks(&harness, &output).is_empty());
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing() {
        use crate::gui::explain::TRACE_LINK;
        // Spec 5.1 and 5.2: a click on the label of a row of the detailed inputs or the
        // assumptions selects its input, whose entry the Explain panel shows in the same frame;
        // a hover shows the row's tooltip and selects nothing (C-2). The face gap's Key design
        // row selects the entry the Size column's gap row selects (decision D-35). No click
        // traces, edits the design or makes an undo step.
        let op_temp = "coupling.op_temp_C";
        let label = label_for(op_temp);
        let mut harness = Harness::new();
        tooltips_at_once(&harness.ctx);
        let output = harness.frame(Vec::new());
        let at = text_rect(&output, label).expect("its row").center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing, "a hover");
        let output = harness.click(at);
        assert_eq!(harness.panel.selection, Selection::Input(op_temp));
        let checks_left = right_column_left(&output);
        let in_explain = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .any(|r| r.left() > checks_left)
        };
        assert!(in_explain(label), "its entry, in the click's frame");
        assert!(in_explain(TRACE_LINK));
        harness.click_text(label_for(FACE_GAP));
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        // An assumption's row, in the Assumptions view.
        let assumption = ASSUMPTIONS[0].paths[0];
        harness.click_text(InputsView::Assumptions.label());
        harness.click_text(label_for(assumption));
        assert_eq!(harness.panel.selection, Selection::Input(assumption));
        // On the opening tab, in the Detailed inputs fold (a tall window: its row in view).
        let mut first = Harness::of(MagcouplingPanel::new(), TALL);
        first.click_text(DETAILED_INPUTS);
        first.click_text(label);
        assert_eq!(first.panel.selection, Selection::Input(op_temp));
        for harness in [&harness, &first] {
            assert_eq!(harness.panel.trace, None);
            assert_eq!(
                harness.panel.design(),
                harness.panel.settled_design().clone()
            );
            assert_eq!(harness.panel.history.undo_len(), 0);
        }
    }

    #[test]
    fn touching_a_detailed_row_s_widget_selects_its_input() {
        // Spec 5.2: touching an input selects it, as part 2b-i's touches do for the Size
        // column (size_controls::touched): the keyboard's focus arriving on a Key design row's
        // slider (a focus that edits nothing), a drag of it, and a click on its reset. A hover
        // over the slider selects nothing (C-2).
        let op_temp = "coupling.op_temp_C";
        let before = KEY_DESIGN[KEY_DESIGN.iter().position(|p| *p == op_temp).unwrap() - 1];
        let mut harness = Harness::new();
        let target = harness.widget(op_temp).id;
        let at = harness.widget(op_temp).rect.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing, "a hover");
        // Tab from the row before (a focus requested between frames is not one that arrives).
        harness.focus(before);
        harness.panel.selection = Selection::Nothing;
        for _ in 0..6 {
            harness.frame(key_tap(egui::Key::Tab));
            if harness.ctx.memory(|m| m.has_focus(target)) {
                break;
            }
        }
        assert!(
            harness.ctx.memory(|m| m.has_focus(target)),
            "Tab reaches it"
        );
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.selection,
            Selection::Input(op_temp),
            "the focus"
        );
        assert_eq!(
            harness.panel.history.undo_len(),
            0,
            "the focus edits nothing"
        );
        harness.ctx.memory_mut(|m| m.surrender_focus(target));
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(vec![egui::Event::PointerMoved(at + egui::vec2(12.0, 0.0))]);
        harness.frame(vec![primary_button(at + egui::vec2(12.0, 0.0), false)]);
        assert_eq!(harness.panel.selection, Selection::Input(op_temp), "a drag");
        assert_ne!(harness.panel.design(), Design::default(), "the drag edits");
        harness.panel.selection = Selection::Nothing;
        harness.click_text(RESET_LABEL);
        assert_eq!(
            harness.panel.selection,
            Selection::Input(op_temp),
            "a reset"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::TraceKind;
        // Spec 5.2: the trace is a link in the input's Explain entry. A click traces the input,
        // a second click ends it (decision O-8); neither edits the design or makes an undo step,
        // and the selection stays.
        let mut harness = Harness::new();
        harness.panel.selection = Selection::Input(FACE_GAP);
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, FACE_GAP)
        );
        let output = harness.frame(Vec::new());
        assert!(
            !trace_marks(&harness, &output).is_empty(),
            "its frames show"
        );
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group() {
        use crate::gui::trace::{CLEAR_TRACE, TraceKind};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0)).on_results();
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
        assert_eq!(count(&output, &trace.banner(false)), 1);
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
        harness.trace_input(label_for(FACE_GAP));
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
        harness.trace_input(InputCatalogue::get().entry(drive).unwrap().meta.label);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(trace.source, drive);
        assert!(trace.paths.is_empty());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner(false)), 1);
        assert!(
            trace
                .banner(false)
                .contains("no result with an equation reads it")
        );
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
        harness.trace_input(label_for(FACE_GAP));
        let output = harness.frame(Vec::new());
        assert!(harness.panel.trace.is_some());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
    }

    #[test]
    fn a_trace_filter_left_ticked_off_the_table_does_not_filter_a_later_trace() {
        use crate::gui::results_table::{TRACED_ONLY, table_entries};
        use crate::gui::trace::CLEAR_TRACE;
        // "Traced only" goes off with the trace even when the table is not on screen: Clear
        // trace and a trace of another input, both with the Geometry view showing, must not
        // leave the box ticked for the new trace (it would filter the table unasked).
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
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
        harness.trace_input(label_for(POLES));
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
        // A result clicked in the Explain panel replaces an input's trace with its own (which marks
        // inputs, and cannot filter the table): the box goes off with it, so an input traced
        // next, with no Clear trace between, does not find it ticked.
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
        harness.click_text(CentreView::Geometry.label());
        // One of the numbers behind a check, in that check's Explain entry.
        let (outside, check) = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .filter_map(|line| {
                let (_, check) = crate::gui::dashboard::DASHBOARD
                    .iter()
                    .find(|&&(path, _)| path == line.path)?;
                Some((line, (*check)?))
            })
            .find(|(line, _)| registry().equation_for(line.path).is_some() && !gap.marks(line.path))
            .expect("a number behind a check that the face gap does not drive");
        let output = harness.select(check);
        // The rightmost: the Explain panel's.
        let value = text_rects(&output, &outside.shown_value())
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .unwrap();
        harness.click(value.center());
        let replaced = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (replaced.kind, replaced.source.as_str()),
            (TraceKind::Result, outside.path)
        );
        harness.trace_input(label_for(POLES));
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
        harness.trace_input(label_for(FACE_GAP));
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
        // A result outside the trace, with a record (in the Explain panel): its trace replaces
        // the input's.
        // One of the numbers behind a check, in that check's Explain entry.
        let (outside, check) = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .filter_map(|line| {
                let (_, check) = crate::gui::dashboard::DASHBOARD
                    .iter()
                    .find(|&&(path, _)| path == line.path)?;
                Some((line, (*check)?))
            })
            .find(|(line, _)| {
                registry().equation_for(line.path).is_some() && !trace.marks(line.path)
            })
            .expect("a number behind a check that the face gap does not drive");
        let output = harness.select(check);
        // The rightmost: the Explain panel's.
        let value = text_rects(&output, &outside.shown_value())
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
        harness.trace_input(label_for(FACE_GAP));
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
        let bondline = search(entries, "bondline", false);
        assert!(!bondline.is_empty());
        assert!(bondline.iter().all(|&i| !trace.marks(&entries[i].path)));
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NO_RESULT), 1);
        assert_eq!(count(&output, NOTHING_TRACED), 0);
    }

    #[test]
    fn a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::CLEAR_TRACE;
        // The banner sits at the foot of the inputs side: a trace moves no row above it (decision
        // O-8). The label selects its input (spec 5.2): a second click on it lands on it again
        // and keeps the trace, which the entry's link ends.
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
        assert_eq!(harness.panel.trace, None, "the label traces nothing");
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
        // The strip sizes from the last frame: let it settle.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner(false)), 1);
        assert_eq!(on_inputs_side(&output, gap), label);
        assert_eq!(on_inputs_side(&output, toggle), toggle_at);
        assert!(on_inputs_side(&output, CLEAR_TRACE).top() > label.bottom());
        // The second click, at the same point: the label again, the trace kept.
        harness.click(label.center());
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
        let output = harness.frame(Vec::new());
        assert_eq!(on_inputs_side(&output, gap), label);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::TraceKind;
        // In Torque -> Magnets the free variable's row is locked (its value widgets disabled);
        // its label still selects the input and its entry traces it, as every row's does.
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
        assert_eq!(harness.panel.selection, Selection::Input(AXIAL_LENGTH));
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, AXIAL_LENGTH)
        );
        // A second click ends it; no click edits the design or makes an undo step.
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.history.undo_len(), undo);
        assert!(!harness.widget(AXIAL_LENGTH).enabled(), "still locked");
    }

    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
        // The M4-1 review's ~930 px window: the inputs (320) and the dashboard (300) leave the
        // table about 294 points. The row's value must end left of the dashboard, with the
        // cell and marker columns (the spreadsheet details) and without them.
        for show_details in [false, true] {
            let mut harness = Harness::on_screen(egui::vec2(930.0, 1024.0));
            harness.panel.show_details = show_details;
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
        // The top bar's button, then the results table's.
        harness.click_text(EXPORT);
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
        harness.click_text(EXPORT);
        let books = queued_spreadsheets(&mut harness);
        let first_group = InputCatalogue::get().groups[0].label;
        assert_eq!(
            books[0][1].1[1][0],
            calamine::Data::String(first_group.to_owned())
        );
        // Torque -> Magnets: the solved length, noted, and the sizing on the Summary.
        harness.size();
        harness.click_text(EXPORT);
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
        let mut harness = Harness::on_screen(TALL).on_results();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_ne!(harness.panel.inputs(), &DesignInputs::default());

        harness.menu(RESET_ALL);
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
    fn the_settled_design_follows_settled_edits_undo_redo_and_an_opened_link() {
        // UX overhaul spec 7.2: what the web app writes the page's address from.
        let mut harness = Harness::new();
        assert_eq!(harness.panel.settled_design(), &Design::default());
        assert_eq!(harness.panel.share_base(), PUBLIC_BASE_URL);
        // A key held down: the edit is in progress, so the settled design waits.
        harness.focus(FACE_GAP);
        harness.frame(vec![key_event(egui::Key::ArrowRight, true)]);
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(harness.panel.settled_design(), &Design::default());
        harness.frame(vec![key_event(egui::Key::ArrowRight, false)]);
        assert_eq!(harness.panel.settled_design(), &gap_design(1.41));
        harness.click_text(UNDO);
        assert_eq!(harness.panel.settled_design(), &Design::default());
        harness.click_text(REDO);
        assert_eq!(harness.panel.settled_design(), &gap_design(1.41));
        // A loaded file settles in the frame that follows it.
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.settled_design(), &gap_design(2.0));
        // A link opened as the session's start is settled at once.
        harness
            .panel
            .open_share_payload(&encode_share_payload(&gap_design(1.6)))
            .unwrap();
        assert_eq!(harness.panel.settled_design(), &gap_design(1.6));
        harness
            .panel
            .set_share_base("http://localhost:8080/tools/magcoupler/");
        assert_eq!(
            harness.panel.share_base(),
            "http://localhost:8080/tools/magcoupler/"
        );
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
        harness.menu(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), gap_design(1.41));
    }

    /// `inputs` as a design with the default sizing state: what a panel whose baseline is
    /// `inputs` opens on and Reset all returns to.
    fn design_at(inputs: &DesignInputs) -> Design {
        Design {
            inputs: inputs.clone(),
            sizing: SizingState::default(),
        }
    }

    #[test]
    fn a_new_panel_keeps_the_spreadsheet_s_values_as_its_baseline() {
        // UX overhaul spec 7.1 (C-21): a host or a test that builds the panel the old way sees
        // the old behaviour.
        assert_eq!(MagcouplingPanel::new().baseline, DesignInputs::default());
        assert_eq!(
            MagcouplingPanel::default().baseline,
            DesignInputs::default()
        );
    }

    #[test]
    fn a_panel_opens_on_its_baseline_and_reset_all_returns_to_it() {
        let baseline = passing_design();
        let start = design_at(&baseline);
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        assert_eq!(harness.panel.design(), start);
        assert_eq!(harness.panel.results(), &compute_all(&baseline));
        // The session starts there: nothing to undo.
        assert!(!harness.panel.history.can_undo(&start));
        // Auto on is part of the design, so Reset all turns it off again.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        assert_ne!(harness.panel.design(), start);
        harness.menu(RESET_TO_START);
        assert_eq!(harness.panel.design(), start);
        assert_eq!(harness.panel.results(), &compute_all(&baseline));
        // Reset all is an edit like any other: Undo brings the changed design back.
        harness.click_text(UNDO);
        assert_ne!(harness.panel.design(), start);
    }

    #[test]
    fn the_status_line_reads_starting_design_at_the_baseline() {
        // UX overhaul spec 7.1: the name follows the design; the count is live, as before.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline, SCREEN);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.05;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Your design: passes all 10 checks"), 1);
        harness.menu(RESET_TO_START);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
    }

    #[test]
    fn the_reset_s_status_names_the_design_its_item_names() {
        // The click and its confirmation use one name for the design.
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        harness.menu(RESET_TO_START);
        assert_eq!(harness.panel.status.as_deref(), Some(STARTING_RESTORED));
        let mut harness = Harness::new();
        harness.menu(RESET_ALL);
        assert_eq!(harness.panel.status.as_deref(), Some(DEFAULT_RESTORED));
    }

    #[test]
    fn a_design_menu_item_run_from_the_keyboard_closes_the_menu() {
        // As a click on an item does: Tab to Save design, Enter.
        let mut harness = Harness::new();
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        let save = text_rect(&output, SAVE_DESIGN).unwrap().center();
        let load = text_rect(&output, LOAD_DESIGN).unwrap().center();
        // Not the menu's own frame, which holds every item and takes a Tab stop too.
        let on_save = |harness: &Harness| {
            harness
                .ctx
                .memory(|m| m.focused())
                .and_then(|id| harness.ctx.read_response(id))
                .is_some_and(|r| r.rect.contains(save) && !r.rect.contains(load))
        };
        for _ in 0..12 {
            if on_save(&harness) {
                break;
            }
            harness.frame(key_tap(egui::Key::Tab));
        }
        assert!(on_save(&harness), "Tab reaches Save design");
        harness.frame(key_tap(egui::Key::Enter));
        assert!(matches!(
            harness.panel.take_requests().as_slice(),
            [PanelRequest::SaveFile { .. }]
        ));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SHOW_DETAILS), 0, "the menu closed");
    }

    #[test]
    fn open_the_example_design_is_one_undo_step_and_keeps_the_baseline() {
        // UX overhaul spec 7.1: the button opens the spreadsheet's design as an edit; the dots
        // then show what the baseline changed; Reset all still returns to the baseline.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness.menu(OPEN_EXAMPLE);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 1);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EXAMPLE_OPENED), 1);
        assert_eq!(count(&output, "Example design: fails 4 of 10 checks"), 1);
        // The Key design group shows two of the inputs the passing test design changes: the
        // face gap and the cup wall.
        assert_eq!(count(&output, CHANGED_DOT), 2);
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), design_at(&baseline));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
        assert_eq!(count(&output, CHANGED_DOT), 0);
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), Design::default());
        harness.menu(RESET_TO_START);
        assert_eq!(harness.panel.design(), design_at(&baseline));
    }

    #[test]
    fn the_first_arrival_s_new_texts_keep_the_text_rules() {
        // UX overhaul spec 8.3: plain words; nothing says the coupling protects the gearbox;
        // none of the words banned for new strings (no pressure boundary is modelled, a ring is
        // not named by a colour, the clamp section is paused); no typed-in number.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        use crate::gui::guidance::{STARTING_DESIGN, STARTING_DESIGN_HOVER, plain_text_violations};
        for text in [
            STARTING_DESIGN,
            STARTING_DESIGN_HOVER,
            EXAMPLE_DESIGN_HOVER,
            OPEN_EXAMPLE,
            EXAMPLE_OPENED,
            DESIGN_MENU,
            RESET_TO_START,
            STARTING_RESTORED,
            DEFAULT_RESTORED,
            SHARE,
            EXPORT,
        ] {
            assert_eq!(plain_text_violations(text), Vec::<&str>::new(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            let lower = text.to_lowercase();
            for banned in [
                "sealed",
                "containment",
                "leak",
                "clamp",
                "red ring",
                "blue ring",
            ] {
                assert!(!lower.contains(banned), "{banned:?} in {text:?}");
            }
            assert!(!text.chars().any(|c| c.is_ascii_digit()), "{text:?}");
        }
    }

    #[test]
    fn a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button() {
        // Reset all already opens it there, and the Design menu keeps the reset's old name
        // (spec 8.1, test 2).
        let mut harness = Harness::new();
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, OPEN_EXAMPLE), 0);
        assert_eq!(
            (count(&output, RESET_ALL), count(&output, RESET_TO_START)),
            (1, 0)
        );
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, OPEN_EXAMPLE), 1);
        assert_eq!(
            (count(&output, RESET_ALL), count(&output, RESET_TO_START)),
            (0, 1)
        );
    }

    #[test]
    fn nothing_moves_the_baseline() {
        // UX overhaul spec 7.1: not an undo, a redo, a file, a link, a link opened as the
        // session's start or Reset all.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        harness.frame(Vec::new());
        harness
            .panel
            .load_share_payload(&encode_share_payload(&gap_design(1.5)))
            .unwrap();
        harness.frame(Vec::new());
        harness.click_text(UNDO);
        harness.click_text(REDO);
        harness.menu(RESET_TO_START);
        harness
            .panel
            .open_share_payload(&encode_share_payload(&Design::default()))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.baseline, baseline);
        assert_eq!(harness.panel.design(), Design::default());
        harness.panel.reset();
        assert_eq!(harness.panel.design(), design_at(&baseline));
    }

    /// The line segments drawn in the brand accent, each as (left, right, y).
    fn accent_lines(output: &egui::FullOutput) -> Vec<(f32, f32, f32)> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::LineSegment { points, stroke } if stroke.color == look::ACCENT => {
                    Some((points[0].x, points[1].x, points[0].y))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_active_tab_is_underlined_in_the_accent_and_no_other() {
        // UX overhaul spec 6.1 (the accent's role) and decision C-5: one line, under the active
        // tab's label, as wide as the tab.
        let under = |output: &egui::FullOutput, label: egui::Rect| {
            let lines = accent_lines(output);
            assert_eq!(lines.len(), 1, "{lines:?}");
            let (left, right, y) = lines[0];
            assert!(
                left <= label.left() && label.right() <= right,
                "{lines:?} {label:?}"
            );
            assert!(
                label.bottom() <= y && y <= label.bottom() + 6.0,
                "{lines:?} {label:?}"
            );
        };
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        // The Size tab's name is drawn twice: the left side's heading, then the tab.
        let size_tab = *text_rects(&output, CentreView::SizeAndTorque.label())
            .last()
            .expect("the tab");
        under(&output, size_tab);
        // A tab still answers a click, and the line follows it on the next frame.
        let geometry = CentreView::Geometry.label();
        harness.click_text(geometry);
        assert_eq!(harness.panel.centre, CentreView::Geometry);
        let output = harness.frame(Vec::new());
        under(&output, text_rect(&output, geometry).expect("the tab"));
        // The active tab's label is in the strong text colour, another's in the quiet one
        // (`look::quiet_text`). egui's own visuals set no weak colour, and their weak text is
        // about 2.6:1, too faint for a tab's name: there it is the text colour (the
        // whole-branch review, R-5).
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(visuals.weak_text_color, None);
        assert_eq!(
            text_color(&output, geometry),
            Some(visuals.strong_text_color())
        );
        let other = PlotKind::GapSweep.label();
        assert_eq!(text_color(&output, other), Some(visuals.text_color()));
        // A theme that sets a weak colour (the standalone app's): another tab's name takes it.
        let weak = egui::Color32::from_rgb(1, 2, 3);
        harness
            .ctx
            .style_mut(|style| style.visuals.weak_text_color = Some(weak));
        assert_eq!(text_color(&harness.frame(Vec::new()), other), Some(weak));
        harness
            .ctx
            .style_mut(|style| style.visuals.weak_text_color = None);
        // The row wraps at this width: a tab on its second line is underlined where it is.
        let last = CentreView::Results.label();
        harness.click_text(last);
        let output = harness.frame(Vec::new());
        let label = text_rect(&output, last).expect("the tab");
        assert!(label.top() > size_tab.bottom(), "{label:?} {size_tab:?}");
        under(&output, label);
    }

    #[test]
    fn the_brand_mark_is_one_accent_square_before_the_title() {
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let marks: Vec<egui::Rect> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect) if rect.fill == look::ACCENT => Some(rect.rect),
                _ => None,
            })
            .collect();
        assert_eq!(marks.len(), 1, "{marks:?}");
        let mark = marks[0];
        assert_eq!(mark.size(), egui::Vec2::splat(look::MARK_SIDE));
        // The title's rect starts at the row's left edge (a wrapping label is indented to
        // where its text goes): its first glyph marks where the title begins.
        let (_, title) = text_shapes(&output, HEADING).next().expect("the title");
        let first = title.galley.rows[0].glyphs[0].pos.x + title.pos.x;
        assert!(mark.right() < first, "{mark:?} {first}");
        let middle = title.galley.rect.translate(title.pos.to_vec2()).center().y;
        assert!((mark.center().y - middle).abs() <= 2.0, "{mark:?} {middle}");
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.menu(SAVE_DESIGN);
        harness.menu(LOAD_DESIGN);
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
        let output = harness.click_text(SHARE);
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
    fn every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp() {
        // egui draws an empty box for a character its default fonts lack (U+2192, the
        // spec's arrow, is one): every text the guard collects, the clamp section paused or not
        // (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts
        // and the named static hovers).
        // Ruling 11 (3A): while the section is paused none of them names the clamp, nor does a
        // text in a table the panel draws from (below). Texts allowed to, each with why: none.
        const ALLOWED: &[(&str, &str)] = &[];
        // The list is a deliberate hook, empty by ruling 11: a text that names the clamp while
        // the section is paused needs the owner's say, so a new entry fails here until that
        // is recorded (change this assertion with the ruling, not around it).
        assert_eq!(
            ALLOWED.len(),
            0,
            "ALLOWED holds a text that may name the paused clamp: ruling 11 allows none; record the owner's say first"
        );
        let names_the_clamp = |text: &str| {
            text.to_lowercase().contains("clamp") && !ALLOWED.iter().any(|&(t, _)| t == text)
        };
        for clamps_paused in [false, true] {
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
            harness.panel.clamps_paused = clamps_paused;
            let mut texts = every_shown_text(&mut harness);
            for text in &texts {
                crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the panel");
            }
            if clamps_paused {
                // The check list, each check's Explain entry and each Size row's (BL-090) on
                // designs failing every check, the clamp's too.
                let mut hot = DesignInputs::default();
                hot_day(&mut hot);
                for inputs in [
                    DesignInputs::default(),
                    only_the_clamp_fails(),
                    hot,
                    short_magnets(),
                    with_poles(40),
                ] {
                    harness.panel.inputs = inputs;
                    for guide in crate::gui::guidance::counted_guides(true) {
                        harness.select(guide.check);
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                    for row in size_controls::SIZE_ROWS {
                        harness.panel.selection = Selection::Input(row.key);
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                }
                // The shaft joints' guide (BL-064) counts while the section is paused: its
                // Problem drawn in its Explain entry, where the inner hub slips, where it is
                // over its stress limit and cannot be fitted either, and on an uncooled wheel
                // shaft.
                let joint = crate::gui::guidance::guide("shaft_joint.check").expect("its guide");
                let mut overstressed = DesignInputs::default();
                hub_overstressed(&mut overstressed);
                let mut uncooled = DesignInputs::default();
                uncooled.shaft_joint.wheel_shaft_fit_C = 20.0;
                for inputs in [only_the_joint_fails(), overstressed, uncooled] {
                    harness.panel.inputs = inputs;
                    harness.select(joint.check);
                    let drawn = drawn_texts(&harness.frame(Vec::new()));
                    let panel = &harness.panel;
                    let problem =
                        (joint.problem)(panel.inputs(), panel.results()).expect("the joints fail");
                    assert!(drawn.contains(&problem), "{problem}");
                    texts.extend(drawn);
                }
                // Not vacuous for the guide (ruling C-8): its label and Why are collected.
                for text in [joint.label, joint.why] {
                    assert!(
                        texts.iter().any(|t| t.contains(text)),
                        "{text:?} is not collected"
                    );
                }
                // The teaching notes shown, their change entries as drawn, their Start here
                // walk and its hover. No change entry or equation of theirs leads into the
                // section.
                use crate::engine::explain::notes;
                for note in notes::NOTES
                    .iter()
                    .filter(|n| notes::shown_when(n.id, true))
                {
                    texts.push(note.title.to_owned());
                    texts.extend(note.sentences.iter().map(|s| s.to_string()));
                    texts.extend(note.change.iter().map(|c| c.text.to_owned()));
                    texts.extend(note.watch_out.iter().map(|w| w.to_string()));
                    texts.extend(note.sources.iter().map(|s| s.to_string()));
                    for change in note.change {
                        assert!(
                            !clamps::hides(change.input, true),
                            "{}: a change entry leads to the hidden {}",
                            note.id,
                            change.input
                        );
                        texts.push(format!(
                            "{} {}",
                            change.step.word(),
                            label_for(change.input)
                        ));
                    }
                    for equation in note.equations {
                        assert!(
                            !clamps::hides(equation, true),
                            "{}: explains the hidden {equation}",
                            note.id
                        );
                    }
                }
                texts.push(crate::gui::explorer::start_here_hover(true).to_owned());
                for &(id, path) in notes::start_here_for(true) {
                    texts.push(id.to_owned());
                    texts.push(label_for(path).to_owned());
                }
                // The check guides that count: their names and Why sentences.
                for guide in crate::gui::guidance::counted_guides(true) {
                    texts.push(guide.label.to_owned());
                    texts.push(guide.why.to_owned());
                }
                // The Why links' reasons where no note explains a shown row (their hover texts).
                for &(path, learn) in crate::gui::guidance::ROW_LEARN.iter() {
                    if let crate::gui::guidance::Learn::None(reason) = learn
                        && !clamps::hides(path, true)
                    {
                        texts.push(reason.to_owned());
                    }
                }
                // The assumptions shown: their names, rationales and sources.
                for a in ASSUMPTIONS.iter().filter(|a| a.shown(true)) {
                    texts.extend([a.label, a.rationale, a.source].map(str::to_owned));
                }
                let named: Vec<&String> = texts.iter().filter(|t| names_the_clamp(t)).collect();
                assert!(named.is_empty(), "{named:#?}");
                // Not vacuous for the shaft joints (BL-064), which show while the clamp is
                // paused: the enumeration reaches the label and the help of each of their inputs
                // and results, their two input sections and their groups' headings. A label or a
                // help may sit inside a hover text. Four input labels are also their assumptions'
                // names, so for those inputs it is the help that shows the input is collected.
                use crate::gui::inputs::section_label;
                use crate::gui::plain_text::help_for;
                use crate::gui::result_groups::result_groups;
                use crate::gui::test_support::{
                    shaft_joint_input_rows, shaft_joint_result_rows, workflow_group,
                };
                let reaches = |text: &str| texts.iter().any(|t| t.contains(text));
                let paths: Vec<String> = shaft_joint_input_rows(&DesignInputs::default())
                    .into_iter()
                    .map(|row| row.path)
                    .chain(
                        shaft_joint_result_rows(&compute_all(&DesignInputs::default()))
                            .into_iter()
                            .map(|row| row.path),
                    )
                    .collect();
                assert_eq!(paths.len(), 15 + 24);
                for path in &paths {
                    for text in [label_for(path), help_for(path)] {
                        assert!(reaches(text), "{path}: {text:?} is not collected");
                    }
                }
                // Each heading whole, as the panel draws it: the two input sections and the
                // original sheet's group by their labels, and the results table's chain group by
                // its label and its row count, as the table draws a group's heading. Part of a
                // text would not do: the check's label, "Shaft joints: keyed shrink fits",
                // contains "Shaft joints".
                let shaft = workflow_group("shaft");
                let chain = result_groups()
                    .iter()
                    .find(|group| group.id == "shaft_joint" && !group.other)
                    .expect("the Shaft joints chain");
                let headings: Vec<String> = shaft
                    .sections
                    .iter()
                    .filter(|section| ["shaft.fit", "shaft.fitting"].contains(&section.id))
                    .map(|section| section.label.to_owned())
                    .chain(section_label("shaft_joint").map(str::to_owned))
                    .chain([format!("{} ({})", chain.label, chain.rows.len())])
                    .collect();
                assert_eq!(headings.len(), 4);
                for heading in &headings {
                    assert!(
                        texts.contains(heading),
                        "{heading:?} is not collected whole"
                    );
                }
            } else {
                // Not vacuous: with the section shown, the same enumeration names the clamp and
                // reaches each surface the pause hides.
                assert!(texts.iter().any(|t| names_the_clamp(t)));
                let friction = ASSUMPTIONS
                    .iter()
                    .find(|a| a.id == "clamp_friction")
                    .expect("the clamp friction")
                    .label;
                for surface in [
                    CentreView::Clamp.label(),
                    label_for("clamps.recommended"),
                    label_for("clamps.boss_od_mm"),
                    "Shaft clamps",
                    friction,
                ] {
                    assert!(
                        texts.iter().any(|t| t.contains(surface)),
                        "{surface}: the enumeration does not reach it with the section shown"
                    );
                }
            }
        }
    }

    /// The texts the guard collects from the panel of `harness`, with its shaft-clamp pause
    /// (`harness.panel.clamps_paused`): every drawn text, in each state, and the hover texts of
    /// inputs, dashboard rows, table rows, callouts and the named static hovers. The glyph
    /// check and the paused section's guard read it.
    fn every_shown_text(harness: &mut Harness) -> Vec<String> {
        let clamps_paused = harness.panel.clamps_paused;
        // With the spreadsheet details on: every provenance line is drawn too (spec S2-5).
        harness.panel.show_details = true;
        let mut texts = drawn_texts(&harness.frame(Vec::new()));
        texts.extend(drawn_texts(&harness.size()));
        // Every view of the centre region, and the geometry callouts' hover texts.
        for view in CentreView::tabs(clamps_paused) {
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
        let geometry =
            crate::gui::geometry::geometry(&shown, harness.panel.results(), clamps_paused);
        texts.extend(
            geometry
                .callouts()
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path, true)),
        );
        // Every group open, in both orders; every workflow heading; the filtered view's runs.
        let catalogue = InputCatalogue::shown(clamps_paused);
        for order in [InputOrder::Workflow, InputOrder::Workbook] {
            harness.click_text(order.label());
            // Bottom up: opening a group moves only the groups below it.
            for group in catalogue.groups_in(order).iter().rev() {
                harness.click_text(group.label);
            }
            for _ in 0..10 {
                harness.frame(Vec::new());
            }
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        for group in &catalogue.workflow {
            texts.push(group.label.to_owned());
            texts.extend(group.sections.iter().map(|s| s.label.to_owned()));
        }
        texts.push(ADVANCED_HEADING.to_owned());
        texts.push(FILTER_HINT.to_owned());
        texts.push(crate::gui::results_table::NOTHING_FAILS.to_owned());
        texts.push(crate::gui::results_table::NO_RESULT.to_owned());
        texts.push(crate::gui::results_table::TRACED_ONLY.to_owned());
        texts.push(crate::gui::explain::TRACE_LINK.to_owned());
        texts.push(crate::gui::trace::CLEAR_TRACE.to_owned());
        texts.push(crate::gui::results_table::NOTHING_TRACED.to_owned());
        for path in [FACE_GAP, "model.pullout_Nm", "coupling.drive_torque_Nm"] {
            texts.push(
                crate::gui::trace::Trace::of(path)
                    .unwrap()
                    .banner(clamps_paused),
            );
        }
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
        texts.extend([SHOW_DETAILS, SHOW_DETAILS_HINT].map(str::to_owned));
        texts.extend([RESET_ALL, RESET_TO_START, SAVE_DESIGN, LOAD_DESIGN].map(str::to_owned));
        texts.push(crate::gui::dashboard::MARKER_LEGEND.to_owned());
        texts.push(crate::gui::guidance::EXAMPLE_DESIGN_HOVER.to_owned());
        // The starting design's name and hover, and the example design's button and status
        // (UX overhaul spec 7.1), which a panel at the spreadsheet's baseline does not draw.
        texts.extend(
            [
                crate::gui::guidance::STARTING_DESIGN,
                crate::gui::guidance::STARTING_DESIGN_HOVER,
                OPEN_EXAMPLE,
                EXAMPLE_OPENED,
            ]
            .map(str::to_owned),
        );
        // The hover texts that are named consts (an inline hover literal is not collected).
        texts.extend(
            [
                crate::gui::size_controls::DIAMETER_HOVER,
                crate::gui::size_controls::LENGTH_HOVER,
                crate::gui::size_controls::AUTO_HOVER,
                crate::gui::size_controls::SET_BY_AUTO,
                crate::gui::input_ui::CHANGED_FROM_DEFAULT,
                crate::gui::input_ui::OUTSIDE_RANGE_HINT,
                crate::gui::dashboard::STORED_3D_NOTE,
                crate::gui::dashboard::NOTE_NOT_REVIEWED,
                RESET_ASSUMPTIONS_HINT,
            ]
            .map(str::to_owned),
        );
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
        texts.extend(
            catalogue
                .all()
                .map(|entry| crate::gui::inputs::input_tooltip(entry, &entry.default, true)),
        );
        // The material warnings the dashboard can show.
        texts.extend(
            crate::engine::warnings::WARNING_RULES
                .iter()
                .map(|rule| rule.text.to_owned()),
        );
        let results = harness.panel.results().clone();
        // The dashboard rows and the table rows the panel draws (its own filters).
        texts.extend(
            crate::gui::dashboard::shown_lines(&results, clamps_paused)
                .into_iter()
                .map(|line| line.tooltip(true)),
        );
        let entries = crate::gui::results_table::table_entries();
        for index in crate::gui::results_table::search(entries, "", clamps_paused) {
            let entry = &entries[index];
            let value = results.get(&entry.path).unwrap_or(Value::None);
            texts.push(crate::gui::results_table::row_tooltip(entry, &value, true));
        }
        texts
    }

    #[test]
    fn the_clamp_tab_and_the_clamp_inputs_show_only_while_the_section_shows() {
        assert_eq!(CentreView::tabs(false).collect::<Vec<_>>(), CentreView::ALL);
        let without: Vec<CentreView> = CentreView::ALL
            .into_iter()
            .filter(|&view| view != CentreView::Clamp)
            .collect();
        assert_eq!(CentreView::tabs(true).collect::<Vec<_>>(), without);
        for clamps_paused in [false, true] {
            let mut harness = Harness::on_screen(TALL);
            harness.panel.clamps_paused = clamps_paused;
            let shown = usize::from(!clamps_paused);
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, CentreView::Clamp.label()), shown, "the tab");
            // By workflow: the bore and the keyway stay, the clamp's sections go.
            harness.click_text("Shaft and hub joint");
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, label_for("coupling.bore_mm")), 1);
            assert_eq!(count(&output, label_for("coupling.keyway_depth_mm")), 1);
            // BL-064: the shaft joints' key and the first rows of their shrink fit and fitting
            // temperatures show in both states: the keyed shrink fits are the joints in force,
            // not the paused clamp.
            for path in [
                "shaft_joint.key_width_mm",
                "shaft_joint.key_contact_mm",
                "shaft_joint.hub_interference_min_um",
                "shaft_joint.hub_fit_max_C",
            ] {
                assert_eq!(count(&output, label_for(path)), 1, "{path}");
            }
            assert_eq!(count(&output, "Shrink fit"), 1);
            assert_eq!(count(&output, "Fitting temperatures"), 1);
            assert_eq!(count(&output, "Clamp geometry"), shown);
            assert_eq!(count(&output, label_for("clamps.boss_od_mm")), shown);
            // By original sheet: the Shaft clamps group.
            harness.click_text(InputOrder::Workbook.label());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, "Shaft clamps"), shown);
            // The filter finds no clamp input while paused.
            harness.click_text(InputOrder::Workflow.label());
            harness.panel.input_filter = "clamp".to_owned();
            let output = harness.frame(Vec::new());
            let total = InputCatalogue::shown(clamps_paused).all().count();
            let matching = filter_inputs(
                InputCatalogue::shown(clamps_paused).groups_in(InputOrder::Workflow),
                "clamp",
            )
            .iter()
            .map(|run| run.entries.len())
            .sum::<usize>();
            assert_eq!(matching == 0, clamps_paused);
            assert_eq!(count(&output, &format!("{matching} of {total} inputs")), 1);
        }
    }

    /// The defaults with the clamp's friction assumption modified.
    fn clamp_friction_modified() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.clamps.friction = 0.1;
        inputs
    }

    #[test]
    fn the_paused_section_s_assumptions_leave_the_view_and_the_banner() {
        let friction = clamp_friction_modified();
        let label = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "clamp_friction")
            .expect("the clamp friction")
            .label;
        assert_eq!(
            assumptions_banner(&friction, false),
            Some(format!("{ASSUMPTIONS_MODIFIED}: {label}"))
        );
        assert_eq!(assumptions_banner(&friction, true), None);
        for clamps_paused in [false, true] {
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
            harness.panel.clamps_paused = clamps_paused;
            harness.click_text(InputsView::Assumptions.label());
            let texts = drawn_texts(&harness.frame(Vec::new()));
            for a in ASSUMPTIONS {
                assert_eq!(
                    texts.contains(&a.label.to_owned()),
                    a.shown(clamps_paused),
                    "{}",
                    a.id
                );
            }
        }
    }

    #[test]
    fn the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace() {
        use crate::gui::guidance::{Direction, fix_text};
        // The shipped panel starts from the switch (the harness overrides it).
        assert_eq!(MagcouplingPanel::new().clamps_paused, clamps::PAUSED);
        // At the clamp's limit, the hot-low torque's one fix is withheld only while the clamp
        // shows (guidance's a_paused_clamp_offers_no_fix_and_holds_back_no_other_guide_s_fix).
        let at_limit = clamp_at_its_limit();
        let fix = fix_text(AXIAL_LENGTH, Direction::Raise);
        let friction = clamp_friction_modified();
        for clamps_paused in [false, true] {
            let mut harness = Harness::on_screen(TALL);
            harness.panel.clamps_paused = clamps_paused;
            harness.panel.inputs = at_limit.clone();
            harness.select(HOT_MIN_CHECK);
            let output = harness.frame(Vec::new());
            assert_eq!(
                count(&output, &fix) > 0,
                clamps_paused,
                "the fix, paused {clamps_paused}"
            );
            // A modified clamp assumption: the header's banner only while the clamp shows.
            harness.panel.inputs = friction.clone();
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(
                drawn_texts(&output)
                    .iter()
                    .any(|t| t.starts_with(ASSUMPTIONS_MODIFIED)),
                !clamps_paused,
                "the banner, paused {clamps_paused}"
            );
            // A trace of the face gap (it drives clamp results): the drawn banner counts as
            // the pause says.
            let trace = crate::gui::trace::Trace::of(FACE_GAP).expect("an input");
            harness.panel.trace = Some(trace.clone());
            let output = harness.frame(Vec::new());
            assert_eq!(
                count(&output, &trace.banner(clamps_paused)),
                1,
                "paused {clamps_paused}"
            );
        }
    }

    #[test]
    fn a_design_file_and_a_share_link_keep_their_clamp_values_while_the_section_is_paused() {
        // A design file written before the pause, its clamp values away from the defaults.
        const PRE_PAUSE: &str = r#"{"format": "magcoupling-design", "version": 1, "inputs": {"clamps.boss_od_mm": 30.0, "clamps.friction": 0.1, "clamps.key_width_mm": 5.0, "clamps.joint_screws": 4, "clamps.screw_class": 2, "materials.screws.proof_10_9_MPa": 900.0}}"#;
        let mut panel = MagcouplingPanel::new();
        panel.clamps_paused = true;
        panel
            .load_design_file(PRE_PAUSE)
            .expect("a pre-pause file loads");
        let design = panel.design();
        let c = &design.inputs.clamps;
        assert_eq!(
            (
                c.boss_od_mm,
                c.friction,
                c.key_width_mm,
                c.joint_screws,
                c.screw_class
            ),
            (30.0, 0.1, 5.0, 4, 2)
        );
        assert_eq!(design.inputs.materials.screws.proof_10_9_MPa, 900.0);
        // The engine still computes the hidden section from them.
        assert_eq!(panel.results().clamps.boss_radius_mm, 15.0);
        // Saved and shared, every clamp value comes back.
        let text = crate::gui::session::design_to_json(&design);
        assert_eq!(design_from_json(&text), Ok(design.clone()));
        let payload = encode_share_payload(&design);
        assert_eq!(decode_share_payload(&payload), Ok(design.clone()));
        let mut reopened = MagcouplingPanel::new();
        reopened.clamps_paused = true;
        reopened
            .load_share_payload(&payload)
            .expect("the link loads");
        assert_eq!(reopened.design(), design);
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
    fn a_new_design_clears_the_lowered_minimum_od_but_undo_and_redo_keep_it() {
        // MGP-10: open, load (file and link), and reset each start from no lowered Minimum OD.
        let mut other = Design::default();
        other.inputs.coupling.npole = 8;
        let text = design_to_json(&other);
        let payload = crate::gui::session::encode_share_payload(&other);
        let cleared = |panel: &MagcouplingPanel| {
            assert_eq!(panel.minimum_diameter_mm, None);
            assert_eq!(panel.results, compute_all(&panel.shown_inputs()));
        };
        let lowered = || {
            let mut panel = MagcouplingPanel::new();
            panel.minimum_diameter_mm = Some(45.0);
            panel
        };
        let mut panel = lowered();
        panel.open_design(other.clone());
        cleared(&panel);
        assert_eq!(panel.design(), other);
        let mut panel = lowered();
        panel.load_design_file(&text).unwrap();
        cleared(&panel);
        let mut panel = lowered();
        panel.load_share_payload(&payload).unwrap();
        cleared(&panel);
        let mut panel = lowered();
        panel.open_share_payload(&payload).unwrap();
        cleared(&panel);
        let mut panel = lowered();
        panel.reset();
        cleared(&panel);
        // A refused load changes nothing.
        let mut panel = lowered();
        assert!(panel.load_design_file("not a design").is_err());
        assert_eq!(panel.minimum_diameter_mm, Some(45.0));
        // Undo and redo are not new designs.
        let mut panel = MagcouplingPanel::new();
        panel.load_design_file(&text).unwrap();
        panel.minimum_diameter_mm = Some(45.0);
        panel.undo();
        assert_eq!(panel.design(), Design::default(), "the load was undone");
        assert_eq!(panel.minimum_diameter_mm, Some(45.0));
        panel.redo();
        assert_eq!(panel.design(), other, "the load was redone");
        assert_eq!(panel.minimum_diameter_mm, Some(45.0));
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
        // The lines' words are the torque card's side words (slice 3 spec 4.2).
        assert_eq!(
            count(
                &output,
                &format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.")
            ),
            1
        );
        assert_eq!(
            count(
                &output,
                &format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}.")
            ),
            1
        );
    }

    use crate::gui::geometry::{MIN_CAP_SKIRT_MM, SKIRT_CAUTION_START};

    /// The example design with its cap's outside and thread diameters set (BL-082 tests).
    fn with_cap(od_mm: f64, thread_mm: f64) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.metal.cap_od_mm = od_mm;
        inputs.metal.cap_thread_dia_mm = thread_mm;
        inputs
    }

    fn skirt_caution(inputs: &DesignInputs) -> Vec<String> {
        wall_cautions(inputs, &compute_all(inputs))
            .into_iter()
            .filter(|line| line.starts_with(SKIRT_CAUTION_START))
            .collect()
    }

    #[test]
    fn a_cap_skirt_under_the_minimum_draws_one_caution_naming_it() {
        // BL-082 (owner answer 4a, 2026-11-22): a 0.49 mm skirt, (42.8 - 41.82) / 2, draws the
        // line, its skirt quoted at four significant digits as every readout is.
        assert_eq!(
            skirt_caution(&with_cap(42.8, 41.82)),
            vec![
                "Cap wall past its thread (skirt): 0.4900 mm; guide minimum is 0.5 mm.".to_owned()
            ]
        );
        // A thread above the outside diameter is a skirt below zero, still under the minimum.
        assert_eq!(skirt_caution(&with_cap(42.0, 42.2)).len(), 1);
    }

    #[test]
    fn a_cap_skirt_at_the_minimum_draws_no_caution() {
        // 0.50 mm, (43.0 - 42.0) / 2 and (42.8 - 41.8) / 2; and a hair under 0.5 that reads
        // 0.5000 mm: compared as quoted, so a skirt never reads equal to the minimum beside a
        // line that says it is under it (owner question 22's rule for the keyway line).
        let hair = 41.8 + 1e-11;
        assert!((42.8 - hair) / 2.0 < MIN_CAP_SKIRT_MM);
        for (od, thread) in [(43.0, 42.0), (42.8, 41.8), (42.8, hair), (42.8, 41.0)] {
            assert_eq!(
                skirt_caution(&with_cap(od, thread)),
                Vec::<String>::new(),
                "{od} {thread}"
            );
        }
        // A diameter that is not a number draws nothing (it cannot be compared).
        assert!(skirt_caution(&with_cap(f64::NAN, 41.0)).is_empty());
    }

    #[test]
    fn the_skirt_caution_is_drawn_amber_on_the_opening_tab_and_counts_no_check() {
        use crate::gui::guidance::{counted_guides, level};
        let mut harness = Harness::new();
        harness.panel.inputs = with_cap(42.8, 41.82);
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        let warn = harness.ctx.style().visuals.warn_fg_color;
        let amber = crate::gui::test_support::texts_in_colors(&output, &[warn]);
        assert_eq!(
            amber
                .iter()
                .filter(|t| t.starts_with(SKIRT_CAUTION_START))
                .count(),
            1,
            "{amber:?}"
        );
        // Not a counted check: no check's level moves with the skirt.
        let thin = harness.panel.inputs.clone();
        let mut thick = thin.clone();
        thick.metal.cap_thread_dia_mm = 41.8;
        let (thin_results, thick_results) = (compute_all(&thin), compute_all(&thick));
        for guide in counted_guides(false) {
            assert_eq!(
                level(guide, &thin, &thin_results),
                level(guide, &thick, &thick_results),
                "{}",
                guide.label
            );
        }
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
        // The three torque tiles read no prediction, and the Hot-low tile shows neither what it
        // needs nor the gap (spec 3.2, 5; UX overhaul spec 4.4).
        assert_eq!(count(&output, NO_PREDICTION), 3);
        let (gap, _) = torque_gap_text(
            harness.panel.results().metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        assert_eq!(count(&output, &gap), 0, "{gap}");
        let texts = drawn_texts(&output);
        assert!(
            !texts
                .iter()
                .any(|t| t.starts_with(crate::gui::tiles::NEEDS)),
            "{texts:?}"
        );
    }

    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls. The
    /// Hot-low tile is selected, so the Explain panel draws the torque card (decision C-10).
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
        let mut harness = Harness::on_screen(screen);
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        harness.panel.inputs = inputs;
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
        harness
    }

    /// Shows the entry of the Size column's row `key` (UX overhaul spec 4.3), as touching its
    /// control does: the Explain panel draws it in the frame this returns.
    fn entry(harness: &mut Harness, key: &'static str) -> egui::FullOutput {
        harness.panel.selection = Selection::Input(key);
        harness.frame(Vec::new())
    }

    /// The rect of `text` drawn in the right column (the check list over the Explain panel):
    /// an entry's copy, where the Size column draws the same words.
    fn explain_rect(output: &egui::FullOutput, text: &str) -> egui::Rect {
        let left = right_column_left(output);
        text_rects(output, text)
            .into_iter()
            .find(|rect| rect.left() >= left - 1.0)
            .unwrap_or_else(|| panic!("no {text:?} in the right column"))
    }

    /// The what-moves-what lines and the torque card of the design the panel shows, as the Size
    /// tab computes them: that design, its own results, the panel's Minimum OD and its
    /// shaft-clamp pause.
    fn shown_consequences(harness: &Harness) -> Consequences {
        let shown = harness.panel.shown_inputs();
        Consequences::of(
            &shown,
            &compute_all(&shown),
            harness.panel.minimum_diameter_mm,
            harness.panel.clamps_paused,
        )
    }

    /// The torque card's lines for the design the panel shows, in the panel's Auto state, as the
    /// Size tab draws them ([`shown_consequences`]).
    fn card_lines(harness: &Harness) -> Vec<String> {
        shown_consequences(harness).card.lines(harness.panel.auto())
    }

    /// The rows egui laid the first drawn text equal to `needle` out on: more than one once it
    /// wrapped.
    fn rows(output: &egui::FullOutput, needle: &str) -> usize {
        text_shapes(output, needle)
            .next()
            .map(|(_, text)| text.galley.rows.len())
            .unwrap_or_else(|| panic!("no {needle:?}"))
    }

    /// The rect of `label` in the centre region: its leftmost copy right of the inputs side.
    /// The dashboard, drawn before the centre, can carry the same name (the Size tab's Hot-low
    /// torque readout and the dashboard row are one quantity, one name), so the first drawn
    /// copy is not always the centre's.
    fn centre_rect(output: &egui::FullOutput, label: &str) -> egui::Rect {
        text_rects(output, label)
            .into_iter()
            .filter(|r| r.left() >= INPUTS_WIDTH)
            .min_by(|a, b| a.left().total_cmp(&b.left()))
            .unwrap_or_else(|| panic!("no {label:?} right of the inputs side"))
    }

    /// The Size column's Required slider's value box in `output`: the first text under the
    /// row's label in the inputs side, the requirement with its unit.
    fn required_box(output: &egui::FullOutput) -> egui::Rect {
        let label = text_rect(output, REQUIRED).expect("the Required row");
        assert!(label.right() <= INPUTS_WIDTH, "{label:?}");
        let mut under: Vec<(String, egui::Rect)> = crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Text(text) => {
                    let rect = text.galley.rect.translate(text.pos.to_vec2());
                    (rect.right() <= INPUTS_WIDTH && rect.top() >= label.bottom() - 0.5)
                        .then(|| (text.galley.text().to_owned(), rect))
                }
                _ => None,
            })
            .collect();
        under.sort_by(|a, b| a.1.top().total_cmp(&b.1.top()));
        let (text, rect) = under.remove(0);
        assert!(text.ends_with(" N\u{b7}m"), "{text}");
        rect
    }

    #[test]
    fn the_tiles_read_hot_low_what_it_needs_its_gap_and_the_nominal_pull_out() {
        // Spec 3.2 and UX overhaul spec 4.4 at the default design, whose hot-low torque is below
        // the requirement: the gap reads short, in red, as the hot minimum check fails.
        use crate::gui::test_support::text_color;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let results = harness.panel.results().clone();
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, label.as_str()] {
            assert_eq!(count(&output, text), 1, "{text}");
        }
        // Hot-low torque is a check's name too (spec 3.5): one tile in the centre.
        let in_centre =
            |r: &egui::Rect| r.left() >= INPUTS_WIDTH && r.right() <= SCREEN.x - DASHBOARD_WIDTH;
        assert_eq!(
            text_rects(&output, HOT_LOW)
                .iter()
                .filter(|r| in_centre(r))
                .count(),
            1
        );
        assert_eq!(
            count(&output, "Expected torque"),
            0,
            "replaced by the nominal pull-out"
        );
        let shown = |path: &str| with_unit(format_value(&results.get(path).unwrap()), "N\u{b7}m");
        let required = harness.panel.inputs().metal.required_min_Nm;
        let (gap, meets) = torque_gap_text(results.metal.torque_hot_low_Nm, required);
        assert!(!meets && gap.starts_with("short "), "{gap}");
        // The Hot-low tile from its top: its number, its label, what it needs, the gap.
        use crate::gui::tiles::{NEEDS, gap_pieces};
        let needs = with_unit(format_value(&Value::Num(required)), "N\u{b7}m");
        let pieces = gap_pieces(&gap);
        let hot_low = shown("metal.torque_hot_low_Nm");
        let tops: Vec<f32> = [hot_low.as_str(), HOT_LOW, NEEDS, pieces[0].as_str()]
            .into_iter()
            .map(|text| centre_rect(&output, text).top())
            .collect();
        assert!(tops.is_sorted(), "{tops:?}");
        // The pull-out's tile: its number over its label.
        let pullout = centre_rect(&output, &shown("model.pullout_Nm"));
        assert!(pullout.bottom() <= centre_rect(&output, &label).top() + 0.5);
        // Red is on what the tile needs and on the gap, never on the number alone.
        let visuals = harness.ctx.style().visuals.clone();
        let bad = Level::Bad.color(&visuals);
        let reds = [NEEDS, needs.as_str()].into_iter();
        for text in reds.chain(pieces.iter().map(String::as_str)) {
            assert_eq!(text_color(&output, text), Some(bad), "{text}");
        }
        assert_eq!(
            text_color(&output, &hot_low),
            Some(visuals.strong_text_color())
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
            let value_box = required_box(&output);
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
        for piece in crate::gui::tiles::gap_pieces(&gap) {
            assert_eq!(
                crate::gui::test_support::text_color(&output, &piece),
                Some(Level::Good.color(&harness.ctx.style().visuals)),
                "{piece}"
            );
        }
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
        let value_box = required_box(&output);
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

    /// The label the torque card replaced under the Size tab's readouts (slice 3 spec 6.1), kept
    /// as the record of what the card took the place of.
    const REPLACED_LABEL: &str = "Build checks are shown in the dashboard. The geometry drawing fits the view; use the dimensions to compare sizes.";

    #[test]
    fn the_hot_low_tile_s_entry_draws_the_torque_card() {
        // Slice 3 spec 6.1 and decision C-10: the card is the Hot-low tile's Explain entry.
        // Until the tile is clicked no line of it is on screen; then its heading and its lines
        // for the design shown are drawn once each, one under the next, in the right column.
        // Another tile's entry has no card. No click edits the design.
        use crate::gui::explain::WHAT_A_CHANGE_DOES;
        use crate::gui::tiles::{NEEDS, Tile};
        let mut harness = Harness::first_view();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, REPLACED_LABEL), 0);
        assert_eq!(count(&output, CARD_HEADING), 0);
        let lines = card_lines(&harness);
        assert_eq!(lines.first().map(String::as_str), Some(CAPTION));
        for line in &lines {
            assert_eq!(count(&output, line), 0, "{line}");
        }
        let needs = drawn_texts(&output)
            .into_iter()
            .find(|text| text.starts_with(NEEDS))
            .expect("the Hot-low tile says what it needs");
        harness.click(text_center(&output, &needs));
        assert_eq!(
            harness.panel.selection,
            Selection::Result(Tile::HotLow.path())
        );
        let output = harness.frame(Vec::new());
        let rect = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text).unwrap_or_else(|| panic!("no {text:?}"))
        };
        assert_eq!(count(&output, WHAT_A_CHANGE_DOES), 1);
        assert_eq!(count(&output, CARD_HEADING), 1);
        let mut above = rect(&output, CARD_HEADING);
        assert!(rect(&output, WHAT_A_CHANGE_DOES).bottom() <= above.top() + 0.5);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
            let at = rect(&output, line);
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            assert!(
                at.left() >= SCREEN.x - DASHBOARD_WIDTH,
                "{line:?} at {at:?}"
            );
            above = at;
        }
        // The pull-out's entry: what it is, and no card.
        let pullout = pullout_label(harness.panel.results().metal.op_temp_C);
        harness.click(centre_rect(&output, &pullout).center());
        assert_eq!(
            harness.panel.selection,
            Selection::Result(Tile::Pullout.path())
        );
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CARD_HEADING), 0);
        assert_eq!(
            count(
                &output,
                crate::gui::plain_text::help_for(Tile::Pullout.path())
            ),
            1
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_size_control_s_edit_reaches_its_entry_in_its_own_frame() {
        // Slice 3 spec 3.5 and UX overhaul spec 5.2: an edit of a Size control selects it, and
        // the frame that applies the edit draws the edited design in the control's entry, not
        // the design before it. The Required box typed into: its entry's title reads the typed
        // 2.4 N·m in the frame that applies it. An arrow key on Magnets per ring: its lines and
        // its torque-card row are the edited design's in the frame of the key.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        let output = harness.frame(Vec::new());
        let value_box = required_box(&output);
        harness.click(value_box.center());
        assert_eq!(harness.panel.selection, Selection::Input(REQUIREMENT_INPUT));
        harness.frame([select_all(), vec![egui::Event::Text("2.4".to_owned())]].concat());
        let output = harness.frame(key_tap(egui::Key::Enter));
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 2.4);
        let unit = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata")
            .meta
            .unit;
        let typed = Value::Num(2.4);
        let title = crate::gui::format::value_with_unit(&typed, format_value(&typed), unit);
        explain_rect(&output, &title);
        let before = shown_consequences(&harness);
        let magnets = harness.panel.design_widgets[3];
        harness.ctx.memory_mut(|m| m.request_focus(magnets));
        harness.frame(Vec::new());
        let output = harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        assert_eq!(harness.panel.selection, Selection::Input(POLES));
        let after = shown_consequences(&harness);
        let row = |c: &Consequences| c.card.control_lines(Control::MagnetsPerRing, None);
        assert_ne!(
            row(&after),
            row(&before),
            "two magnets fewer move the count row"
        );
        for line in row(&after).iter().chain(&after.magnet_lines(8, None)) {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in row(&before)
            .iter()
            .filter(|line| !row(&after).contains(line))
        {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
    }

    // The torque card's open or closed and its room (slice 3 spec 6.2, tests 19 to 25). Each
    // test pauses the shaft-clamp section before the card's first frame, the one that decides
    // its default (`card_tab`), and steps frames past the collapse animation before it measures.

    #[test]
    fn each_size_control_s_entry_draws_its_what_moves_what_lines_under_what_sets_it() {
        // Spec 3.3 at the default design, in the Explain panel since UX overhaul spec 4.3: the
        // Set-by line in Outside diameter's entry, the length split in Overall length's, and the
        // magnet count's lines in Magnets per ring's, the ring-size line first, each under the
        // one before and under the entry's "What sets it". The Size column draws none of them.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let consequences = shown_consequences(&harness);
        let diameter = consequences
            .diameter_line()
            .expect("the default design has a breakdown");
        let length = consequences.length_line().expect("and a length split");
        let magnets = consequences.magnet_lines(10, None);
        assert_eq!(magnets.first().map(String::as_str), Some(RING_SIZE_STAYS));
        // The ring-size line and the fit line (slice 3, decision 7A: the lines for two magnets
        // fewer and two more are the torque card's count row).
        assert_eq!(magnets.len(), 2, "{magnets:?}");
        let column = harness.frame(Vec::new());
        for (key, lines) in [
            (DIAMETER_HINT_KEY, vec![diameter]),
            (LENGTH_HINT_KEY, vec![length]),
            (POLES, magnets),
        ] {
            for line in &lines {
                assert_eq!(count(&column, line), 0, "the column draws no {line:?}");
            }
            let output = entry(&mut harness, key);
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{key}: {line}");
                let at = explain_rect(&output, line);
                assert!(
                    above.bottom() <= at.top() + 0.5,
                    "{key}: {line:?} is not under the line above it"
                );
                above = at;
            }
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "drawing them edits nothing"
        );
    }

    /// The Size column's five controls in drawing order, each with its hint key (spec 3.5), which
    /// is its row's key: the gap and the magnet count's hints are their Key design rows'.
    const SIZE_TAB_HINTS: [(&str, &str); 5] = [
        (MINIMUM, MINIMUM_OD_HINT_KEY),
        (DIAMETER, DIAMETER_HINT_KEY),
        (LENGTH, LENGTH_HINT_KEY),
        (GAP, FACE_GAP),
        (MAGNETS, POLES),
    ];

    #[test]
    fn key_design_draws_each_hint_line_under_its_row() {
        // Slice 2 spec 3.5 (S2-7): one weak line under each Key design input, between its row
        // and the next; not repeated when the input's own group is open.
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        for (i, &path) in KEY_DESIGN.iter().enumerate() {
            let hint = hint_for(path).unwrap_or_else(|| panic!("{path} has no hint"));
            assert_eq!(count(&output, hint), 1, "{path}: {hint}");
            let at = rect(hint);
            assert!(
                harness.widget(path).rect.bottom() <= at.top() + 0.5,
                "{path}: the hint is not under its row"
            );
            if let Some(&next) = KEY_DESIGN.get(i + 1) {
                assert!(
                    at.bottom() <= harness.widget(next).rect.top() + 0.5,
                    "{path}: the hint runs into the next row"
                );
            }
        }
        // The face gap's own group open: its row there draws no second hint.
        let gap = WORKFLOW
            .iter()
            .find(|group| group.id == "gap")
            .expect("the gap group")
            .label;
        harness.click_text(gap);
        // The header opens over a few frames (its animation).
        let mut output = harness.frame(Vec::new());
        for _ in 0..10 {
            output = harness.frame(Vec::new());
        }
        let residual = label_for("metal.residual_target_mm");
        assert_eq!(count(&output, residual), 1, "the gap group is open");
        let hint = hint_for(FACE_GAP).expect("the face gap has a hint");
        assert_eq!(count(&output, hint), 1, "{hint}");
    }

    #[test]
    fn each_size_control_s_entry_draws_its_hint_under_what_sets_it() {
        // Slice 2 spec 3.5 (S2-7), moved by UX overhaul spec 4.3: each of the five controls'
        // hints shows once, in its entry, under the entry's title and its "What sets it"; the
        // Size column draws none of them.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let column = harness.frame(Vec::new());
        for (label, key) in SIZE_TAB_HINTS {
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
            assert_eq!(count(&column, hint), 0, "{label}: the column draws no hint");
            let output = entry(&mut harness, key);
            assert_eq!(count(&output, hint), 1, "{label}: {hint}");
            let heading = explain_rect(&output, WHAT_SETS_IT);
            assert!(
                explain_rect(&output, label).bottom() <= heading.top() + 0.5,
                "{label}: the entry's title leads"
            );
            assert!(
                heading.bottom() <= explain_rect(&output, hint).top() + 0.5,
                "{label}: the hint is not under its heading"
            );
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "drawing them edits nothing"
        );
    }

    #[test]
    fn hint_lines_stay_inside_their_sides_in_a_narrow_window() {
        // Spec 5 "A narrow window: hint lines wrap": at the M4-1 review's 930-point window every
        // hint wraps inside its side: the Key design group's in the inputs side, each Size
        // control's in its entry in the right column (UX overhaul spec 4.3).
        let narrow = egui::vec2(930.0, 3000.0);
        let side = |harness: &Harness, id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("{id} is drawn"))
                .rect
        };
        let mut wrapped = false;
        let mut inside = |output: &egui::FullOutput, hint: &str, left: f32, right: f32| {
            let at = text_rect(output, hint).unwrap_or_else(|| panic!("no {hint:?}"));
            assert!(
                at.left() >= left - 1.0 && at.right() <= right + 1.0,
                "{hint:?} at {at:?} runs out of its side ({left} to {right})"
            );
            wrapped |= rows(output, hint) > 1;
        };
        let mut harness = Harness::on_screen(narrow);
        let output = harness.frame(Vec::new());
        let left = side(&harness, "magcoupling_inputs").left();
        for path in KEY_DESIGN {
            let hint = hint_for(path).unwrap_or_else(|| panic!("{path} has no hint"));
            inside(&output, hint, left, left + INPUTS_WIDTH);
        }
        let mut harness = size_tab(narrow, DesignInputs::default());
        for (_, key) in SIZE_TAB_HINTS {
            let output = entry(&mut harness, key);
            let right = side(&harness, "magcoupling_dashboard");
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
            inside(&output, hint, right.left(), right.right());
        }
        assert!(
            wrapped,
            "no hint wrapped: the test no longer exercises wrapping"
        );
    }

    #[test]
    fn touching_a_size_control_selects_its_row_and_a_hover_selects_nothing() {
        // UX overhaul spec 5.2: a click on a Size control's label, the keyboard's focus
        // arriving on its slider, or a drag of its slider, selects its row; a hover over a label
        // shows its hover text and selects nothing (C-2). A label's click and the focus edit
        // nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
        let label_at = |output: &egui::FullOutput, label: &str| {
            text_rects(output, label)
                .into_iter()
                .find(|rect| rect.right() <= INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no {label:?} in the Size column"))
                .center()
        };
        for row in size_controls::SIZE_ROWS {
            let output = harness.frame(Vec::new());
            harness.frame(vec![egui::Event::PointerMoved(label_at(
                &output, row.label,
            ))]);
            harness.frame(Vec::new());
            assert_eq!(harness.panel.selection, tile, "a hover on {:?}", row.label);
        }
        // The six sliders: the four controls, Minimum OD, then Required (size_column_ui).
        let sliders: Vec<egui::Id> = harness.panel.design_widgets[..6].to_vec();
        for id in sliders {
            let at = harness
                .ctx
                .read_response(id)
                .expect("the slider has a response")
                .rect
                .center();
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            harness.frame(Vec::new());
            assert_eq!(
                harness.panel.selection, tile,
                "a hover on the slider at {at:?}"
            );
        }
        for row in size_controls::SIZE_ROWS {
            harness.panel.selection = Selection::Nothing;
            let output = harness.frame(Vec::new());
            harness.click(label_at(&output, row.label));
            assert_eq!(
                harness.panel.selection,
                Selection::Input(row.key),
                "{:?}",
                row.label
            );
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "a click edits nothing"
        );
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        // Tab from the slider before it (a focus requested between frames is not one that
        // arrives) onto the gap's slider: two presses, its value box then the gap's slider, so
        // the gap's label takes no Tab stop.
        let gap = harness.size_widget(GAP).id;
        let length = harness.size_widget(LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(length));
        harness.frame(Vec::new());
        harness.panel.selection = Selection::Nothing;
        harness.frame(key_tap(egui::Key::Tab));
        assert!(
            !harness.ctx.memory(|m| m.has_focus(gap)),
            "one press: the value box"
        );
        harness.frame(key_tap(egui::Key::Tab));
        assert!(
            harness.ctx.memory(|m| m.has_focus(gap)),
            "Tab reaches the gap in two presses"
        );
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.selection,
            Selection::Input(FACE_GAP),
            "the keyboard's focus"
        );
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the focus edits nothing"
        );
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        let at = harness.size_widget(GAP).rect.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(vec![egui::Event::PointerMoved(at + egui::vec2(12.0, 0.0))]);
        harness.frame(vec![primary_button(at + egui::vec2(12.0, 0.0), false)]);
        assert_eq!(
            harness.panel.selection,
            Selection::Input(FACE_GAP),
            "a drag"
        );
    }

    #[test]
    fn the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them() {
        // UX overhaul spec 4.3: each control is its label and its slider; what left the column
        // (every hint, Set-by, floor and what-moves-what line, the Why links and the footnote
        // on what the controls move) is drawn nowhere until a row is selected. Under the
        // controls, the folds Detailed inputs and Assumptions, both closed.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let shown = harness.panel.shown_inputs();
        let consequences = shown_consequences(&harness);
        let size = dimensions(&shown).expect("the starting design has size controls");
        let mut gone: Vec<String> = SIZE_TAB_HINTS
            .iter()
            .map(|&(_, key)| hint_for(key).expect("a hint").to_owned())
            .collect();
        gone.extend(consequences.magnet_lines(shown.coupling.npole, harness.panel.auto()));
        gone.extend(consequences.diameter_line());
        gone.extend(consequences.length_line());
        gone.push(size_controls::floor_line(size.minimum_diameter_range.min));
        gone.push(size_controls::LINKED_HOUSING.to_owned());
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE] {
            gone.push(why_note_text(reviewed_note(id).expect("a reviewed note")));
        }
        for text in &gone {
            assert_eq!(count(&output, text), 0, "{text:?} is drawn");
        }
        let column = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .find(|rect| rect.right() <= INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no {text:?} in the Size column"))
        };
        for row in size_controls::SIZE_ROWS {
            column(row.label);
        }
        let detailed = column(DETAILED_INPUTS);
        let assumptions = column(ASSUMPTIONS_FOLD);
        assert!(
            detailed.bottom() <= assumptions.top(),
            "{detailed:?} {assumptions:?}"
        );
        assert_eq!(
            count(&output, ASSUMPTIONS_NOTE),
            0,
            "the Assumptions fold starts closed"
        );
    }

    #[test]
    fn the_assumptions_fold_scrolls_with_the_size_column_to_its_last_assumption() {
        // UXQ-2 of the 2026-11-23 review: the fold drew the Assumptions view's own scroll area
        // inside the column's, a pane of two rationales at 1024x700 the column could not scroll
        // past. Opened at 1024x700, a wheel over the column brings the last assumption's
        // rationale into the window.
        let size = egui::vec2(1024.0, 700.0);
        let mut harness = size_tab(size, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let header = text_rects(&output, ASSUMPTIONS_FOLD)
            .into_iter()
            .find(|rect| rect.right() <= INPUTS_WIDTH)
            .expect("the fold's header in the Size column");
        harness.click(header.center());
        let paused = harness.panel.clamps_paused;
        let last = assumptions::states(&harness.panel.inputs)
            .into_iter()
            .rev()
            .find(|state| state.assumption.shown(paused))
            .expect("an assumption is shown")
            .assumption
            .rationale;
        let output = harness.frame(Vec::new());
        assert!(
            text_rect(&output, last).is_none_or(|rect| rect.top() > size.y),
            "the last rationale is in view before a scroll: the test would prove nothing"
        );
        harness.frame(vec![egui::Event::PointerMoved(header.center())]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -20000.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        // egui spreads a wheel step over several frames.
        let mut output = harness.frame(Vec::new());
        for _ in 0..20 {
            output = harness.frame(Vec::new());
        }
        let rect = text_rect(&output, last).expect("the last rationale is drawn");
        assert!(rect.bottom() <= size.y, "{rect:?} below the window's foot");
    }

    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
        // Spec 6 test 10, the owner's case of 2026-10-13: from 10 to 8 magnets the coupling did
        // not shrink and nothing said why. At 8 magnets Magnets per ring's entry says the ring
        // keeps its size, and Outside diameter's Set-by line shows the cap as the largest part:
        // the outside diameter is the cap's, so fewer magnets cannot lower it (in the Explain
        // panel since UX overhaul spec 4.3).
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, with_poles(8));
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
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        assert_eq!(count(&output, &set_by), 1);
        assert!(
            explain_rect(&output, WHAT_SETS_IT).bottom() <= explain_rect(&output, &set_by).top(),
            "in Outside diameter's entry, under What sets it"
        );
        let output = entry(&mut harness, POLES);
        assert_eq!(count(&output, RING_SIZE_STAYS), 1);
        // T5/T6 amendment: with Minimum OD unset the slider floor, not the coupling, stops the
        // fit search. The line says how far it could go and that Minimum OD unlocks it.
        let fit = consequences
            .magnet_lines(8, None)
            .into_iter()
            .find(|line| line.starts_with("Fits down to "))
            .expect("a fit line at the slider's floor");
        assert!(fit.ends_with("set Minimum OD lower to reach it."), "{fit}");
        assert_eq!(count(&output, &fit), 1, "{fit}");
        // Slice 3: two lines for Magnets per ring, and the torque card answers the owner's
        // question in numbers (decision 7A moved the neighbour lines there): the entry draws
        // the card's count row, and the Hot-low tile's entry every one of its lines.
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
        for line in consequences
            .card
            .control_lines(Control::MagnetsPerRing, None)
        {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let output = harness.frame(Vec::new());
        for line in consequences.card.lines(None) {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
    }

    #[test]
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): Magnets per ring's
        // first line says what a new count does to the ring: it stays with Auto off and with
        // Auto on the magnet length; with Auto on the ring size the torque card's Auto line says
        // Auto sizes the ring again (decision 5A), and with Auto on the count (its slider
        // locked) no line speaks of changing it. The control's lines are drawn one under the
        // next in its entry, and the card draws the Auto line of the state and no other.
        use crate::gui::explain::WHAT_SETS_IT;
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
        let states = [
            (None, Some(RING_SIZE_STAYS), None),
            (
                Some(FreeVariable::AxialLength),
                Some(RING_SIZE_STAYS),
                Some(AUTO_HOLDS_LENGTH),
            ),
            (Some(FreeVariable::RingRadius), None, Some(AUTO_HOLDS_RING)),
            (
                Some(FreeVariable::MagnetsPerRing),
                None,
                Some(AUTO_HOLDS_COUNT),
            ),
        ];
        for (auto, first, auto_line) in states {
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            if let Some(variable) = auto {
                harness.panel.sizing.variable = variable;
                harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            }
            harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let output = entry(&mut harness, POLES);
            // The entry's card row keeps the same Auto line (owner 5A), between the caption
            // and the row.
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: the entry's {line}");
            }
            assert_eq!(
                count(&output, RING_SIZE_STAYS),
                usize::from(first.is_some()),
                "{auto:?}"
            );
            let shown = harness.panel.shown_inputs();
            let lines = shown_consequences(&harness).magnet_lines(shown.coupling.npole, auto);
            assert_eq!(
                lines.len(),
                1 + usize::from(first.is_some()),
                "{auto:?}: {lines:?}"
            );
            assert_eq!(
                lines.first().map(String::as_str) == first,
                first.is_some(),
                "{auto:?}"
            );
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{auto:?}: {line}");
                let at = explain_rect(&output, line);
                assert!(above.bottom() <= at.top() + 0.5, "{auto:?}: {line:?}");
                above = at;
            }
        }
    }

    #[test]
    fn size_tab_what_moves_what_lines_follow_each_edit() {
        // Spec 6 test 5: the lines are recomputed when the design changes (the cache keys on
        // it), from the magnet count's own slider and from any other input; so is the torque
        // card that took over the neighbour lines (slice 3, decision 7A). The arrow key selects
        // the control, whose entry draws its lines and its card row (UX overhaul spec 4.3); the
        // other controls' entries and the Hot-low tile's whole card follow too.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
        let row = |c: &Consequences| c.card.control_lines(Control::MagnetsPerRing, None);
        let row_at_10 = row(&shown_consequences(&harness));
        // The controls' widgets in drawing order: diameter, length, gap, magnets, Minimum OD.
        let magnets = harness.panel.design_widgets[3];
        harness.ctx.memory_mut(|m| m.request_focus(magnets));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowLeft));
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        assert_eq!(harness.panel.selection, Selection::Input(POLES));
        let at_8 = shown_consequences(&harness).magnet_lines(8, None);
        assert_ne!(at_8, at_10);
        let row_at_8 = row(&shown_consequences(&harness));
        assert_ne!(row_at_8, row_at_10);
        for (now, then) in [(&at_8, &at_10), (&row_at_8, &row_at_10)] {
            for line in now {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            for line in then.iter().filter(|line| !now.contains(line)) {
                assert_eq!(count(&output, line), 0, "stale: {line}");
            }
        }
        // Another input: a wider face gap moves the smallest diameter that fits and every step
        // of the card.
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        let output = harness.frame(Vec::new());
        let wider = shown_consequences(&harness);
        let lines = wider.magnet_lines(8, None);
        assert_ne!(lines, at_8);
        let row_wider = row(&wider);
        assert_ne!(row_wider, row_at_8);
        for (now, then) in [(&lines, &at_8), (&row_wider, &row_at_8)] {
            for line in now {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            for line in then.iter().filter(|line| !now.contains(line)) {
                assert_eq!(count(&output, line), 0, "stale: {line}");
            }
        }
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        let diameter = wider.diameter_line().expect("a Set-by line");
        assert_eq!(count(&output, &diameter), 1, "{diameter}");
        let output = entry(&mut harness, LENGTH_HINT_KEY);
        let length = wider.length_line().expect("a length split");
        assert_eq!(count(&output, &length), 1, "{length}");
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let output = harness.frame(Vec::new());
        for line in wider.card.lines(None) {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
    }

    #[test]
    fn size_tab_smallest_diameter_line_searches_with_the_minimum_od() {
        // Spec 3.3 and 3.5: the fit search covers the Size tab's diameter range, the Minimum OD
        // preference included. At 8 magnets the blocks fit at the slider's floor, so the line
        // takes its minimum form; a lower Minimum OD lets the search find the smallest fit.
        let mut harness = size_tab(SCREEN, with_poles(8));
        harness.panel.selection = Selection::Input(POLES);
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
        // Spec 5, as the torque card's count row has it (slice 3, decision 7A): at 4 or 40
        // magnets per ring the side past the slider reads END_OF_RANGE, drawn in the control's
        // entry as its card row, the side inside it is a step, and no text names 2 or 42
        // magnets. The default blocks do not fit 40 magnets (the card reads its one line
        // there); forty narrow blocks do.
        use crate::gui::test_support::forty_narrow_magnets;
        use crate::gui::torque_card::END_OF_RANGE;
        for (inputs, past, inside) in [
            (with_poles(4), "-2", "+2"),
            (forty_narrow_magnets(), "+2", "-2"),
        ] {
            let npole = inputs.coupling.npole;
            let mut harness = size_tab(SCREEN, inputs);
            let output = entry(&mut harness, POLES);
            let consequences = shown_consequences(&harness);
            let count_line = row_line(&consequences.card, Control::MagnetsPerRing, None);
            assert!(
                count_line.contains(&format!("{past} -> {END_OF_RANGE}")),
                "{count_line}"
            );
            assert!(
                !count_line.contains(&format!("{inside} -> {END_OF_RANGE}")),
                "{count_line}"
            );
            assert_eq!(count(&output, &count_line), 1, "{count_line}");
            // Two lines for Magnets per ring at a slider end (three before the card): the
            // ring-size line and the fit line.
            let lines = consequences.magnet_lines(npole, None);
            assert_eq!(lines.len(), 2, "{lines:?}");
            assert!(lines.contains(&RING_SIZE_STAYS.to_owned()), "{lines:?}");
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            // No drawn text names the count past the slider's end as "<count> magnets", the form
            // in which the Size tab names a count (the fit line's "... fits 10 magnets: ...").
            // The count must stand alone: after a letter or a digit ("12 magnets", "N42
            // magnets") it is part of another word.
            let outside = if past == "-2" { npole - 2 } else { npole + 2 };
            let named = format!("{outside} magnets");
            let names_outside = |text: &str| {
                text.match_indices(&named)
                    .any(|(at, _)| !text[..at].ends_with(char::is_alphanumeric))
            };
            let texts = drawn_texts(&output);
            assert!(
                !texts.iter().any(|t| names_outside(t)),
                "{npole}: {texts:?}"
            );
        }
    }

    #[test]
    fn size_tab_narrow_window_wraps_the_new_lines_inside_their_sides() {
        // Spec 5 "Narrow windows", at the M4-1 review's ~930 px window and the owner's 8 magnets:
        // the what-moves-what lines wrap inside the right column, in their controls' entries
        // (UX overhaul spec 4.3), one under the next; the Required row is the inputs side's
        // (its 320 points), and the tiles stay in the centre region, left of the right column.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
        let output = harness.frame(Vec::new());
        let side = |harness: &Harness, id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("{id} is drawn"))
                .rect
        };
        let inputs_left = side(&harness, "magcoupling_inputs").left();
        let inputs_right = inputs_left + INPUTS_WIDTH;
        let dashboard_left = side(&harness, "magcoupling_dashboard").left();
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let results = harness.panel.results().clone();
        let (gap, _) = torque_gap_text(
            results.metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        let label = pullout_label(results.metal.op_temp_C);
        let pieces = crate::gui::tiles::gap_pieces(&gap);
        let texts = [HOT_LOW, label.as_str()].into_iter();
        for text in texts.chain(pieces.iter().map(String::as_str)) {
            let at = centre_rect(&output, text);
            assert!(
                at.left() >= inputs_right - 1.0 && at.right() <= dashboard_left + 1.0,
                "{text:?} at {at:?} runs out of the centre ({inputs_right} to {dashboard_left})"
            );
        }
        let required = rect(REQUIRED);
        assert!(
            required.left() >= inputs_left - 1.0 && required.right() <= inputs_right + 1.0,
            "{required:?}"
        );
        let consequences = shown_consequences(&harness);
        let mut wrapped = false;
        for (key, lines) in [
            (POLES, consequences.magnet_lines(8, None)),
            (
                DIAMETER_HINT_KEY,
                consequences.diameter_line().into_iter().collect(),
            ),
            (
                LENGTH_HINT_KEY,
                consequences.length_line().into_iter().collect(),
            ),
        ] {
            assert!(!lines.is_empty(), "{key}");
            let output = entry(&mut harness, key);
            let right = side(&harness, "magcoupling_dashboard");
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
                let at = explain_rect(&output, line);
                assert!(
                    above.bottom() <= at.top() + 0.5,
                    "{line:?} overlaps the line above it"
                );
                assert!(
                    at.left() >= right.left() - 1.0 && at.right() <= right.right() + 1.0,
                    "{line:?} at {at:?} runs out of the right column ({right:?})"
                );
                wrapped |= rows(&output, line) > 1;
                above = at;
            }
        }
        assert!(
            wrapped,
            "no line wrapped: the test no longer exercises wrapping"
        );
    }

    #[test]
    fn size_tab_new_texts_have_glyphs_and_plain_words() {
        // Spec 6 tests 3 and 11 for the Size tab, in each state its new texts take: the
        // default, a met requirement, both ends of the magnet slider, the owner's 8 magnets, no
        // prediction (short magnets) and a ring too small for its magnets; the controls' lines
        // drawn in their entries (UX overhaul spec 4.3). Every drawn text has glyphs in the
        // default fonts; the new ones name no workbook cell, correction or sheet.
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
            // A 0.49 mm cap skirt: its amber caution (BL-082).
            (with_cap(42.8, 41.82), None),
        ];
        let mut harness = size_tab(egui::vec2(1280.0, 4000.0), DesignInputs::default());
        let mut drawn = Vec::new();
        let mut new_texts = vec![
            REQUIRED.to_owned(),
            HOT_LOW.to_owned(),
            CARD_HEADING.to_owned(),
            size_controls::MINIMUM_HOVER.to_owned(),
            crate::gui::explain::SPREADSHEET_CELL.to_owned(),
        ];
        // The Hot-low tile's entry (the whole card) and each Size control's (UX overhaul spec
        // 4.3: the lines that were drawn under the controls).
        let selections = std::iter::once(Selection::Result(crate::gui::tiles::Tile::HotLow.path()))
            .chain(size_controls::SIZE_ROWS.map(|row| Selection::Input(row.key)));
        let selections: Vec<Selection> = selections.collect();
        for (inputs, minimum) in states {
            harness.panel.inputs = inputs;
            harness.panel.minimum_diameter_mm = minimum;
            // The second frame draws with any minimum the first one pinned.
            harness.frame(Vec::new());
            let mut all = Vec::new();
            for &selection in &selections {
                harness.panel.selection = selection;
                all.extend(drawn_texts(&harness.frame(Vec::new())));
            }
            let shown = harness.panel.shown_inputs();
            let results = compute_all(&shown);
            let consequences = shown_consequences(&harness);
            let mut texts = consequences.magnet_lines(shown.coupling.npole, None);
            // The torque card's lines (slice 3), the neighbour lines' numbers among them.
            texts.extend(consequences.card.lines(None));
            if let Some(size) = dimensions(&shown) {
                texts.extend(consequences.diameter_line());
                texts.extend(consequences.length_line());
                texts.push(size_controls::floor_line(size.minimum_diameter_range.min));
            }
            texts.push(size_controls::LINKED_HOUSING.to_owned());
            texts.push(pullout_label(results.metal.op_temp_C));
            // The amber wall lines (the keyway's and the cap skirt's, BL-082).
            texts.extend(wall_cautions(&shown, &results));
            let hot_low = results.metal.torque_hot_low_Nm;
            if predicts(&results, hot_low) {
                let gap = torque_gap_text(hot_low, shown.metal.required_min_Nm).0;
                texts.extend(crate::gui::tiles::gap_pieces(&gap));
            } else {
                texts.push(NO_PREDICTION.to_owned());
            }
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
        let mut harness = Harness::on_screen(TALL).on_results();
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
        // On a tall screen: with Auto's lines and the hint lines the measured drag row sits
        // below a laptop screen's fold, and egui paints no label scrolled out.
        let mut harness = Harness::on_screen(TALL);
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
        harness.select(SPACE_CLAIM_CHECK);
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
        let wrap = text_shapes(&output, &status)
            .next()
            .map(|(_, text)| text.galley.job.wrap.max_width);
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
        tooltips_at_once(&harness.ctx);
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
        let line = "No valid value of ring radius in range";
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
        tooltips_at_once(&harness.ctx);
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
    fn the_size_tab_s_control_labels_show_their_hovers() {
        use crate::gui::plain_text::help_for;
        use crate::gui::size_controls::{DIAMETER_HOVER, LENGTH_HOVER};
        // The gap and magnet-count labels hover their input's plain_text::help_for (F19's
        // second help site); Outside diameter and Overall length, which are no inputs, say what
        // they are and what moving them changes (R-3).
        for (label, hover) in [
            (GAP, help_for(FACE_GAP)),
            (MAGNETS, help_for(POLES)),
            (DIAMETER, DIAMETER_HOVER),
            (LENGTH, LENGTH_HOVER),
        ] {
            assert!(!hover.is_empty(), "{label}");
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            tooltips_at_once(&harness.ctx);
            let output = harness.frame(Vec::new());
            assert!(
                !drawn_texts(&output).iter().any(|t| t == hover),
                "{label}: {hover:?} is drawn before the hover"
            );
            let at = text_center(&output, label);
            // egui shows a tooltip only after time has passed with the pointer on the widget.
            harness.frame_after(0.5, vec![egui::Event::PointerMoved(at)]);
            let output = harness.frame_after(0.5, Vec::new());
            assert!(
                drawn_texts(&output).iter().any(|t| t == hover),
                "{label}: {hover:?}"
            );
        }
    }

    #[test]
    fn hovering_a_headline_value_shows_its_equation() {
        use crate::gui::readouts::OPEN_HINT;
        let mut harness = Harness::new().on_results();
        tooltips_at_once(&harness.ctx);
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
        // D_cup = 2 (r_corner + t_wall): hovering the cup OD in the Results tab's Headline group
        // frames the Key design row of the wall in its term's colour, from the next frame, and
        // its tooltip draws t_wall in that colour; moving the pointer away clears every mark.
        let mut harness = Harness::new().on_results();
        tooltips_at_once(&harness.ctx);
        // The right column settles its width on the second frame: measure where the row rests.
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
    fn the_equation_panel_starts_closed_with_no_button_on_the_bar_and_a_value_opens_it() {
        // Decision C-11: the Equation panel opens from a value (a click on a readout, an Explain
        // entry's Equation link), not from the top bar; its Close button closes it.
        use crate::gui::explorer::{CLOSE, EQUATION_PANEL};
        let mut harness = Harness::new();
        assert!(!harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EQUATION_PANEL), 0);
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EQUATION_PANEL), 1);
        harness.click_text(CLOSE);
        assert!(!harness.panel.explorer.open);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns() {
        use crate::gui::explorer::TERMS;
        let mut harness = Harness::new().on_results();
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
            harness.panel.selection = Selection::Note("br_temperature");
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
        // above the panel and the window keeps its size, the Size tab with its torque card
        // closed, as the Equation panel's opening leaves it, and opened by the user (slice 3
        // test 21).
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
            // The cell line only with the spreadsheet details (spec S2-5).
            for show_details in [false, true] {
                let mut harness = Harness::new();
                harness.panel.show_details = show_details;
                let shut = harness.frame(Vec::new());
                harness.panel.explorer.open_path(&entry.path);
                harness.frame(Vec::new());
                let output = harness.frame(Vec::new());
                assert_eq!(count(&output, NO_EQUATION), 1);
                // The panel adds a line to what the views draw: the cell, or NO_SPREADSHEET_CELL.
                let cell = entry
                    .info
                    .cell
                    .as_deref()
                    .unwrap_or(crate::gui::plain_text::NO_SPREADSHEET_CELL);
                assert_eq!(
                    count(&output, cell),
                    count(&shut, cell) + usize::from(show_details),
                    "{cell} (details {show_details})"
                );
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
    }

    #[test]
    fn the_open_equation_shows_its_label_cell_value_and_corrections() {
        // Decision M43-14: the corrections the equation embodies, upstream of the pull-out;
        // the cell and the corrections only with the spreadsheet details (spec S2-5), the
        // label alone without them.
        let value_line = format!("T_pull = {}", displayed_pullout(&DesignInputs::default()));
        let label = label_for("model.pullout_Nm");
        let header = format!("{label} (Calculator!C93)");
        for show_details in [false, true] {
            let mut harness = Harness::new();
            harness.panel.show_details = show_details;
            let shut = harness.frame(Vec::new());
            harness.panel.explorer.open_path("model.pullout_Nm");
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, &value_line), 1, "{value_line:?}");
            for line in [header.as_str(), "Embodies corrections: E7, E8, E3"] {
                assert_eq!(count(&output, line), usize::from(show_details), "{line:?}");
            }
            assert_eq!(
                count(&output, label),
                count(&shut, label) + usize::from(!show_details),
                "{label:?}"
            );
        }
    }

    #[test]
    fn the_dashboard_keeps_its_width_with_the_spreadsheet_details_on() {
        // Each value and its correction ids flow as one wrapped text: a long value (the cup
        // wall's verdict in plain words, ruling R9, among the numbers of its check's entry)
        // with its ids never runs past the row, which would widen the resizable right column a
        // little more every frame.
        for size in [egui::vec2(930.0, 1024.0), SCREEN] {
            let mut harness = Harness::on_screen(size);
            harness.panel.show_details = true;
            harness.select("materials.cup_wall_check");
            let lefts: Vec<f32> = (0..4)
                .map(|_| right_column_left(&harness.frame(Vec::new())))
                .collect();
            assert!(
                lefts.iter().all(|&left| left == lefts[0]),
                "{size:?}: the right column moved {lefts:?}"
            );
        }
    }

    #[test]
    fn the_panel_s_own_strings_read_in_plain_words() {
        // Spec 3.8 (map-locks J1-J3, J7, J8, J10, J12, J13, J16; map-jargon C-4 to C-6, C-8,
        // C-11, A-10): the panel's own words name no spreadsheet cell, correction, file or code,
        // and none of the engine's internal words.
        use crate::engine::explain::{TermKind, TermStyle};
        use crate::gui::clamp_drawing::ONE_PIECE_ONLY;
        use crate::gui::dashboard::END_EFFECT_GREYED;
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::explorer::{NO_EQUATION, term_tag};
        use crate::gui::input_ui::{CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT};
        use crate::gui::inputs::ASSUMPTION_LINE;
        use crate::gui::plain_text::NO_SPREADSHEET_CELL;
        use crate::gui::results_table::NOTHING_TRACED;
        // Words of the engine and its tests that mean nothing to a reader of the page.
        const INTERNAL: [&str; 7] = [
            "explained result",
            "equation record",
            "differential test",
            "refresh after a 3d run",
            "f_end",
            "autofit",
            "rust-only",
        ];
        let mut texts: Vec<String> = [
            RESET_ASSUMPTIONS,
            RESET_ASSUMPTIONS_HINT,
            ASSUMPTIONS_RESET,
            CHANGED_FROM_DEFAULT,
            NO_EQUATION,
            TRACE_LINK,
            OUTSIDE_RANGE_HINT,
            NOTHING_TRACED,
            ONE_PIECE_ONLY,
            NO_SPREADSHEET_CELL,
            ASSUMPTION_LINE,
            END_EFFECT_GREYED,
        ]
        .map(str::to_owned)
        .to_vec();
        texts.extend(InputOrder::ALL.map(|order| order.label().to_owned()));
        for group in &WORKFLOW {
            texts.push(group.label.to_owned());
            texts.extend(
                group
                    .sections
                    .iter()
                    .map(|section| section.label.to_owned()),
            );
        }
        texts.push(term_tag(TermStyle {
            kind: TermKind::CellOnly,
            changed_from_default: false,
            affected_by_modified_assumption: false,
        }));
        for path in [
            "coupling.drive_torque_Nm",
            "metal.face_gap_mm",
            "model.pullout_Nm",
        ] {
            texts.push(Trace::of(path).expect("a traceable path").banner(false));
        }
        texts.push(
            end_effect_banner(&compute_all(&short_magnets()))
                .expect("short magnets are out of the end-effect range"),
        );
        let defaults = DesignInputs::default();
        let notes = crate::gui::geometry::geometry(&defaults, &compute_all(&defaults), false).notes;
        assert_eq!(
            notes.len(),
            3,
            "the wall rule and both design checks at the defaults"
        );
        texts.extend(notes.into_iter().map(|note| note.text));
        for text in &texts {
            assert_eq!(
                crate::gui::guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text:?}"
            );
            let lower = text.to_lowercase();
            for word in INTERNAL {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }
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
        // The source line only with the spreadsheet details (spec S2-5).
        for show_details in [false, true] {
            // Tall enough for all fourteen without scrolling.
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
            harness.panel.show_details = show_details;
            harness.click_text(InputsView::Assumptions.label());
            assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
            let output = harness.frame(Vec::new());
            let texts = drawn_texts(&output);
            for assumption in ASSUMPTIONS {
                for want in [assumption.label.to_owned(), assumption.rationale.to_owned()] {
                    assert!(texts.contains(&want), "missing {want:?}");
                }
                let source = format!("{SOURCE}: {}", assumption.source);
                assert_eq!(texts.contains(&source), show_details, "{source:?}");
            }
            // Idle frames rewrite no assumption (the rows are the inputs' own).
            for _ in 0..3 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            harness.click_text(InputsView::Design.label());
            assert_eq!(harness.panel.inputs_view, InputsView::Design);
        }
    }

    #[test]
    fn the_equation_panel_s_changed_tag_follows_the_baseline() {
        // UX overhaul spec 7.1 (C-22). The passing test design asks for 1.6 N·m, the
        // spreadsheet for 2.5 N·m; the hot minimum check's equation reads the requirement.
        let input = "input: click to find its row";
        let changed = format!("{CHANGED_DOT} changed; {input}");
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        harness.panel.explorer.open_path("metal.hot_min_check");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            (count(&output, input), count(&output, &changed)),
            (1, 0),
            "at the baseline"
        );
        harness.panel.inputs.metal.required_min_Nm = 2.5;
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            (count(&output, input), count(&output, &changed)),
            (0, 1),
            "at the spreadsheet's requirement"
        );
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
            text_rect(&output, ASSUMPTION_TAG)
                .is_none_or(|tag| tag.right() > text_clip_rect(&output, ASSUMPTION_TAG).right()),
            "the tag fits without scrolling: the test would prove nothing"
        );
        // The pointer on the start of the set's value, in view at the left of the list.
        let at = text_rect(&output, "5: 1, 3, 5 (default)")
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
        let tag = text_rect(&output, ASSUMPTION_TAG).expect("the tag is drawn");
        let clip = text_clip_rect(&output, ASSUMPTION_TAG);
        assert!(clip.contains_rect(tag), "{tag:?} cut by {clip:?}");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn closing_the_equation_panel_clears_the_input_row_s_frame() {
        // Decision M43-12: a leaf term frames its row while its equation is shown; closing
        // the panel, by its Close button (the only one: decision C-11), takes the frame away.
        use crate::gui::explorer::CLOSE;
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
        assert_eq!(focus_rects(&output).len(), 1, "the row is framed");
        harness.click_text(CLOSE);
        assert!(!harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.focus(), None);
        // The inputs side paints before the dock: the frame goes on the next frame.
        let output = harness.frame(Vec::new());
        assert!(focus_rects(&output).is_empty());
        // Shown again, the panel keeps its equation but frames no row.
        harness.panel.explorer.open = true;
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.gearbox_input_ripple_Nm"]
        );
        let output = harness.frame(Vec::new());
        assert!(focus_rects(&output).is_empty(), "shown again");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
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
    fn the_open_equation_s_lesson_line_shows_its_note_in_the_explain_panel() {
        // Decision C-11: the Equation panel's Explain toggle is gone. Under the open equation
        // its note's title follows "Full lesson:", and a click shows the note in the Explain
        // panel; the equation stays open (C-25). The line follows the equation.
        use crate::engine::explain::notes::{note, note_for};
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::{NO_NOTE, NoteFallback, START_HERE_HINT, note_fallback};
        let mut harness = Harness::on_screen(TALL);
        let pullout = note("pullout_angle").unwrap();
        let end = note("end_effect").unwrap();
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Explain"), 0, "no toggle");
        assert_eq!(count(&output, FULL_LESSON), 1);
        assert_eq!(count(&output, pullout.title), 1, "the lesson line alone");
        assert_eq!(count(&output, pullout.sentences[0]), 0);
        assert_eq!(
            diagram_points(&output),
            0,
            "no diagram in the Equation panel"
        );
        harness.click_text(pullout.title);
        assert_eq!(harness.panel.selection, Selection::Note(pullout.id));
        assert_eq!(harness.panel.explorer.current(), Some("model.pullout_Nm"));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, pullout.sentences[0]),
            1,
            "in the Explain panel"
        );
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end: the line names its note; the Explain panel keeps the pull-out's.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, end.title), 1);
        assert_eq!(harness.panel.selection, Selection::Note(pullout.id));
        // A value no reviewed note explains, with nothing upstream that has a note or an
        // assumption (never a dashboard row, spec 3.7): the reason and where the notes are.
        let bare = registry()
            .equations()
            .iter()
            .map(|eq| eq.target.as_str())
            .find(|p| note_for(p).is_none() && note_fallback(p) == NoteFallback::Reason(NO_NOTE))
            .expect("an explained result with nothing upstream to read");
        harness.panel.explorer.open_path(bare);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 1);
        assert_eq!(count(&output, START_HERE_HINT), 1);
    }

    #[test]
    fn start_here_opens_the_matching_equations_in_turn() {
        use crate::engine::explain::notes::{START_HERE, note};
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        // The Explain panel's orientation starts the walk; each step opens its equation and
        // selects its note, which the Explain panel shows (UX overhaul spec 5.5).
        let mut harness = Harness::new();
        harness.click_text(START_HERE_BUTTON);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[0].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[0].0));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &start_here_text(0)), 1);
        let first = note(START_HERE[0].0).unwrap();
        // The note's title heads its entry, and the lesson line under the equation names it.
        assert_eq!(count(&output, first.title), 2);
        assert_eq!(
            count(&output, first.sentences[0]),
            1,
            "the note in the Explain panel"
        );
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[2].1));
        harness.click_text(PREVIOUS);
        assert_eq!(harness.panel.explorer.start_here(), Some(1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        harness.click_text(STOP);
        assert_eq!(harness.panel.explorer.start_here(), None);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn with_auto_on_the_step_1_note_names_the_length_auto_holds_and_draws_the_design_shown() {
        // Review Focus RF-1 (A-25 T-1): Auto on, its default free variable the axial magnet
        // length. torque_from_shear offers "Raise <axial magnet length>": that entry is plain
        // text with the reason, a click on its label asks for no row, and the note's live
        // values are the design shown's (the solved length in place), not the stored inputs'
        // (the axial override blank).
        use crate::engine::explain::note_text;
        use crate::engine::explain::notes::note;
        use crate::gui::explorer::{HELD_BY_AUTO, live_value};
        let id = "torque_from_shear";
        let mut harness = Harness::on_screen(TALL);
        harness.size();
        assert_eq!(
            harness.panel.sizing(),
            &SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::AxialLength,
            }
        );
        harness.panel.selection = Selection::Note(id);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!(" ({HELD_BY_AUTO})")), 1);
        let shown_inputs = harness.panel.shown_inputs();
        assert_ne!(
            &shown_inputs,
            harness.panel.inputs(),
            "the solved length, not the blank override"
        );
        let shown = compute_all(&shown_inputs);
        let stored = compute_all(harness.panel.inputs());
        let step_1 = note(id).unwrap();
        let drawn = note_text::render(step_1.sentences[1], |p| live_value(&shown, p));
        assert_eq!(count(&output, &drawn), 1, "{drawn:?}");
        // σ takes no magnet length, so sentence 1 reads alike on both designs. The first
        // watch-out's T_2D and pull-out move with the length: the design shown's copy is drawn,
        // the stored inputs' is not.
        let watch_out = |results: &DesignResults| {
            note_text::render(step_1.watch_out[0], |p| live_value(results, p))
        };
        let (drawn, not_drawn) = (watch_out(&shown), watch_out(&stored));
        assert_eq!(count(&output, &drawn), 1, "{drawn:?}");
        assert_eq!(count(&output, &not_drawn), 0, "{not_drawn:?}");
        let label = centre_rect(&output, label_for(AXIAL_LENGTH));
        harness.click(label.center());
        assert_eq!(
            harness.panel.explorer.focus(),
            None,
            "a held input is no link"
        );
        assert_eq!(
            harness.panel.selection,
            Selection::Note(id),
            "the note stays open"
        );
    }

    #[test]
    fn every_start_here_step_shows_with_glyphs_in_the_default_fonts() {
        // The bar names each step's note (glyph-safe, decision M43-6); it sits above
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
        // Each note selected (the Explain panel draws it), then its sources fold opened (it
        // starts closed): every drawn text, sources included, has glyphs.
        use crate::engine::explain::notes::NOTES;
        use crate::gui::explorer::WHERE_FROM;
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        for note in NOTES {
            harness.panel.selection = Selection::Note(note.id);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let texts = drawn_texts(&output);
            assert!(texts.contains(&glyph_safe(note.title)), "{}", note.id);
            let sources: Vec<String> = note.sources.iter().map(|s| glyph_safe(s)).collect();
            for source in &sources {
                assert!(
                    !texts.contains(source),
                    "{}: the fold starts closed",
                    note.id
                );
            }
            harness.click_text(WHERE_FROM);
            let mut output = harness.frame(Vec::new());
            for _ in 0..15 {
                output = harness.frame(Vec::new());
            }
            let texts = drawn_texts(&output);
            for source in &sources {
                assert!(texts.contains(source), "{}: {source}", note.id);
            }
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
        assert_eq!(harness.panel.selection, Selection::Note(rule.note_id));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, note.sentences[0]), 1);
        assert!(
            diagram_points(&output) == 0,
            "the flux-path diagram marks no point"
        );
        // Its one equation is a link: the circuit in effect, opened like a value in the
        // Equation panel; the note stays selected (decision C-25).
        harness.click_text("circuit");
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.selection, Selection::Note(rule.note_id));
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
        // folds the inputs under "Detailed inputs"; a new panel opens there). The Results tab's
        // Headline group lists the headline values.
        panel.centre = CentreView::Results;
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
        // A window sizes itself on its first frame and paints on the next; the right column
        // settles its width, which a third frame takes in.
        frame(&mut panel, Vec::new());
        frame(&mut panel, Vec::new());
        let output = frame(&mut panel, Vec::new());
        // The check list's status line, and the table's first headline value.
        let defaults = DesignInputs::default();
        let status = design_status_text(&design_status(
            &defaults,
            &compute_all(&defaults),
            clamps::PAUSED,
        ));
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
        use crate::gui::explain::CHECKS_IT_TOUCHES;
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text, guide};
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
        // from the Geometry view (the input's group or the Assumptions view opens) and from
        // Size and torque (its Detailed inputs open).
        let mut harness = Harness::on_screen(TALL);
        let defaults = DesignInputs::default();
        let fixes: Vec<(&str, &str, String)> = CHECK_GUIDES
            .iter()
            .zip(all_fixes(&defaults, &compute_all(&defaults), false))
            .flat_map(|(guide, fixes)| {
                fixes
                    .into_iter()
                    .map(|(path, dir)| (guide.check, path, fix_text(path, dir)))
            })
            .collect();
        assert!(!fixes.is_empty(), "the red checks have fixes");
        for view in [CentreView::Geometry, CentreView::SizeAndTorque] {
            for (check, path, text) in &fixes {
                // No frame left from the last fix.
                harness.panel.explorer.close();
                harness.panel.centre = view;
                // The right column settles its width on the second frame: click where the link
                // rests.
                harness.select(check);
                harness.click_text(text);
                assert_eq!(
                    harness.panel.explorer.focus().map(|f| f.path.as_str()),
                    Some(*path),
                    "{view:?}"
                );
                // Spec 5.2: the link selects its input too.
                assert_eq!(harness.panel.selection, Selection::Input(path), "{view:?}");
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
                // Decision D-54: the input's entry links back to the check under "Checks it
                // touches"; a click on it selects the check again.
                let heading = text_rect(&output, CHECKS_IT_TOUCHES)
                    .unwrap_or_else(|| panic!("{view:?} {path}: no {CHECKS_IT_TOUCHES:?}"));
                let back = text_rects(&output, guide(check).expect("a guide").label)
                    .into_iter()
                    .find(|r| r.top() >= heading.bottom())
                    .unwrap_or_else(|| panic!("{view:?} {path}: no link back to {check}"));
                harness.click(back.center());
                assert_eq!(
                    harness.panel.selection,
                    Selection::Check(check),
                    "{view:?} {path}"
                );
            }
        }
        // Following a fix edits nothing: no input, no undo step.
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_fix_link_on_the_opening_tab_shows_its_row_inside_a_short_window() {
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text};
        // Plan review C-9: the Size column scrolls (decision D-7) and Detailed inputs keeps its
        // own scroll area inside it. In a laptop window and in the smallest one, each fix link
        // of the example's red checks that the Explain panel shows opens Detailed inputs and
        // brings its input's row inside the window, not under its foot.
        let defaults = DesignInputs::default();
        let fixes: Vec<(&str, &str, String)> = CHECK_GUIDES
            .iter()
            .zip(all_fixes(&defaults, &compute_all(&defaults), false))
            .flat_map(|(guide, fixes)| {
                fixes
                    .into_iter()
                    .map(|(path, dir)| (guide.check, path, fix_text(path, dir)))
            })
            .collect();
        for size in [egui::vec2(1280.0, 800.0), egui::vec2(1024.0, 700.0)] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut followed = 0;
            for (check, path, text) in &fixes {
                let mut harness = Harness::on_screen(size);
                harness.select(check);
                let output = harness.frame(Vec::new());
                let link = text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
                if !screen.contains_rect(link) {
                    continue;
                }
                harness.click(link.center());
                assert_eq!(
                    harness.panel.explorer.focus().map(|f| f.path.as_str()),
                    Some(*path)
                );
                // The column scrolls there in an animation of up to half a second.
                let mut output = harness.frame(Vec::new());
                for _ in 0..40 {
                    output = harness.frame(Vec::new());
                }
                let rows: Vec<egui::Rect> = text_rects(&output, label_for(path))
                    .into_iter()
                    .filter(|r| r.left() < INPUTS_WIDTH)
                    .collect();
                let focus = focus_rects(&output);
                assert!(
                    rows.iter()
                        .any(|r| screen.contains_rect(*r)
                            && focus.iter().any(|f| f.contains_rect(*r))),
                    "{size:?} {path}: {rows:?} {focus:?}"
                );
                followed += 1;
            }
            assert!(followed > 0, "{size:?}: no fix link on screen");
        }
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
            all_fixes(&inputs, &compute_all(&inputs), false)
                .iter()
                .flatten()
                .any(|&fix| fix == (ring, Direction::Raise))
        );
        let text = fix_text(ring, Direction::Raise);
        assert_eq!(text, format!("Raise {DIAMETER}"));
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs = inputs.clone();
        harness.frame(Vec::new());
        harness.select(crate::gui::guidance::FIT_CHECK);
        let undo = harness.panel.history.undo_len();
        assert_eq!(harness.panel.centre, CentreView::Geometry);
        harness.click_text(&text);
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.explorer.focus(), None, "no row is asked for");
        // Spec 5.2: the link selects the control it shows.
        assert_eq!(harness.panel.selection, Selection::Input(DIAMETER_HINT_KEY));
        assert!(!harness.panel.explorer.open);
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &inputs);
        assert_eq!(harness.panel.history.undo_len(), undo);
    }

    #[test]
    fn a_raised_minimum_od_takes_lower_outside_diameter_off_the_space_claim_s_entry() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::guidance::{CHECK_GUIDES, Direction, all_fixes, fix_text};
        // The space claim of a 12 mm ring (the cup sets the diameter, past its claim) offers
        // "Lower Outside diameter"; with the Minimum OD at the diameter the slider cannot go
        // down, so the link goes and every other fix stays.
        let mut inputs = crate::gui::test_support::passing_design();
        inputs.coupling.inner_back_apothem_mm = 12.0;
        let ring = "coupling.inner_back_apothem_mm";
        let claim = CHECK_GUIDES
            .iter()
            .position(|guide| guide.check == SPACE_CLAIM_CHECK)
            .expect("the space claim's guide");
        let all = all_fixes(&inputs, &compute_all(&inputs), false);
        let lower = fix_text(ring, Direction::Lower);
        let others: Vec<(&str, String)> = CHECK_GUIDES
            .iter()
            .zip(&all)
            .flat_map(|(guide, fixes)| fixes.iter().map(move |&fix| (guide.check, fix)))
            .filter(|&(_, fix)| fix != (ring, Direction::Lower))
            .map(|(check, (path, dir))| (check, fix_text(path, dir)))
            .collect();
        assert_eq!(all[claim], [(ring, Direction::Lower)]);
        assert!(!others.is_empty(), "another check has a fix");
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs = inputs.clone();
        let output = harness.select(SPACE_CLAIM_CHECK);
        assert_eq!(count(&output, &lower), 1);
        assert_eq!(count(&output, crate::gui::guidance::NO_FIX), 0);
        harness.panel.minimum_diameter_mm = Some(dimensions(&inputs).unwrap().diameter_mm);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &lower), 0);
        // Its one fix gone, the entry says no listed change helps; every other check keeps
        // its fixes.
        assert_eq!(count(&output, crate::gui::guidance::NO_FIX), 1);
        for (check, other) in &others {
            let output = harness.select(check);
            assert_eq!(count(&output, other), 1, "{other}");
        }
        assert_eq!(harness.panel.inputs(), &inputs);
    }

    #[test]
    fn a_check_s_lesson_link_shows_its_note_in_the_explain_panel() {
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
        let notes: Vec<(&'static str, &'static str)> = CHECK_GUIDES
            .iter()
            .filter_map(|g| match g.learn {
                Learn::Note(id) => Some((g.check, id)),
                _ => None,
            })
            .collect();
        assert!(!notes.is_empty(), "a guide links a reviewed note");
        let mut harness = Harness::on_screen(TALL);
        for (check, id) in notes {
            let note = crate::gui::explorer::reviewed_note(id).expect("a guide's note is reviewed");
            harness.panel.explorer.close();
            // The lesson link of the check's Explain entry (the right column settles its width
            // on the second frame: click where the link rests).
            harness.select(check);
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
            assert_eq!(harness.panel.selection, Selection::Note(id));
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, note.sentences[0]), 1, "{id}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_size_tab_s_why_links_open_their_notes() {
        // Spec 3.7 (S2-8), moved by UX overhaul spec 4.3: the Why links the Size column drew are
        // its rows' entries' lessons: Outside diameter's and Magnets per ring's how the coupling
        // makes torque, Required torque's the hot minimum check's note; the torque card's
        // heading keeps its link (slice 3 spec 6.1; in the Hot-low tile's entry). Each shows its
        // note in the Explain panel (spec 5.2). The column draws no Why link.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::typeset::glyph_safe;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
        let note = |id| reviewed_note(id).expect("a Size-tab link's note is reviewed");
        harness.panel.selection = Selection::Nothing;
        let column = harness.frame(Vec::new());
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE] {
            assert_eq!(count(&column, &why_note_text(note(id))), 0, "{id}");
        }
        for (key, id) in [
            (DIAMETER_HINT_KEY, SIZE_TAB_NOTE),
            (POLES, SIZE_TAB_NOTE),
            (REQUIREMENT_INPUT, TORQUE_BAND_NOTE),
        ] {
            let output = entry(&mut harness, key);
            explain_rect(&output, FULL_LESSON);
            harness.click(explain_rect(&output, &glyph_safe(note(id).title)).center());
            assert_eq!(harness.panel.selection, Selection::Note(id), "{key}");
        }
        harness.panel.selection = tile;
        let output = harness.frame(Vec::new());
        let at = last_characters(&output, &why_note_text(note(SIZE_TAB_NOTE)));
        assert_eq!(at.len(), 1, "the card's heading's link");
        harness.click(at[0]);
        assert_eq!(harness.panel.selection, Selection::Note(SIZE_TAB_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring() {
        // An odd magnet count has no diameter range: the Outside diameter control is not drawn;
        // Magnets per ring's entry still links its note, and the torque card's heading its (the
        // card keeps its gap and count rows, slice 3 T-5). Outside diameter's entry, if it was
        // selected before the count went odd, draws no value and no card row.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explain::WHAT_A_CHANGE_DOES;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::guidance::SIZE_TAB_NOTE;
        use crate::gui::typeset::glyph_safe;
        let odd = with_poles(7);
        assert!(
            dimensions(&odd).is_none(),
            "an odd count has no diameter range"
        );
        let mut harness = size_tab(SCREEN, odd);
        let note = reviewed_note(SIZE_TAB_NOTE).unwrap();
        let output = harness.frame(Vec::new());
        assert_eq!(text_rects(&output, &why_note_text(note)).len(), 1);
        assert_eq!(count(&output, DIAMETER), 0, "no Outside diameter control");
        let output = entry(&mut harness, POLES);
        assert_eq!(count(&output, &glyph_safe(note.title)), 1);
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        assert_eq!(count(&output, WHAT_A_CHANGE_DOES), 0);
        assert_eq!(count(&output, size_controls::DIAMETER_HOVER), 1);
    }

    #[test]
    fn the_end_effect_banner_links_to_its_note_over_the_centre() {
        // Spec 3.7 (CG-10): the link under the banner opens the end-effect check's note;
        // without the banner there is no link.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::guidance::BANNER_NOTE;
        let text =
            why_note_text(reviewed_note(BANNER_NOTE).expect("the banner's note is reviewed"));
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &text), 0, "no banner, no link");
        harness.panel.inputs = short_magnets();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let links = last_characters(&output, &text);
        assert_eq!(links.len(), 1, "under the centre region's banner");
        harness.click(links[0]);
        assert_eq!(harness.panel.selection, Selection::Note(BANNER_NOTE));
        assert_eq!(harness.panel.inputs(), &short_magnets());
    }

    /// Opens `path` and clicks its closest-note link: the one copy of `note`'s title on the
    /// lead's line, left of the right column (whose entries draw note titles in their lesson
    /// links), clicked on its text (it ends a wrapped row). A second copy there would make the
    /// click ambiguous: the helper says so rather than pass. The click shows `note` in the
    /// Explain panel and edits nothing.
    fn click_the_closest_note_link(path: &str, note: &crate::engine::explain::notes::Note) {
        use crate::gui::explorer::NEAREST_NOTE;
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(TALL);
        harness.panel.explorer.open_path(path);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let lead = text_rect(&output, NEAREST_NOTE).expect("the lead");
        let beside: Vec<egui::Pos2> = last_characters(&output, &glyph_safe(note.title))
            .into_iter()
            .filter(|at| {
                at.x < TALL.x - DASHBOARD_WIDTH && (at.y - lead.center().y).abs() < lead.height()
            })
            .collect();
        assert_eq!(
            beside.len(),
            1,
            "{path}: the link beside the lead: {beside:?}"
        );
        harness.click(beside[0]);
        assert_eq!(harness.panel.selection, Selection::Note(note.id), "{path}");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
    }

    #[test]
    fn explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it() {
        // BL-059: the gearbox check has no equation and no note of its own. Its row offers
        // the note its Why link leads to, the torque band (a row's own link comes
        // first), and the link beside the lead shows it in the Explain panel. Its "no note" and lead counts
        // are the dashboard loop's (`explain_never_shows_no_teaching_note_on_a_dashboard_row`).
        use crate::engine::explain::notes::note_for;
        use crate::gui::explorer::{NoteFallback, note_fallback};
        use crate::gui::guidance::{Learn, TORQUE_BAND_NOTE, row_learn};
        assert!(note_for(GEARBOX_CHECK).is_none());
        assert_eq!(
            row_learn(GEARBOX_CHECK),
            Some(Learn::Note(TORQUE_BAND_NOTE))
        );
        let NoteFallback::Note(note) = note_fallback(GEARBOX_CHECK) else {
            panic!("{GEARBOX_CHECK} offers no note");
        };
        assert_eq!(note.id, TORQUE_BAND_NOTE);
        click_the_closest_note_link(GEARBOX_CHECK, note);
    }

    #[test]
    fn explain_never_shows_no_teaching_note_on_a_dashboard_row() {
        // Spec section 6 test 11: each dashboard row opened shows its note's lesson line, the
        // closest note, an assumption's rationale, or its own reason with where the notes are.
        use crate::engine::explain::notes::note_for;
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::explorer::{
            NEAREST_ASSUMPTION, NEAREST_NOTE, NO_NOTE, NoteFallback, START_HERE_HINT, note_fallback,
        };
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(TALL);
        for &(path, _) in &DASHBOARD {
            harness.panel.explorer.open_path(path);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, NO_NOTE), 0, "{path}");
            if note_for(path).is_some() {
                continue;
            }
            match note_fallback(path) {
                NoteFallback::Note(note) => {
                    assert_eq!(count(&output, NEAREST_NOTE), 1, "{path}");
                    assert!(count(&output, &glyph_safe(note.title)) >= 1, "{path}");
                }
                NoteFallback::Assumption(assumption) => assert_eq!(
                    count(
                        &output,
                        &format!("{NEAREST_ASSUMPTION} {}", assumption.label)
                    ),
                    1,
                    "{path}"
                ),
                NoteFallback::Reason(reason) => {
                    assert_eq!(count(&output, reason), 1, "{path}");
                    assert_eq!(count(&output, START_HERE_HINT), 1, "{path}");
                }
            }
        }
    }

    #[test]
    fn the_closest_note_link_opens_that_note() {
        // TC-29a: a dashboard row whose equation has no note and whose Why link leads to none
        // offers the closest note upstream; its link shows it in the Explain panel.
        use crate::engine::explain::notes::note_for;
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::explorer::{NoteFallback, note_fallback};
        use crate::gui::guidance::{Learn, row_learn};
        let (path, note) = DASHBOARD
            .iter()
            .find_map(|&(path, _)| match (note_for(path), note_fallback(path)) {
                (None, NoteFallback::Note(note))
                    if row_learn(path) != Some(Learn::Note(note.id)) =>
                {
                    Some((path, note))
                }
                _ => None,
            })
            .expect("a dashboard row offered its closest note upstream");
        click_the_closest_note_link(path, note);
    }

    /// Where a click lands on each copy of the reviewed note `id`'s title drawn in the right
    /// column of `output`: its last character (`last_characters`). A check's Explain entry
    /// draws its note's title as its lesson link.
    fn lesson_links(output: &egui::FullOutput, id: &str) -> Vec<egui::Pos2> {
        let note = crate::gui::explorer::reviewed_note(id)
            .unwrap_or_else(|| panic!("the {id} note is reviewed"));
        let left = right_column_left(output);
        last_characters(output, &crate::gui::typeset::glyph_safe(note.title))
            .into_iter()
            .filter(|at| at.x > left)
            .collect()
    }

    /// Clicks the note link at `at` and checks that the click selected note `id` (the Explain
    /// panel shows it) and edited no input. The harness must hold the default design.
    fn click_opens_note(harness: &mut Harness, at: egui::Pos2, id: &'static str) {
        harness.click(at);
        assert_eq!(harness.panel.selection, Selection::Note(id));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{id}");
    }

    #[test]
    fn the_gearbox_check_s_lesson_link_opens_the_torque_band_note() {
        // BL-059: the gearbox check's lesson link leads to the torque-band note, as the hot
        // minimum's does. Its Explain entry draws the note's title once; the click edits
        // nothing.
        use crate::gui::guidance::TORQUE_BAND_NOTE;
        let mut harness = Harness::on_screen(TALL);
        let output = harness.select(GEARBOX_CHECK);
        let links = lesson_links(&output, TORQUE_BAND_NOTE);
        assert_eq!(links.len(), 1, "one link in the gearbox check's entry");
        click_opens_note(&mut harness, links[0], TORQUE_BAND_NOTE);
    }

    #[test]
    fn the_shaft_joint_check_s_lesson_link_opens_the_shaft_joint_note() {
        // BL-064: the shaft joints check's lesson link leads to the shaft-joint note. Its
        // Explain entry draws the note's title once; the click edits nothing.
        const SHAFT_JOINT_CHECK: &str = "shaft_joint.check";
        let mut harness = Harness::on_screen(TALL);
        let output = harness.select(SHAFT_JOINT_CHECK);
        let links = lesson_links(&output, "shaft_joint");
        assert_eq!(links.len(), 1, "one link to the note");
        click_opens_note(&mut harness, links[0], "shaft_joint");
    }

    #[test]
    fn the_failing_filter_lists_the_gearbox_check_with_its_red_badge() {
        use crate::gui::dashboard::failing_checks;
        use crate::gui::results_table::{FAILING_ONLY, entry_index, table_entries};
        // BL-059: at the defaults the gearbox check fails beside the cup ring, the hot minimum,
        // the clearance and the cup wall, all red. The filter lists the five, then the amber
        // daily-cycle screen at the entered bondline (BL-063), and the gearbox check's row once
        // in the table, with a red badge.
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results(), false);
        assert_eq!(failing.len(), 6, "{failing:?}");
        assert!(
            failing[..5].iter().all(|&(_, level)| level == Level::Bad),
            "{failing:?}"
        );
        assert!(
            failing.contains(&(GEARBOX_CHECK, Level::Bad)),
            "{failing:?}"
        );
        assert_eq!(
            failing[5],
            (
                "temperature.adhesive_life.daily_entered_screen",
                Level::Caution
            )
        );
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("6 of {total} results")), 1);
        let in_table = |x: f32| x > INPUTS_WIDTH && x < SCREEN.x - DASHBOARD_WIDTH;
        // The row is found by the label the table draws: its entry's (`row_ui`), which an
        // overlay name for the dashboard row would not change.
        let entry = &table_entries()[entry_index(GEARBOX_CHECK).expect("the gearbox row")];
        let rows: Vec<egui::Rect> = text_rects(&output, entry.info.meta.label)
            .into_iter()
            .filter(|r| in_table(r.center().x))
            .collect();
        assert_eq!(rows.len(), 1, "the gearbox check's row in the table");
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(
            table_badges_on(&output, &visuals, rows[0], in_table),
            [Level::Bad]
        );
    }

    #[test]
    fn the_status_line_counts_the_checks_at_default_and_after_a_fix() {
        use crate::gui::guidance::{design_status, design_status_text};
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
        // The workbook example fails its hot minimum, its clearance, its cup wall and its
        // gearbox check, of the ten guided checks (the shaft joints pass).
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (4, 0, 10, true)
        );
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // A fix: the cup wall at 2.4 mm (its rule asks for 2.0 mm and up).
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
        assert_eq!((status.bad, status.is_default), (3, false));
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // Reset all: the example again.
        harness.menu(RESET_ALL);
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
        assert_eq!((status.bad, status.is_default), (4, true));
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
    fn a_drag_computes_the_card_once_for_each_design_it_draws() {
        // Slice 3 spec 3.5 and decision 13A: on a drag frame the card asks for the design after
        // the frame's edit (one miss) and the next frame's controls ask for that same design (a
        // hit), so a drag computes the consequences, and the design's results they are handed
        // (T5 M-6), once per design drawn, never twice; at the smoke design with Auto on the
        // magnet length, the drag the browser frame check times.
        let mut harness = size_tab(SCREEN, smoke_inputs());
        harness.panel.clamps_paused = true;
        harness.panel.sizing.variable = FreeVariable::AxialLength;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let gap = harness.size_widget(GAP).rect;
        let mut at = gap.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        let results = harness.panel.size_tab_results.computed;
        let mut designs = 0;
        for i in 0..20 {
            let before = harness.panel.shown_inputs();
            at += egui::vec2(if i % 2 == 0 { 3.0 } else { -3.0 }, 0.0);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            designs += usize::from(harness.panel.shown_inputs() != before);
        }
        harness.frame(vec![primary_button(at, false)]);
        harness.frame(Vec::new());
        assert!(designs >= 10, "the drag moved the gap on {designs} frames");
        assert_eq!(harness.panel.consequences.computed - computed, designs);
        assert_eq!(harness.panel.size_tab_results.computed - results, designs);
    }

    #[test]
    fn idle_frames_compute_the_card_once_and_an_edit_again() {
        // Slice 3 test 10: the torque card lives in the ConsequenceCache, keyed on the design
        // shown, the Minimum OD and the clamp pause (spec 3.2), with the Auto marking applied
        // outside it (spec 3.4). Idle frames compute nothing; a gap edit computes once (the frame
        // that draws it asks with the edited design, the next frames hit); a toggle of
        // the clamp pause computes once; so does a new Minimum OD, which the drawn Outside
        // diameter row follows. The design's results the cache is handed are computed once per
        // design (T5 M-6): none on idle frames, one for the gap edit, none for the pause or the
        // Minimum OD, which leave the design as it is.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        let results = harness.panel.size_tab_results.computed;
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.consequences.computed, computed, "idle frames");
        assert_eq!(
            harness.panel.size_tab_results.computed, results,
            "idle frames' results"
        );
        let gap = harness.panel.design_widgets[2];
        harness.ctx.memory_mut(|m| m.request_focus(gap));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert!(
            harness.panel.inputs().metal.face_gap_mm > DesignInputs::default().metal.face_gap_mm
        );
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 1,
            "one gap edit"
        );
        assert_eq!(
            harness.panel.size_tab_results.computed,
            results + 1,
            "one gap edit's results"
        );
        harness.panel.clamps_paused = false;
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 2,
            "the pause toggled"
        );
        assert_eq!(
            harness.panel.size_tab_results.computed,
            results + 1,
            "the pause toggled: the same design's results"
        );
        // The key's third part, the Minimum OD (owner 9A: the diameter row steps the slider at
        // the user's Minimum OD): set at the diameter shown, it computes once, and the drawn
        // Outside diameter row reads END_OF_RANGE on its -1 mm side, where with none set it read
        // that the magnets do not fit. Outside diameter's entry draws the row (UX overhaul spec
        // 4.3); the gap edit above selected the gap's.
        use crate::gui::torque_card::END_OF_RANGE;
        harness.panel.selection = Selection::Input(DIAMETER_HINT_KEY);
        let diameter_row = |harness: &Harness| {
            row_line(
                &shown_consequences(harness).card,
                Control::OutsideDiameter,
                None,
            )
        };
        let unset = diameter_row(&harness);
        assert!(
            unset.contains(&format!("-1 mm -> {SIDE_DOES_NOT_FIT};")),
            "{unset}"
        );
        let shown = dimensions(&harness.panel.shown_inputs()).expect("size controls");
        harness.panel.minimum_diameter_mm = Some(shown.diameter_mm);
        let mut output = harness.frame(Vec::new());
        for _ in 0..3 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 3,
            "the Minimum OD set"
        );
        assert_eq!(
            harness.panel.size_tab_results.computed,
            results + 1,
            "the Minimum OD set: the same design's results"
        );
        let at_shown = diameter_row(&harness);
        assert!(
            at_shown.contains(&format!("-1 mm -> {END_OF_RANGE};")),
            "{at_shown}"
        );
        assert_eq!(count(&output, &at_shown), 1, "{at_shown}");
        assert_eq!(count(&output, &unset), 0, "stale: {unset}");
        // (i) The smoke design, Auto on the magnet length, solved: turning Auto off adopts the
        // solved length (decision M41-7), so the design shown, and the card, stay; nothing is
        // computed, and the Overall length row, held while Auto was on, reads its numbers.
        let mut harness = size_tab(SCREEN, smoke_inputs());
        harness.panel.clamps_paused = true;
        harness.panel.sizing.variable = FreeVariable::AxialLength;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let held = shown_consequences(&harness).card;
        let held_row = held_line(Control::OverallLength);
        assert_eq!(
            count(&output, &held_row),
            1,
            "the held row while Auto is on"
        );
        let computed = harness.panel.consequences.computed;
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            harness.panel.consequences.computed, computed,
            "Auto turned off"
        );
        assert_eq!(shown_consequences(&harness).card, held);
        assert_eq!(count(&output, &held_row), 0);
        let length = row_line(&held, Control::OverallLength, None);
        assert_eq!(count(&output, &length), 1, "{length}");
        // (ii) The defaults, Auto on the magnet count: the best count is the one shown (10), so
        // turning Auto on, letting it solve and turning it off computes nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        harness.panel.sizing.variable = FreeVariable::MagnetsPerRing;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.shown_inputs(), DesignInputs::default());
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.consequences.computed, computed,
            "Auto on the count"
        );
    }

    #[test]
    fn a_narrow_window_wraps_each_check_s_entry_inside_the_right_column() {
        // The M4-1 review's ~930 px window, tall enough for every row: every guide line of the
        // workbook example's red checks and the status line wrap inside the dashboard, never past
        // its right margin (a line past it would widen the dashboard into the centre region).
        use crate::gui::guidance::{
            CHECK_GUIDES, all_fixes, design_status, design_status_text, fix_text, level,
        };
        let width = 930.0;
        let mut harness = Harness::on_screen(egui::vec2(width, 3000.0));
        let inputs = DesignInputs::default();
        let results = harness.panel.results().clone();
        let fixes = all_fixes(&inputs, &results, false);
        let status = design_status_text(&design_status(&inputs, &results, false));
        let mut entries = 0;
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
            if level(guide, &inputs, &results) != Level::Bad {
                continue;
            }
            entries += 1;
            // The right column settles its width on the second frame.
            harness.select(guide.check);
            let output = harness.frame(Vec::new());
            let mut texts = vec![status.clone(), guide.why.to_owned()];
            texts.extend((guide.problem)(&inputs, &results));
            texts.extend(fixes.iter().map(|&(path, dir)| fix_text(path, dir)));
            // Right, the side panel's margin inside the frame's (8 points each), give or take
            // egui's rounding of a widget's edges (a line that wraps to the full width can end
            // one rounding step past it).
            let column = right_column_left(&output)..=width - 16.0 + egui::emath::GUI_ROUNDING;
            for text in &texts {
                let rects = text_rects(&output, text);
                assert!(!rects.is_empty(), "{text:?} is drawn");
                for rect in rects {
                    assert!(
                        column.contains(&rect.left()) && column.contains(&rect.right()),
                        "{text:?} at {rect:?}, the right column {column:?}"
                    );
                }
            }
        }
        assert_eq!(entries, 4, "the example design's four red checks");
    }
}
