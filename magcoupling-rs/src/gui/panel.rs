//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: the results table; plan M4-2 adds the
//! geometry view and the plots). Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.
//!
//! The panel never touches files or the network: what needs the platform (saving a file,
//! picking one) it queues as a [`PanelRequest`] for the host, which drains them with
//! [`MagcouplingPanel::take_requests`] after each frame.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
use crate::gui::session::Design;
use crate::gui::sizing::SizingState;
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";

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
}

/// The views of the centre region. Plan M4-2 adds the geometry view and the plot tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreView {
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 1] = [CentreView::Results];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
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
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
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
            results,
            key_widgets: Vec::new(),
            last_error: None,
            centre: CentreView::Results,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
        }
    }

    /// The platform work queued since the last call (the host saves files), oldest first.
    pub fn take_requests(&mut self) -> Vec<PanelRequest> {
        std::mem::take(&mut self.requests)
    }

    /// The design as a file holds it.
    fn design(&self) -> Design {
        Design {
            inputs: self.inputs.clone(),
            sizing: SizingState::default(),
        }
    }

    /// The design inputs.
    pub fn inputs(&self) -> &DesignInputs {
        &self.inputs
    }

    /// The results of [`MagcouplingPanel::inputs`], as of the last frame or reset.
    pub fn results(&self) -> &DesignResults {
        &self.results
    }

    /// Back to the default design.
    pub fn reset(&mut self) {
        self.inputs = DesignInputs::default();
        self.results = compute_all(&self.inputs);
        self.last_error = None;
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            // After the inputs: the readouts show this frame's edits.
            self.results = compute_all(&self.inputs);
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui));
        });
    }

    /// The centre region: the view tabs, then the view.
    fn centre_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
            }
        });
        ui.separator();
        match self.centre {
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
                        contents: results_json(&self.design(), &self.results),
                    }),
                    None => {}
                }
            }
        }
    }

    /// The header line: heading, the session buttons, the last refusal.
    fn header_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(HEADING);
            ui.separator();
            if ui.button(RESET_ALL).clicked() {
                self.reset();
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
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
                                    self.input_row_ui(ui, entry);
                                }
                            }
                        });
                }
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
        let current = self.inputs.get(&entry.path).unwrap_or(Value::None);
        let seed = self.seed(entry);
        let output = input_row(ui, entry, &current, seed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_tap, primary_button, select_all, short_magnets, sized_frame,
        text_rect,
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
        harness.panel.inputs.metal.face_gap_mm = 1.4123;
        harness.panel.inputs.metal.measured_drag_Nm = Some(0.012345);
        let design = harness.panel.inputs.clone();
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
        assert_eq!(harness.panel.inputs(), &design);
        assert_eq!(harness.panel.last_error, None);
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
        // Over the dashboard and over the results table.
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
        let output = harness.frame(Vec::new());
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
