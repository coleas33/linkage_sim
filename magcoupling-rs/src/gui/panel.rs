//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! The M4 infrastructure tracer: a few key-input sliders generated from the
//! engine's input metadata and the headline numbers, recomputed every frame
//! with every approved correction on ([`compute_all`]). The M4 plans grow it
//! into the spec's layout (inputs, geometry view, dashboard, plots).

use crate::engine::api::HEADLINE;
use crate::engine::meta::{
    FieldType, InputMeta, InputSet, ResultMeta, Value, input_rows, result_rows,
};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults, compute_all, headline};

/// Input paths of the tracer's sliders: face gap, pole count and axial length,
/// the first entries of the spec's Key design group.
///
/// The axial length is the manual inner magnet length, which the engine uses
/// only when the inner part is not a library part (its help says so); at the
/// default part (B842SH) it moves no result.
pub const KEY_INPUTS: [&str; 3] = [
    "metal.face_gap_mm",
    "coupling.npole",
    "coupling.magnets.manual_inner_length_mm",
];

/// The calculator panel: design inputs, their results, and the UI that edits
/// the one and shows the other.
///
/// Hostable by any egui app: the standalone app shows it as a full page
/// (`app::MagcouplingApp`), the linkage app in an `egui::Window` (M5). Call
/// [`MagcouplingPanel::ui`] once per frame.
pub struct MagcouplingPanel {
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// Metadata of each [`KEY_INPUTS`] path, in order.
    key_inputs: [(&'static str, &'static InputMeta); KEY_INPUTS.len()],
    /// Metadata of each [`HEADLINE`] result, in order.
    headline_meta: [&'static ResultMeta; HEADLINE.len()],
    /// The id of each key input's slider in the last frame (the slider rail).
    slider_ids: [Option<egui::Id>; KEY_INPUTS.len()],
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
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
        let input_meta = input_rows(&inputs);
        let result_meta = result_rows(&results);
        let key_inputs = KEY_INPUTS.map(|path| {
            let row = input_meta.iter().find(|row| row.path == path);
            (
                path,
                row.unwrap_or_else(|| panic!("KEY_INPUTS: no input {path}"))
                    .meta,
            )
        });
        let headline_meta = HEADLINE.map(|(key, path)| {
            let row = result_meta.iter().find(|row| row.path == path);
            row.unwrap_or_else(|| panic!("HEADLINE {key}: no result {path}"))
                .meta
        });
        Self {
            inputs,
            results,
            key_inputs,
            headline_meta,
            slider_ids: [None; KEY_INPUTS.len()],
            last_error: None,
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
    ///
    /// Recomputes every result after the inputs are drawn, so the readout
    /// shows this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            ui.heading("Magnetic coupling calculator");
            for (index, (path, meta)) in self.key_inputs.into_iter().enumerate() {
                self.slider_ids[index] = self.slider(ui, path, meta);
            }
            if let Some(error) = &self.last_error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            if ui.button("Reset all").clicked() {
                self.reset();
            }
            self.results = compute_all(&self.inputs);
            ui.separator();
            self.headline_ui(ui);
        });
    }

    /// One key input's slider, set up from its metadata; returns its id, or
    /// `None` when the input cannot have a slider.
    fn slider(
        &mut self,
        ui: &mut egui::Ui,
        path: &'static str,
        meta: &'static InputMeta,
    ) -> Option<egui::Id> {
        let (range, current) = match (meta.range, self.inputs.get(path)) {
            (Some(range), Some(Value::Num(x))) => (range, x),
            (Some(range), Some(Value::Int(i))) => (range, i as f64),
            (range, value) => {
                ui.label(format!(
                    "{}: no slider (range {range:?}, value {value:?})",
                    meta.label
                ));
                return None;
            }
        };
        let mut value = current;
        let mut slider = egui::Slider::new(&mut value, range.min..=range.max)
            .text(meta.label)
            .logarithmic(range.log)
            // Edits: the slider, arrow keys and typed values stay in the range,
            // and a value already outside it is kept until edited (never
            // rewritten by an idle frame).
            .clamping(egui::SliderClamping::Edits);
        if meta.ty == FieldType::I64 {
            slider = slider.integer();
        }
        // After integer(), which sets a step of 1: pole counts step by 2.
        slider = slider.step_by(range.step);
        if meta.unit != "-" {
            slider = slider.suffix(format!(" {}", meta.unit));
        }
        let response = ui.add(slider).on_hover_text(input_tooltip(path, meta));
        if response.changed() && value != current {
            let new = match meta.ty {
                FieldType::I64 => Value::Int(value.round() as i64),
                _ => Value::Num(value),
            };
            self.last_error = self.inputs.set(path, new).err().map(|e| e.to_string());
        }
        Some(response.id)
    }

    /// The headline numbers: label, then value with unit; the hover shows the
    /// Python key, the result path and the workbook cell.
    fn headline_ui(&self, ui: &mut egui::Ui) {
        egui::Grid::new("magcoupling_headline")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let rows = headline(&self.results)
                    .into_iter()
                    .zip(HEADLINE)
                    .zip(self.headline_meta);
                for (((key, value), (_, path)), meta) in rows {
                    ui.label(meta.label).on_hover_text(format!(
                        "{key}\n{path}\n{}",
                        meta.cell.unwrap_or("no workbook cell")
                    ));
                    ui.label(with_unit(format_value(&value), meta.unit));
                    ui.end_row();
                }
            });
    }
}

/// Hover text of an input: its help, path and workbook cell.
fn input_tooltip(path: &str, meta: &InputMeta) -> String {
    let mut text = String::new();
    if !meta.help.is_empty() {
        text.push_str(meta.help);
        text.push('\n');
    }
    text.push_str(path);
    text.push('\n');
    text.push_str(meta.cell.unwrap_or("no workbook cell"));
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_press, primary_button, text_rect,
    };

    const FACE_GAP: usize = 0;
    const POLES: usize = 1;
    const AXIAL_LENGTH: usize = 2;

    /// A panel and the egui context it is drawn in, frame by frame.
    struct Harness {
        ctx: egui::Context,
        panel: MagcouplingPanel,
    }

    impl Harness {
        fn new() -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
            };
            harness.frame(Vec::new());
            harness
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            central_panel_frame(&self.ctx, events, |ui| panel.ui(ui))
        }

        /// The key input's slider (its rail) as drawn in the last frame.
        fn slider(&self, key: usize) -> egui::Response {
            let id = self.panel.slider_ids[key].expect("the slider was drawn");
            self.ctx
                .read_response(id)
                .expect("the slider has a response")
        }

        /// Gives the key input's slider keyboard focus.
        fn focus(&mut self, key: usize) {
            let id = self.slider(key).id;
            self.ctx.memory_mut(|m| m.request_focus(id));
            self.frame(Vec::new());
            assert!(
                self.ctx.memory(|m| m.has_focus(id)),
                "slider {key} has focus"
            );
        }

        /// A click (move, press, release) at `at`.
        fn click(&mut self, at: egui::Pos2) -> egui::FullOutput {
            self.frame(vec![egui::Event::PointerMoved(at)]);
            self.frame(vec![primary_button(at, true)]);
            self.frame(vec![primary_button(at, false)])
        }

        fn number(&self, key: usize) -> f64 {
            match self.panel.inputs.get(KEY_INPUTS[key]) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{}: {other:?}", KEY_INPUTS[key]),
            }
        }
    }

    /// The headline of `inputs`, as the panel displays it (value with unit).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
        let panel = MagcouplingPanel::new();
        headline(&compute_all(inputs))
            .into_iter()
            .zip(panel.headline_meta)
            .map(|((_, value), meta)| with_unit(format_value(&value), meta.unit))
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

    #[test]
    fn key_inputs_are_numeric_inputs_with_slider_ranges() {
        let panel = MagcouplingPanel::new();
        for (path, meta) in panel.key_inputs {
            assert!(
                matches!(meta.ty, FieldType::F64 | FieldType::I64),
                "{path}: {:?}",
                meta.ty
            );
            let range = meta.range.unwrap_or_else(|| panic!("{path}: no range"));
            assert!(
                range.min < range.max && range.step > 0.0,
                "{path}: {range:?}"
            );
            assert!(meta.choices.is_empty(), "{path} is a selector");
        }
        assert_eq!(panel.key_inputs.map(|(path, _)| path), KEY_INPUTS);
    }

    #[test]
    fn a_new_panel_holds_the_default_design_and_its_corrected_results() {
        let panel = MagcouplingPanel::new();
        assert_eq!(panel.inputs(), &DesignInputs::default());
        assert_eq!(panel.results(), &compute_all(&DesignInputs::default()));
        assert_eq!(MagcouplingPanel::default().inputs(), panel.inputs());
    }

    #[test]
    fn every_headline_number_is_drawn_with_its_label() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        for meta in harness.panel.headline_meta {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing label {:?}",
                meta.label
            );
        }
        for (_, meta) in harness.panel.key_inputs {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing slider {:?}",
                meta.label
            );
        }
        // The corrected headline (E2: M4 x 14), not the workbook's (M4 x 12).
        assert!(texts.iter().any(|t| t.contains("M4 x 14")), "{texts:?}");
    }

    #[test]
    fn idle_frames_change_no_input() {
        let mut harness = Harness::new();
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn an_input_outside_its_slider_range_is_kept_until_edited() {
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 7.5; // range 0.3 to 5.0
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 7.5);
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 7.5;
        assert_eq!(harness.panel.results(), &compute_all(&inputs));
    }

    #[test]
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        let output = harness.frame(vec![key_press(egui::Key::ArrowRight)]);

        // One step (0.01 mm) up from 1.4 mm, snapped from the range start.
        assert!(
            (harness.number(FACE_GAP) - 1.41).abs() < 1e-12,
            "{}",
            harness.number(FACE_GAP)
        );
        let mut expected = DesignInputs::default();
        expected.metal.face_gap_mm = harness.panel.inputs.metal.face_gap_mm;
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
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::new();
        let rail = harness.slider(FACE_GAP).rect;
        let output = harness.click(rail.right_center() - egui::vec2(1.0, 0.0));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_drew_headline(&output, harness.panel.inputs());
        assert_ne!(
            displayed_pullout(harness.panel.inputs()),
            displayed_pullout(&DesignInputs::default())
        );
    }

    #[test]
    fn the_pole_count_steps_by_two_and_stays_even() {
        let mut harness = Harness::new();
        harness.focus(POLES);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 12);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 8);
        let rail = harness.slider(POLES).rect;
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
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 40);
        harness.focus(FACE_GAP);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 0.3);
    }

    #[test]
    fn the_axial_length_slider_edits_the_manual_inner_length() {
        let mut harness = Harness::new();
        harness.focus(AXIAL_LENGTH);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        let length = harness.panel.inputs.coupling.magnets.manual_inner_length_mm;
        assert!((length - 12.71).abs() < 1e-12, "{length}");
        // A library part (the default) ignores the manual length.
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        harness.focus(POLES);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_ne!(harness.panel.inputs(), &DesignInputs::default());

        let output = harness.frame(Vec::new());
        let button = text_rect(&output, "Reset all").expect("the reset button is drawn");
        harness.click(button.center());
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
                egui::Window::new("Magnetic coupling").show(ctx, |ui| panel.ui(ui));
            }));
        }
        assert_drew_headline(&output.expect("two frames ran"), &DesignInputs::default());
    }

    #[test]
    fn input_tooltips_carry_help_path_and_cell() {
        let panel = MagcouplingPanel::new();
        let (path, meta) = panel.key_inputs[FACE_GAP];
        assert_eq!(
            input_tooltip(path, meta),
            "Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119"
        );
        let (path, meta) = panel.key_inputs[AXIAL_LENGTH];
        assert!(
            input_tooltip(path, meta).starts_with("Used only if the part is not in the library.")
        );
    }
}
