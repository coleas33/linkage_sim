//! The readout hook (spec Addendum A2 "Hover": "Any displayed value (dashboard, results
//! table, geometry callouts) shows a tooltip with its equation. Each term is coloured, and the
//! same colour marks that term wherever its value appears on screen").
//!
//! Every view hands each value it displays to [`Readouts::show`], with its result path and
//! its hover text, built with the frame's [`Readouts::show_details`] (the header's Show
//! spreadsheet details, spec S2-5): the dashboard rows, the results-table rows, the geometry
//! callouts, the clamp summary and screw table, the plot readouts. While the value is hovered
//! the tooltip
//! shows that text, then the value's equation typeset with its terms in colour
//! ([`TermColors::of`]), no wider than the window; a click asks the panel to open it in the
//! Equation panel, or, for a dimension tag, a callout row and a results-table row
//! ([`Readouts::select`]), to select its result for the Explain panel. Both the
//! text and the typesetting run only inside `on_hover_ui`, so a frame pays for the one value
//! hovered. A value whose path is a term of the equation in view (the one hovered in the last
//! frame, else the one open in the Equation panel) gets a frame in that term's colour, and so
//! do the input rows of the equation's leaf terms ([`Readouts::mark`]).
//!
//! The equation registry is built once per process ([`registry`], decision M43-5): the panel
//! builds it at start-up (about 3 ms in a release build), and every frame only looks paths up.

use std::sync::OnceLock;

use egui::{Rect, Stroke, StrokeKind};

use crate::engine::explain::Registry;
use crate::gui::explain::{Selection, result_selection};
use crate::gui::typeset::{TermColors, equation_ui};

/// The hint under a tooltip's equation.
pub const OPEN_HINT: &str = "Click to open it in the Equation panel";

/// The text size of a tooltip's equation [points]; an equation wider than the window at this
/// size is set smaller (`typeset::equation_ui`).
pub const TOOLTIP_SIZE: f32 = 15.0;

/// The width of a term's mark [points].
pub const MARK_WIDTH: f32 = 2.0;

/// The start of the log line written when the registry is built (the web smoke looks for it).
pub const REGISTRY_LOG_PREFIX: &str = "magcoupling explorer: ";

/// The equation registry, built on first use and kept for the life of the process: every
/// record parsed and checked once (`Registry::build` panics only on a broken record, which
/// `tests/explain.rs` rules out).
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry = Registry::build();
        log::info!(
            "{REGISTRY_LOG_PREFIX}{} equations",
            registry.equations().len()
        );
        registry
    })
}

/// What the user did with the readouts in one frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadoutEvents {
    /// The path of the value under the pointer.
    pub hovered: Option<String>,
    /// The path of the value clicked, which opens in the Equation panel.
    pub clicked: Option<String>,
    /// The path of the result a click selected for the Explain panel (a dimension tag, a
    /// callout row, a results-table row: UX overhaul spec 5.2).
    pub selected: Option<&'static str>,
    /// The teaching note a warning's or a check's Why link asked for, by id.
    pub note: Option<&'static str>,
    /// The input whose row a check guide's link asked to show (a fix, an assumption's Why
    /// link), by path.
    pub focus: Option<&'static str>,
}

/// One frame's readouts: the term colours they mark, whether their hover texts carry the
/// spreadsheet details, and what the user did.
#[derive(Clone, Debug, Default)]
pub struct Readouts {
    marks: TermColors,
    events: ReadoutEvents,
    /// The header's Show spreadsheet details as the frame began ([`crate::gui::SHOW_DETAILS`]).
    show_details: bool,
    /// The Explain panel's selection as the frame began ([`Readouts::with_selection`]).
    selection: Selection,
}

impl Readouts {
    /// The readouts of a frame that marks the terms in `marks`, its hover texts with the
    /// spreadsheet details when `show_details`.
    pub fn new(marks: TermColors, show_details: bool) -> Self {
        Self {
            marks,
            events: ReadoutEvents::default(),
            show_details,
            selection: Selection::Nothing,
        }
    }

    /// These readouts with `selection`, the Explain panel's as the frame began: the selecting
    /// value ([`Readouts::select`]) whose click selects it is outlined.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Whether this frame shows the spreadsheet details (spec S2-5): the internal names, the
    /// original spreadsheet's cells and the correction ids the header's toggle gates.
    pub fn show_details(&self) -> bool {
        self.show_details
    }

    /// The colours this frame marks.
    pub fn marks(&self) -> &TermColors {
        &self.marks
    }

    /// Frames `rect` in the colour of the term at `path`, if the equation in view shows it.
    pub fn mark(&self, ui: &egui::Ui, rect: Rect, path: &str) {
        if let Some(color) = self.marks.get(path) {
            ui.painter().rect_stroke(
                rect.expand(1.0),
                3.0,
                Stroke::new(MARK_WIDTH, color),
                StrokeKind::Outside,
            );
        }
    }

    /// Shows the value of the result at `path` that `response` displays: its mark, and while
    /// hovered `text(show_details)` and its equation; a click asks for it in the Equation
    /// panel. `text` is called only while the value is hovered, with
    /// [`Readouts::show_details`].
    pub fn show(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce(bool) -> String,
    ) {
        if response.clicked() {
            self.events.clicked = Some(path.to_owned());
        }
        self.tip(ui, response, path, text, true);
    }

    /// [`Readouts::show`] without the click, for a value whose click is its owner's (a check's
    /// row and a tile select it, UX overhaul spec 5.2): its mark, and while hovered
    /// `text(show_details)` and its equation.
    pub fn hover(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce(bool) -> String,
    ) {
        self.tip(ui, response, path, text, false);
    }

    /// [`Readouts::show`] for a result a click selects (a dimension tag, a callout row, a
    /// results-table row: UX overhaul spec 5.2) rather than opens in the Equation panel (spec
    /// 5.5): its mark, and while hovered `text(show_details)` and its equation; a click, or
    /// Enter or Space on it focused, asks for it in the Explain panel. The keyboard's focus
    /// arriving asks for nothing. Outlined while focused or selected ([`Readouts::with_selection`]).
    pub fn select(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &'static str,
        text: impl FnOnce(bool) -> String,
    ) {
        if response.clicked() {
            self.events.selected = Some(path);
        }
        // Selected or keyboard-focused: outlined as a selected tile or check row is.
        if response.has_focus() || self.selection == result_selection(path) {
            ui.painter().rect_stroke(
                response.rect,
                2.0,
                Stroke::new(1.0, ui.visuals().strong_text_color()),
                StrokeKind::Inside,
            );
        }
        self.tip(ui, response, path, text, false);
    }

    /// [`Readouts::select`] for a result drawn in `rect` by widgets that take no clicks: a
    /// click sensor over the rect, keyed by `path` in `ui`.
    pub fn select_over(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        path: &'static str,
        text: impl FnOnce(bool) -> String,
    ) {
        self.select(ui, sensor(ui, rect, path), path, text);
    }

    /// The mark of the value at `path` and, while `response` is hovered, its tooltip:
    /// `text(show_details)`, its equation and, with `open_hint`, [`OPEN_HINT`].
    fn tip(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce(bool) -> String,
        open_hint: bool,
    ) {
        self.mark(ui, response.rect, path);
        if response.hovered() {
            self.events.hovered = Some(path.to_owned());
        }
        let show_details = self.show_details;
        response.on_hover_ui(|ui| {
            ui.label(text(show_details));
            if let Some(eq) = registry().equation_for(path) {
                ui.separator();
                // The tooltip stays inside the window: a long equation is set smaller, and
                // the text over it wraps at the tooltip's width.
                let frame = egui::Frame::popup(ui.style()).total_margin().sum().x;
                let room = ui.ctx().screen_rect().width() - frame;
                equation_ui(ui, registry(), eq, &TermColors::of(eq), TOOLTIP_SIZE, room);
            }
            if open_hint {
                ui.weak(OPEN_HINT);
            }
        });
    }

    /// [`Readouts::show`] for a value drawn in `rect` by widgets that take no clicks (labels
    /// keep their look): a click sensor over the rect, keyed by `path` in `ui`.
    pub fn show_over(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        path: &str,
        text: impl FnOnce(bool) -> String,
    ) {
        self.show(ui, sensor(ui, rect, path), path, text);
    }

    /// Asks for the teaching note `id` (a warning's link, a lesson link), which the panel shows
    /// in the Explain panel.
    pub fn open_note(&mut self, id: &'static str) {
        self.events.note = Some(id);
    }

    /// Asks for the row of the input at `path` (a check guide's fix, an assumption's Why
    /// link).
    pub fn focus_input(&mut self, path: &'static str) {
        self.events.focus = Some(path);
    }

    /// What the user did this frame.
    pub fn events(&self) -> &ReadoutEvents {
        &self.events
    }

    /// Ends the frame: what the user did.
    pub fn finish(self) -> ReadoutEvents {
        self.events
    }
}

/// The click sensor over `rect` of the readout at `path`, keyed by `path` in `ui`.
fn sensor(ui: &egui::Ui, rect: Rect, path: &str) -> egui::Response {
    ui.interact(rect, ui.id().with(("readout", path)), egui::Sense::click())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{
        drawn_texts, flat_shapes, hover_frames, sized_frame, sized_frame_at, text_center,
        tooltips_at_once,
    };

    /// A label of `text` showing the result at `path`, drawn with `readouts`.
    fn draw(ui: &mut egui::Ui, readouts: &mut Readouts, path: &str, text: &str) -> Rect {
        let rect = ui.label(text).rect;
        readouts.show_over(ui, rect, path, |_| format!("hover text of {path}"));
        rect
    }

    #[test]
    fn the_registry_is_built_once() {
        assert!(std::ptr::eq(registry(), registry()));
        assert!(registry().equation_for("model.pullout_Nm").is_some());
    }

    #[test]
    fn a_hovered_value_shows_its_text_and_equation_and_a_click_asks_for_it() {
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let size = egui::vec2(800.0, 600.0);
        let mut rect = Rect::NOTHING;
        sized_frame(&ctx, size, Vec::new(), |ui| {
            rect = draw(
                ui,
                &mut Readouts::default(),
                "model.pullout_Nm",
                "2.688 N·m",
            );
        });
        let at = rect.center();
        let mut events = ReadoutEvents::default();
        let mut output = None;
        for (time, input) in [
            (0.1, vec![egui::Event::PointerMoved(at)]),
            (0.2, Vec::new()),
        ] {
            output = Some(sized_frame_at(&ctx, size, Some(time), input, |ui| {
                let mut readouts = Readouts::default();
                draw(ui, &mut readouts, "model.pullout_Nm", "2.688 N·m");
                events = readouts.finish();
            }));
        }
        assert_eq!(events.hovered.as_deref(), Some("model.pullout_Nm"));
        assert_eq!(events.clicked, None);
        let texts = drawn_texts(&output.unwrap());
        assert!(texts.contains(&"hover text of model.pullout_Nm".to_owned()));
        // The equation T_pull = T_2D f_end f_cal, typeset run by run.
        for run in ["T", "pull", " = ", "2D", "end", "cal"] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
        assert!(texts.contains(&OPEN_HINT.to_owned()));
        // A click.
        for pressed in [true, false] {
            sized_frame(
                &ctx,
                size,
                vec![crate::gui::test_support::primary_button(at, pressed)],
                |ui| {
                    let mut readouts = Readouts::default();
                    draw(ui, &mut readouts, "model.pullout_Nm", "2.688 N·m");
                    events = readouts.finish();
                },
            );
        }
        assert_eq!(events.clicked.as_deref(), Some("model.pullout_Nm"));
    }

    #[test]
    fn an_equation_wider_than_the_window_is_set_smaller_and_another_keeps_its_size() {
        // The Verdict row's equation is the widest (1279 points at the tooltip's size in egui's
        // built-in font): its tooltip ran off a window narrower than that. The pull-out's fits.
        use crate::gui::test_support::{cut_texts, text_shapes};
        let size = egui::vec2(1024.0, 700.0);
        let screen = Rect::from_min_size(egui::Pos2::ZERO, size);
        for (path, fits) in [
            ("temperature.summary.verdict", false),
            ("model.pullout_Nm", true),
        ] {
            let (_, output) = hover_frames(
                &egui::Context::default(),
                size,
                |output| text_center(output, "a value"),
                |ui| {
                    draw(ui, &mut Readouts::default(), path, "a value");
                },
            );
            assert!(drawn_texts(&output).contains(&OPEN_HINT.to_owned()));
            assert_eq!(cut_texts(&output, screen), Vec::<String>::new(), "{path}");
            // The equation's own text: the sign between its symbol and its formula.
            let sizes: Vec<f32> = text_shapes(&output, " = ")
                .map(|(_, text)| text.galley.job.sections[0].format.font_id.size)
                .collect();
            assert!(!sizes.is_empty(), "{path}");
            for set in sizes {
                assert_eq!(set == TOOLTIP_SIZE, fits, "{path}: {set}");
                assert!(set >= crate::gui::typeset::SMALLEST_FITTED_SIZE, "{path}");
            }
        }
    }

    #[test]
    fn a_value_whose_path_is_a_term_in_view_is_framed_in_its_colour() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let marks = TermColors::of(eq);
        let f_end = marks.get("model.f_end").unwrap();
        let ctx = egui::Context::default();
        let output = sized_frame(&ctx, egui::vec2(600.0, 400.0), Vec::new(), |ui| {
            let mut readouts = Readouts::new(marks.clone(), false);
            draw(ui, &mut readouts, "model.f_end", "0.9046");
            draw(ui, &mut readouts, "mass.total_g", "48.15 g");
        });
        let frames: Vec<egui::Color32> = flat_shapes(&output)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r) if r.stroke.width == MARK_WIDTH => Some(r.stroke.color),
                _ => None,
            })
            .collect();
        assert_eq!(frames, [f_end], "only the term's value is framed");
    }

    #[test]
    fn a_hover_text_is_built_with_the_frame_s_details_setting() {
        // The header's Show spreadsheet details reaches every readout's hover text through the
        // frame's readouts (spec S2-5); readouts made without it have it off.
        assert!(!Readouts::default().show_details());
        for show_details in [false, true] {
            let (_, output) = hover_frames(
                &egui::Context::default(),
                egui::vec2(800.0, 600.0),
                |output| text_center(output, "2.688 N·m"),
                |ui| {
                    let mut readouts = Readouts::new(TermColors::default(), show_details);
                    assert_eq!(readouts.show_details(), show_details);
                    let rect = ui.label("2.688 N·m").rect;
                    readouts.show_over(ui, rect, "model.pullout_Nm", |details| {
                        format!("details shown: {details}")
                    });
                },
            );
            let want = format!("details shown: {show_details}");
            assert!(drawn_texts(&output).contains(&want), "{want}");
        }
    }

    #[test]
    fn a_warning_link_asks_for_its_note() {
        let mut readouts = Readouts::default();
        readouts.open_note("a5.low_saturation");
        assert_eq!(readouts.events().note, Some("a5.low_saturation"));
        assert_eq!(readouts.finish().note, Some("a5.low_saturation"));
    }

    #[test]
    fn a_check_guide_link_asks_for_its_input_s_row() {
        let mut readouts = Readouts::default();
        readouts.focus_input("metal.cup_wall_corner_mm");
        assert_eq!(readouts.events().focus, Some("metal.cup_wall_corner_mm"));
        assert_eq!(readouts.finish().focus, Some("metal.cup_wall_corner_mm"));
    }

    #[test]
    fn a_selecting_value_asks_for_the_explain_panel_by_click_or_key_and_not_on_focus() {
        // UX overhaul spec 5.2 and 5.5: a dimension tag, a callout row and a results-table row
        // select their result. The hover shows its text and its equation without the line
        // that says a click opens the Equation panel; a click asks for the result in the
        // Explain panel and opens no equation; the keyboard's focus arriving asks for nothing,
        // and Enter or Space on the focused value asks as a click does (BL-093).
        use crate::gui::test_support::{key_tap, primary_button};
        const PATH: &str = "model.pullout_Nm";
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let size = egui::vec2(800.0, 600.0);
        let frame = |input: Vec<egui::Event>, time: Option<f64>| {
            let mut events = ReadoutEvents::default();
            let mut rect = Rect::NOTHING;
            let output = sized_frame_at(&ctx, size, time, input, |ui| {
                let mut readouts = Readouts::default();
                rect = ui.label("2.688 N·m").rect;
                readouts.select_over(ui, rect, PATH, |_| format!("hover text of {PATH}"));
                events = readouts.finish();
            });
            (output, events, rect)
        };
        let (_, _, rect) = frame(Vec::new(), Some(0.0));
        let at = rect.center();
        frame(vec![egui::Event::PointerMoved(at)], Some(0.1));
        let (output, events, _) = frame(Vec::new(), Some(0.2));
        assert_eq!(events.hovered.as_deref(), Some(PATH));
        let texts = drawn_texts(&output);
        assert!(
            texts.contains(&format!("hover text of {PATH}")),
            "{texts:?}"
        );
        for run in ["T", "pull", " = ", "2D", "end", "cal"] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
        assert!(!texts.contains(&OPEN_HINT.to_owned()), "{texts:?}");
        frame(vec![primary_button(at, true)], Some(0.3));
        let (_, events, _) = frame(vec![primary_button(at, false)], Some(0.4));
        assert_eq!((events.selected, events.clicked), (Some(PATH), None));
        // The keyboard: the pointer gone, a fresh focus on the value, then Enter and Space.
        frame(vec![egui::Event::PointerGone], Some(0.5));
        let (_, events, _) = frame(key_tap(egui::Key::Tab), Some(0.6));
        assert_eq!(events.selected, None, "the focus arriving selects nothing");
        assert!(
            ctx.memory(|m| m.focused().is_some()),
            "the value takes the focus"
        );
        for key in [egui::Key::Enter, egui::Key::Space] {
            let (_, events, _) = frame(key_tap(key), Some(0.7));
            assert_eq!(events.selected, Some(PATH), "{key:?}");
        }
    }

    #[test]
    fn a_selecting_value_is_outlined_while_selected_or_focused() {
        // A dimension tag, a callout row or a results-table row shows where it is as a tile and
        // a check row do: outlined while its result is the selection, or while it has the
        // keyboard's focus; with neither (another result selected), no outline.
        use crate::gui::explain::Selection;
        use crate::gui::test_support::key_tap;
        const PATH: &str = "model.pullout_Nm";
        let ctx = egui::Context::default();
        let size = egui::vec2(800.0, 600.0);
        let outlines = |selection: Selection, input: Vec<egui::Event>| {
            let output = sized_frame(&ctx, size, input, |ui| {
                let mut readouts = Readouts::default().with_selection(selection);
                let rect = ui.label("2.688 N·m").rect;
                readouts.select_over(ui, rect, PATH, |_| String::new());
            });
            let strong = ctx.style().visuals.strong_text_color();
            flat_shapes(&output)
                .into_iter()
                .filter(|s| matches!(s, egui::Shape::Rect(r) if r.stroke.width == 1.0 && r.stroke.color == strong))
                .count()
        };
        assert_eq!(outlines(Selection::Nothing, Vec::new()), 0);
        assert_eq!(outlines(Selection::Result("mass.total_g"), Vec::new()), 0);
        assert_eq!(outlines(Selection::Result(PATH), Vec::new()), 1);
        // The keyboard's focus arriving outlines it with nothing selected.
        outlines(Selection::Nothing, key_tap(egui::Key::Tab));
        assert!(
            ctx.memory(|m| m.focused().is_some()),
            "the value takes the focus"
        );
        assert_eq!(outlines(Selection::Nothing, Vec::new()), 1);
    }
}
