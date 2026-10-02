//! The readout hook (spec Addendum A2 "Hover": "Any displayed value (dashboard, results
//! table, geometry callouts) shows a tooltip with its equation. Each term is coloured, and the
//! same colour marks that term wherever its value appears on screen").
//!
//! Every view hands each value it displays to [`Readouts::show`], with its result path and
//! its hover text: the dashboard rows, the results-table rows, the geometry callouts, the
//! clamp summary and screw table, the plot readouts. While the value is hovered the tooltip
//! shows that text, then the value's equation typeset with its terms in colour
//! ([`TermColors::of`]); a click asks the panel to open it in the Equation panel. Both the
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
use crate::gui::typeset::{TermColors, equation_ui};

/// The hint under a tooltip's equation.
pub const OPEN_HINT: &str = "Click to open it in the Equation panel";

/// The text size of a tooltip's equation [points].
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
    /// The path of the value clicked.
    pub clicked: Option<String>,
    /// The teaching note a warning's link asked for, by id.
    pub note: Option<&'static str>,
}

/// One frame's readouts: the term colours they mark and what the user did.
#[derive(Clone, Debug, Default)]
pub struct Readouts {
    marks: TermColors,
    events: ReadoutEvents,
}

impl Readouts {
    /// The readouts of a frame that marks the terms in `marks`.
    pub fn new(marks: TermColors) -> Self {
        Self {
            marks,
            events: ReadoutEvents::default(),
        }
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
    /// hovered `text()` and its equation; a click asks for it in the Equation panel. `text` is
    /// called only while the value is hovered.
    pub fn show(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce() -> String,
    ) {
        self.mark(ui, response.rect, path);
        if response.hovered() {
            self.events.hovered = Some(path.to_owned());
        }
        if response.clicked() {
            self.events.clicked = Some(path.to_owned());
        }
        response.on_hover_ui(|ui| {
            ui.label(text());
            if let Some(eq) = registry().equation_for(path) {
                ui.separator();
                equation_ui(ui, registry(), eq, &TermColors::of(eq), TOOLTIP_SIZE);
            }
            ui.weak(OPEN_HINT);
        });
    }

    /// [`Readouts::show`] for a value drawn in `rect` by widgets that take no clicks (labels
    /// keep their look): a click sensor over the rect, keyed by `path` in `ui`.
    pub fn show_over(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        path: &str,
        text: impl FnOnce() -> String,
    ) {
        let response = ui.interact(rect, ui.id().with(("readout", path)), egui::Sense::click());
        self.show(ui, response, path, text);
    }

    /// Asks for the teaching note `id` (a warning's link).
    pub fn open_note(&mut self, id: &'static str) {
        self.events.note = Some(id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{drawn_texts, flat_shapes, sized_frame, sized_frame_at};

    /// A label of `text` showing the result at `path`, drawn with `readouts`.
    fn draw(ui: &mut egui::Ui, readouts: &mut Readouts, path: &str, text: &str) -> Rect {
        let rect = ui.label(text).rect;
        readouts.show_over(ui, rect, path, || format!("hover text of {path}"));
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
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
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
    fn a_value_whose_path_is_a_term_in_view_is_framed_in_its_colour() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let marks = TermColors::of(eq);
        let f_end = marks.get("model.f_end").unwrap();
        let ctx = egui::Context::default();
        let output = sized_frame(&ctx, egui::vec2(600.0, 400.0), Vec::new(), |ui| {
            let mut readouts = Readouts::new(marks.clone());
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
    fn a_warning_link_asks_for_its_note() {
        let mut readouts = Readouts::default();
        readouts.open_note("a5.low_saturation");
        assert_eq!(readouts.events().note, Some("a5.low_saturation"));
        assert_eq!(readouts.finish().note, Some("a5.low_saturation"));
    }
}
