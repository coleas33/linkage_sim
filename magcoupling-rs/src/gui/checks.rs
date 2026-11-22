//! The check list (UX overhaul spec 4.5): the top of the right column. The design status line,
//! then one row per check that counts, in [`CHECK_GUIDES`]' order, which never changes (decision
//! C-3): the check's state as a shape and a word as well as a colour (decision C-4), and its
//! name. A click on a row selects the check, and the Explain panel under the list
//! ([`crate::gui::explain`]) shows its problem, why it matters and what to change. Under the
//! rows, the material warnings that fire, and the line that says two of the checks read stored
//! 3D fields ([`STORED_3D_LABEL`]): at the foot of the list, it is on the first view.
//!
//! [`CHECK_GUIDES`]: crate::gui::guidance::CHECK_GUIDES

use crate::gui::dashboard::{
    Level, STORED_3D_LABEL, STORED_3D_NOTE, check_level, greyed_by_end_effect, shown_lines,
    status_ui, warnings_ui,
};
use crate::gui::guidance::{self, CheckGuide, FIT_CHECK, counted_guides};
use crate::gui::look;
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults};

/// The heading of the check list.
pub const HEADING: &str = "CHECKS";

/// The most of the right column's height the check list takes before it scrolls (spec 4.5:
/// "about 45 %"); the Explain panel has the rest.
pub const CHECKS_SHARE: f32 = 0.45;

/// The side of a state mark's cell [points].
const MARK_SIDE: f32 = 12.0;

/// The state a check's row shows: its guide's level, or none where the check gives no badge (it
/// does not apply, as the cup wall without back iron, or it is computed from a pull-out the
/// end-effect model cannot predict). The Magnets fit row always has one.
pub(crate) fn mark_level(
    guide: &CheckGuide,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Option<Level> {
    if guide.check == FIT_CHECK {
        Some(guidance::level(guide, inputs, results))
    } else {
        check_level(results, guide.check)
    }
}

/// A check's state as a shape in its level's colour (decision C-4): a filled disc where it
/// fails, a triangle where it asks for a look, a ring where it passes; an empty cell for no
/// state. The level's word is its hover text.
pub(crate) fn state_mark(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) =
        ui.allocate_exact_size(egui::Vec2::splat(MARK_SIDE), egui::Sense::hover());
    let Some(level) = level else {
        return;
    };
    let color = level.color(ui.visuals());
    let centre = rect.center();
    match level {
        Level::Bad => {
            ui.painter().circle_filled(centre, 5.0, color);
        }
        Level::Caution => {
            let corner = |x: f32, y: f32| centre + egui::vec2(x, y);
            ui.painter().add(egui::Shape::convex_polygon(
                vec![corner(0.0, -5.0), corner(5.0, 4.0), corner(-5.0, 4.0)],
                color,
                egui::Stroke::NONE,
            ));
        }
        Level::Good => {
            ui.painter()
                .circle_stroke(centre, 4.0, egui::Stroke::new(1.5, color));
        }
    }
    response.on_hover_text(level.word());
}

/// Draws the check list of `results`, computed from `inputs`: [`HEADING`], the design status
/// line (named against the session's `baseline`), a row per guide that counts while the
/// shaft-clamp section is paused (`clamps_paused`) or not, the material warnings that fire and
/// the stored-3D line. The row of the check `selected` is drawn on the raised panel colour. A
/// row with a result is a readout (`readouts`): its hover text and equation while hovered.
/// Returns the check whose row was clicked.
pub(crate) fn checks_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    baseline: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
    selected: Option<&'static str>,
    readouts: &mut Readouts,
) -> Option<&'static str> {
    ui.colored_label(look::quiet_text(ui.visuals()), HEADING);
    status_ui(ui, inputs, baseline, results, clamps_paused);
    let lines = shown_lines(results, clamps_paused);
    let mut clicked = None;
    for guide in counted_guides(clamps_paused) {
        let level = mark_level(guide, inputs, results);
        let background = ui.painter().add(egui::Shape::Noop);
        let row = ui
            .horizontal(|ui| {
                state_mark(ui, level);
                let label = egui::RichText::new(guide.label);
                let label = if greyed_by_end_effect(results, guide.check) {
                    label.weak()
                } else {
                    label
                };
                ui.label(label);
                if let Some(level) = level {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.colored_label(level.color(ui.visuals()), level.word());
                    });
                }
            })
            .response
            .rect;
        let rect = row.with_max_x(ui.max_rect().right());
        let response = ui.interact(
            rect,
            ui.id().with(("check_row", guide.check)),
            egui::Sense::click(),
        );
        if selected == Some(guide.check) || response.hovered() {
            ui.painter().set(
                background,
                egui::Shape::rect_filled(
                    rect.expand2(egui::vec2(2.0, 1.0)),
                    4.0,
                    ui.visuals().faint_bg_color,
                ),
            );
        }
        if response.clicked() {
            clicked = Some(guide.check);
        }
        if let Some(line) = lines.iter().find(|line| line.path == guide.check) {
            readouts.hover(ui, response, line.path, |show_details| {
                line.tooltip(show_details)
            });
        }
    }
    ui.separator();
    warnings_ui(ui, results, readouts);
    ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::dashboard::{WARNINGS_HEADING, severity_level, warning_lines};
    use crate::gui::guidance::{
        CHECK_GUIDES, EXAMPLE_DESIGN, EXAMPLE_DESIGN_HOVER, STARTING_DESIGN, STARTING_DESIGN_HOVER,
        YOUR_DESIGN, design_status, design_status_text,
    };
    use crate::gui::test_support::{
        drawn_texts, flat_shapes, hover_frames, only_the_clamp_fails, only_the_gearbox_fails,
        only_the_joint_fails, passing_design, primary_button, reversed_hub_band, short_magnets,
        sized_frame, text_center, text_color, text_rect, with_poles,
    };

    /// The defaults with `edit` applied.
    fn design(edit: impl FnOnce(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(crate::gui::test_support::hot_day)
    }

    /// Frames of the check list of `inputs` against the spreadsheet's baseline in `ctx`, with
    /// the shaft-clamp section paused (`clamps_paused`) or not and `selected` selected: a
    /// first one with no input, then one per entry of `events`. What the last drew and the
    /// check it returned.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        clamps_paused: bool,
        selected: Option<&'static str>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<&'static str>) {
        let results = compute_all(inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut clicked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 1200.0), input, |ui| {
                clicked = checks_ui(
                    ui,
                    inputs,
                    &DesignInputs::default(),
                    &results,
                    clamps_paused,
                    selected,
                    &mut Readouts::default(),
                );
            });
            last = Some((output, clicked));
        }
        last.expect("one frame at least")
    }

    /// What the check list of `inputs` draws.
    fn list_texts(inputs: &DesignInputs, clamps_paused: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        drawn_texts(&frames(&ctx, inputs, clamps_paused, None, Vec::new()).0)
    }

    #[test]
    fn the_list_is_the_status_line_a_row_per_counted_check_in_the_guides_order_and_the_3d_line() {
        // Spec 4.5 and decision C-3: one fixed order, failing checks not sorted to the top. At
        // the example design four checks fail and five pass.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for paused in [false, true] {
            let mut want = vec![
                HEADING.to_owned(),
                design_status_text(&design_status(&inputs, &results, paused)),
            ];
            for guide in counted_guides(paused) {
                want.push(guide.label.to_owned());
                want.push(guidance::level(guide, &inputs, &results).word().to_owned());
            }
            want.push(STORED_3D_LABEL.to_owned());
            assert_eq!(list_texts(&inputs, paused), want, "paused {paused}");
        }
        let labels: Vec<&str> = counted_guides(true).map(|guide| guide.label).collect();
        assert_eq!(
            labels,
            [
                "Hot-low torque",
                "Running clearance",
                "Cup wall",
                "Temperature limits",
                "Shaft joints",
                "Gearbox input torque",
                "Space claim",
                "End-effect model",
                "Magnets fit"
            ]
        );
        // Ruling 11: a design that fails only the clamp reads as passing while it is paused,
        // and its row is drawn only while the section shows.
        let inputs = only_the_clamp_fails();
        let clamp = guidance::guide("clamps.recommended").expect("the clamp's guide");
        for paused in [false, true] {
            let texts = list_texts(&inputs, paused);
            let rows = texts.iter().filter(|t| *t == clamp.label).count();
            assert_eq!(rows, usize::from(!paused), "paused {paused}");
            let fails = texts.iter().filter(|t| *t == Level::Bad.word()).count();
            assert_eq!(fails, usize::from(!paused), "paused {paused}");
        }
        assert_eq!(
            list_texts(&inputs, true)[1],
            "Your design: passes all 9 checks"
        );
    }

    /// The marks the rows of the check list of `inputs` draw, top to bottom: the level each
    /// one's shape stands for. (A material warning under the rows has a badge of its own.)
    fn marks(ctx: &egui::Context, inputs: &DesignInputs) -> Vec<Level> {
        let visuals = ctx.style().visuals.clone();
        let (output, _) = frames(ctx, inputs, false, None, Vec::new());
        let foot = text_rect(&output, CHECK_GUIDES.last().unwrap().label)
            .expect("the last row")
            .bottom();
        flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                // A filled disc: fails.
                egui::Shape::Circle(c)
                    if c.center.y < foot && c.fill == Level::Bad.color(&visuals) =>
                {
                    Some(Level::Bad)
                }
                // A ring: passes.
                egui::Shape::Circle(c)
                    if c.fill == egui::Color32::TRANSPARENT
                        && c.stroke.color == Level::Good.color(&visuals) =>
                {
                    Some(Level::Good)
                }
                // A triangle: asks for a look.
                egui::Shape::Path(p)
                    if p.closed
                        && p.points.len() == 3
                        && p.points[0].y < foot
                        && p.fill == Level::Caution.color(&visuals) =>
                {
                    Some(Level::Caution)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_state_has_its_own_shape_and_word_as_well_as_its_colour() {
        // Spec 8.2, test 12 (decision C-4): a filled disc where a check fails, a triangle where
        // it asks for a look, a ring where it passes, each with its word beside the name.
        let ctx = egui::Context::default();
        let mut reached = std::collections::BTreeSet::new();
        for inputs in [
            DesignInputs::default(),
            hot_day(),
            passing_design(),
            with_poles(40),
        ] {
            let results = compute_all(&inputs);
            let want: Vec<Level> = CHECK_GUIDES
                .iter()
                .map(|guide| guidance::level(guide, &inputs, &results))
                .collect();
            assert_eq!(marks(&ctx, &inputs), want);
            let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
            let visuals = ctx.style().visuals.clone();
            for level in &want {
                assert_eq!(
                    text_color(&output, level.word()),
                    Some(level.color(&visuals)),
                    "{level:?}"
                );
            }
            reached.extend(want);
        }
        assert_eq!(reached.len(), 3, "the designs reach every state");
        let words = [Level::Good, Level::Caution, Level::Bad].map(Level::word);
        assert_eq!(words, ["OK", "Check", "Fails"]);
    }

    #[test]
    fn a_check_that_gives_no_badge_has_no_mark_and_no_word() {
        // Short magnets: the end-effect model is out of range, so the checks computed from the
        // pull-out are greyed and judged by the End-effect model row alone. Without back iron
        // the cup wall check does not apply.
        let ctx = egui::Context::default();
        for (inputs, unjudged) in [
            (
                short_magnets(),
                vec![
                    "metal.hot_min_check",
                    "clamps.recommended",
                    "shaft_joint.check",
                    "gearbox.cold_high_check",
                ],
            ),
            (
                design(|i| i.coupling.backiron = 0),
                vec!["materials.cup_wall_check"],
            ),
        ] {
            let results = compute_all(&inputs);
            for guide in &CHECK_GUIDES {
                let mark = mark_level(guide, &inputs, &results);
                assert_eq!(
                    mark.is_none(),
                    unjudged.contains(&guide.check),
                    "{}",
                    guide.check
                );
            }
            let judged = CHECK_GUIDES.len() - unjudged.len();
            assert_eq!(marks(&ctx, &inputs).len(), judged);
            let words = [Level::Good, Level::Caution, Level::Bad].map(Level::word);
            let texts = list_texts(&inputs, false);
            let drawn = texts.iter().filter(|t| words.contains(&t.as_str())).count();
            assert_eq!(drawn, judged);
        }
    }

    #[test]
    fn a_click_on_a_row_returns_its_check_and_the_selected_row_is_raised() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let gearbox = guidance::guide("gearbox.cold_high_check").unwrap();
        let (output, clicked) = frames(&ctx, &inputs, false, None, Vec::new());
        assert_eq!(clicked, None);
        let at = text_rect(&output, gearbox.label).unwrap().center();
        // A hover returns nothing; the click returns the check on its release.
        let (_, hovered) = frames(
            &ctx,
            &inputs,
            false,
            None,
            vec![vec![egui::Event::PointerMoved(at)]],
        );
        assert_eq!(hovered, None);
        let click = vec![
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ];
        let (_, clicked) = frames(&ctx, &inputs, false, None, click);
        assert_eq!(clicked, Some(gearbox.check));
        // The selected row sits on the raised panel colour, and no other does.
        let raised = |selected| {
            let ctx = egui::Context::default();
            let fill = ctx.style().visuals.faint_bg_color;
            let (output, _) = frames(&ctx, &inputs, false, selected, Vec::new());
            let label = text_rect(&output, gearbox.label).unwrap();
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Rect(r) if r.fill == fill && r.rect.contains_rect(label))
                })
                .count()
        };
        assert_eq!((raised(None), raised(Some(gearbox.check))), (0, 1));
        assert_eq!(raised(Some("metal.hot_min_check")), 0);
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let mut passing_hot_day = passing_design();
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
        let mut passing_reversed = passing_design();
        reversed_hub_band(&mut passing_reversed);
        let cases = [
            // The workbook example: four red checks.
            (DesignInputs::default(), 4, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                3,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The passing design with the example's gearbox rating: the gearbox check alone.
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's minimum at 8 µm: the shaft joints alone.
            (only_the_joint_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's band reversed: the shaft joints ask for a
            // look.
            (passing_reversed, 0, 1, Level::Caution),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 4, 1, Level::Bad),
            // The passing design on the hot day: amber alone.
            (passing_hot_day, 0, 1, Level::Caution),
        ];
        for (inputs, bad, caution, colour) in cases {
            let results = compute_all(&inputs);
            let status = design_status(&inputs, &results, false);
            assert_eq!((status.bad, status.caution), (bad, caution));
            assert_eq!(status.total, CHECK_GUIDES.len());
            assert_eq!(status.is_default, inputs == DesignInputs::default());
            let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
            let text = design_status_text(&status);
            assert_eq!(
                drawn_texts(&output).get(1),
                Some(&text),
                "the status line tops the list, under its heading"
            );
            assert_eq!(text_color(&output, &text), Some(colour.color(&visuals)));
        }
    }

    /// The texts drawn while the status line `text` of `inputs` against `baseline` is hovered.
    fn status_hover(inputs: &DesignInputs, baseline: &DesignInputs, text: &str) -> Vec<String> {
        let results = compute_all(inputs);
        let (_, output) = hover_frames(
            &egui::Context::default(),
            egui::vec2(320.0, 1200.0),
            |output| text_center(output, text),
            |ui| {
                checks_ui(
                    ui,
                    inputs,
                    baseline,
                    &results,
                    false,
                    None,
                    &mut Readouts::default(),
                );
            },
        );
        drawn_texts(&output)
    }

    #[test]
    fn the_example_design_s_status_line_says_on_hover_where_its_values_come_from() {
        // Spec S2-5: "Example design" at the defaults, its hover the original spreadsheet;
        // "Your design" has no hover.
        for (inputs, example) in [(DesignInputs::default(), true), (passing_design(), false)] {
            let results = compute_all(&inputs);
            let text = design_status_text(&design_status(&inputs, &results, false));
            assert_eq!(text.starts_with(EXAMPLE_DESIGN), example, "{text}");
            let texts = status_hover(&inputs, &DesignInputs::default(), &text);
            let hover = EXAMPLE_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), example, "{text}");
        }
    }

    #[test]
    fn the_starting_design_s_status_line_says_on_hover_what_it_is() {
        // UX overhaul spec 7.1: at the session's baseline, when that is not the spreadsheet's
        // design, the line reads "Starting design" and its hover says what the design is; the
        // same design on a panel whose baseline is the spreadsheet's is "Your design", with no
        // hover.
        let inputs = passing_design();
        for (baseline, name, starting) in [
            (passing_design(), STARTING_DESIGN, true),
            (DesignInputs::default(), YOUR_DESIGN, false),
        ] {
            let text = format!("{name}: passes all 10 checks");
            let texts = status_hover(&inputs, &baseline, &text);
            let hover = STARTING_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), starting, "{text}");
            let example = EXAMPLE_DESIGN_HOVER.to_owned();
            assert!(!texts.contains(&example), "{text}");
        }
    }

    #[test]
    fn the_warnings_that_fire_follow_the_rows_and_the_stored_3d_line_closes_the_list() {
        // An aluminium hub behind the inner magnets fires a material warning.
        let inputs = design(|i| i.coupling.backiron = 0);
        let results = compute_all(&inputs);
        let warnings = warning_lines(&results);
        assert!(!warnings.is_empty(), "a design that fires a warning");
        let ctx = egui::Context::default();
        let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
        let y = |text: &str| {
            text_rect(&output, text)
                .unwrap_or_else(|| panic!("no {text:?}"))
                .top()
        };
        let last_row = CHECK_GUIDES.last().unwrap().label;
        assert!(y(last_row) < y(WARNINGS_HEADING));
        for (_, text) in &warnings {
            assert!(y(WARNINGS_HEADING) < y(text) && y(text) < y(STORED_3D_LABEL));
        }
        assert_eq!(drawn_texts(&output).last().unwrap(), STORED_3D_LABEL);
        // Each warning's mark is its level's shape (decision C-4): the expansion mismatch's
        // caution a triangle, never the fail disc in the caution colour.
        let caution = Level::Caution.color(&ctx.style().visuals);
        let cautions = warnings
            .iter()
            .filter(|(rule, _)| severity_level(rule.severity) == Level::Caution);
        assert!(cautions.count() > 0, "a design that fires a caution");
        let shapes = flat_shapes(&output);
        let disc = |s: &&egui::Shape| matches!(s, egui::Shape::Circle(c) if c.fill == caution);
        let triangle =
            |s: &&egui::Shape| matches!(s, egui::Shape::Path(p) if p.closed && p.fill == caution);
        assert!(!shapes.iter().any(disc));
        assert!(shapes.iter().any(triangle));
        // Without a warning the heading is not drawn.
        let texts = list_texts(&DesignInputs::default(), false);
        assert!(!texts.iter().any(|t| t == WARNINGS_HEADING));
    }
}
