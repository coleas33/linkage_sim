//! The four tiles of the opening tab (UX overhaul spec 4.4): the nominal pull-out, hot-low with
//! what it must reach, cold-high and the rotating mass. On each the number leads, large, and
//! its label follows, small (decision C-6). A tile is a readout: its hover is the result's hover
//! text and equation, and a click selects it, so the Explain panel shows what it is.
//!
//! The numbers are the results' own, through the formatters every readout uses. Hot-low and
//! the requirement are quoted as a pair ([`hot_low_quotes`], BL-068), so a failing tile never
//! shows two equal numbers. Where the model makes no prediction a torque tile says so in place
//! of its number, in the words the Size tab's readouts had.

use crate::gui::checks::state_mark;
use crate::gui::dashboard::{Level, end_effect_banner, hover_text};
use crate::gui::guidance::{hot_low_quotes, result_number, result_text};
use crate::gui::look;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::Readouts;
use crate::gui::size_controls::{
    HOT_LOW, NO_PREDICTION, NO_TORQUE, predicts, pullout_label, torque_gap_text,
};
use crate::{DesignInputs, DesignResults};

/// The start of the line under the Hot-low tile's label: what hot-low must reach, the
/// requirement ("needs 1.350 N·m").
pub const NEEDS: &str = "needs";

/// The size of a tile's number [points].
pub const NUMBER_SIZE: f32 = 22.0;

/// The narrowest a tile gets in a row of four [points]: in a narrower region the tiles wrap to
/// two rows of two instead of clipping.
pub const MIN_TILE_WIDTH: f32 = 130.0;

/// The gap between tiles [points].
const TILE_GAP: f32 = 6.0;

/// A tile, by the result it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    /// The nominal pull-out torque at the operating temperature.
    Pullout,
    /// The hot-low torque, with the requirement it must reach.
    HotLow,
    /// The cold-high torque.
    ColdHigh,
    /// The rotating mass.
    Mass,
}

impl Tile {
    /// Every tile, in the order drawn.
    pub const ALL: [Tile; 4] = [Tile::Pullout, Tile::HotLow, Tile::ColdHigh, Tile::Mass];

    /// The result the tile shows.
    pub const fn path(self) -> &'static str {
        match self {
            Tile::Pullout => "model.pullout_Nm",
            Tile::HotLow => "metal.torque_hot_low_Nm",
            Tile::ColdHigh => "metal.torque_cold_high_Nm",
            Tile::Mass => "mass.total_g",
        }
    }

    /// The tile's label: the result's on-screen name; the pull-out's with its operating
    /// temperature, as the Size tab's readout named it.
    pub fn label(self, results: &DesignResults) -> String {
        match self {
            Tile::Pullout => pullout_label(results.metal.op_temp_C),
            Tile::HotLow => HOT_LOW.to_owned(),
            Tile::ColdHigh | Tile::Mass => label_for(self.path()).to_owned(),
        }
    }

    /// The tile's number with its unit, or, for a torque the model does not predict
    /// ([`predicts`]), the words that say so.
    pub fn number(
        self,
        inputs: &DesignInputs,
        results: &DesignResults,
    ) -> Result<String, &'static str> {
        if self != Tile::Mass && !predicts(results, result_number(results, self.path())) {
            return Err(if end_effect_banner(results).is_some() {
                NO_PREDICTION
            } else {
                NO_TORQUE
            });
        }
        Ok(match self {
            Tile::HotLow => hot_low_quotes(inputs, results).0,
            Tile::Pullout | Tile::ColdHigh | Tile::Mass => result_text(results, self.path()),
        })
    }
}

/// The lines under the Hot-low tile's label while the model predicts hot-low: the requirement
/// it must reach, as [`hot_low_quotes`] quotes it (drawn after [`NEEDS`]), its gap to the
/// requirement ([`torque_gap_text`]) and whether it meets it.
pub(crate) fn needs_lines(
    inputs: &DesignInputs,
    results: &DesignResults,
) -> (String, String, bool) {
    let (_, required) = hot_low_quotes(inputs, results);
    let (gap, meets) = torque_gap_text(
        result_number(results, Tile::HotLow.path()),
        inputs.metal.required_min_Nm,
    );
    (required, gap, meets)
}

/// The gap text in the pieces a line may break between: the torque, then its percent in
/// brackets, so a number never leaves its unit ("short 0.21 N·m", "(-8.6 %)").
pub(crate) fn gap_pieces(gap: &str) -> Vec<String> {
    match gap.split_once(" (") {
        Some((torque, percent)) => vec![torque.to_owned(), format!("({percent}")],
        None => vec![gap.to_owned()],
    }
}

/// Draws `pieces` as one line that wraps only between them, a space apart, after the shape of
/// `mark` where there is one.
fn unbroken_line(ui: &mut egui::Ui, mark: Option<Level>, pieces: Vec<egui::RichText>) {
    ui.horizontal_wrapped(|ui| {
        if mark.is_some() {
            state_mark(ui, mark);
        }
        let body = egui::TextStyle::Body.resolve(ui.style());
        ui.spacing_mut().item_spacing.x = ui.fonts(|fonts| fonts.glyph_width(&body, ' '));
        for piece in pieces {
            ui.add(egui::Label::new(piece).extend());
        }
    });
}

/// Draws the tiles of `results`, computed from `inputs`, across the width `ui` has: four in a
/// row, or two rows of two where a row of four would make a tile narrower than
/// [`MIN_TILE_WIDTH`]. The tile `selected` is outlined in the text colour. Each is a readout
/// (`readouts`: its hover). Returns the tile clicked.
pub(crate) fn tiles_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    selected: Option<Tile>,
    readouts: &mut Readouts,
) -> Option<Tile> {
    let available = ui.available_width();
    let columns = if available >= 4.0 * MIN_TILE_WIDTH + 3.0 * TILE_GAP {
        4
    } else {
        2
    };
    let width = ((available - (columns - 1) as f32 * TILE_GAP) / columns as f32).max(0.0);
    let mut clicked = None;
    for row in Tile::ALL.chunks(columns) {
        // One height per row: the tallest tile's content as the last frame drew it.
        let id = ui.id().with(("magcoupling_tile_row", row[0].path()));
        let height: f32 = ui.data(|data| data.get_temp(id)).unwrap_or(0.0);
        let mut tallest: f32 = 0.0;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = TILE_GAP;
            for &tile in row {
                let is_selected = selected == Some(tile);
                let (content, response) =
                    tile_ui(ui, tile, (width, height), is_selected, inputs, results);
                tallest = tallest.max(content);
                if response.clicked() {
                    clicked = Some(tile);
                }
                readouts.hover(ui, response, tile.path(), |show_details| {
                    hover_text(tile.path(), show_details).unwrap_or_default()
                });
            }
        });
        if tallest != height {
            ui.data_mut(|data| data.insert_temp(id, tallest));
            // Discard this pass so the first painted frame already has even heights (BL-111).
            ui.ctx().request_discard("tile row height changed");
        }
    }
    clicked
}

/// One tile, `size.0` points wide with its content at least `size.1` points tall: its number
/// (or the words for no prediction), its label and, on the Hot-low tile, what it must reach.
/// Returns its content's own height and its click sensor.
fn tile_ui(
    ui: &mut egui::Ui,
    tile: Tile,
    size: (f32, f32),
    selected: bool,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> (f32, egui::Response) {
    let id = ui.id().with(("magcoupling_tile", tile.path()));
    // A tile the keyboard reached is outlined as a selected one is. At rest a tile, a click
    // target, shows the resting outline of a framed control (BL-109); one width in both
    // states, whatever the host's resting width, so selecting a tile moves no row.
    let focused = ui.memory(|memory| memory.has_focus(id));
    let visuals = ui.visuals();
    let outline = if selected || focused {
        visuals.strong_text_color()
    } else {
        // A host with no resting outline (width 0) gets the noninteractive hairline (BL-110).
        let rest = visuals.widgets.inactive.bg_stroke;
        if rest.width > 0.0 {
            rest.color
        } else {
            visuals.widgets.noninteractive.bg_stroke.color
        }
    };
    let frame = egui::Frame::new()
        .fill(visuals.faint_bg_color)
        .stroke(egui::Stroke::new(1.0, outline))
        .corner_radius(4.0)
        .inner_margin(8.0);
    let inner = frame.show(ui, |ui| {
        ui.set_width((size.0 - frame.total_margin().sum().x).max(0.0));
        let content = ui
            .vertical(|ui| {
                let number = tile.number(inputs, results);
                match &number {
                    Ok(number) => {
                        ui.label(egui::RichText::new(number).size(NUMBER_SIZE).strong());
                    }
                    Err(words) => {
                        ui.add(egui::Label::new(egui::RichText::new(*words).weak()).wrap());
                    }
                }
                let quiet = look::quiet_text(ui.visuals());
                let label = egui::RichText::new(tile.label(results)).color(quiet);
                ui.add(egui::Label::new(label).wrap());
                if tile == Tile::HotLow && number.is_ok() {
                    let (required, gap, meets) = needs_lines(inputs, results);
                    let level = if meets { Level::Good } else { Level::Bad };
                    let color = level.color(ui.visuals());
                    // A failing tile is marked by the fail shape and red on this line, never on
                    // the number alone (spec 4.4). A passing one: the word muted, the value in
                    // the text colour.
                    let needs = egui::RichText::new(NEEDS);
                    let required = egui::RichText::new(required);
                    if meets {
                        unbroken_line(ui, None, vec![needs.color(quiet), required]);
                    } else {
                        let line = vec![needs.color(color), required.color(color)];
                        unbroken_line(ui, Some(level), line);
                    }
                    let pieces = gap_pieces(&gap).into_iter();
                    let pieces = pieces
                        .map(|p| egui::RichText::new(p).color(color))
                        .collect();
                    unbroken_line(ui, None, pieces);
                }
            })
            .response
            .rect
            .height();
        // The frame's own layout is its row's, left to right: a spacer would sit beside the
        // content, so the row's height is set as the frame's least height.
        ui.set_min_height(size.1);
        content
    });
    let response = ui.interact(inner.response.rect, id, egui::Sense::click());
    (inner.inner, response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::dashboard::END_EFFECT_BANNER;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, flat_shapes, glyph_context, hover_frames, key_tap,
        no_back_iron, passing_design, primary_button, short_magnets, sized_frame, text_color,
        text_rect, zero_torque,
    };

    /// Frames of the tiles of `inputs` in a region `width` points wide in `ctx`, with
    /// `selected` selected: a first one with no input, a second so each row has its height,
    /// then one per entry of `events`. What the last drew and the tile it returned.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        width: f32,
        selected: Option<Tile>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Tile>) {
        let results = compute_all(inputs);
        let mut last = None;
        for input in [Vec::new(), Vec::new()].into_iter().chain(events) {
            let mut clicked = None;
            let output = sized_frame(ctx, egui::vec2(width, 600.0), input, |ui| {
                clicked = tiles_ui(ui, inputs, &results, selected, &mut Readouts::default());
            });
            last = Some((output, clicked));
        }
        last.expect("two frames at least")
    }

    /// The rect of `text` in `output`.
    fn rect(output: &egui::FullOutput, text: &str) -> egui::Rect {
        text_rect(output, text).unwrap_or_else(|| panic!("no {text:?} drawn"))
    }

    /// The frames of the tiles, each a filled rect with a hairline, as drawn left to right and
    /// top to bottom.
    fn tile_rects(ctx: &egui::Context, output: &egui::FullOutput) -> Vec<egui::Rect> {
        let fill = ctx.style().visuals.faint_bg_color;
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.fill == fill => Some(r.rect),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_four_tiles_share_one_height_in_the_first_painted_frame() {
        // BL-111: a row's height came from the last frame, so the first frame drew uneven tiles.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let output = sized_frame(&ctx, egui::vec2(700.0, 600.0), Vec::new(), |ui| {
            tiles_ui(ui, &inputs, &results, None, &mut Readouts::default());
        });
        let tiles = tile_rects(&ctx, &output);
        assert_eq!(tiles.len(), 4);
        assert!(
            tiles
                .iter()
                .all(|t| (t.height() - tiles[0].height()).abs() < 0.5),
            "{tiles:?}"
        );
    }

    #[test]
    fn a_resting_tile_shows_an_edge_in_a_plain_egui_context() {
        // BL-110 (F-1): under egui::Visuals::dark(), what a host gets that does not apply the
        // calculator's theme, the resting outline is Stroke::NONE; the tile falls back to the
        // noninteractive hairline, so it still has a visible edge.
        let ctx = egui::Context::default();
        assert_eq!(ctx.style().visuals.widgets.inactive.bg_stroke.width, 0.0);
        let (output, _) = frames(&ctx, &DesignInputs::default(), 700.0, None, Vec::new());
        let fill = ctx.style().visuals.faint_bg_color;
        let edges: Vec<egui::Color32> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.fill == fill => Some(r.stroke.color),
                _ => None,
            })
            .collect();
        assert_eq!(edges.len(), Tile::ALL.len());
        let hairline = ctx.style().visuals.widgets.noninteractive.bg_stroke.color;
        assert!(
            edges.iter().all(|&c| c == hairline && c.a() > 0),
            "{edges:?}"
        );
    }

    #[test]
    fn the_four_tiles_show_the_results_own_numbers_over_their_labels() {
        // UX overhaul spec 4.4 and decision C-6: the nominal pull-out, hot-low, cold-high and
        // the rotating mass, each number as every readout formats it, leading its label.
        let ctx = egui::Context::default();
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
            // One row of body text: a label may wrap onto two.
            let body = egui::TextStyle::Body.resolve(&ctx.style());
            let row = ctx.fonts(|fonts| fonts.row_height(&body));
            let mut left = f32::NEG_INFINITY;
            for tile in Tile::ALL {
                let number = result_text(&results, tile.path());
                assert_eq!(tile.number(&inputs, &results), Ok(number.clone()));
                let (number, label) =
                    (rect(&output, &number), rect(&output, &tile.label(&results)));
                assert!(number.bottom() <= label.top() + 0.5, "{tile:?}");
                assert!(number.height() > 1.5 * row, "{tile:?}: the number is large");
                assert!(number.left() > left, "{tile:?}: in order from the left");
                left = number.left();
            }
            // Four tiles in one row, equally wide and equally tall.
            let tiles = tile_rects(&ctx, &output);
            assert_eq!(tiles.len(), 4);
            for tile in &tiles {
                assert!((tile.top() - tiles[0].top()).abs() < 0.5, "{tiles:?}");
                assert!((tile.width() - tiles[0].width()).abs() < 0.5, "{tiles:?}");
                assert!((tile.height() - tiles[0].height()).abs() < 0.5, "{tiles:?}");
            }
        }
        assert_eq!(
            Tile::ALL.map(Tile::path),
            [
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "mass.total_g"
            ]
        );
    }

    #[test]
    fn a_label_is_quiet_and_its_number_keeps_the_text_colour() {
        // UX overhaul spec 6.1: a reader scanning for a number finds it as the bright line of
        // its tile. In a theme that sets a weak colour the labels take it.
        let ctx = egui::Context::default();
        let weak = egui::Color32::from_rgb(1, 2, 3);
        ctx.style_mut(|style| style.visuals.weak_text_color = Some(weak));
        let strong = ctx.style().visuals.strong_text_color();
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        for tile in Tile::ALL {
            let number = result_text(&results, tile.path());
            assert_eq!(text_color(&output, &number), Some(strong), "{tile:?}");
            let label = tile.label(&results);
            assert_eq!(text_color(&output, &label), Some(weak), "{tile:?}");
        }
        // A tile that meets its requirement says what it needs: the word as a caption, the
        // requirement as a value.
        let (required, _, meets) = needs_lines(&inputs, &results);
        assert!(meets);
        assert_eq!(text_color(&output, NEEDS), Some(weak));
        let text = ctx.style().visuals.text_color();
        assert_eq!(text_color(&output, &required), Some(text));
    }

    #[test]
    fn the_hot_low_tile_says_what_it_needs_and_its_gap_and_fails_by_shape_and_red_on_that_line() {
        // Spec 4.4: "needs <required>" under the label, then the existing gap text. A failing
        // tile is marked by the fail shape and red on its "needs" line, never on the number.
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let discs = |output: &egui::FullOutput| {
            flat_shapes(output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c) if c.fill == Level::Bad.color(&visuals))
                })
                .count()
        };
        // The example design fails its hot minimum.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (required, gap, meets) = needs_lines(&inputs, &results);
        assert_eq!((required.as_str(), meets), ("2.500 N·m", false));
        assert_eq!(gap, "short 0.21 N·m (-8.6 %)");
        assert_eq!(gap_pieces(&gap), ["short 0.21 N·m", "(-8.6 %)"]);
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        let bad = Level::Bad.color(&visuals);
        for text in [NEEDS, required.as_str(), "short 0.21 N·m", "(-8.6 %)"] {
            assert_eq!(text_color(&output, text), Some(bad), "{text}");
        }
        assert_eq!(
            text_color(&output, "2.285 N·m"),
            Some(visuals.strong_text_color())
        );
        assert_eq!(discs(&output), 1);
        let label = rect(&output, HOT_LOW);
        assert!(label.bottom() <= rect(&output, NEEDS).top() + 0.5);
        assert!(rect(&output, NEEDS).bottom() <= rect(&output, "short 0.21 N·m").top() + 0.5);
        // A passing design: the margin in the pass colour, and no fail shape.
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let (required, gap, meets) = needs_lines(&inputs, &results);
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        for piece in gap_pieces(&gap) {
            let good = Level::Good.color(&visuals);
            assert_eq!(text_color(&output, &piece), Some(good), "{piece}");
        }
        assert_eq!(discs(&output), 0);
        assert!(text_rect(&output, NEEDS).is_some() && text_rect(&output, &required).is_some());
    }

    #[test]
    fn the_gap_line_breaks_only_between_its_torque_and_its_percent() {
        // UX overhaul plan review E-3: in a narrow tile the gap wrapped inside its brackets,
        // "short 0.21 N·m (-8.6" over "%)". Each piece is drawn whole, on one row, inside its
        // tile, at every width.
        let ctx = egui::Context::default();
        frames(&ctx, &DesignInputs::default(), 700.0, None, Vec::new());
        let body = egui::TextStyle::Body.resolve(&ctx.style());
        let row = ctx.fonts(|fonts| fonts.row_height(&body));
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let (_, gap, _) = needs_lines(&inputs, &results);
            assert_eq!(gap_pieces(&gap).concat().len(), gap.len() - 1, "{gap}");
            for width in [700.0, 560.0, 300.0] {
                let (output, _) = frames(&ctx, &inputs, width, None, Vec::new());
                let tile = tile_rects(&ctx, &output)[1];
                for piece in gap_pieces(&gap) {
                    let at = rect(&output, &piece);
                    assert!(at.height() < 1.5 * row, "{width}: {piece:?} wrapped");
                    assert!(tile.contains_rect(at), "{width}: {piece:?} at {at:?}");
                }
            }
        }
        assert_eq!(gap_pieces("margin +0 N·m"), ["margin +0 N·m"]);
    }

    #[test]
    fn a_failing_hot_low_tile_never_shows_two_equal_numbers() {
        // Spec 4.4 (BL-068): where hot-low is a hair under the requirement, both would read
        // the same at four significant digits. The tile quotes the pair as the hot-minimum
        // Problem does: hot-low one last digit below.
        let mut inputs = DesignInputs::default();
        let hot_low = compute_all(&inputs).metal.torque_hot_low_Nm;
        inputs.metal.required_min_Nm = hot_low + 1e-7;
        let results = compute_all(&inputs);
        assert!(results.metal.torque_hot_low_Nm < inputs.metal.required_min_Nm);
        let number = Tile::HotLow.number(&inputs, &results).unwrap();
        let (required, _, meets) = needs_lines(&inputs, &results);
        assert!(!meets);
        assert_eq!(required, result_text(&results, Tile::HotLow.path()));
        assert_ne!(required, number);
        // The Problem quotes the same two numbers.
        let problem = (crate::gui::guidance::guide("metal.hot_min_check")
            .unwrap()
            .problem)(&inputs, &results)
        .expect("the check fails");
        assert!(
            problem.contains(&number) && problem.contains(&required),
            "{problem}"
        );
        // A passing tile reads as its readout does, with no digit moved.
        inputs.metal.required_min_Nm = hot_low;
        let results = compute_all(&inputs);
        assert_eq!(
            Tile::HotLow.number(&inputs, &results),
            Ok(result_text(&results, Tile::HotLow.path()))
        );
    }

    #[test]
    fn a_failing_hot_low_tile_never_reads_a_gap_of_zero() {
        // BL-083: at Required = hot-low + 1e-7 the tile read "short 0.00 N·m (-0.0 %)" beside
        // hot-low quoted one last digit below the requirement (BL-068). A shortfall that reads
        // zero at its decimals reads four significant digits, as the readouts write values.
        for mut inputs in [crate::app::starting_design(), DesignInputs::default()] {
            let hot_low = compute_all(&inputs).metal.torque_hot_low_Nm;
            inputs.metal.required_min_Nm = hot_low + 1e-7;
            let results = compute_all(&inputs);
            let (_, gap, meets) = needs_lines(&inputs, &results);
            assert!(!meets, "{gap}");
            assert_eq!(gap, "short 1.000e-7 N·m (-0.0 %)");
        }
    }

    #[test]
    fn a_torque_the_model_does_not_predict_reads_so_in_place_of_its_number() {
        // Spec 4.4: the no-prediction texts replace the number as on the Size tab's readouts,
        // and the Hot-low tile shows neither what it needs nor a gap. The mass is still a
        // number.
        let ctx = egui::Context::default();
        // Without back iron, a design file's calibration coefficient of 0: no torque is finite
        // while the end effect is in range. A variation of -1e308: the pull-out stays 2.69 N·m
        // and the two torques of the variation band are not finite.
        let mut zero_coefficient = no_back_iron();
        zero_coefficient.calibration.f_cal_original = 0.0;
        let mut huge_variation = DesignInputs::default();
        huge_variation.metal.variation = -1e308;
        let all = [Tile::Pullout, Tile::HotLow, Tile::ColdHigh];
        for (inputs, words, unpredicted) in [
            (short_magnets(), NO_PREDICTION, &all[..]),
            (zero_coefficient, NO_TORQUE, &all[..]),
            (
                huge_variation,
                NO_TORQUE,
                &[Tile::HotLow, Tile::ColdHigh][..],
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(
                end_effect_banner(&results).is_some(),
                words == NO_PREDICTION,
                "{END_EFFECT_BANNER}"
            );
            for tile in all {
                let number = tile.number(&inputs, &results);
                if unpredicted.contains(&tile) {
                    assert_eq!(number, Err(words), "{tile:?}");
                } else {
                    assert_eq!(number, Ok(result_text(&results, tile.path())), "{tile:?}");
                }
            }
            assert!(Tile::Mass.number(&inputs, &results).is_ok());
            let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
            let texts = drawn_texts(&output);
            let drawn = texts.iter().filter(|t| *t == words).count();
            assert_eq!(drawn, unpredicted.len(), "{texts:?}");
            assert!(!texts.iter().any(|t| t.starts_with(NEEDS)), "{texts:?}");
        }
        // An unmagnetized inner ring is a prediction of zero, drawn as a number.
        let zero = zero_torque();
        let results = compute_all(&zero);
        assert_eq!(
            Tile::Pullout.number(&zero, &results),
            Ok(result_text(&results, Tile::Pullout.path()))
        );
    }

    #[test]
    fn in_a_narrow_region_the_tiles_wrap_to_two_rows_and_nothing_is_cut() {
        // Spec 8.2, test 15: four in a row where each keeps MIN_TILE_WIDTH, else two rows of
        // two. No tile runs past the region, and every text stays inside its tile.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for (width, per_row) in [(700.0, 4), (560.0, 4), (520.0, 2), (300.0, 2)] {
            let (output, _) = frames(&ctx, &inputs, width, None, Vec::new());
            let tiles = tile_rects(&ctx, &output);
            assert_eq!(tiles.len(), 4, "{width}");
            let first_row = tiles
                .iter()
                .filter(|t| (t.top() - tiles[0].top()).abs() < 0.5);
            assert_eq!(first_row.count(), per_row, "{width}: {tiles:?}");
            for (tile, rect_of) in Tile::ALL.iter().zip(&tiles) {
                assert!(rect_of.right() <= width + 0.5, "{width}: {rect_of:?}");
                for text in [result_text(&results, tile.path()), tile.label(&results)] {
                    let at = rect(&output, &text);
                    assert!(rect_of.contains_rect(at), "{width}: {text:?} at {at:?}");
                }
            }
        }
    }

    #[test]
    fn a_click_returns_the_tile_a_hover_shows_its_readout_and_the_selected_tile_is_outlined() {
        // Spec 5.2 and decision C-2: a click selects, a hover never does. The hover is the
        // result's own text; the selected tile's hairline is the text colour.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (output, clicked) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        assert_eq!(clicked, None);
        let at = rect(&output, &Tile::ColdHigh.label(&results)).center();
        let (_, hovered) = frames(
            &ctx,
            &inputs,
            700.0,
            None,
            vec![vec![egui::Event::PointerMoved(at)]],
        );
        assert_eq!(hovered, None);
        let click = vec![
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ];
        let (_, clicked) = frames(&ctx, &inputs, 700.0, None, click);
        assert_eq!(clicked, Some(Tile::ColdHigh));
        // The hover text.
        let label = Tile::Mass.label(&results);
        let (_, hover) = hover_frames(
            &egui::Context::default(),
            egui::vec2(700.0, 600.0),
            |output| rect(output, &label).center(),
            |ui| {
                tiles_ui(ui, &inputs, &results, None, &mut Readouts::default());
            },
        );
        let help = hover_text(Tile::Mass.path(), false).unwrap();
        assert!(drawn_texts(&hover).contains(&help), "{help}");
        // The outline.
        let outlined = |selected, events| {
            let ctx = egui::Context::default();
            let stroke = ctx.style().visuals.strong_text_color();
            let (output, _) = frames(&ctx, &inputs, 700.0, selected, events);
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| matches!(shape, egui::Shape::Rect(r) if r.stroke.color == stroke))
                .count()
        };
        assert_eq!(
            (
                outlined(None, Vec::new()),
                outlined(Some(Tile::Pullout), Vec::new())
            ),
            (0, 1)
        );
        // A tile the keyboard reached is outlined too (review UXQ-3).
        assert_eq!(outlined(None, vec![key_tap(egui::Key::Tab), Vec::new()]), 1);
    }

    #[test]
    fn the_tiles_own_words_have_glyphs_and_plain_words() {
        // Spec 8.3: the one new string of the tiles is NEEDS; the rest are the readouts'.
        let ctx = glyph_context();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (required, gap, _) = needs_lines(&inputs, &results);
        let mut texts = vec![required, gap, NEEDS.to_owned()];
        texts.extend(Tile::ALL.map(|tile| tile.label(&results)));
        for text in &texts {
            assert_glyphs(&ctx, text, "a tile");
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }
}
