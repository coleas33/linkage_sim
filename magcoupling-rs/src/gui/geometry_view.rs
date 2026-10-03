//! The geometry view (spec M4 "Layout": the centre region's default view, decision M42-2):
//! paints [`crate::gui::geometry`] to scale with egui's painter, both views at one scale
//! (decision M42-3), then lists the dimension callouts and the notes under them.
//!
//! Each dimension is a line with arrowheads and a tag; the list repeats the tag with the
//! callout's text. The line and the text are readouts ([`crate::gui::readouts::Readouts`]):
//! hovering either shows the hover text of the result it shows
//! ([`crate::gui::dashboard::hover_text`]) and its equation, built only while hovered, and a
//! click opens it in the Equation panel. A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
//! every frame, so it follows a slider while it is dragged.
//!
//! [`side_by_side`] (views at one scale, nothing at all when there is no room) and [`arrowhead`]
//! are shared with the clamp drawing.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};

use crate::gui::dashboard::{Level, hover_text};
use crate::gui::geometry::{Callout, Geometry, Mm, Outline, Part, View, finite, geometry};
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults};

/// The colour of a dimension that is fine: drawing.py's dimension blue, lightened for a dark
/// background.
pub const DIMENSION: Color32 = Color32::from_rgb(90, 170, 230);

/// The gap between the end view and the side view [mm at the drawing's scale].
const VIEW_GAP_MM: f64 = 3.0;

/// The smallest height of the drawing [points] while the list under it keeps
/// [`MIN_LIST_ROWS`]; the list scrolls when space is short. Shorter still, the drawing shrinks
/// below it (it is drawn to scale, so it only scales down).
const MIN_DRAWING_HEIGHT: f32 = 160.0;

/// The rows of the callout list the drawing leaves room for when space is short.
const MIN_LIST_ROWS: f32 = 3.0;

/// The margin inside the drawing's area [points].
const MARGIN: f32 = 8.0;

/// Maps millimetres (x right, y up) to screen points (y down) at one scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// The screen point of the millimetre origin.
    pub origin: Pos2,
    /// Points per millimetre.
    pub scale: f32,
}

impl Transform {
    /// The screen point of `p`.
    pub fn to_px(&self, p: Mm) -> Pos2 {
        Pos2::new(
            self.origin.x + p[0] as f32 * self.scale,
            self.origin.y - p[1] as f32 * self.scale,
        )
    }

    /// A length in millimetres as points.
    pub fn len(&self, mm: f64) -> f32 {
        mm as f32 * self.scale
    }
}

/// What the geometry view drew: the drawing's area, the shared scale, each view's transform
/// (`None` when it is not drawable) and each dimension's hover area by tag.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryLayout {
    pub rect: Rect,
    pub scale: f32,
    pub end: Option<Transform>,
    pub side: Option<Transform>,
    pub dimensions: Vec<(usize, Rect)>,
}

/// The fill colour of a part.
pub fn part_color(part: Part, visuals: &egui::Visuals) -> Color32 {
    match part {
        Part::Body => Color32::from_rgb(128, 132, 140),
        Part::Cavity => visuals.extreme_bg_color,
        Part::Magnet { north: true } => Color32::from_rgb(190, 110, 90),
        Part::Magnet { north: false } => Color32::from_rgb(95, 125, 175),
        Part::Retainer => Color32::from_rgb(150, 175, 165),
        Part::Cap => Color32::from_rgb(175, 180, 190),
        Part::Shaft => Color32::from_rgb(100, 104, 112),
        Part::Key => Color32::from_rgb(205, 180, 95),
    }
}

/// The colour of a callout or dashed line: its level's, else `plain`.
fn level_color(level: Option<Level>, plain: Color32, visuals: &egui::Visuals) -> Color32 {
    level.map_or(plain, |l| l.color(visuals))
}

/// Places views of the extents `extents` (each its min and max corner [mm]) side by side in
/// `rect`, `gap_mm` apart, centred, at one scale, the largest that fits: that scale [points per
/// mm] and each view's transform. `None` when there is no room (a region of no size or an
/// extent that is not finite, where the scale would be zero, negative or not a number and the
/// views mirrored): then nothing is painted.
pub(crate) fn side_by_side(
    rect: Rect,
    extents: &[(Mm, Mm)],
    gap_mm: f64,
) -> Option<(f32, Vec<Transform>)> {
    // f32::min passes over NaN, so an extent that is not finite is refused here.
    if !extents
        .iter()
        .all(|(min, max)| finite(*min) && finite(*max))
    {
        return None;
    }
    let gaps = extents.len().saturating_sub(1);
    let width: f64 = extents
        .iter()
        .map(|(min, max)| max[0] - min[0])
        .sum::<f64>()
        + gap_mm * gaps as f64;
    let height = extents
        .iter()
        .map(|(min, max)| max[1] - min[1])
        .fold(0.0, f64::max);
    let scale = (rect.width() / width as f32).min(rect.height() / height as f32);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let total: f32 = extents
        .iter()
        .map(|(min, max)| (max[0] - min[0]) as f32 * scale)
        .sum::<f32>()
        + (gap_mm as f32 * scale) * gaps as f32;
    let mut left = rect.center().x - total / 2.0;
    let mut transforms = Vec::with_capacity(extents.len());
    for (min, max) in extents {
        transforms.push(Transform {
            origin: Pos2::new(
                left - min[0] as f32 * scale,
                rect.center().y + ((min[1] + max[1]) / 2.0) as f32 * scale,
            ),
            scale,
        });
        left += ((max[0] - min[0]) + gap_mm) as f32 * scale;
    }
    Some((scale, transforms))
}

/// Draws an arrowhead at `tip` pointing along `dir` (a unit vector): two 6-point strokes back
/// from the tip, 0.45 rad either side of the line.
pub(crate) fn arrowhead(painter: &egui::Painter, tip: Pos2, dir: Vec2, stroke: Stroke) {
    for side in [-1.0, 1.0] {
        let back = egui::emath::Rot2::from_angle(side * 0.45) * (-dir) * 6.0;
        painter.line_segment([tip, tip + back], stroke);
    }
}

/// Draws the geometry of the design shown (`inputs` and the `results` computed from them),
/// each callout a readout (`readouts`), and returns what it drew.
pub fn geometry_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) -> GeometryLayout {
    let g = geometry(inputs, results);
    let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
    let rows = g.end.callouts.len() + g.side.callouts.len() + g.notes.len();
    let list = rows as f32 * row * 1.5;
    let height = drawing_height(ui.available_height(), list, row);
    let rect = if height > 0.0 {
        ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover())
            .0
    } else {
        // No room for the drawing: the list gets the whole height (an empty allocation would
        // still take an item spacing).
        Rect::from_min_size(
            ui.available_rect_before_wrap().min,
            Vec2::new(ui.available_width(), 0.0),
        )
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let views: Vec<&View> = [&g.end, &g.side]
        .into_iter()
        .filter(|v| v.is_drawable())
        .collect();
    let mut layout = GeometryLayout {
        rect,
        scale: 0.0,
        end: None,
        side: None,
        dimensions: Vec::new(),
    };
    let extents: Vec<(Mm, Mm)> = views.iter().map(|v| (v.min, v.max)).collect();
    if let Some((scale, transforms)) = side_by_side(rect.shrink(MARGIN), &extents, VIEW_GAP_MM) {
        layout.scale = scale;
        for (view, transform) in views.into_iter().zip(transforms) {
            paint_view(
                ui,
                &painter,
                view,
                transform,
                &mut layout.dimensions,
                readouts,
            );
            if std::ptr::eq(view, &g.end) {
                layout.end = Some(transform);
            } else {
                layout.side = Some(transform);
            }
        }
    }
    list_ui(ui, &g, readouts);
    layout
}

/// The drawing's height [points] in `available` points over a callout list wanting `list`
/// points in rows `row` points apart: what the list leaves, at least [`MIN_DRAWING_HEIGHT`]
/// while the list keeps [`MIN_LIST_ROWS`] (all of it, if shorter), then less, down to none.
/// The list scrolls in what is left.
fn drawing_height(available: f32, list: f32, row: f32) -> f32 {
    let kept = list.min(MIN_LIST_ROWS * row);
    (available - list)
        .max(MIN_DRAWING_HEIGHT)
        .min(available - kept)
        .max(0.0)
}

/// Paints one view's pieces, dashed lines and dimensions, and registers each dimension as a
/// readout.
fn paint_view(
    ui: &egui::Ui,
    painter: &egui::Painter,
    view: &View,
    t: Transform,
    dimensions: &mut Vec<(usize, Rect)>,
    readouts: &mut Readouts,
) {
    let visuals = ui.visuals();
    let centre = t.to_px([0.0, 0.0]);
    for piece in &view.pieces {
        let color = part_color(piece.part, visuals);
        match &piece.outline {
            Outline::Disc { r } => {
                painter.circle_filled(centre, t.len(*r), color);
            }
            Outline::Ring { r_in, r_out } => {
                let width = t.len(r_out - r_in);
                painter.circle_stroke(
                    centre,
                    t.len((r_in + r_out) / 2.0),
                    Stroke::new(width, color),
                );
            }
            Outline::Polygon(points) => {
                let points = points.iter().map(|p| t.to_px(*p)).collect();
                painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
            }
            Outline::Sector {
                r_in,
                r_out,
                from,
                to,
            } => {
                let r = (r_in + r_out) / 2.0;
                let points: Vec<Pos2> = (0..=16)
                    .map(|i| {
                        let a = from + (to - from) * f64::from(i) / 16.0;
                        t.to_px([r * a.cos(), r * a.sin()])
                    })
                    .collect();
                painter.add(egui::Shape::line(
                    points,
                    Stroke::new(t.len(r_out - r_in), color),
                ));
            }
            Outline::Rect { min, max } => {
                painter.rect_filled(Rect::from_two_pos(t.to_px(*min), t.to_px(*max)), 0.0, color);
            }
        }
    }
    for dashed in &view.dashed {
        let color = level_color(dashed.level, visuals.weak_text_color(), visuals);
        let points: Vec<Pos2> = dashed.points.iter().map(|p| t.to_px(*p)).collect();
        painter.extend(egui::Shape::dashed_line(
            &points,
            Stroke::new(1.0, color),
            6.0,
            4.0,
        ));
    }
    for callout in &view.callouts {
        if !(finite(callout.from) && finite(callout.to)) {
            continue;
        }
        let rect = paint_dimension(painter, callout, t, visuals);
        let response = ui.interact(
            rect,
            ui.id().with(("geometry_dimension", callout.tag)),
            Sense::click(),
        );
        readouts.show(ui, response, callout.path, || callout_text(callout));
        dimensions.push((callout.tag, rect));
    }
}

/// Paints a dimension line with its arrowheads and tag; returns its hover area.
fn paint_dimension(
    painter: &egui::Painter,
    callout: &Callout,
    t: Transform,
    visuals: &egui::Visuals,
) -> Rect {
    let color = level_color(callout.level, DIMENSION, visuals);
    let stroke = Stroke::new(1.5, color);
    let (a, b) = (t.to_px(callout.from), t.to_px(callout.to));
    painter.line_segment([a, b], stroke);
    let along = (b - a).normalized();
    if (b - a).length() >= 10.0 {
        for (tip, dir) in [(a, -along), (b, along)] {
            arrowhead(painter, tip, dir, stroke);
        }
    }
    let away = if along == Vec2::ZERO { Vec2::X } else { along };
    // Past the outer end the tag lands on a block or the liner: on a plate, it stays readable.
    plated_text(
        painter,
        b + away * 9.0,
        callout.tag.to_string(),
        egui::FontId::proportional(12.0),
        color,
        visuals.extreme_bg_color,
    );
    Rect::from_two_pos(a, b).expand(6.0)
}

/// Paints `text` centred at `at` on a plate of `plate` one point larger than the text, so it
/// reads over whatever is drawn beneath it.
pub(crate) fn plated_text(
    painter: &egui::Painter,
    at: Pos2,
    text: String,
    font: egui::FontId,
    color: Color32,
    plate: Color32,
) {
    let galley = painter.layout_no_wrap(text, font, color);
    let rect = egui::Align2::CENTER_CENTER.anchor_size(at, galley.size());
    painter.rect_filled(rect.expand(1.0), 0.0, plate);
    painter.galley(rect.min, galley, color);
}

/// The hover text of the callout's result.
fn callout_text(callout: &Callout) -> String {
    hover_text(callout.path).unwrap_or_else(|| callout.path.to_owned())
}

/// The callouts (tag and text, in their colours, each a readout) and the notes, scrolling in
/// the height left (egui's 64-point floor for a scroll area lowered to none).
fn list_ui(ui: &mut egui::Ui, g: &Geometry, readouts: &mut Readouts) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_geometry_list")
        .auto_shrink([false, true])
        .min_scrolled_height(0.0)
        .show(ui, |ui| {
            for callout in g.callouts() {
                let color = callout.level.map(|l| l.color(ui.visuals()));
                let rich = |text: String| match color {
                    Some(c) => egui::RichText::new(text).color(c),
                    None => egui::RichText::new(text),
                };
                let rect = ui
                    .horizontal(|ui| {
                        ui.label(rich(callout.tag.to_string()).strong());
                        ui.add(egui::Label::new(rich(callout.text.clone())).wrap());
                    })
                    .response
                    .rect;
                readouts.show_over(ui, rect, callout.path, || callout_text(callout));
            }
            for note in &g.notes {
                let text = egui::RichText::new(&note.text);
                let text = match note.level {
                    Some(level) => text.color(level.color(ui.visuals())),
                    None => text.color(ui.visuals().weak_text_color()),
                };
                ui.add(egui::Label::new(text).wrap());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::geometry::{NOT_DRAWN, mm};
    use crate::gui::test_support::{
        drawn_texts, flat_shapes, sized_frame, sized_frame_at, text_color, text_rects,
    };

    /// One frame of the geometry view of `inputs` on a `size` screen, with `events`.
    fn frame(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, GeometryLayout) {
        let results = compute_all(inputs);
        let mut layout = None;
        let output = sized_frame(ctx, size, events, |ui| {
            layout = Some(geometry_ui(ui, inputs, &results, &mut Readouts::default()));
        });
        (output, layout.expect("drawn"))
    }

    /// Every filled circle's and every stroked circle's radius [points].
    fn circle_radii(output: &egui::FullOutput) -> Vec<f32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) => Some(c.radius),
                _ => None,
            })
            .collect()
    }

    /// Every painted rectangle's width [points].
    fn rect_widths(output: &egui::FullOutput) -> Vec<f32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) => Some(r.rect.width()),
                _ => None,
            })
            .collect()
    }

    fn has_radius(radii: &[f32], want: f32) -> bool {
        radii.iter().any(|r| (r - want).abs() < 1e-3)
    }

    #[test]
    fn the_views_share_one_scale_that_fits_the_screen() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (_, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        let g = geometry(&inputs, &compute_all(&inputs));
        let end = layout.end.expect("the end view is drawn");
        let side = layout.side.expect("the side view is drawn");
        assert_eq!(end.scale, side.scale);
        // Both views side by side, at the largest scale that fits inside the margins.
        let width = (g.end.max[0] - g.end.min[0]) + VIEW_GAP_MM + (g.side.max[0] - g.side.min[0]);
        let height = (g.end.max[1] - g.end.min[1]).max(g.side.max[1] - g.side.min[1]);
        let inner = layout.rect.shrink(MARGIN);
        let want = (inner.width() / width as f32).min(inner.height() / height as f32);
        assert_eq!(layout.scale, want);
        // The end view's extent lands inside the drawing.
        assert!(inner.contains(end.to_px(g.end.min)) && inner.contains(end.to_px(g.end.max)));
        assert!(layout.rect.width() > 900.0 && layout.rect.height() >= MIN_DRAWING_HEIGHT);
    }

    #[test]
    fn known_dimensions_map_to_their_pixel_distances() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        let s = layout.scale;
        let radii = circle_radii(&output);
        // The cup OD (41.33 mm) and the 10 mm shaft, to scale.
        assert!(
            has_radius(&radii, (r.model.cup_od_mm / 2.0) as f32 * s),
            "{radii:?}"
        );
        assert!(has_radius(&radii, 5.0 * s), "{radii:?}");
        // The face gap's dimension spans 1.4 mm at the scale.
        let end = layout.end.unwrap();
        let g = geometry(&inputs, &r);
        let face = g.callouts().find(|c| c.tag == 1).unwrap();
        let span = (end.to_px(face.to) - end.to_px(face.from)).length();
        assert!((span - 1.4 * s).abs() < 1e-2, "{span} vs {}", 1.4 * s);
        // The side view, painted at the same scale: the cup wall spans the cup depth in effect
        // and the cap its 0.8 mm.
        assert!(layout.side.is_some());
        let widths = rect_widths(&output);
        for mm in [r.housing.cup_depth_mm, inputs.metal.cap_axial_mm] {
            assert!(
                widths.iter().any(|w| (w - mm as f32 * s).abs() < 1e-2),
                "{mm} mm in {widths:?}"
            );
        }
    }

    #[test]
    fn the_callouts_and_notes_are_listed_with_their_colours() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, _) = frame(&ctx, &inputs, egui::vec2(1000.0, 900.0), Vec::new());
        let texts = drawn_texts(&output);
        for want in [
            "Face gap 1.400 mm",
            "Corner gap 1.027 mm",
            "Overall length: axial stack 31.80 mm of 35.00 mm",
            "Large-diameter bay: stack 18.80 mm of 20.00 mm",
            "Diameter: rotating OD 42.80 mm of 43.00 mm",
        ] {
            assert!(
                texts.iter().any(|t| t == want),
                "missing {want:?} in {texts:?}"
            );
            assert_eq!(
                text_color(&output, want),
                Some(ctx.style().visuals.text_color())
            );
        }
        // The default running clearance is below zero: its text and its line are red.
        let run =
            "Running clearance -0.1032 mm (sleeve-to-liner gap 0.6768 mm less movement 0.7800 mm)";
        let red = Level::Bad.color(&ctx.style().visuals);
        assert_eq!(text_color(&output, run), Some(red));
        assert!(segment_colors(&output).contains(&red));
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("Design check: the boss OD"))
        );
        assert!(texts.iter().any(|t| t.starts_with("Autofit:")));
    }

    /// The stroke colour of every line segment.
    fn segment_colors(output: &egui::FullOutput) -> Vec<Color32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::LineSegment { stroke, .. } => Some(stroke.color),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_design_past_the_space_claim_draws_the_exceeded_axes_red() {
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 900.0), Vec::new());
        let red = Level::Bad.color(&ctx.style().visuals);
        let text = "Overall length: axial stack 69.90 mm is 34.90 mm over the 35.00 mm claim";
        assert_eq!(text_color(&output, text), Some(red));
        // The views still fit: the stack's 69.9 mm sets the scale.
        assert!(layout.side.is_some() && layout.scale > 0.0);
        assert!(
            segment_colors(&output)
                .iter()
                .filter(|c| **c == red)
                .count()
                >= 2
        );
    }

    #[test]
    fn each_dimension_tag_sits_on_a_plate_of_the_drawing_s_background() {
        // A tag lands past its line's outer end, on a magnet block or the liner: a plate of the
        // drawing's background under it keeps it readable.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        let background = ctx.style().visuals.extreme_bg_color;
        let plates: Vec<Rect> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.fill == background => Some(r.rect),
                _ => None,
            })
            .collect();
        assert_eq!(layout.dimensions.len(), 6);
        for (tag, _) in &layout.dimensions {
            let on_drawing: Vec<Rect> = text_rects(&output, &tag.to_string())
                .into_iter()
                .filter(|r| layout.rect.contains_rect(*r))
                .collect();
            assert_eq!(on_drawing.len(), 1, "tag {tag}: {on_drawing:?}");
            let text = on_drawing[0];
            // Its own plate, not the drawing's background as a whole.
            assert!(
                plates
                    .iter()
                    .any(|p| p.contains_rect(text) && p.width() < text.width() + 8.0),
                "tag {tag} at {text:?}: no plate in {plates:?}"
            );
        }
    }

    #[test]
    fn hovering_a_dimension_shows_its_result_s_hover_text() {
        let ctx = egui::Context::default();
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
        let inputs = DesignInputs::default();
        let size = egui::vec2(1000.0, 700.0);
        let (_, layout) = frame(&ctx, &inputs, size, Vec::new());
        let (_, rect) = layout
            .dimensions
            .iter()
            .find(|(tag, _)| *tag == 1)
            .copied()
            .expect("the face gap is drawn");
        let at = rect.center();
        let want = hover_text("model.face_gap_mm").unwrap();
        let (before, _) = frame(&ctx, &inputs, size, vec![egui::Event::PointerMoved(at)]);
        assert!(
            !drawn_texts(&before).contains(&want),
            "not before the pointer is there"
        );
        let results = compute_all(&inputs);
        let mut output = None;
        for dt in [0.1, 0.2] {
            output = Some(sized_frame_at(&ctx, size, Some(dt), Vec::new(), |ui| {
                geometry_ui(ui, &inputs, &results, &mut Readouts::default());
            }));
        }
        assert!(
            want.starts_with("Gap at the flat centres (effective gap)\n"),
            "{want}"
        );
        let texts = drawn_texts(&output.unwrap());
        assert!(texts.contains(&want), "no hover text in {texts:?}");
    }

    #[test]
    fn a_dimension_that_is_not_a_number_draws_without_panicking() {
        // validate() refuses NaN at every boundary; a struct literal can still hold it.
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.metal.cup_wall_corner_mm = f64::NAN;
        inputs.metal.max_diameter_mm = f64::NAN;
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        assert!(layout.end.is_none());
        assert!(layout.side.is_some());
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t == "Not drawn: a dimension of the end view is not a number")
        );
        // A tiny screen still draws.
        frame(
            &ctx,
            &DesignInputs::default(),
            egui::vec2(120.0, 90.0),
            Vec::new(),
        );
    }

    #[test]
    fn a_region_with_no_room_draws_no_view() {
        // A window shrunk to nothing: no scale, so neither view is painted (nor mirrored).
        let ctx = egui::Context::default();
        for size in [egui::vec2(0.0, 0.0), egui::vec2(1.0, 1.0)] {
            let (_, layout) = frame(&ctx, &DesignInputs::default(), size, Vec::new());
            assert!(layout.scale >= 0.0, "{size:?}: {layout:?}");
            assert!(
                layout.end.is_none() && layout.side.is_none(),
                "{size:?}: {layout:?}"
            );
            assert!(layout.dimensions.is_empty());
        }
        // side_by_side itself: no room, no extent, or an extent that is not finite; else the
        // largest scale that fits, the views centred.
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(100.0, 100.0));
        let unit: (Mm, Mm) = ([0.0, 0.0], [1.0, 1.0]);
        let empty = Rect::from_min_size(Pos2::ZERO, Vec2::ZERO);
        assert!(side_by_side(empty, &[unit], 0.0).is_none());
        assert!(side_by_side(rect, &[], 0.0).is_none());
        assert!(side_by_side(rect, &[([0.0, 0.0], [f64::NAN, 1.0])], 0.0).is_none());
        let (scale, transforms) = side_by_side(rect, &[unit, unit], 2.0).unwrap();
        assert_eq!(scale, 25.0, "1 + 2 + 1 mm across 100 points");
        assert_eq!(transforms[0].to_px([0.0, 0.0]), Pos2::new(0.0, 62.5));
        assert_eq!(transforms[1].to_px([0.0, 0.0]), Pos2::new(75.0, 62.5));
    }

    #[test]
    fn the_drawing_keeps_its_floor_only_while_the_list_keeps_a_few_rows() {
        // A list of 300 points in 20-point rows: room for both, then the floor with the list
        // scrolling, then the drawing shrinking below the floor over three rows, then none.
        assert_eq!(drawing_height(1000.0, 300.0, 20.0), 700.0);
        assert_eq!(drawing_height(460.0, 300.0, 20.0), MIN_DRAWING_HEIGHT);
        assert_eq!(drawing_height(220.0, 300.0, 20.0), MIN_DRAWING_HEIGHT);
        assert_eq!(drawing_height(200.0, 300.0, 20.0), 140.0);
        assert_eq!(drawing_height(60.0, 300.0, 20.0), 0.0);
        assert_eq!(drawing_height(0.0, 300.0, 20.0), 0.0);
        // A list shorter than three rows keeps all of it.
        assert_eq!(drawing_height(100.0, 30.0, 20.0), 70.0);
    }

    #[test]
    fn the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling() {
        // The space left above the Equation panel, from roomy to none: the drawing and the
        // callout list end at the region's foot. The drawing keeps MIN_DRAWING_HEIGHT while
        // the list keeps a few rows under it, then shrinks (to scale) below it; the list
        // scrolls in what is left.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let size = egui::vec2(1000.0, 700.0);
        for height in [400.0, 300.0, 200.0, 120.0, 60.0, 10.0, 0.0] {
            let region = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(700.0, height));
            let mut layout = None;
            for _ in 0..2 {
                let (_, used) = crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                    layout = Some(geometry_ui(ui, &inputs, &results, &mut Readouts::default()));
                });
                assert!(
                    used.bottom() <= region.bottom() + 0.01,
                    "{height}: the view runs {} points past the region",
                    used.bottom() - region.bottom()
                );
            }
            let layout = layout.unwrap();
            assert!(layout.rect.bottom() <= region.bottom() + 0.01, "{height}");
            if height >= 300.0 {
                assert!(
                    layout.rect.height() >= MIN_DRAWING_HEIGHT,
                    "{height}: {layout:?}"
                );
            } else if height >= 120.0 {
                // Shorter than the floor, still drawn to one scale.
                assert!(
                    layout.rect.height() < MIN_DRAWING_HEIGHT,
                    "{height}: {layout:?}"
                );
                assert!(
                    layout.scale > 0.0 && layout.end.is_some(),
                    "{height}: {layout:?}"
                );
            }
        }
    }

    #[test]
    fn a_claim_far_past_the_pieces_keeps_the_drawing_to_scale() {
        // A design file's finite but huge claim is left off: the pieces still fill the drawing.
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.metal.max_diameter_mm = 1e300;
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        assert!(layout.end.is_some() && layout.side.is_some());
        let g = geometry(&inputs, &compute_all(&inputs));
        let tall = (g.end.max[1] - g.end.min[1]) as f32 * layout.scale;
        assert!(
            layout.scale > 0.0 && tall > 0.5 * layout.rect.height(),
            "{} points per mm",
            layout.scale
        );
        let note = format!(
            "{NOT_DRAWN}: the diameter claim {} is off the drawing",
            mm(1e300)
        );
        assert_eq!(
            text_color(&output, &note),
            Some(Level::Caution.color(&ctx.style().visuals))
        );
    }
}
