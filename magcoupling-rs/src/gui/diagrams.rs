//! The teaching notes' small diagrams (spec Addendum A4: "an optional small diagram (for
//! example the square-wave magnetization and its harmonics, and the flux path with and without
//! back iron)"), drawn with egui's painter: no image files.
//!
//! [`diagram_shapes`] builds the shapes of one [`Diagram`] kind inside a rect (pure, testable);
//! [`diagram_ui`] allocates [`DIAGRAM_SIZE`] and paints them. Each diagram is a schematic of
//! the idea its note explains, not a plot of the design's numbers: the square wave of fill 0.85
//! and the sum of its harmonics 1, 3 and 5 (amplitudes 4/(nπ) sin(nπλ/2)); the flux closing
//! through steel against spreading behind the magnets, each block lettered N or S; torque
//! against electrical angle, the fundamental alone dashed (its peak at 90°) beside a sum with a
//! strong third harmonic and its peak; the field fringing at a magnet's ends, the length the end
//! factor takes off bracketed at each end; the intrinsic
//! demagnetization curve with its knee and three load lines; first-order heating with the time
//! constant at 63 %, a limit below the steady temperature and the time the rise takes to reach
//! it.

use std::f32::consts::PI;

use egui::text::Fonts;
use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, Vec2, pos2, vec2};

use crate::engine::explain::notes::Diagram;

/// The size a diagram takes [points].
pub const DIAGRAM_SIZE: Vec2 = vec2(300.0, 130.0);

/// The text size of a diagram's labels [points].
const LABEL_SIZE: f32 = 11.0;

/// The fraction of each pole the square wave's blocks fill (λ): a fill where none of the
/// harmonics 1, 3 and 5 vanishes (0.4 and 0.8 zero the fifth, 2/3 the third), so the curve
/// labelled "1 + 3 + 5" sums three; the third (−0.32) visibly pulls the fundamental's 1.24
/// overshoot down at a block's centre.
const FILL: f32 = 0.85;

/// The torque-angle diagram's third harmonic against its fundamental, A3/A1: above 1/9 the
/// summed curve dips at 90° and peaks on either side (correction E7).
const TORQUE_THIRD: f32 = 0.15;

/// The torque-angle diagram's names for its two curves (TC-22), each label in its curve's
/// colour: the fundamental alone, dashed, and the sum with a strong third harmonic, solid.
const FUNDAMENTAL_ALONE: &str = "fundamental alone";
const STRONG_THIRD: &str = "strong third harmonic";

/// The end-fringing diagram's label under the magnet (TC-24), in the colour of the brackets
/// under the magnet's ends: the length the end factor takes off.
const END_LOSS_LABEL: &str = "magnet length L, c_end·τ_p/2 lost per end";

/// The share of the magnet's length each end bracket marks: a schematic share, not the
/// design's (f_end = 1 − c_end·τ_p/L takes c_end·τ_p/2 off each end).
const END_LOSS: f32 = 0.08;

/// The colours a diagram draws with.
struct Pens {
    ink: Color32,
    weak: Color32,
    north: Color32,
    south: Color32,
    steel: Color32,
    /// The curves' accent.
    first: Color32,
    /// The marked points' accent.
    second: Color32,
}

impl Pens {
    fn of(visuals: &egui::Visuals) -> Self {
        Pens {
            ink: visuals.text_color(),
            weak: visuals.weak_text_color(),
            north: Color32::from_rgb(220, 110, 90),
            south: Color32::from_rgb(90, 130, 210),
            steel: Color32::from_gray(120),
            // A teal and a lime, hues no term colour has (`typeset::TERM_PALETTE`, decision
            // M43-3): a diagram sits under the open equation, whose terms those colours key.
            first: Color32::from_rgb(0, 165, 165),
            second: Color32::from_rgb(120, 180, 20),
        }
    }
}

/// `n` points of `f` over `[0, 1]`, mapped into `rect` (y up, `f` from `lo` to `hi`).
fn curve(rect: Rect, lo: f32, hi: f32, n: usize, f: impl Fn(f32) -> f32) -> Vec<Pos2> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let y = (f(t) - lo) / (hi - lo);
            pos2(
                rect.left() + t * rect.width(),
                rect.bottom() - y * rect.height(),
            )
        })
        .collect()
}

/// The points of an elliptic arc around `centre` with radii `r`, from angle `from` to `to`.
fn arc(centre: Pos2, r: Vec2, from: f32, to: f32) -> Vec<Pos2> {
    (0..=16)
        .map(|i| {
            let a = from + (to - from) * i as f32 / 16.0;
            centre + vec2(r.x * a.cos(), -r.y * a.sin())
        })
        .collect()
}

fn label(fonts: &Fonts, at: Pos2, anchor: Align2, text: &str, color: Color32) -> Shape {
    Shape::text(
        fonts,
        at,
        anchor,
        text,
        FontId::proportional(LABEL_SIZE),
        color,
    )
}

/// The shapes of the diagram `kind` inside `rect`.
pub fn diagram_shapes(
    fonts: &Fonts,
    kind: Diagram,
    rect: Rect,
    visuals: &egui::Visuals,
) -> Vec<Shape> {
    let pens = Pens::of(visuals);
    let inner = rect.shrink(12.0);
    match kind {
        Diagram::SquareWaveHarmonics => square_wave(fonts, inner, &pens),
        Diagram::FluxPathBackIron => flux_paths(fonts, inner, &pens),
        Diagram::TorqueAngle => torque_angle(fonts, inner, &pens),
        Diagram::EndFringing => end_fringing(fonts, inner, &pens),
        Diagram::DemagKnee => demag_knee(fonts, inner, &pens),
        Diagram::HeatingCurve => heating_curve(fonts, inner, &pens),
    }
}

/// The magnetization of two pole pairs at fill [`FILL`] (blocks and spaces), the fundamental and
/// the sum of harmonics 1, 3 and 5.
fn square_wave(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let poles = 4.0;
    // +1 over the north blocks, -1 over the south ones, 0 in the gaps.
    let wave = |t: f32| {
        let x = t * poles;
        let pole = x.floor();
        let within = (x - pole - 0.5).abs() * 2.0;
        if within <= FILL {
            if pole as i32 % 2 == 0 { 1.0 } else { -1.0 }
        } else {
            0.0
        }
    };
    let harmonic = |n: f32, t: f32| {
        let amp = 4.0 / (n * PI) * (n * PI * FILL / 2.0).sin();
        // Pole centres at x = 0.5, 1.5, ... (in poles): cos(n π (x - 0.5)).
        amp * (n * PI * (t * poles - 0.5)).cos()
    };
    let plot = Rect::from_min_max(r.min, pos2(r.max.x, r.max.y - 14.0));
    let axis = plot.center().y;
    let mut shapes = vec![Shape::line_segment(
        [pos2(plot.left(), axis), pos2(plot.right(), axis)],
        Stroke::new(1.0, pens.weak),
    )];
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, wave),
        Stroke::new(1.5, pens.ink),
    ));
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, |t| harmonic(1.0, t)),
        Stroke::new(1.0, pens.second),
    ));
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, |t| {
            harmonic(1.0, t) + harmonic(3.0, t) + harmonic(5.0, t)
        }),
        Stroke::new(1.5, pens.first),
    ));
    shapes.push(label(
        fonts,
        pos2(r.left(), r.bottom()),
        Align2::LEFT_BOTTOM,
        &format!("blocks and spaces (fill {FILL})"),
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        pos2(r.center().x + 10.0, r.bottom()),
        Align2::LEFT_BOTTOM,
        "n = 1",
        pens.second,
    ));
    shapes.push(label(
        fonts,
        pos2(r.right(), r.bottom()),
        Align2::RIGHT_BOTTOM,
        "1 + 3 + 5",
        pens.first,
    ));
    shapes
}

/// One panel of the flux-path diagram: a ring of four blocks above the gap and one below,
/// with steel behind both (`steel`) or not, each block lettered N or S, and two flux loops.
fn flux_panel(fonts: &Fonts, r: Rect, steel: bool, pens: &Pens, title: &str) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let block_w = r.width() / 4.0;
    let block_h = 12.0;
    let gap = 14.0;
    let mid = r.center().y + 4.0;
    let top = mid - gap / 2.0 - block_h;
    let bottom = mid + gap / 2.0;
    for i in 0..4 {
        let x = r.left() + i as f32 * block_w;
        let (upper, lower) = if i % 2 == 0 {
            (pens.north, pens.south)
        } else {
            (pens.south, pens.north)
        };
        for (y, color) in [(top, upper), (bottom, lower)] {
            let block = Rect::from_min_size(pos2(x + 1.0, y), vec2(block_w - 2.0, block_h));
            shapes.push(Shape::rect_filled(block, 1.0, color));
            // TC-24: each block names its pole, so the two colours cannot read as the inner
            // and the outer ring.
            let pole = if color == pens.north { "N" } else { "S" };
            shapes.push(label(
                fonts,
                block.center(),
                Align2::CENTER_CENTER,
                pole,
                Color32::WHITE,
            ));
        }
    }
    if steel {
        for y in [top - 8.0, bottom + block_h] {
            shapes.push(Shape::rect_filled(
                Rect::from_min_size(pos2(r.left(), y), vec2(r.width(), 8.0)),
                1.0,
                pens.steel,
            ));
        }
    }
    // Two loops, each across the gap between neighbouring blocks: through the steel they stay
    // inside it; in free space they bulge out behind the magnets.
    for i in [0.5f32, 2.5] {
        let cx = r.left() + (i + 0.5) * block_w;
        let (rx, ry) = if steel {
            (block_w * 0.55, gap / 2.0 + block_h + 4.0)
        } else {
            (block_w * 0.9, gap / 2.0 + block_h + 28.0)
        };
        shapes.push(Shape::closed_line(
            arc(pos2(cx, mid), vec2(rx, ry), 0.0, 2.0 * PI),
            Stroke::new(1.0, pens.first),
        ));
    }
    shapes.push(label(
        fonts,
        pos2(r.center().x, r.top()),
        Align2::CENTER_TOP,
        title,
        pens.ink,
    ));
    shapes
}

fn flux_paths(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let half = r.width() / 2.0 - 8.0;
    let left = Rect::from_min_size(r.min, vec2(half, r.height()));
    let right = Rect::from_min_size(pos2(r.right() - half, r.top()), vec2(half, r.height()));
    let mut shapes = flux_panel(fonts, left, true, pens, "with back iron");
    shapes.extend(flux_panel(fonts, right, false, pens, "free space"));
    shapes
}

/// Torque against electrical angle over 0 to π (TC-22): the fundamental alone, dashed, its peak
/// at 90° (half a pole pitch, the tick "90°"), and the sum with a strong third harmonic, solid,
/// its peak marked.
fn torque_angle(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left() + 14.0, r.top()),
        pos2(r.right(), r.bottom() - 14.0),
    );
    let torque = |t: f32| (t * PI).sin() + TORQUE_THIRD * (3.0 * t * PI).sin();
    let points = curve(plot, 0.0, 1.2, 181, torque);
    let fundamental = curve(plot, 0.0, 1.2, 181, |t| (t * PI).sin());
    // The fundamental's top: its middle sample, t = 0.5.
    let fundamental_top = fundamental[90];
    let peak = points
        .iter()
        .copied()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .unwrap_or(plot.center());
    let half_pitch = pos2(plot.center().x, plot.bottom());
    let mut shapes = vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [half_pitch, half_pitch - vec2(0.0, 4.0)],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(points, Stroke::new(1.5, pens.first)),
        Shape::circle_filled(peak, 3.5, pens.second),
        label(
            fonts,
            peak - vec2(6.0, 0.0),
            Align2::RIGHT_CENTER,
            "pull-out",
            pens.second,
        ),
        label(
            fonts,
            fundamental_top - vec2(0.0, 2.0),
            Align2::CENTER_BOTTOM,
            FUNDAMENTAL_ALONE,
            pens.ink,
        ),
        label(
            fonts,
            plot.left_bottom(),
            Align2::LEFT_TOP,
            STRONG_THIRD,
            pens.first,
        ),
        label(fonts, half_pitch, Align2::CENTER_TOP, "90°", pens.weak),
        label(
            fonts,
            plot.right_bottom(),
            Align2::RIGHT_TOP,
            "φ: 0 to 180°",
            pens.ink,
        ),
        label(fonts, plot.left_top(), Align2::RIGHT_TOP, "T", pens.ink),
    ];
    shapes.extend(Shape::dashed_line(
        &fundamental,
        Stroke::new(1.0, pens.ink),
        4.0,
        3.0,
    ));
    shapes
}

/// A magnet block with straight field lines over its middle and lines bulging out at its ends,
/// the length the end factor takes off bracketed under each end.
fn end_fringing(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let block = Rect::from_center_size(
        pos2(r.center().x, r.bottom() - 22.0),
        vec2(r.width() * 0.6, 14.0),
    );
    let mut shapes = vec![Shape::rect_filled(block, 1.0, pens.north)];
    let lines = 7;
    for i in 0..lines {
        let t = i as f32 / (lines - 1) as f32;
        let x = block.left() + t * block.width();
        // The outer lines lean out: the more, the nearer the end.
        let lean = (t - 0.5) * 2.0;
        let bulge = lean * lean.abs() * 40.0;
        let points: Vec<Pos2> = (0..=10)
            .map(|k| {
                let s = k as f32 / 10.0;
                pos2(
                    x + bulge * s * s,
                    block.top() - s * (block.top() - r.top() - 4.0),
                )
            })
            .collect();
        shapes.push(Shape::line(points, Stroke::new(1.0, pens.first)));
    }
    // TC-24: the length the end factor takes off, c_end·τ_p/2 at each end, bracketed under the
    // magnet's ends in the colour of the label that names it.
    let lost = END_LOSS * block.width();
    let y = block.bottom() + 2.0;
    for (from, to) in [
        (block.left(), block.left() + lost),
        (block.right() - lost, block.right()),
    ] {
        shapes.push(Shape::line_segment(
            [pos2(from, y), pos2(to, y)],
            Stroke::new(2.0, pens.second),
        ));
    }
    shapes.push(label(
        fonts,
        pos2(block.center().x, block.bottom() + 4.0),
        Align2::CENTER_TOP,
        END_LOSS_LABEL,
        pens.second,
    ));
    shapes.push(label(
        fonts,
        pos2(r.right(), r.top()),
        Align2::RIGHT_TOP,
        "fringing at the ends",
        pens.second,
    ));
    shapes
}

/// The intrinsic curve J(H) in the second quadrant with its knee, and three load lines.
fn demag_knee(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left(), r.top() + 4.0),
        pos2(r.right() - 14.0, r.bottom() - 14.0),
    );
    // t = 0 at H = -Hcj (left), t = 1 at H = 0 (right); J flat near Br, falling past the knee.
    let knee_t = 0.1;
    let intrinsic = |t: f32| {
        if t >= knee_t {
            1.0 - 0.05 * (1.0 - t)
        } else {
            let s = t / knee_t;
            (1.0 - 0.05 * (1.0 - knee_t)) * s.sqrt()
        }
    };
    let points = curve(plot, 0.0, 1.1, 201, intrinsic);
    let knee = curve(plot, 0.0, 1.1, 2, |_| intrinsic(knee_t))[0];
    let knee = pos2(plot.left() + knee_t * plot.width(), knee.y);
    let origin = plot.right_bottom();
    let mut shapes = vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.right_bottom(), plot.right_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(points, Stroke::new(1.5, pens.first)),
        Shape::circle_filled(knee, 3.5, pens.second),
        label(
            fonts,
            knee + vec2(6.0, -4.0),
            Align2::LEFT_BOTTOM,
            "knee",
            pens.second,
        ),
    ];
    // Load lines from the origin: steeper is a higher permeance coefficient.
    for slope in [0.4f32, 0.8, 1.6] {
        let dx = (plot.height() / slope).min(plot.width());
        let end = pos2(origin.x - dx, origin.y - dx * slope);
        shapes.extend(Shape::dashed_line(
            &[origin, end],
            Stroke::new(1.0, pens.weak),
            4.0,
            3.0,
        ));
    }
    shapes.push(label(
        fonts,
        plot.left_bottom(),
        Align2::LEFT_TOP,
        "H (reverse field)",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        plot.right_top(),
        Align2::LEFT_TOP,
        "J",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        pos2(plot.center().x, plot.bottom() - 4.0),
        Align2::CENTER_BOTTOM,
        "load lines",
        pens.weak,
    ));
    shapes
}

/// The heating curve's temperature limit as a share of its steady rise: below the steady
/// temperature, so the rise crosses it (the case "time to the limit" names), and late enough
/// (2.66 time constants) that the crossing's guide clears the τ_th label.
const LIMIT_SHARE: f32 = 0.93;

/// First-order heating toward a steady temperature over five time constants: τ_th at 63 %, a
/// temperature limit below the steady level, and the time the rise takes to reach it.
fn heating_curve(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left() + 14.0, r.top() + 4.0),
        pos2(r.right(), r.bottom() - 14.0),
    );
    let span = 5.0;
    let rise = |t: f32| 1.0 - (-t * span).exp();
    let tau = pos2(
        plot.left() + plot.width() / span,
        plot.bottom() - rise(1.0 / span) / 1.1 * plot.height(),
    );
    let steady = plot.bottom() - plot.height() / 1.1;
    let limit = plot.bottom() - LIMIT_SHARE / 1.1 * plot.height();
    // The rise reaches LIMIT_SHARE of the steady rise after -ln(1 - LIMIT_SHARE) time
    // constants, five of which span the plot.
    let reach = plot.left() - (1.0 - LIMIT_SHARE).ln() / span * plot.width();
    let mut shapes = vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(
            curve(plot, 0.0, 1.1, 121, rise),
            Stroke::new(1.5, pens.first),
        ),
        Shape::circle_filled(tau, 3.5, pens.second),
        label(
            fonts,
            tau + vec2(6.0, 2.0),
            Align2::LEFT_TOP,
            "τ_th: 63 %",
            pens.second,
        ),
        label(
            fonts,
            plot.right_bottom(),
            Align2::RIGHT_TOP,
            "time slipping",
            pens.ink,
        ),
    ];
    shapes.extend(Shape::dashed_line(
        &[pos2(plot.left(), steady), pos2(plot.right(), steady)],
        Stroke::new(1.0, pens.weak),
        4.0,
        3.0,
    ));
    shapes.push(label(
        fonts,
        pos2(plot.right(), steady - 2.0),
        Align2::RIGHT_BOTTOM,
        "steady temperature",
        pens.weak,
    ));
    shapes.extend(Shape::dashed_line(
        &[pos2(plot.left(), limit), pos2(plot.right(), limit)],
        Stroke::new(1.0, pens.north),
        4.0,
        3.0,
    ));
    shapes.push(label(
        fonts,
        pos2(plot.left() + 3.0, limit + 2.0),
        Align2::LEFT_TOP,
        "limit",
        pens.north,
    ));
    shapes.extend(Shape::dashed_line(
        &[pos2(reach, limit), pos2(reach, plot.bottom())],
        Stroke::new(1.0, pens.weak),
        3.0,
        3.0,
    ));
    shapes.push(label(
        fonts,
        pos2(reach, plot.bottom() + 1.0),
        Align2::CENTER_TOP,
        "time to the limit",
        pens.weak,
    ));
    shapes
}

/// Allocates [`DIAGRAM_SIZE`] and paints the diagram `kind` there.
pub fn diagram_ui(ui: &mut egui::Ui, kind: Diagram) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(DIAGRAM_SIZE, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let shapes = ui.fonts(|f| diagram_shapes(f, kind, rect, ui.visuals()));
    painter.extend(shapes);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::typeset::TERM_PALETTE;

    const KINDS: [Diagram; 6] = [
        Diagram::SquareWaveHarmonics,
        Diagram::FluxPathBackIron,
        Diagram::TorqueAngle,
        Diagram::EndFringing,
        Diagram::DemagKnee,
        Diagram::HeatingCurve,
    ];

    /// Every point a shape draws (its path, its circle's centre, its rect's corners, its
    /// text's rect corners).
    fn points(shape: &Shape) -> Vec<Pos2> {
        match shape {
            Shape::Path(p) => p.points.clone(),
            Shape::LineSegment { points, .. } => points.to_vec(),
            Shape::Circle(c) => vec![c.center],
            Shape::Rect(r) => vec![r.rect.min, r.rect.max],
            Shape::Text(t) => {
                let rect = t.galley.rect.translate(t.pos.to_vec2());
                vec![rect.min, rect.max]
            }
            Shape::Vec(v) => v.iter().flat_map(points).collect(),
            _ => Vec::new(),
        }
    }

    fn shapes_of(kind: Diagram, rect: Rect) -> (Vec<Shape>, Vec<String>) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let visuals = egui::Visuals::dark();
        let shapes = ctx.fonts(|f| diagram_shapes(f, kind, rect, &visuals));
        let texts: Vec<String> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Text(t) => Some(t.galley.text().to_owned()),
                _ => None,
            })
            .collect();
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&ctx, text, &format!("{kind:?}"));
        }
        (shapes, texts)
    }

    #[test]
    fn every_diagram_draws_inside_its_rect_with_finite_points_and_labels() {
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let mut signatures = Vec::new();
        for kind in KINDS {
            let (shapes, texts) = shapes_of(kind, rect);
            assert!(shapes.len() >= 5, "{kind:?}: {} shapes", shapes.len());
            assert!(!texts.is_empty(), "{kind:?} is labelled");
            for shape in &shapes {
                for p in points(shape) {
                    assert!(p.x.is_finite() && p.y.is_finite(), "{kind:?}: {p:?}");
                    assert!(
                        rect.expand(1.0).contains(p),
                        "{kind:?}: {p:?} outside {rect:?}"
                    );
                }
            }
            signatures.push(texts);
        }
        // Six diagrams, six sets of labels.
        for (i, a) in signatures.iter().enumerate() {
            for b in &signatures[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn no_two_labels_of_a_diagram_overlap() {
        // Each label reads on its own: the torque-angle diagram's curve names, its 90° tick and
        // its pull-out mark, and the flux diagram's pole letters, share their rect with no
        // other label.
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        for kind in KINDS {
            let (shapes, _) = shapes_of(kind, rect);
            let labels: Vec<(String, Rect)> = shapes
                .iter()
                .filter_map(|s| match s {
                    Shape::Text(t) => Some((
                        t.galley.text().to_owned(),
                        t.galley.rect.translate(t.pos.to_vec2()),
                    )),
                    _ => None,
                })
                .collect();
            for (i, (a, a_rect)) in labels.iter().enumerate() {
                for (b, b_rect) in &labels[i + 1..] {
                    assert!(
                        !a_rect.intersects(*b_rect),
                        "{kind:?}: {a:?} {a_rect:?} overlaps {b:?} {b_rect:?}"
                    );
                }
            }
        }
    }

    /// Every colour a shape paints: its strokes, its fills and its text.
    fn colors(shape: &Shape) -> Vec<Color32> {
        match shape {
            Shape::Path(p) => match &p.stroke.color {
                egui::epaint::ColorMode::Solid(c) => vec![p.fill, *c],
                egui::epaint::ColorMode::UV(_) => vec![p.fill],
            },
            Shape::LineSegment { stroke, .. } => vec![stroke.color],
            Shape::Circle(c) => vec![c.fill, c.stroke.color],
            Shape::Rect(r) => vec![r.fill, r.stroke.color],
            Shape::Text(t) => t
                .galley
                .job
                .sections
                .iter()
                .map(|s| s.format.color)
                .chain([t.fallback_color])
                .chain(t.override_text_color)
                .collect(),
            Shape::Vec(v) => v.iter().flat_map(colors).collect(),
            _ => Vec::new(),
        }
    }

    /// The distance between two colours in RGB.
    fn rgb_distance(a: Color32, b: Color32) -> f32 {
        let d = |x: u8, y: u8| f32::from(x) - f32::from(y);
        (d(a.r(), b.r()).powi(2) + d(a.g(), b.g()).powi(2) + d(a.b(), b.b()).powi(2)).sqrt()
    }

    #[test]
    fn the_diagrams_paint_in_no_term_colour() {
        // Decision M43-3: a term colour on screen is read against the open equation's key, and
        // a diagram sits under that equation, above its term list.
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        for kind in KINDS {
            let (shapes, _) = shapes_of(kind, rect);
            for color in shapes.iter().flat_map(colors) {
                assert!(
                    !TERM_PALETTE.contains(&color),
                    "{kind:?} paints the term colour {color:?}"
                );
            }
        }
        // Nor one a reader could take for a term colour: the accents stand well apart.
        let pens = Pens::of(&egui::Visuals::dark());
        for accent in [pens.first, pens.second] {
            for term in TERM_PALETTE {
                let d = rgb_distance(accent, term);
                assert!(d >= 80.0, "{accent:?} is {d} from the term colour {term:?}");
            }
        }
    }

    /// The points of the one path stroked in `color`.
    fn path_in(shapes: &[Shape], color: Color32) -> &[Pos2] {
        let paths: Vec<&[Pos2]> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Path(p)
                    if matches!(p.stroke.color, egui::epaint::ColorMode::Solid(c) if c == color) =>
                {
                    Some(&p.points[..])
                }
                _ => None,
            })
            .collect();
        assert_eq!(paths.len(), 1, "paths stroked in {color:?}");
        paths[0]
    }

    /// The points of the diagram's one path (its curve).
    fn only_path(shapes: &[Shape]) -> &[Pos2] {
        let paths: Vec<&[Pos2]> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Path(p) => Some(&p.points[..]),
                _ => None,
            })
            .collect();
        assert_eq!(paths.len(), 1, "one curve");
        paths[0]
    }

    /// The centre of the diagram's one marked point.
    fn only_marker(shapes: &[Shape]) -> Pos2 {
        let marks: Vec<Pos2> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Circle(c) => Some(c.center),
                _ => None,
            })
            .collect();
        assert_eq!(marks.len(), 1, "one marked point");
        marks[0]
    }

    /// The y of every horizontal line segment (an axis, a dashed level's dashes).
    fn horizontal_lines(shapes: &[Shape]) -> Vec<f32> {
        shapes
            .iter()
            .filter_map(|s| match s {
                Shape::LineSegment { points: [a, b], .. } if a.y == b.y => Some(a.y),
                _ => None,
            })
            .collect()
    }

    /// The colour of the label `text`.
    fn label_color(shapes: &[Shape], text: &str) -> Color32 {
        shapes
            .iter()
            .find_map(|s| match s {
                Shape::Text(t) if t.galley.text() == text => Some(t.fallback_color),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no label {text:?}"))
    }

    /// The y of `curve` at `x`, linear between its samples.
    fn y_at(curve: &[Pos2], x: f32) -> f32 {
        let i = curve
            .windows(2)
            .position(|w| w[0].x <= x && x <= w[1].x)
            .unwrap_or_else(|| panic!("{x} is outside the curve"));
        let (a, b) = (curve[i], curve[i + 1]);
        a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)
    }

    #[test]
    fn the_square_wave_paints_the_sum_of_its_harmonics_at_its_fill() {
        // Harmonic n of a square wave of fill λ has the amplitude 4/(nπ) sin(nπλ/2) (the
        // harmonics note; engine::model::harmonic_amplitude), +1 over a north block's centre.
        let amp = |n: f32| 4.0 / (n * PI) * (n * PI * FILL / 2.0).sin();
        let (a1, a3, a5) = (amp(1.0), amp(3.0), amp(5.0));
        // The curve labelled "1 + 3 + 5" sums three harmonics a reader can see: no fill that
        // zeroes one (0.4 or 0.8 the fifth, 2/3 the third).
        assert!(a3.abs() >= 0.2, "the third: {a3}");
        assert!(a5.abs() >= 0.05, "the fifth: {a5}");
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::SquareWaveHarmonics, rect);
        // Each curve found by its label's colour, as a reader matches them.
        let wave_label = format!("blocks and spaces (fill {FILL})");
        let wave = path_in(&shapes, label_color(&shapes, &wave_label));
        let fundamental = path_in(&shapes, label_color(&shapes, "n = 1"));
        let sum = path_in(&shapes, label_color(&shapes, "1 + 3 + 5"));
        assert_eq!((wave.len(), fundamental.len()), (sum.len(), sum.len()));
        // The painted wave sets the scale: its gaps are 0, its north blocks +1 (the topmost).
        let zero = wave[0].y;
        let plateau = wave.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let value = |p: Pos2| (zero - p.y) / (zero - plateau);
        let levels: Vec<f32> = wave.iter().map(|&p| value(p)).collect();
        for level in &levels {
            assert!(
                [-1.0, 0.0, 1.0].iter().any(|l| (level - l).abs() < 1e-4),
                "{level}"
            );
        }
        // The blocks fill the fraction the label says (60 samples a pole).
        let filled = levels.iter().filter(|l| l.abs() > 0.5).count() as f32 / levels.len() as f32;
        assert!((filled - FILL).abs() < 0.03, "{filled}");
        // Each block's and each gap's centre: the middle of a run of one level.
        let mut runs: Vec<(f32, usize)> = Vec::new();
        let mut start = 0;
        for i in 1..=levels.len() {
            if i == levels.len() || (levels[i] - levels[start]).abs() > 0.5 {
                if start > 0 && i < levels.len() {
                    runs.push((levels[start].round(), (start + i - 1) / 2));
                }
                start = i;
            }
        }
        let blocks = runs.iter().filter(|(l, _)| *l != 0.0).count();
        assert_eq!(blocks, 4, "two pole pairs: {runs:?}");
        for &(level, i) in &runs {
            // ±(a1 + a3 + a5) over a north or a south block's centre; 0 between blocks, where
            // every odd harmonic crosses zero.
            let (want_sum, want_first) = (level * (a1 + a3 + a5), level * a1);
            let (got_sum, got_first) = (value(sum[i]), value(fundamental[i]));
            assert!(
                (got_sum - want_sum).abs() < 1e-3,
                "sum at {i}: {got_sum} != {want_sum}"
            );
            assert!(
                (got_first - want_first).abs() < 1e-3,
                "fundamental at {i}: {got_first} != {want_first}"
            );
            if level != 0.0 {
                // Near the wave's plateau, and visibly off the fundamental alone.
                assert!((got_sum - level).abs() < 0.25, "{got_sum}");
                assert!((got_sum - got_first).abs() >= 0.2, "{got_sum} {got_first}");
            }
        }
    }

    #[test]
    fn the_heating_curve_marks_tau_at_63_percent_of_the_steady_rise() {
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::HeatingCurve, rect);
        let curve = only_path(&shapes);
        // The rise starts at the time axis's origin (t = 0, no rise).
        let (left, right, bottom) = (curve[0].x, curve[curve.len() - 1].x, curve[0].y);
        let lines = horizontal_lines(&shapes);
        assert!(lines.contains(&bottom), "the time axis: {lines:?}");
        // The steady temperature: the topmost dashed level above the axis (the limit, the
        // other level, lies below it).
        let steady = lines
            .iter()
            .copied()
            .filter(|y| *y < bottom - 1.0)
            .reduce(f32::min)
            .expect("the steady temperature");
        let rise = |y: f32| (bottom - y) / (bottom - steady);
        let tau = only_marker(&shapes);
        assert!(
            ((tau.x - left) / (right - left) - 0.2).abs() < 1e-4,
            "{tau:?}"
        );
        let e = std::f32::consts::E;
        assert!(
            (rise(tau.y) - (1.0 - 1.0 / e)).abs() < 1e-4,
            "{}",
            rise(tau.y)
        );
        // On the curve, which then reaches 95 % at 3τ (the note) and nears the steady level.
        assert!((y_at(curve, tau.x) - tau.y).abs() < 0.05);
        let three_tau = left + 3.0 * (tau.x - left);
        let at_three = rise(y_at(curve, three_tau));
        assert!((at_three - (1.0 - e.powi(-3))).abs() < 1e-3, "{at_three}");
        assert!(rise(curve[curve.len() - 1].y) > 0.99);
    }

    #[test]
    fn the_heating_curve_marks_the_limit_and_the_time_to_reach_it() {
        // The thermal note's "time to the limit" (TC-24): how long unbroken slip takes to heat
        // the coupling to its limit, a level below the steady temperature that the rise crosses.
        // Five labels, the time constant's named as the panel names it; no two overlap
        // (`no_two_labels_of_a_diagram_overlap`).
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, texts) = shapes_of(Diagram::HeatingCurve, rect);
        assert_eq!(
            texts,
            [
                "\u{3c4}_th: 63 %",
                "time slipping",
                "steady temperature",
                "limit",
                "time to the limit",
            ]
        );
        let curve = only_path(&shapes);
        let (left, right, bottom) = (curve[0].x, curve[curve.len() - 1].x, curve[0].y);
        let levels: Vec<f32> = horizontal_lines(&shapes)
            .into_iter()
            .filter(|y| *y < bottom - 1.0)
            .collect();
        let steady = levels
            .iter()
            .copied()
            .reduce(f32::min)
            .expect("the steady level");
        let limit = levels.iter().copied().reduce(f32::max).expect("the limit");
        assert!(
            limit > steady + 1.0,
            "a limit below the steady level: {levels:?}"
        );
        let rise = |y: f32| (bottom - y) / (bottom - steady);
        assert!((rise(limit) - LIMIT_SHARE).abs() < 1e-4, "{}", rise(limit));
        // One dashed guide from the limit down to the time axis, where the curve reaches the
        // limit: rise = 1 - e^(-t/τ_th), five time constants across the plot.
        let guides: Vec<[Pos2; 2]> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::LineSegment { points: [a, b], .. }
                    if a.x == b.x && a.y != b.y && (a.x - left).abs() > 1.0 =>
                {
                    Some([*a, *b])
                }
                _ => None,
            })
            .collect();
        assert!(!guides.is_empty(), "the time-to-the-limit guide");
        let x = guides[0][0].x;
        assert!(
            guides.iter().all(|[a, _]| (a.x - x).abs() < 1e-3),
            "{guides:?}"
        );
        let ys = || guides.iter().flat_map(|g| g.iter().map(|p| p.y));
        let top = ys().fold(f32::INFINITY, f32::min);
        let foot = ys().fold(f32::NEG_INFINITY, f32::max);
        assert!((top - limit).abs() < 1e-3, "{top} {limit}");
        assert!(bottom - foot <= 6.5, "{foot} {bottom}");
        let time_constants = (x - left) / (right - left) * 5.0;
        assert!(
            (time_constants + (1.0 - LIMIT_SHARE).ln()).abs() < 1e-3,
            "{time_constants}"
        );
        assert!(
            (y_at(curve, x) - limit).abs() < 0.5,
            "{} {limit}",
            y_at(curve, x)
        );
    }

    #[test]
    fn the_demag_knee_sits_at_0_9_hcj_on_the_intrinsic_curve() {
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::DemagKnee, rect);
        let curve = only_path(&shapes);
        // H runs from −Hcj at the left to 0 at the right, along the H axis (the one horizontal
        // line: the load lines slope).
        let (left, right) = (curve[0].x, curve[curve.len() - 1].x);
        let [axis] = horizontal_lines(&shapes)[..] else {
            panic!("one H axis")
        };
        let j = |y: f32| axis - y;
        // Hcj is where J reaches zero.
        assert!(j(curve[0].y).abs() < 1e-3, "{}", j(curve[0].y));
        // The knee: H_k = 0.9 Hcj (the knee-fraction input's default), on the curve.
        let knee = only_marker(&shapes);
        let h = |x: f32| -(right - x) / (right - left);
        assert!((h(knee.x) + 0.9).abs() < 1e-4, "{}", h(knee.x));
        assert!((y_at(curve, knee.x) - knee.y).abs() < 0.05);
        // Flat near Br from the knee to H = 0; falling steeply past it toward −Hcj.
        let br = j(curve[curve.len() - 1].y);
        assert!(j(knee.y) >= 0.9 * br, "{} {br}", j(knee.y));
        let midway = j(y_at(curve, (left + knee.x) / 2.0));
        assert!(midway <= 0.8 * j(knee.y), "{midway}");
    }

    #[test]
    fn the_torque_peak_leaves_90_degrees_for_a_strong_third_harmonic() {
        // T = sin φ + k sin 3φ: while k ≤ 1/9 the peak is at 90°; above it 90° is a dip and the
        // peaks sit at cos²φ = (9k − 1)/(12k), so the calculator searches the summed curve for
        // its highest point (correction E7, the pull-out angle note).
        let k = TORQUE_THIRD;
        assert!(k > 1.0 / 9.0, "{k}");
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::TorqueAngle, rect);
        let curve = only_path(&shapes);
        let (left, right) = (curve[0].x, curve[curve.len() - 1].x);
        let degrees = |x: f32| (x - left) / (right - left) * 180.0;
        let peak = only_marker(&shapes);
        // The marker is the curve's highest point (the least y).
        let top = curve.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        assert!((peak.y - top).abs() < 1e-3, "{peak:?} {top}");
        let want = ((9.0 * k - 1.0) / (12.0 * k)).sqrt().acos().to_degrees();
        let got = degrees(peak.x);
        assert!(
            (got - want).abs() < 1.0 || (got - (180.0 - want)).abs() < 1.0,
            "peak at {got}°, want {want}° or {}°",
            180.0 - want
        );
        assert!((got - 90.0).abs() > 15.0, "{got}");
        // The dip at 90°: less torque (a lower point) than the peak.
        let middle = left + (right - left) / 2.0;
        assert!(
            y_at(curve, middle) > peak.y + 0.5,
            "{}",
            y_at(curve, middle)
        );
    }

    #[test]
    fn the_torque_angle_diagram_dashes_the_fundamental_alone_peaking_at_90_degrees() {
        // TC-22: the summed curve with a strong third harmonic is the unusual case (the test
        // above keeps its E7 peak); beside it the fundamental alone, dashed, peaks at 90°, half
        // a pole pitch, where a tick and its label sit. Each curve is named in its colour.
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::TorqueAngle, rect);
        let summed = path_in(&shapes, label_color(&shapes, STRONG_THIRD));
        let (left, right) = (summed[0].x, summed[summed.len() - 1].x);
        let degrees = |x: f32| (x - left) / (right - left) * 180.0;
        let dashed = label_color(&shapes, FUNDAMENTAL_ALONE);
        let dashes: Vec<Pos2> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::LineSegment { points, stroke } if stroke.color == dashed => Some(*points),
                _ => None,
            })
            .flatten()
            .collect();
        assert!(
            dashes.len() >= 40,
            "a dashed curve: {} points",
            dashes.len()
        );
        let top = dashes
            .iter()
            .copied()
            .min_by(|a, b| a.y.total_cmp(&b.y))
            .expect("the dashes");
        assert!(
            (degrees(top.x) - 90.0).abs() < 5.0,
            "peak at {}°",
            degrees(top.x)
        );
        // It rises from the axis at 0° and comes back to it near 180°, as the summed curve does.
        let axis = summed[0].y;
        assert!(
            dashes
                .iter()
                .any(|p| degrees(p.x).abs() < 1e-3 && (p.y - axis).abs() < 1e-3)
        );
        assert!(dashes.iter().any(|p| (degrees(p.x) - 180.0).abs() < 5.0));
        assert!(dashes.iter().all(|p| p.y <= axis + 1e-3), "above the axis");
        // The 90° label sits under 90°.
        let tick = shapes
            .iter()
            .find_map(|s| match s {
                Shape::Text(t) if t.galley.text() == "90°" => {
                    Some(t.galley.rect.translate(t.pos.to_vec2()))
                }
                _ => None,
            })
            .expect("the 90° label");
        assert!(
            (degrees(tick.center().x) - 90.0).abs() < 0.5,
            "{}",
            degrees(tick.center().x)
        );
    }

    #[test]
    fn the_flux_path_blocks_are_lettered_n_and_s() {
        // TC-24: red and blue alone read as the inner and the outer ring; each block carries
        // its pole's letter, centred on it.
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::FluxPathBackIron, rect);
        let pens = Pens::of(&egui::Visuals::dark());
        for (fill, letter) in [(pens.north, "N"), (pens.south, "S")] {
            let blocks: Vec<Rect> = shapes
                .iter()
                .filter_map(|s| match s {
                    Shape::Rect(r) if r.fill == fill => Some(r.rect),
                    _ => None,
                })
                .collect();
            let letters: Vec<Pos2> = shapes
                .iter()
                .filter_map(|s| match s {
                    Shape::Text(t) if t.galley.text() == letter => {
                        Some(t.galley.rect.translate(t.pos.to_vec2()).center())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(blocks.len(), 8, "{letter}: four blocks a panel, two panels");
            assert_eq!(letters.len(), blocks.len(), "{letter}");
            for block in &blocks {
                assert_eq!(
                    letters.iter().filter(|c| block.contains(**c)).count(),
                    1,
                    "{letter} on {block:?}"
                );
            }
        }
    }

    #[test]
    fn the_end_fringing_brackets_the_length_lost_at_each_end() {
        // TC-24: f_end = 1 − c_end·τ_p/L takes c_end·τ_p/2 off each end of the magnet; a bracket
        // under each end marks that share, in the colour of the label that names it.
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let (shapes, _) = shapes_of(Diagram::EndFringing, rect);
        let pens = Pens::of(&egui::Visuals::dark());
        let block = shapes
            .iter()
            .find_map(|s| match s {
                Shape::Rect(r) if r.fill == pens.north => Some(r.rect),
                _ => None,
            })
            .expect("the magnet");
        let color = label_color(&shapes, END_LOSS_LABEL);
        let mut brackets: Vec<(f32, f32, f32, f32)> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::LineSegment {
                    points: [a, b],
                    stroke,
                } if stroke.color == color => Some((a.x.min(b.x), a.x.max(b.x), a.y, b.y)),
                _ => None,
            })
            .collect();
        brackets.sort_by(|p, q| p.0.total_cmp(&q.0));
        let [(l0, l1, ly0, ly1), (r0, r1, ry0, ry1)] = brackets[..] else {
            panic!("two brackets: {brackets:?}")
        };
        let lost = END_LOSS * block.width();
        assert!((l0 - block.left()).abs() < 1e-3 && (l1 - l0 - lost).abs() < 1e-3);
        assert!((r1 - block.right()).abs() < 1e-3 && (r1 - r0 - lost).abs() < 1e-3);
        for y in [ly1, ry0, ry1] {
            assert_eq!(y, ly0, "level, at one height");
        }
        assert!(ly0 > block.bottom(), "under the magnet: {ly0} {block:?}");
    }
}
