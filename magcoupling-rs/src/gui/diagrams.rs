//! The teaching notes' small diagrams (spec Addendum A4: "an optional small diagram (for
//! example the square-wave magnetization and its harmonics, and the flux path with and without
//! back iron)"), drawn with egui's painter: no image files.
//!
//! [`diagram_shapes`] builds the shapes of one [`Diagram`] kind inside a rect (pure, testable);
//! [`diagram_ui`] allocates [`DIAGRAM_SIZE`] and paints them. Each diagram is a schematic of
//! the idea its note explains, not a plot of the design's numbers: the square wave of fill 0.8
//! and the sum of its harmonics 1, 3 and 5 (amplitudes 4/(nπ) sin(nπλ/2)); the flux closing
//! through steel against spreading behind the magnets; torque against electrical angle with
//! a third harmonic and its peak; the field fringing at a magnet's ends; the intrinsic
//! demagnetization curve with its knee and three load lines; first-order heating with the time
//! constant at 63 %.

use std::f32::consts::PI;

use egui::text::Fonts;
use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, Vec2, pos2, vec2};

use crate::engine::explain::notes::Diagram;
use crate::gui::typeset::TERM_PALETTE;

/// The size a diagram takes [points].
pub const DIAGRAM_SIZE: Vec2 = vec2(300.0, 130.0);

/// The text size of a diagram's labels [points].
const LABEL_SIZE: f32 = 11.0;

/// The colours a diagram draws with.
struct Pens {
    ink: Color32,
    weak: Color32,
    north: Color32,
    south: Color32,
    steel: Color32,
    first: Color32,
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
            first: TERM_PALETTE[0],
            second: TERM_PALETTE[1],
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

/// The magnetization of two pole pairs at fill 0.8 (blocks and gaps), the fundamental and the
/// sum of harmonics 1, 3 and 5.
fn square_wave(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let fill = 0.8;
    let poles = 4.0;
    // +1 over the north blocks, -1 over the south ones, 0 in the gaps.
    let wave = |t: f32| {
        let x = t * poles;
        let pole = x.floor();
        let within = (x - pole - 0.5).abs() * 2.0;
        if within <= fill {
            if pole as i32 % 2 == 0 { 1.0 } else { -1.0 }
        } else {
            0.0
        }
    };
    let harmonic = |n: f32, t: f32| {
        let amp = 4.0 / (n * PI) * (n * PI * fill / 2.0).sin();
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
        "blocks and gaps (fill 0.8)",
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
/// with steel behind both (`steel`) or not, and two flux loops.
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
        shapes.push(Shape::rect_filled(
            Rect::from_min_size(pos2(x + 1.0, top), vec2(block_w - 2.0, block_h)),
            1.0,
            upper,
        ));
        shapes.push(Shape::rect_filled(
            Rect::from_min_size(pos2(x + 1.0, bottom), vec2(block_w - 2.0, block_h)),
            1.0,
            lower,
        ));
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

/// Torque against electrical angle over 0 to π with a third harmonic, its peak marked.
fn torque_angle(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left() + 14.0, r.top()),
        pos2(r.right(), r.bottom() - 14.0),
    );
    let torque = |t: f32| (t * PI).sin() + 0.15 * (3.0 * t * PI).sin();
    let points = curve(plot, 0.0, 1.2, 181, torque);
    let peak = points
        .iter()
        .copied()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .unwrap_or(plot.center());
    vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(points, Stroke::new(1.5, pens.first)),
        Shape::circle_filled(peak, 3.5, pens.second),
        label(
            fonts,
            peak + vec2(6.0, 0.0),
            Align2::LEFT_CENTER,
            "pull-out",
            pens.second,
        ),
        label(
            fonts,
            plot.right_bottom(),
            Align2::RIGHT_TOP,
            "φ: 0 to 180°",
            pens.ink,
        ),
        label(fonts, plot.left_top(), Align2::RIGHT_TOP, "T", pens.ink),
    ]
}

/// A magnet block with straight field lines over its middle and lines bulging out at its ends.
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
    shapes.push(label(
        fonts,
        pos2(block.center().x, block.bottom() + 2.0),
        Align2::CENTER_TOP,
        "magnet length L",
        pens.ink,
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

/// First-order heating toward a steady temperature over five time constants, τ at 63 %.
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
            "τ: 63 %",
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
    fn the_harmonics_sum_follows_the_square_wave() {
        // At a north block's centre the wave is 1 and the sum of 1, 3, 5 is near it; its
        // fundamental alone is 4/π sin(0.4π) = 1.211.
        let fill = 0.8f32;
        let sum: f32 = [1.0f32, 3.0, 5.0]
            .iter()
            .map(|n| 4.0 / (n * PI) * (n * PI * fill / 2.0).sin())
            .sum();
        assert!((sum - 1.0).abs() < 0.25, "{sum}");
        let first = 4.0 / PI * (PI * fill / 2.0).sin();
        assert!((first - 1.211).abs() < 1e-3, "{first}");
    }
}
