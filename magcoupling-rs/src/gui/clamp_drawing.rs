//! The clamp tab (spec M4 "Layout": "clamp table plus the clamp drawing (end and top views,
//! egui painter port of `drawing.py`)").
//!
//! [`clamp_drawing`] ports `reference/magcoupling-py/magcoupling/drawing.py` (`clamp_layout`):
//! the end view and the top view of the one-piece slotted clamp for the recommended screw, as
//! marks in millimetres with drawing.py's coordinates, texts and limits, its patches clipped to
//! the boss as `set_clip_path` clips them. Python raises `ValueError` when no screw size fits;
//! here the tab shows that message instead of a drawing. [`clamp_ui`] paints both views at one
//! scale with drawing.py's four pens mapped to the panel's dark theme (decision M42-7), then
//! the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw sizes'
//! table with the recommended size's column marked; every value shows its result's hover text.
//!
//! A design file or share link can hold any finite slit, and a struct written by hand any
//! dimension, which make a cut rectangle millions of millimetres long. The painter therefore
//! cuts each area and centre line to the part of the plane it shows (`visible_mm`, exact on the
//! box's sides) before it dashes it: no outline is longer than the region, whatever the design.

use std::sync::OnceLock;

use egui::{Color32, Stroke};

use crate::engine::clamps::MACHINING_STEPS;
use crate::engine::compat::fmt_fixed;
use crate::engine::meta::{NumOrText, ResultSet, Value};
use crate::gui::dashboard::{Level, hover_text};
use crate::gui::format::{format_value, with_unit};
use crate::gui::geometry::{Mm, finite};
use crate::gui::geometry_view::{DIMENSION, Transform, arrowhead, plated_text, side_by_side};
use crate::gui::readouts::Readouts;
use crate::gui::results_table::table_entries;
use crate::{DesignInputs, DesignResults};

/// drawing.py's `ValueError` text, shown when no screw size fits.
pub const NO_SCREW_FITS: &str = "No screw size fits; enlarge the boss or the clamp length first.";

/// The note under the drawing when the two-piece clamp is selected: drawing.py draws only the
/// one-piece layout.
pub const ONE_PIECE_ONLY: &str =
    "The drawing shows the one-piece slotted clamp (drawing.py's only layout).";

/// The most screws the top view draws (a design file can hold any clamp).
pub const MAX_DRAWN_SCREWS: i64 = 50;

/// The heading of the screw sizes table.
pub const SCREW_SIZES: &str = "Clamp screw sizes";

/// drawing.py's pens: INK, CUT, HID and DIM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pen {
    Ink,
    Cut,
    Hidden,
    Dimension,
}

/// drawing.py's fills: none, white (the background), CUT and the light cut `#f4d9d5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    None,
    Background,
    Cut,
    LightCut,
}

/// One mark of a view [mm].
#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    /// An unfilled circle.
    Circle {
        centre: Mm,
        r: f64,
        pen: Pen,
        width: f32,
    },
    /// A closed convex area: a rectangle, or a rectangle clipped to the boss.
    Area {
        points: Vec<Mm>,
        fill: Fill,
        pen: Pen,
        dashed: bool,
        width: f32,
    },
    /// A centre line (drawing.py's dash-dot `HID` lines).
    CentreLine { from: Mm, to: Mm },
    /// drawing.py's `_dim`: a double arrow with its text at the middle plus `offset`.
    Dimension {
        from: Mm,
        to: Mm,
        text: String,
        offset: Mm,
    },
    /// drawing.py's `annotate`: `text` at `at` with an arrow to `tip`.
    Note {
        text: String,
        tip: Mm,
        at: Mm,
        pen: Pen,
    },
}

/// One view: its title, marks and limits [mm].
#[derive(Clone, Debug, PartialEq)]
pub struct DrawingView {
    pub title: &'static str,
    pub marks: Vec<Mark>,
    pub min: Mm,
    pub max: Mm,
}

/// The clamp drawing: drawing.py's suptitle and its two views.
#[derive(Clone, Debug, PartialEq)]
pub struct ClampDrawing {
    pub title: String,
    pub end: DrawingView,
    pub top: DrawingView,
}

/// Python's `f"{x:g}"`: six significant digits, trailing zeros dropped, scientific below 1e-4
/// and from 1e6 (`25.0` gives `25`, `0.8` gives `0.8`, `1e-5` gives `1e-05`).
pub fn fmt_g(x: f64) -> String {
    if !x.is_finite() {
        return match x {
            x if x.is_nan() => "nan".to_owned(),
            x if x > 0.0 => "inf".to_owned(),
            _ => "-inf".to_owned(),
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let scientific = format!("{x:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust's {:e} has an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s.to_owned()
        }
    };
    if (-4..6).contains(&exponent) {
        let decimals = (5 - exponent).max(0) as usize;
        trim(&format!("{x:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    }
}

/// A rectangle from its corner `(x, y)` and its size, as matplotlib's `Rectangle` takes it.
fn rect(x: f64, y: f64, w: f64, h: f64) -> Vec<Mm> {
    vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

/// The point of the edge `p` to `q` where a function linear in the point, `sp` at `p` and `sq`
/// at `q` (of opposite signs), is zero.
fn crossing(p: Mm, q: Mm, sp: f64, sq: f64) -> Mm {
    let t = sp / (sp - sq);
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/// One Sutherland–Hodgman step: the convex polygon `points` clipped to the half-plane where
/// `side` (linear in the point) is not negative; an edge that crosses it is cut where `side`
/// changes sign, and `snap` puts that new point exactly on the plane. A polygon wholly inside
/// comes back unchanged, point for point.
fn clip_half_plane(points: &[Mm], side: impl Fn(Mm) -> f64, snap: impl Fn(Mm) -> Mm) -> Vec<Mm> {
    let mut out = Vec::with_capacity(points.len() + 1);
    for (j, &p) in points.iter().enumerate() {
        let q = points[(j + 1) % points.len()];
        let (sp, sq) = (side(p), side(q));
        if sp >= 0.0 {
            out.push(p);
        }
        if (sp >= 0.0) != (sq >= 0.0) {
            out.push(snap(crossing(p, q, sp, sq)));
        }
    }
    out
}

/// The convex polygon `points` clipped to the disc of radius `r` at the origin (approximated by
/// the regular 96-gon inside it): Sutherland–Hodgman against each of its edges. Empty when
/// nothing is inside.
pub fn clip_to_disc(points: &[Mm], r: f64) -> Vec<Mm> {
    const SIDES: usize = 96;
    let corner = |i: usize| {
        let a = std::f64::consts::TAU * i as f64 / SIDES as f64;
        [r * a.cos(), r * a.sin()]
    };
    let mut out = points.to_vec();
    for i in 0..SIDES {
        let (a, b) = (corner(i), corner(i + 1));
        // Inside: to the left of the edge a -> b (the polygon runs anticlockwise).
        let left = |p: Mm| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        out = clip_half_plane(&out, left, |p| p);
        if out.is_empty() {
            break;
        }
    }
    out
}

/// One side of an axis-aligned clip box: the part where coordinate `axis` is at least `bound`
/// (`keep_above`) or at most `bound`.
#[derive(Clone, Copy)]
struct BoxSide {
    axis: usize,
    bound: f64,
    keep_above: bool,
}

impl BoxSide {
    /// The four sides of the box `lo` to `hi`.
    fn of_box(lo: Mm, hi: Mm) -> [BoxSide; 4] {
        let side = |axis, bound, keep_above| BoxSide {
            axis,
            bound,
            keep_above,
        };
        [
            side(0, lo[0], true),
            side(0, hi[0], false),
            side(1, lo[1], true),
            side(1, hi[1], false),
        ]
    }

    /// Not negative where `p` is on the side to keep.
    fn depth(self, p: Mm) -> f64 {
        let over = p[self.axis] - self.bound;
        if self.keep_above { over } else { -over }
    }

    /// `p` with the coordinate the side bounds set to the bound itself. A point cut from an edge
    /// is on the side's line only up to rounding, which at coordinates of 1e30 mm is a distance
    /// of 1e14 mm: far outside the box, and an outline that long never ends.
    fn snap(self, mut p: Mm) -> Mm {
        p[self.axis] = self.bound;
        p
    }
}

/// The convex polygon `points` clipped to the box `lo` to `hi` [mm]. Empty when nothing is
/// inside (and for a box with `lo` past `hi`); a polygon wholly inside comes back unchanged.
/// Every point of the result is inside the box.
fn clip_to_box(points: &[Mm], lo: Mm, hi: Mm) -> Vec<Mm> {
    let mut out = points.to_vec();
    for side in BoxSide::of_box(lo, hi) {
        out = clip_half_plane(&out, |p| side.depth(p), |p| side.snap(p));
        if out.is_empty() {
            break;
        }
    }
    out
}

/// The part of the segment `a` to `b` inside the box `lo` to `hi` [mm], `None` when none is. A
/// segment wholly inside comes back unchanged; both ends of the result are inside the box.
fn clip_segment(a: Mm, b: Mm, lo: Mm, hi: Mm) -> Option<[Mm; 2]> {
    let (mut a, mut b) = (a, b);
    for side in BoxSide::of_box(lo, hi) {
        let (da, db) = (side.depth(a), side.depth(b));
        match (da >= 0.0, db >= 0.0) {
            (true, true) => {}
            (false, false) => return None,
            (true, false) => b = side.snap(crossing(a, b, da, db)),
            (false, true) => a = side.snap(crossing(a, b, da, db)),
        }
    }
    Some([a, b])
}

/// A number result that may be text: the number, or NaN.
fn num(value: &NumOrText) -> f64 {
    match value {
        NumOrText::Num(x) => *x,
        NumOrText::Text(_) => f64::NAN,
    }
}

/// drawing.py's `clamp_layout`, as marks; `Err(NO_SCREW_FITS)` where Python raises.
#[allow(non_snake_case)] // drawing.py's names (D, L, R)
pub fn clamp_drawing(
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Result<ClampDrawing, &'static str> {
    let (c, cl, md) = (&inputs.clamps, &results.clamps, &inputs.metal);
    let row = usize::try_from(cl.index)
        .ok()
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| cl.table.get(i))
        .ok_or(NO_SCREW_FITS)?;
    let (D, d, L) = (c.boss_od_mm, cl.shaft_mm, c.clamp_length_mm);
    let (R, r) = (D / 2.0, d / 2.0);
    let (e, hole, cb, tap, thr) = (
        row.offset_mm,
        row.hole_mm,
        row.cbore_dia_mm,
        row.tap_drill_mm,
        row.d_mm,
    );
    let x_out = (R.powi(2) - e.powi(2)).sqrt();
    let x_seat = row.grip_mm + c.slit_mm / 2.0;
    let (key_w, key_d) = (c.key_width_mm, inputs.coupling.keyway_depth_mm);
    let slit = c.slit_mm;
    let clipped = |points: Vec<Mm>, fill: Fill, dashed: bool, width: f32| Mark::Area {
        points: clip_to_disc(&points, R),
        fill,
        pen: Pen::Cut,
        dashed,
        width,
    };

    // End view.
    let mut end = vec![
        Mark::Circle {
            centre: [0.0, 0.0],
            r: R,
            pen: Pen::Ink,
            width: 2.0,
        },
        Mark::Circle {
            centre: [0.0, 0.0],
            r,
            pen: Pen::Ink,
            width: 2.0,
        },
        Mark::Area {
            points: rect(r - 0.2, -key_w / 2.0, key_d + 0.2, key_w),
            fill: Fill::Background,
            pen: Pen::Ink,
            dashed: false,
            width: 1.5,
        },
        clipped(
            rect(-slit / 2.0, r, slit, R - r + 1.0),
            Fill::Cut,
            false,
            1.0,
        ),
        clipped(
            rect(-R - 1.0, e - cb / 2.0, R + 1.0 - x_seat, cb),
            Fill::LightCut,
            true,
            1.2,
        ),
        clipped(
            rect(-x_seat, e - hole / 2.0, x_seat - slit / 2.0, hole),
            Fill::LightCut,
            true,
            1.2,
        ),
        clipped(
            rect(slit / 2.0, e - thr / 2.0, R + 1.0 - slit / 2.0, thr),
            Fill::LightCut,
            true,
            1.2,
        ),
    ];
    for y in [e, 0.0] {
        end.push(Mark::CentreLine {
            from: [-R - 2.0, y],
            to: [R + 2.0, y],
        });
    }
    end.push(Mark::CentreLine {
        from: [0.0, -R - 2.0],
        to: [0.0, R + 2.0],
    });
    end.extend([
        Mark::Dimension {
            from: [R + 3.5, 0.0],
            to: [R + 3.5, e],
            text: fmt_fixed(e, 2),
            offset: [1.6, 0.0],
        },
        Mark::Dimension {
            from: [-R, -R - 3.0],
            to: [R, -R - 3.0],
            text: format!("Ø{} boss", fmt_g(D)),
            offset: [0.0, -1.2],
        },
        Mark::Dimension {
            from: [-r, -2.2],
            to: [r, -2.2],
            text: format!("Ø{} H7", fmt_g(d)),
            offset: [0.0, -1.1],
        },
        Mark::Note {
            text: format!("Slit {} wide,\nbore to OD", fmt_g(slit)),
            tip: [0.0, R - 1.0],
            at: [4.5, R + 4.5],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!(
                "Counterbore Ø{} from the OD;\nhead seat {} from the slit",
                fmt_g(cb),
                fmt_fixed(x_seat, 2)
            ),
            tip: [-x_out + 2.0, e + 1.0],
            at: [-R - 11.0, R + 3.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("Clearance Ø{}", fmt_g(hole)),
            tip: [-2.5, e - hole / 2.0],
            at: [-R - 9.0, 1.5],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("{} tapped\n(drill Ø{})", row.size, fmt_g(tap)),
            tip: [x_out - 2.0, e + thr / 2.0],
            at: [R + 1.0, R + 3.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("Keyway {} wide,\n90° from slit", fmt_g(key_w)),
            tip: [r + key_d, 0.0],
            at: [R + 1.5, -7.5],
            pen: Pen::Ink,
        },
    ]);

    // Top view onto the slit.
    let (fl_D, fl_t, pl_D, pl_t) = (
        md.adapter_flange_dia_mm,
        md.adapter_flange_mm,
        md.adapter_pilot_dia_mm,
        md.adapter_pilot_mm,
    );
    let relief = c.relief_mm;
    let area = |points: Vec<Mm>, fill: Fill, pen: Pen, dashed: bool, width: f32| Mark::Area {
        points,
        fill,
        pen,
        dashed,
        width,
    };
    let mut top = vec![
        area(rect(0.0, -R, L, D), Fill::Background, Pen::Ink, false, 2.0),
        area(rect(L, -R, relief, D), Fill::LightCut, Pen::Cut, false, 1.2),
        area(
            rect(L + relief, -fl_D / 2.0, fl_t, fl_D),
            Fill::Background,
            Pen::Ink,
            false,
            2.0,
        ),
        area(
            rect(L + relief + fl_t, -pl_D / 2.0, pl_t, pl_D),
            Fill::Background,
            Pen::Ink,
            false,
            2.0,
        ),
        area(
            rect(0.0, -slit / 2.0, L, slit),
            Fill::Cut,
            Pen::Cut,
            false,
            1.0,
        ),
        Mark::CentreLine {
            from: [-2.0, 0.0],
            to: [L + relief + fl_t + pl_t + 2.0, 0.0],
        },
    ];
    let first = num(&cl.layout_first_mm);
    let pitch = num(&cl.layout_pitch_mm);
    // drawing.py draws `range(int(row.screws_needed))`, whatever fits; at most
    // MAX_DRAWN_SCREWS here.
    let screws = row.screws_needed.clamp(0, MAX_DRAWN_SCREWS);
    for i in 0..screws {
        let zc = first + i as f64 * pitch;
        top.push(Mark::CentreLine {
            from: [zc, -R - 1.5],
            to: [zc, R + 1.5],
        });
        top.push(area(
            rect(zc - cb / 2.0, -x_out, cb, x_out - x_seat),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
        top.push(area(
            rect(zc - hole / 2.0, -x_seat, hole, x_seat - slit / 2.0),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
        top.push(area(
            rect(zc - thr / 2.0, slit / 2.0, thr, x_out - slit / 2.0),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
    }
    top.extend([
        Mark::Dimension {
            from: [0.0, R + 3.0],
            to: [L, R + 3.0],
            text: format!("{} clamp", fmt_g(L)),
            offset: [0.0, 1.2],
        },
        Mark::Dimension {
            from: [0.0, -R - 3.0],
            to: [first, -R - 3.0],
            text: fmt_g(first),
            offset: [0.0, -1.2],
        },
        Mark::Note {
            text: format!(
                "Relief cut {} wide,\n{} deep from the slit side\n(leaves {} hinge)",
                fmt_g(relief),
                fmt_g(D - c.hinge_mm),
                fmt_g(c.hinge_mm)
            ),
            tip: [L + relief / 2.0, R - 2.0],
            at: [L + 7.0, R + 7.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: "Screw axis\n(square to the slit)".to_owned(),
            tip: [first, -R + 3.0],
            at: [-9.0, -R - 5.0],
            pen: Pen::Hidden,
        },
        Mark::Note {
            text: "Flange to the steel cup web\n(pilot + 3 x M3 + dowel)".to_owned(),
            tip: [L + relief + fl_t / 2.0, -fl_D / 2.0 + 2.0],
            at: [L + 9.0, -R - 7.0],
            pen: Pen::Ink,
        },
    ]);

    Ok(ClampDrawing {
        title: format!(
            "One-piece slotted clamp, Ø{} keyed shaft: {}, {} N·m, {} mm key",
            fmt_g(d),
            cl.recommended,
            fmt_fixed(num(&cl.tightening_Nm), 1),
            fmt_g(num(&cl.hex_mm))
        ),
        end: DrawingView {
            title: "End view from the free end (screw hidden, dashed)",
            marks: end,
            min: [-R - 12.0, -R - 6.0],
            max: [R + 12.0, R + 8.0],
        },
        top: DrawingView {
            title: "Top view onto the slit",
            marks: top,
            min: [-12.0, -R - 10.0],
            max: [L + relief + fl_t + pl_t + 16.0, R + 12.0],
        },
    })
}

/// The colour of a pen in the panel's theme.
fn pen_color(pen: Pen, visuals: &egui::Visuals) -> Color32 {
    match pen {
        Pen::Ink => visuals.strong_text_color(),
        Pen::Cut => Color32::from_rgb(225, 95, 80),
        Pen::Hidden => visuals.weak_text_color(),
        Pen::Dimension => DIMENSION,
    }
}

/// The colour of a fill in the panel's theme.
fn fill_color(fill: Fill, visuals: &egui::Visuals) -> Color32 {
    match fill {
        Fill::None => Color32::TRANSPARENT,
        Fill::Background => visuals.extreme_bg_color,
        Fill::Cut => pen_color(Pen::Cut, visuals),
        Fill::LightCut => Color32::from_rgba_unmultiplied(225, 95, 80, 60),
    }
}

/// How far beyond the painter's clip rectangle a mark is still built [points]: more than any
/// stroke is wide, so the edges that cutting a mark to it adds lie off the screen.
const CLIP_MARGIN: f32 = 16.0;

/// The largest size of a view's title [points]; the band above the views holds one row of it.
const TITLE_POINTS: f32 = 14.0;

/// The part of the drawing plane [mm] the painter shows through `t` (the corners `lo` and `hi`):
/// its clip rectangle, widened by [`CLIP_MARGIN`]. `t.scale` is positive and finite
/// ([`side_by_side`] returns no transform otherwise).
fn visible_mm(clip: egui::Rect, t: Transform) -> (Mm, Mm) {
    let clip = clip.expand(CLIP_MARGIN);
    let scale = f64::from(t.scale);
    let x = |px: f32| (f64::from(px) - f64::from(t.origin.x)) / scale;
    let y = |px: f32| (f64::from(t.origin.y) - f64::from(px)) / scale;
    (
        [x(clip.min.x), y(clip.max.y)],
        [x(clip.max.x), y(clip.min.y)],
    )
}

/// Paints one view with `t`, its title from `title_top` down (the band above the views). The
/// texts scale with the drawing (drawing.py's 9-point text is about 0.9 mm on its figure),
/// between 8 and 12 points, so they do not crowd a small drawing.
///
/// Areas and centre lines are cut to [`visible_mm`] before they are converted to points and
/// dashed: a design file or share link can hold any finite slit, and a struct written by hand
/// any dimension, which make a cut rectangle millions of millimetres long (or a point beyond
/// f32). Dashed as it stands, such an outline never ends (`Shape::dashed_line` walks it with an
/// f32 position that stops advancing); cut to the part the painter shows, it costs what the
/// region's size allows, and the part it shows is the same.
fn paint_view(
    painter: &egui::Painter,
    view: &DrawingView,
    t: Transform,
    title_top: f32,
    visuals: &egui::Visuals,
) {
    let (lo, hi) = visible_mm(painter.clip_rect(), t);
    let font = egui::FontId::proportional((t.scale * 0.9).clamp(8.0, 12.0));
    let arrow = |from: egui::Pos2, to: egui::Pos2, color: Color32| {
        let stroke = Stroke::new(1.0, color);
        painter.line_segment([from, to], stroke);
        arrowhead(painter, to, (to - from).normalized(), stroke);
    };
    for mark in &view.marks {
        match mark {
            Mark::Circle {
                centre,
                r,
                pen,
                width,
            } if finite(*centre) && r.is_finite() => {
                painter.circle_stroke(
                    t.to_px(*centre),
                    t.len(*r),
                    Stroke::new(*width, pen_color(*pen, visuals)),
                );
            }
            Mark::Area {
                points,
                fill,
                pen,
                dashed,
                width,
            } if points.len() >= 3 && points.iter().all(|p| finite(*p)) => {
                let points = clip_to_box(points, lo, hi);
                // Nothing of it shows (or its cut overflowed): nothing to paint.
                if points.len() < 3 || !points.iter().all(|p| finite(*p)) {
                    continue;
                }
                let px: Vec<egui::Pos2> = points.iter().map(|p| t.to_px(*p)).collect();
                let stroke = Stroke::new(*width, pen_color(*pen, visuals));
                let fill = fill_color(*fill, visuals);
                if *dashed {
                    painter.add(egui::Shape::convex_polygon(px.clone(), fill, Stroke::NONE));
                    let mut closed = px;
                    closed.push(closed[0]);
                    painter.extend(egui::Shape::dashed_line(&closed, stroke, 4.0, 3.0));
                } else {
                    painter.add(egui::Shape::convex_polygon(px, fill, stroke));
                }
            }
            Mark::CentreLine { from, to } if finite(*from) && finite(*to) => {
                let visible = clip_segment(*from, *to, lo, hi);
                if let Some(line) = visible.filter(|line| line.iter().all(|p| finite(*p))) {
                    let points = line.map(|p| t.to_px(p));
                    let stroke = Stroke::new(0.8, pen_color(Pen::Hidden, visuals));
                    painter.extend(egui::Shape::dashed_line(&points, stroke, 8.0, 3.0));
                }
            }
            Mark::Dimension {
                from,
                to,
                text,
                offset,
            } if finite(*from) && finite(*to) => {
                let color = pen_color(Pen::Dimension, visuals);
                let (a, b) = (t.to_px(*from), t.to_px(*to));
                let middle = a + (b - a) / 2.0;
                arrow(middle, a, color);
                arrow(middle, b, color);
                let at = t.to_px([
                    (from[0] + to[0]) / 2.0 + offset[0],
                    (from[1] + to[1]) / 2.0 + offset[1],
                ]);
                plated_text(
                    painter,
                    at,
                    text.clone(),
                    font.clone(),
                    color,
                    visuals.extreme_bg_color,
                );
            }
            Mark::Note { text, tip, at, pen } if finite(*tip) && finite(*at) => {
                let color = pen_color(*pen, visuals);
                let start = t.to_px(*at);
                arrow(start, t.to_px(*tip), color);
                // drawing.py's annotations may run past the axes into the figure's margin (the
                // flange note does); here a note that would pass the drawing's right edge moves
                // back inside it, its arrow still from `at`.
                let galley = painter.layout_no_wrap(text.clone(), font.clone(), color);
                let rect = egui::Align2::LEFT_BOTTOM.anchor_size(start, galley.size());
                let over = (rect.max.x - painter.clip_rect().max.x).max(0.0);
                painter.galley(rect.min - egui::vec2(over, 0.0), galley, color);
            }
            _ => {}
        }
    }
    // The title in the band above the view (`title_top`), as matplotlib puts an axes title
    // above its axes: drawn inside the limits it ran under the notes near the top.
    let title_at = egui::pos2(
        t.to_px([(view.min[0] + view.max[0]) / 2.0, view.max[1]]).x,
        title_top,
    );
    painter.text(
        title_at,
        egui::Align2::CENTER_TOP,
        view.title,
        egui::FontId::proportional((t.scale * 1.1).clamp(9.0, TITLE_POINTS)),
        pen_color(Pen::Ink, visuals),
    );
}

/// Draws the clamp drawing (its title, then both views side by side at one scale,
/// [`side_by_side`], under a band for the view titles) in `size` and returns the scale [points
/// per mm], 0 when there is no room.
pub fn drawing_ui(ui: &mut egui::Ui, drawing: &ClampDrawing, size: egui::Vec2) -> f32 {
    let visuals = ui.visuals().clone();
    ui.label(egui::RichText::new(&drawing.title).strong());
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);
    let band = ui.fonts(|f| f.row_height(&egui::FontId::proportional(TITLE_POINTS))) + 2.0;
    let views = [&drawing.end, &drawing.top];
    let extents: Vec<(Mm, Mm)> = views.iter().map(|v| (v.min, v.max)).collect();
    let below_band = rect.with_min_y((rect.min.y + band).min(rect.max.y));
    let Some((scale, transforms)) = side_by_side(below_band, &extents, 0.0) else {
        return 0.0;
    };
    for (view, t) in views.into_iter().zip(transforms) {
        paint_view(&painter, view, t, rect.min.y + 1.0, &visuals);
    }
    scale
}

/// The Shaft clamps results the clamp table lists first.
pub const SUMMARY: [&str; 11] = [
    "clamps.recommended",
    "clamps.length_note",
    "clamps.screws",
    "clamps.tightening_Nm",
    "clamps.hex_mm",
    "clamps.capacity_Nm",
    "clamps.sf_coupling",
    "clamps.head_check",
    "clamps.vent_port",
    "clamps.key_sf",
    "clamps.joint_sf",
];

/// The screw sizes table's rows, built once: (label, unit, field), in the table's order.
pub fn screw_rows() -> &'static [(&'static str, &'static str, String)] {
    static ROWS: OnceLock<Vec<(&'static str, &'static str, String)>> = OnceLock::new();
    ROWS.get_or_init(|| {
        table_entries()
            .iter()
            .filter_map(|entry| {
                let field = entry.path.strip_prefix("clamps.table[0].")?;
                Some((
                    entry.info.meta.label,
                    entry.info.meta.unit,
                    field.to_owned(),
                ))
            })
            .collect()
    })
}

/// The clamp tab: the drawing (or why there is none), then the clamp table, each value a
/// readout (`readouts`).
pub fn clamp_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    // In the height left, however short (egui's 64-point floor lowered to none).
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_clamp_scroll")
        .auto_shrink([false, false])
        .min_scrolled_height(0.0)
        .show(ui, |ui| {
            match clamp_drawing(inputs, results) {
                Ok(drawing) => {
                    // The two views' proportions (about 2:1), at most 80 % of the height, at
                    // least 160 points (f32::clamp would panic where the width allows less).
                    let width = ui.available_width();
                    let height = (width * 0.5).min(ui.available_height() * 0.8).max(160.0);
                    drawing_ui(ui, &drawing, egui::vec2(width, height));
                }
                Err(message) => {
                    ui.colored_label(Level::Bad.color(ui.visuals()), message);
                }
            }
            if inputs.clamps.clamp_type != 1 {
                ui.weak(ONE_PIECE_ONLY);
            }
            ui.separator();
            // One line per result, label then value, wrapping at the region's width; the hover
            // text is built only while hovered.
            for path in SUMMARY {
                let value = results.get(path).unwrap_or(Value::None);
                if value == Value::Text(String::new()) {
                    continue; // an empty length note
                }
                let Some(info) = crate::gui::dashboard::result_info(path) else {
                    continue;
                };
                let rect = ui
                    .horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(info.meta.label).weak());
                        ui.label(with_unit(format_value(&value), info.meta.unit));
                    })
                    .response
                    .rect;
                readouts.show_over(ui, rect, path, || hover_text(path).unwrap_or_default());
            }
            ui.add_space(4.0);
            for step in MACHINING_STEPS {
                ui.add(egui::Label::new(egui::RichText::new(step).weak()).wrap());
            }
            egui::CollapsingHeader::new(SCREW_SIZES)
                .id_salt("magcoupling_screw_sizes")
                .default_open(false)
                .show(ui, |ui| screw_table_ui(ui, results, readouts));
        });
}

/// The 'Clamp screw sizes' table: a row per field, a column per size, the recommended size's
/// column in green; each value a readout.
fn screw_table_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let sizes = results.clamps.table.len();
    let pick = usize::try_from(results.clamps.index)
        .ok()
        .and_then(|i| i.checked_sub(1));
    let good = Level::Good.color(ui.visuals());
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_screw_sizes_scroll")
        .show(ui, |ui| {
            egui::Grid::new("magcoupling_screw_sizes_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (label, unit, field) in screw_rows() {
                        ui.label(with_unit((*label).to_owned(), unit));
                        for i in 0..sizes {
                            let path = format!("clamps.table[{i}].{field}");
                            let text = format_value(&results.get(&path).unwrap_or(Value::None));
                            let rich = egui::RichText::new(text);
                            let rich = if Some(i) == pick {
                                rich.color(good).strong()
                            } else {
                                rich
                            };
                            let rect = ui.label(rich).rect;
                            readouts.show_over(ui, rect, &path, || {
                                hover_text(&path).unwrap_or_else(|| path.clone())
                            });
                        }
                        ui.end_row();
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::test_support::{drawn_texts, flat_shapes, sized_frame, text_rects};

    fn texts(view: &DrawingView) -> Vec<String> {
        view.marks
            .iter()
            .filter_map(|m| match m {
                Mark::Dimension { text, .. } | Mark::Note { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn fmt_g_matches_python() {
        let cases = [
            (25.0, "25"),
            (0.8, "0.8"),
            (2.9, "2.9"),
            (27.436000000000003, "27.436"),
            (1e-5, "1e-05"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (123456.0, "123456"),
            (-2.5, "-2.5"),
            (0.0, "0"),
            (f64::NAN, "nan"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g(x), want, "{x}");
        }
    }

    #[test]
    fn clipping_keeps_what_is_inside_the_disc() {
        let inside = rect(-1.0, -1.0, 2.0, 2.0);
        assert_eq!(clip_to_disc(&inside, 10.0).len(), 4);
        let straddling = clip_to_disc(&rect(5.0, -1.0, 10.0, 2.0), 10.0);
        assert!(straddling.len() > 4, "the arc adds corners");
        for p in &straddling {
            assert!(p[0].hypot(p[1]) <= 10.0 + 1e-9, "{p:?}");
        }
        assert!(clip_to_disc(&rect(20.0, 20.0, 1.0, 1.0), 10.0).is_empty());
    }

    /// Every point of `points` is inside the box `lo` to `hi`.
    fn assert_inside(points: &[Mm], lo: Mm, hi: Mm, what: &str) {
        for p in points {
            assert!(
                (lo[0]..=hi[0]).contains(&p[0]) && (lo[1]..=hi[1]).contains(&p[1]),
                "{what}: {p:?} is outside {lo:?} to {hi:?} in {points:?}"
            );
        }
    }

    #[test]
    fn clipping_a_polygon_to_a_box_keeps_the_inside_and_bounds_the_rest() {
        let (lo, hi) = ([-5.0, -3.0], [5.0, 3.0]);
        // Wholly inside: the same points in the same order.
        let inside = rect(-1.0, -1.0, 2.0, 2.0);
        assert_eq!(clip_to_box(&inside, lo, hi), inside);
        // Wholly outside, and a box that is empty: nothing.
        assert!(clip_to_box(&rect(6.0, 0.0, 1.0, 1.0), lo, hi).is_empty());
        assert!(clip_to_box(&inside, hi, lo).is_empty());
        // A cut rectangle 1e8 mm long (a slit of -1e8): the part inside the box, exactly.
        let long = clip_to_box(&rect(0.0, -1.0, 1e8, 2.0), lo, hi);
        assert_eq!(long, rect(0.0, -1.0, 5.0, 2.0));
        // A rectangle reaching 1e8 mm up and to the right: its corner inside the box.
        let corner = clip_to_box(&rect(2.0, 1.0, 1e8, 1e8), lo, hi);
        assert_eq!(corner, rect(2.0, 1.0, 3.0, 2.0));
        // Past what rounding keeps apart: a coordinate of 1e30 cannot tell where the box's side
        // is, but no point may land outside the box (a point 1e13 mm off would make an outline
        // that never ends).
        for (y0, h) in [
            (-7e29, 1.7e30),
            (-1e30, 2e30),
            (-3e29, 9e29),
            (-1e300, 2e300),
        ] {
            let cut = clip_to_box(&rect(-1e30, y0, 2e30, h), lo, hi);
            assert!(cut.len() >= 3, "{y0}: {cut:?}");
            assert_inside(&cut, lo, hi, &format!("{y0}"));
        }
    }

    #[test]
    fn clipping_a_segment_to_a_box() {
        let (lo, hi) = ([-5.0, -3.0], [5.0, 3.0]);
        // Wholly inside: unchanged, exactly.
        let inside = [[-1.0, 0.1], [2.0, 0.3]];
        assert_eq!(clip_segment(inside[0], inside[1], lo, hi), Some(inside));
        // Wholly outside, parallel to a side and outside it, and crossing no side inside.
        assert_eq!(clip_segment([6.0, 0.0], [9.0, 1.0], lo, hi), None);
        assert_eq!(clip_segment([-9.0, 4.0], [9.0, 4.0], lo, hi), None);
        assert_eq!(clip_segment([-9.0, 2.0], [-2.0, 9.0], lo, hi), None);
        // Along an axis and through the box: the part inside, exactly, in either direction.
        assert_eq!(
            clip_segment([-1e8, 1.0], [1e8, 1.0], lo, hi),
            Some([[-5.0, 1.0], [5.0, 1.0]])
        );
        assert_eq!(
            clip_segment([2.0, 1e8], [2.0, -1e8], lo, hi),
            Some([[2.0, 3.0], [2.0, -3.0]])
        );
        // One end inside: the other end moves to the side.
        assert_eq!(
            clip_segment([0.0, 0.0], [10.0, 0.0], lo, hi),
            Some([[0.0, 0.0], [5.0, 0.0]])
        );
        // Oblique, through a corner region, and at a size rounding cannot resolve.
        let [a, b] = clip_segment([-1e30, -1e30], [1e30, 1e30], lo, hi).expect("it crosses");
        assert_inside(&[a, b], lo, hi, "1e30");
        let [a, b] = clip_segment([0.0, -1e30], [0.0, 1e30], lo, hi).expect("it crosses");
        assert_eq!([a, b], [[0.0, -3.0], [0.0, 3.0]]);
    }

    #[test]
    fn the_visible_part_of_the_plane_follows_the_transform() {
        // 2 points per mm, the origin at (100, 200) on the screen, y up.
        let t = Transform {
            origin: egui::pos2(100.0, 200.0),
            scale: 2.0,
        };
        let clip = egui::Rect::from_min_max(egui::pos2(90.0, 150.0), egui::pos2(130.0, 210.0));
        let m = f64::from(CLIP_MARGIN);
        let (lo, hi) = visible_mm(clip, t);
        // x: (90 - m - 100) / 2 to (130 + m - 100) / 2; y: (200 - (210 + m)) / 2 to
        // (200 - (150 - m)) / 2.
        assert_eq!(lo, [(-10.0 - m) / 2.0, (-10.0 - m) / 2.0]);
        assert_eq!(hi, [(30.0 + m) / 2.0, (50.0 + m) / 2.0]);
        // A mark at the screen's corner is inside; one far past it is not.
        assert_eq!(
            clip_to_box(&[[-5.0, -5.0], [15.0, -5.0], [5.0, 25.0]], lo, hi).len(),
            3
        );
        assert!(clip_to_box(&rect(1e3, 1e3, 1.0, 1.0), lo, hi).is_empty());
    }

    #[test]
    fn the_default_clamp_is_drawn_with_drawing_py_s_texts() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let drawing = clamp_drawing(&inputs, &r).expect("M4 fits at the defaults");
        assert_eq!(
            drawing.title,
            "One-piece slotted clamp, Ø10 keyed shaft: ISO 4762 M4 x 14, class 12.9, 5.1 N·m, 3 mm key"
        );
        assert_eq!(
            texts(&drawing.end),
            [
                "8.25",
                "Ø25 boss",
                "Ø10 H7",
                "Slit 0.8 wide,\nbore to OD",
                "Counterbore Ø7.5 from the OD;\nhead seat 4.26 from the slit",
                "Clearance Ø4.5",
                "M4 tapped\n(drill Ø3.3)",
                "Keyway 4 wide,\n90° from slit",
            ]
        );
        assert_eq!(
            texts(&drawing.top),
            [
                "10 clamp",
                "5",
                "Relief cut 1 wide,\n17 deep from the slit side\n(leaves 8 hinge)",
                "Screw axis\n(square to the slit)",
                "Flange to the steel cup web\n(pilot + 3 x M3 + dowel)",
            ]
        );
        // The boss and the bore, unfilled; every cut clipped to the boss.
        assert_eq!(
            drawing.end.marks[0],
            Mark::Circle {
                centre: [0.0, 0.0],
                r: 12.5,
                pen: Pen::Ink,
                width: 2.0
            }
        );
        for mark in &drawing.end.marks {
            if let Mark::Area {
                points,
                pen: Pen::Cut,
                ..
            } = mark
            {
                assert!(points.iter().all(|p| p[0].hypot(p[1]) <= 12.5 + 1e-9));
            }
        }
        // One screw (M4 x 1): one screw axis across the top view, at the first position.
        let axes: Vec<&Mark> = drawing
            .top
            .marks
            .iter()
            .filter(|m| matches!(m, Mark::CentreLine { from, .. } if from[0] == 5.0))
            .collect();
        assert_eq!(axes.len(), 1);
        assert_eq!(drawing.end.min, [-24.5, -18.5]);
        assert_eq!(drawing.top.max, [33.5, 24.5]);
    }

    #[test]
    fn no_fitting_screw_gives_drawing_py_s_message() {
        let mut inputs = DesignInputs::default();
        inputs.clamps.boss_od_mm = 12.0;
        inputs.clamps.clamp_length_mm = 3.0;
        let r = compute_all(&inputs);
        assert_eq!(r.clamps.index, 0);
        assert_eq!(clamp_drawing(&inputs, &r), Err(NO_SCREW_FITS));
        // An index outside the table (a struct written by hand) is no fit either.
        let mut odd = compute_all(&DesignInputs::default());
        odd.clamps.index = 9;
        assert_eq!(
            clamp_drawing(&DesignInputs::default(), &odd),
            Err(NO_SCREW_FITS)
        );
        odd.clamps.index = -1;
        assert_eq!(
            clamp_drawing(&DesignInputs::default(), &odd),
            Err(NO_SCREW_FITS)
        );
    }

    #[test]
    fn a_screw_count_past_the_table_draws_a_bounded_number() {
        let inputs = DesignInputs::default();
        let mut r = compute_all(&inputs);
        let row = (r.clamps.index - 1) as usize;
        r.clamps.table[row].screws_needed = i64::MAX;
        r.clamps.table[row].screws_fit = i64::MAX;
        let axes = |r: &DesignResults| {
            clamp_drawing(&inputs, r)
                .unwrap()
                .top
                .marks
                .iter()
                .filter(|m| matches!(m, Mark::CentreLine { from, to } if from[0] == to[0]))
                .count()
        };
        assert_eq!(axes(&r), MAX_DRAWN_SCREWS as usize);
        // drawing.py draws the screws needed, whatever fits (range(int(row.screws_needed))).
        r.clamps.table[row].screws_needed = 3;
        r.clamps.table[row].screws_fit = 1;
        assert_eq!(axes(&r), 3);
        r.clamps.table[row].screws_needed = -2;
        assert_eq!(axes(&r), 0);
    }

    #[test]
    fn the_screw_table_rows_follow_the_sheet() {
        let rows = screw_rows();
        assert_eq!(rows.len(), 34, "the size and the 33 celled rows");
        assert_eq!(rows[0].2, "size");
        assert_eq!(rows[1], ("Nominal diameter", "mm", "d_mm".to_owned()));
        let r = compute_all(&DesignInputs::default());
        for (_, _, field) in rows {
            for i in 0..r.clamps.table.len() {
                assert!(
                    r.get(&format!("clamps.table[{i}].{field}")).is_some(),
                    "{field}"
                );
            }
        }
    }

    #[test]
    fn the_clamp_tab_paints_both_views_to_scale_and_the_table() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let drawing = clamp_drawing(&inputs, &r).unwrap();
        let mut scale = 0.0;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 700.0), Vec::new(), |ui| {
            scale = drawing_ui(ui, &drawing, egui::vec2(980.0, 420.0));
        });
        assert!(scale > 0.0);
        let radii: Vec<f32> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) => Some(c.radius),
                _ => None,
            })
            .collect();
        // The 25 mm boss and the 10 mm bore at one scale.
        assert!(
            radii.iter().any(|r| (r - 12.5 * scale).abs() < 1e-3),
            "{radii:?}"
        );
        assert!(
            radii.iter().any(|r| (r - 5.0 * scale).abs() < 1e-3),
            "{radii:?}"
        );
        let texts = drawn_texts(&output);
        for want in [
            drawing.title.as_str(),
            drawing.end.title,
            drawing.top.title,
            "Ø25 boss",
            "M4 tapped\n(drill Ø3.3)",
        ] {
            assert!(
                texts.iter().any(|t| t == want),
                "missing {want:?} in {texts:?}"
            );
        }
        // The whole tab: the summary and the machining steps; no fit shows the message.
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &inputs, &r, &mut Readouts::default())
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "ISO 4762 M4 x 14, class 12.9"));
        assert!(texts.iter().any(|t| t == MACHINING_STEPS[0]));
        assert!(texts.iter().any(|t| t == SCREW_SIZES));
        let mut tight = DesignInputs::default();
        tight.clamps.boss_od_mm = 12.0;
        tight.clamps.clamp_length_mm = 3.0;
        tight.clamps.clamp_type = 2;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &tight, &compute_all(&tight), &mut Readouts::default())
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == NO_SCREW_FITS));
        assert!(texts.iter().any(|t| t == ONE_PIECE_ONLY));
        // A narrow, short region still draws the tab (a ~930 px window leaves about 294
        // points).
        for size in [egui::vec2(294.0, 900.0), egui::vec2(120.0, 90.0)] {
            let output = sized_frame(&ctx, size, Vec::new(), |ui| {
                clamp_ui(ui, &inputs, &r, &mut Readouts::default())
            });
            assert!(drawn_texts(&output).iter().any(|t| t == &drawing.title));
        }
        // No room at all: no scale, nothing painted.
        let mut scale = None;
        sized_frame(&ctx, egui::vec2(1000.0, 700.0), Vec::new(), |ui| {
            scale = Some(drawing_ui(ui, &drawing, egui::Vec2::ZERO));
        });
        assert_eq!(scale, Some(0.0));
    }

    #[test]
    fn the_drawing_s_texts_stay_inside_it_and_clear_of_the_view_titles() {
        // The view titles sit above the views, as matplotlib's axes titles do, so no note runs
        // under them; a note that would run past the drawing's right edge (drawing.py's flange
        // note runs into the figure's margin) moves back inside it (at 540 points it would
        // pass the edge). From about 480 points wide down, the texts' 8-point floor lets the
        // three-line relief note reach the title band: the limit of a fixed smallest text.
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let drawing = clamp_drawing(&inputs, &r).unwrap();
        let ctx = egui::Context::default();
        let background = ctx.style().visuals.extreme_bg_color;
        for size in [
            egui::vec2(540.0, 270.0),
            egui::vec2(660.0, 330.0),
            egui::vec2(980.0, 420.0),
        ] {
            let output = sized_frame(&ctx, egui::vec2(1000.0, 700.0), Vec::new(), |ui| {
                drawing_ui(ui, &drawing, size);
            });
            let area = flat_shapes(&output)
                .into_iter()
                .find_map(|shape| match shape {
                    egui::Shape::Rect(r) if r.fill == background && r.rect.size() == size => {
                        Some(r.rect)
                    }
                    _ => None,
                })
                .expect("the drawing's area");
            let rects = |text: &str| {
                let found = text_rects(&output, text);
                assert_eq!(found.len(), 1, "{size:?}: {text:?} drawn {found:?}");
                found[0]
            };
            let titles = [rects(drawing.end.title), rects(drawing.top.title)];
            for view in [&drawing.end, &drawing.top] {
                for text in texts(view) {
                    let rect = rects(&text);
                    assert!(
                        area.expand(0.5).contains_rect(rect),
                        "{size:?}: {text:?} at {rect:?} outside {area:?}"
                    );
                    for title in titles {
                        assert!(
                            !rect.intersects(title),
                            "{size:?}: {text:?} at {rect:?} under the title at {title:?}"
                        );
                    }
                }
            }
            for title in titles {
                assert!(area.contains_rect(title), "{size:?}: {title:?}");
            }
        }
    }

    /// What one frame of the clamp tab painted: the shapes, the line segments among them (dashes,
    /// dimension lines and arrowheads), the texts and the circles' radii.
    struct Painted {
        shapes: usize,
        segments: usize,
        texts: Vec<String>,
        radii: Vec<f32>,
    }

    /// One frame of the clamp tab for `inputs` and `results`, run on its own thread: a frame
    /// that stalls (a dashed line has no end) fails the test after 20 s instead of hanging the
    /// whole test binary.
    fn paint_clamp_tab(inputs: DesignInputs, results: DesignResults) -> Painted {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let ctx = egui::Context::default();
            let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
                clamp_ui(ui, &inputs, &results, &mut Readouts::default())
            });
            let shapes = flat_shapes(&output);
            let painted = Painted {
                shapes: shapes.len(),
                segments: shapes
                    .iter()
                    .filter(|s| matches!(s, egui::Shape::LineSegment { .. }))
                    .count(),
                texts: drawn_texts(&output),
                radii: shapes
                    .iter()
                    .filter_map(|s| match s {
                        egui::Shape::Circle(c) => Some(c.radius),
                        _ => None,
                    })
                    .collect(),
            };
            let _ = sender.send(painted);
        });
        match receiver.recv_timeout(std::time::Duration::from_secs(20)) {
            Ok(painted) => painted,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("the clamp tab's frame took over 20 s: a mark outside the view stalls it")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("the clamp tab's frame panicked")
            }
        }
    }

    /// The drawing was painted (not drawing.py's message), its boss circle at the default
    /// drawing's scale (the pieces keep their scale), and the frame holds a bounded number of
    /// shapes: what is outside the drawing's area adds none.
    fn assert_painted_like_the_default(painted: &Painted, default: &Painted, what: &str) {
        assert!(
            painted.texts.iter().any(|t| t == "Ø25 boss"),
            "{what}: no drawing painted"
        );
        assert!(!painted.texts.iter().any(|t| t == NO_SCREW_FITS), "{what}");
        let boss = default.radii.iter().copied().fold(0.0, f32::max);
        assert!(boss > 0.0);
        assert!(
            painted.radii.contains(&boss),
            "{what}: the scale moved: {:?} against the boss {boss}",
            painted.radii
        );
        assert!(
            painted.shapes < 10 * default.shapes,
            "{what}: {} shapes against the default's {}",
            painted.shapes,
            default.shapes
        );
    }

    #[test]
    fn a_slit_far_outside_its_slider_paints_a_bounded_frame() {
        // A design file or share link can hold any finite slit (session.rs checks no range);
        // a negative one makes the head-side jaw (grip = 4.26 - slit / 2) and the cut rectangles
        // built on it astronomically long, past anything the dashed outline can walk.
        let inputs = DesignInputs::default();
        let default = paint_clamp_tab(inputs.clone(), compute_all(&inputs));
        assert!(
            default.segments > 50,
            "the default's segments: {}",
            default.segments
        );
        for slit in [-1e7, -1e8, -1e12] {
            let mut inputs = DesignInputs::default();
            inputs.clamps.slit_mm = slit;
            let results = compute_all(&inputs);
            assert_eq!(
                results.clamps.index, 3,
                "slit {slit}: the M4 is still the size"
            );
            assert!(clamp_drawing(&inputs, &results).is_ok(), "slit {slit}");
            let painted = paint_clamp_tab(inputs, results);
            assert_painted_like_the_default(&painted, &default, &format!("slit {slit}"));
            // What the cuts show inside the drawing is still dashed.
            assert!(painted.segments > default.segments / 2, "slit {slit}");
        }
    }

    #[test]
    fn a_clamp_result_far_outside_its_range_paints_a_bounded_frame() {
        // A struct written by hand (Review Focus 6) can hold a row's dimension of any size, up
        // to a size whose pixel position is not even a finite f32.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let default = paint_clamp_tab(inputs.clone(), results.clone());
        let row = (results.clamps.index - 1) as usize;
        type Set = fn(&mut DesignResults, usize, f64);
        let fields: [(&str, Set); 9] = [
            ("hole_mm", |r, row, x| r.clamps.table[row].hole_mm = x),
            ("cbore_dia_mm", |r, row, x| {
                r.clamps.table[row].cbore_dia_mm = x
            }),
            ("d_mm", |r, row, x| r.clamps.table[row].d_mm = x),
            ("grip_mm", |r, row, x| r.clamps.table[row].grip_mm = x),
            ("offset_mm", |r, row, x| r.clamps.table[row].offset_mm = x),
            ("tap_drill_mm", |r, row, x| {
                r.clamps.table[row].tap_drill_mm = x
            }),
            ("shaft_mm", |r, _, x| r.clamps.shaft_mm = x),
            ("layout_first_mm", |r, _, x| {
                r.clamps.layout_first_mm = NumOrText::Num(x)
            }),
            ("layout_pitch_mm", |r, _, x| {
                r.clamps.layout_pitch_mm = NumOrText::Num(x)
            }),
        ];
        for (name, set) in fields {
            for x in [1e30, -1e30, 1e300, -1e300] {
                let mut odd = results.clone();
                set(&mut odd, row, x);
                let painted = paint_clamp_tab(inputs.clone(), odd);
                assert_painted_like_the_default(&painted, &default, &format!("{name} {x:e}"));
            }
        }
    }

    #[test]
    fn the_clamp_view_stays_in_a_short_region() {
        // The space left above the Equation panel: the clamp view scrolls in exactly that
        // space (no 64-point scroll area floor), down to a region a few points tall or none.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let size = egui::vec2(1000.0, 700.0);
        for height in [300.0, 60.0, 20.0, 5.0, 0.0] {
            let region =
                egui::Rect::from_min_size(egui::pos2(20.0, 30.0), egui::vec2(700.0, height));
            for _ in 0..2 {
                let (_, used) = crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                    clamp_ui(ui, &inputs, &results, &mut Readouts::default())
                });
                assert!(
                    used.bottom() <= region.bottom() + 0.01,
                    "{height}: the view runs {} points past the region",
                    used.bottom() - region.bottom()
                );
            }
        }
    }
}
