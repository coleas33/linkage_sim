//! The geometry view's drawing, in millimetres (spec M4 "Layout": "Center — geometry view, to
//! scale: end view of both rings with blocks on polygon faces, cup, sleeve, liner, shaft and key;
//! side view of the axial stack against the 20 mm bay and 35 mm overall length. ... Dimension
//! callouts for face gap, corner gap, running clearance; violations (negative clearance, envelope
//! exceeded) draw red"; Addendum A1: the space claim "drawn as a dashed outline", exceeding it
//! "shows a red callout on the view naming the overshoot in mm per axis").
//!
//! Pure: [`geometry`] turns the design shown (its inputs and results) into two [`View`]s, each a
//! list of filled pieces, dashed lines and dimension [`Callout`]s in millimetres, and the
//! [`Note`]s under them; `gui::geometry_view` paints them to scale. Every dimension is read
//! from the engine's results, the axial ones from the housing in effect (`housing.*`, decision
//! A2-8), so the drawing follows every edit and the length override.
//!
//! - **End view** (x right, y up, the shaft axis at the origin, seen from the free end): the cup
//!   with its pocket, the outer blocks, the liner, the sleeve, the inner blocks, the hub, the
//!   shaft and the key, and the diameter claim as a dashed circle. Faceted blocks sit on the
//!   polygon flats; arcs (any other `coupling.faceted` code, as the engine reads it) are sectors
//!   of the width at their mid-radius. Face `k`'s centre is at 90° + 360° k / N: face 0 at the
//!   top carries the face gap; face 1 the corner gap, along its normal from the circle its inner
//!   block's corners sweep (dashed) to its outer block's flat; the running clearance is at the
//!   bottom.
//! - **Side view** (decision M42-3: the upper half section through the flat centres, x along
//!   the axis from the cap's front face, y the radius): cap, cup wall, web and boss, and inside
//!   the cavity the outer and inner blocks, the liner, the sleeve and the hub, each centred on
//!   the cavity (the workbook gives no axial positions); the space claim's overall length, bay
//!   and diameter as dashed lines; the axial stack, the large-diameter stack and the rotating OD
//!   as dimensions.
//!
//! A callout is red ([`Level::Bad`]) when it is violated (decision M42-4): a gap below zero, the
//! running clearance below its target (the dashboard's clearance check), a space-claim axis
//! exceeded; amber ([`Level::Caution`]) when its value is not a number.
//!
//! A design file can hold any finite number (`InputSet::set` checks no range), so each view's
//! extent comes from its pieces: a claim line farther than [`CLAIM_REACH`] times the pieces'
//! extent along its axis is left off with a note, and a piece or line that holds a number that
//! is not finite is dropped and counted in a note.

use std::f64::consts::PI;

use crate::engine::meta::{NumOrText, Value};
use crate::gui::dashboard::{Level, verdict_level};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults};

/// A point [mm]: x right, y up.
pub type Mm = [f64; 2];

/// The most poles per ring the end view draws blocks for (a design file can hold any count;
/// the sliders stop at 40).
pub const MAX_DRAWN_POLES: i64 = 200;

/// How far a space-claim line may lie from the axis or the cap's front face: at most this many
/// times the pieces' extent along its axis (the body's largest OD, the axial stack). Farther, a
/// finite but huge claim from a design file would shrink the pieces to nothing; it is left off
/// the drawing with a note.
pub const CLAIM_REACH: f64 = 10.0;

/// What a piece is a section of (its fill colour in the painter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The steel (or, without back iron, the body material's) hub, cup, web and boss.
    Body,
    /// The empty cup pocket: the drawing's background.
    Cavity,
    /// A magnet block; `north` alternates around a ring.
    Magnet { north: bool },
    /// The 316L sleeve over the inner blocks and the liner inside the outer blocks.
    Retainer,
    /// The aluminium front cap.
    Cap,
    /// The keyed shaft.
    Shaft,
    /// The key.
    Key,
}

/// A filled outline [mm].
#[derive(Clone, Debug, PartialEq)]
pub enum Outline {
    /// A disc of radius `r` centred on the axis.
    Disc { r: f64 },
    /// A ring between radii `r_in` and `r_out`, centred on the axis.
    Ring { r_in: f64, r_out: f64 },
    /// A convex polygon.
    Polygon(Vec<Mm>),
    /// An annular sector between radii `r_in` and `r_out`, from angle `from` to `to` [rad].
    Sector {
        r_in: f64,
        r_out: f64,
        from: f64,
        to: f64,
    },
    /// An axis-aligned rectangle.
    Rect { min: Mm, max: Mm },
}

impl Outline {
    /// Whether every number of the outline is finite (a design written straight into the
    /// struct can hold NaN; such a piece is not drawn).
    pub fn is_finite(&self) -> bool {
        match self {
            Outline::Disc { r } => r.is_finite(),
            Outline::Ring { r_in, r_out } => r_in.is_finite() && r_out.is_finite(),
            Outline::Polygon(points) => points.iter().all(|p| finite(*p)),
            Outline::Sector {
                r_in,
                r_out,
                from,
                to,
            } => [r_in, r_out, from, to].iter().all(|x| x.is_finite()),
            Outline::Rect { min, max } => finite(*min) && finite(*max),
        }
    }
}

/// One filled piece of a view.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub part: Part,
    pub outline: Outline,
}

/// A dashed line [mm]: the space claim and the axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Dashed {
    pub points: Vec<Mm>,
    /// Red when the claim axis it draws is exceeded; `None` for the plain line.
    pub level: Option<Level>,
}

/// A dimension: a line in its view, a tag drawn beside it, and its text in the list under the
/// views. Hovering it shows the hover text of the result it shows (`path`: the M4-3 hook).
#[derive(Clone, Debug, PartialEq)]
pub struct Callout {
    /// The tag beside the dimension and before its text: 1, 2, ...
    pub tag: usize,
    /// The result the dimension shows.
    pub path: &'static str,
    pub text: String,
    /// `Some(Level::Bad)` violated, `Some(Level::Caution)` not a number, `None` fine.
    pub level: Option<Level>,
    /// The dimension line [mm].
    pub from: Mm,
    pub to: Mm,
}

/// One view: what it draws, in paint order, and its extent [mm].
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub pieces: Vec<Piece>,
    pub dashed: Vec<Dashed>,
    pub callouts: Vec<Callout>,
    /// The corners of the area the view needs [mm].
    pub min: Mm,
    pub max: Mm,
}

impl View {
    /// Whether the view can be drawn: a finite extent of positive size.
    pub fn is_drawable(&self) -> bool {
        finite(self.min)
            && finite(self.max)
            && self.max[0] > self.min[0]
            && self.max[1] > self.min[1]
    }

    /// Drops every piece and dashed line that holds a number that is not finite and returns
    /// how many it dropped (a callout stays: the list shows its text, and the painter skips a
    /// line that is not finite).
    fn retain_finite(&mut self) -> usize {
        let before = self.pieces.len() + self.dashed.len();
        self.pieces.retain(|p| p.outline.is_finite());
        self.dashed.retain(|d| d.points.iter().all(|p| finite(*p)));
        before - self.pieces.len() - self.dashed.len()
    }
}

/// A line of text under the views: the autofit hint, a design check, or why something is not
/// drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub text: String,
    /// `Some(Level::Caution)` for a design check or a part not drawn; `None` for a hint.
    pub level: Option<Level>,
}

/// The two views and the notes of a design.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub end: View,
    pub side: View,
    pub notes: Vec<Note>,
}

impl Geometry {
    /// Every callout of both views, in tag order.
    pub fn callouts(&self) -> impl Iterator<Item = &Callout> {
        self.end.callouts.iter().chain(self.side.callouts.iter())
    }
}

/// The start of the note shown when the blocks are not drawn.
pub const BLOCKS_NOT_DRAWN: &str = "Blocks not drawn";

/// The start of the note shown when a view holds a number that is not finite.
pub const NOT_DRAWN: &str = "Not drawn";

/// The start of each design-check note (the workbook's two inconsistencies, decision 28; the
/// user's decision of 2026-10-01: flagged in the GUI, no engine change).
pub const DESIGN_CHECK: &str = "Design check";

/// The start of the autofit hint.
pub const AUTOFIT: &str = "Autofit";

/// Whether both coordinates of `p` are finite.
pub(crate) fn finite(p: Mm) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

/// A length as the callouts show it: four significant digits and the unit.
pub fn mm(x: f64) -> String {
    with_unit(format_value(&Value::Num(x)), "mm")
}

/// The unit vector at angle `a` [rad].
fn radial(a: f64) -> Mm {
    [a.cos(), a.sin()]
}

/// `r` along the unit vector at angle `a`, plus `t` along its tangent (90° further).
fn at(r: f64, a: f64, t: f64) -> Mm {
    [r * a.cos() - t * a.sin(), r * a.sin() + t * a.cos()]
}

/// The centre angle of face `k` of `n` [rad]: face 0 at the top.
fn face_angle(k: i64, n: i64) -> f64 {
    PI / 2.0 + 2.0 * PI * k as f64 / n as f64
}

/// A block on face `k` of `n`: radial from apothem `r0` to `r1`, `width` wide; faceted on the
/// flat, or (`faceted` false) an arc of `width` at its mid-radius.
fn block(k: i64, n: i64, r0: f64, r1: f64, width: f64, faceted: bool) -> Outline {
    let a = face_angle(k, n);
    if faceted {
        let w = width / 2.0;
        Outline::Polygon(vec![
            at(r0, a, -w),
            at(r0, a, w),
            at(r1, a, w),
            at(r1, a, -w),
        ])
    } else {
        let half = width / 2.0 / ((r0 + r1) / 2.0);
        Outline::Sector {
            r_in: r0,
            r_out: r1,
            from: a - half,
            to: a + half,
        }
    }
}

/// A regular `n`-gon with its flats at `apothem`, flat 0 at the top; a disc of that radius when
/// not `faceted`.
fn polygon_or_disc(n: i64, apothem: f64, faceted: bool) -> Outline {
    if !faceted {
        return Outline::Disc { r: apothem };
    }
    let corner = apothem / (PI / n as f64).cos();
    Outline::Polygon(
        (0..n)
            .map(|k| {
                let a = face_angle(k, n) + PI / n as f64;
                [corner * a.cos(), corner * a.sin()]
            })
            .collect(),
    )
}

/// The level of a gap: red below zero, amber when not a number.
fn gap_level(x: f64) -> Option<Level> {
    if x.is_nan() {
        Some(Level::Caution)
    } else if x < 0.0 {
        Some(Level::Bad)
    } else {
        None
    }
}

/// The body's largest outside diameter, the cup's, the cap's or the boss's [mm]: the pieces'
/// extent across the axis in both views.
fn body_od(inputs: &DesignInputs, results: &DesignResults) -> f64 {
    let md = &inputs.metal;
    results.model.cup_od_mm.max(md.cap_od_mm).max(md.boss_od_mm)
}

/// A space-claim axis as the views draw it: `Some(claim)`, or `None` with a note when the claim
/// is off the drawing, farther than [`CLAIM_REACH`] times `extent`, the pieces' extent along its
/// axis. A claim that is not a number stays: its line holds NaN and is dropped as not finite.
fn drawn_claim(axis: &str, claim: f64, extent: f64, notes: &mut Vec<Note>) -> Option<f64> {
    if claim.abs() > CLAIM_REACH * extent.abs() {
        notes.push(Note {
            text: format!(
                "{NOT_DRAWN}: the {axis} claim {} is off the drawing",
                mm(claim)
            ),
            level: Some(Level::Caution),
        });
        None
    } else {
        Some(claim)
    }
}

/// The end view: the section through the magnets, seen from the free end, with the diameter
/// claim when it is drawn (`diameter`, [`drawn_claim`]).
fn end_view(
    inputs: &DesignInputs,
    results: &DesignResults,
    diameter: Option<f64>,
    notes: &mut Vec<Note>,
) -> View {
    let (c, m, ret) = (&inputs.coupling, &results.model, &results.retainers);
    let md = &inputs.metal;
    let n = c.npole;
    let faceted = c.faceted == 1;
    let draw_blocks = (2..=MAX_DRAWN_POLES).contains(&n);
    // Without blocks there are no flats to draw: the pocket and the hub become discs.
    let flats = faceted && draw_blocks;
    let a_i = c.inner_back_apothem_mm;
    let pocket = m.outer_back_apothem_mm + md.bond_outer_mm;
    let mut pieces = vec![
        Piece {
            part: Part::Body,
            outline: Outline::Disc {
                r: m.cup_od_mm / 2.0,
            },
        },
        Piece {
            part: Part::Cavity,
            outline: polygon_or_disc(n, pocket, flats),
        },
    ];
    let ring = |k: i64, r0: f64, r1: f64, width: f64, north_first: bool| Piece {
        part: Part::Magnet {
            north: (k % 2 == 0) == north_first,
        },
        outline: block(k, n, r0, r1, width, faceted),
    };
    if draw_blocks {
        for k in 0..n {
            pieces.push(ring(
                k,
                m.outer_face_apothem_mm,
                m.outer_back_apothem_mm,
                m.outer_width_mm,
                false,
            ));
        }
    } else {
        notes.push(Note {
            text: format!(
                "{BLOCKS_NOT_DRAWN}: {n} poles per ring is outside 2 to {MAX_DRAWN_POLES}"
            ),
            level: Some(Level::Caution),
        });
    }
    pieces.push(Piece {
        part: Part::Retainer,
        outline: Outline::Ring {
            r_in: ret.liner_id_mm / 2.0,
            r_out: ret.liner_od_mm / 2.0,
        },
    });
    pieces.push(Piece {
        part: Part::Retainer,
        outline: Outline::Ring {
            r_in: ret.sleeve_id_mm / 2.0,
            r_out: ret.sleeve_od_mm / 2.0,
        },
    });
    if draw_blocks {
        for k in 0..n {
            pieces.push(ring(k, a_i, m.inner_face_radius_mm, m.inner_width_mm, true));
        }
    }
    pieces.push(Piece {
        part: Part::Body,
        outline: polygon_or_disc(n, a_i - md.bond_inner_mm, flats),
    });
    let bore = c.bore_mm / 2.0;
    pieces.push(Piece {
        part: Part::Shaft,
        outline: Outline::Disc { r: bore },
    });
    // The key in the hub's keyway, at 0°: from where its sides cross the bore to the keyway's
    // depth past the bore (Calculator C40), the key's width (Shaft clamps C58).
    let w = inputs.clamps.key_width_mm / 2.0;
    let seat = (bore * bore - w * w).max(0.0).sqrt();
    let tip = bore + c.keyway_depth_mm;
    pieces.push(Piece {
        part: Part::Key,
        outline: Outline::Polygon(vec![[seat, -w], [tip, -w], [tip, w], [seat, w]]),
    });

    // Dimensions: the face gap at face 0; the corner gap along face 1's normal, from the circle
    // its inner block's corners sweep to its outer block's flat (the corner gap is the outer
    // face apothem less the inner corner radius); the running clearance at the bottom.
    let top = face_angle(0, n.max(1));
    let face_1 = if n >= 1 { face_angle(1, n) } else { top };
    // The angle from face 1's centre to its inner block's corners.
    let corner_half = if faceted {
        (m.inner_width_mm / 2.0).atan2(m.inner_face_radius_mm)
    } else {
        m.inner_width_mm / 2.0 / ((a_i + m.inner_face_radius_mm) / 2.0)
    };
    let scale = |r: f64, a: f64| {
        let u = radial(a);
        [r * u[0], r * u[1]]
    };
    let mt = &results.metal;
    let clearance_level = if !mt.min_running_clearance_mm.is_finite() {
        Some(Level::Caution)
    } else {
        match verdict_level("metal.clearance_check", &mt.clearance_check) {
            Some(Level::Bad) => Some(Level::Bad),
            _ => None,
        }
    };
    let callouts = vec![
        Callout {
            tag: 1,
            path: "model.face_gap_mm",
            text: format!("Face gap {}", mm(m.face_gap_mm)),
            level: gap_level(m.face_gap_mm),
            from: scale(m.inner_face_radius_mm, top),
            to: scale(m.outer_face_apothem_mm, top),
        },
        Callout {
            tag: 2,
            path: "model.corner_gap_mm",
            text: format!("Corner gap {}", mm(m.corner_gap_mm)),
            level: gap_level(m.corner_gap_mm),
            from: scale(m.inner_corner_radius_mm, face_1),
            to: scale(m.outer_face_apothem_mm, face_1),
        },
        Callout {
            tag: 3,
            path: "metal.min_running_clearance_mm",
            text: format!(
                "Running clearance {} (sleeve-to-liner gap {} less movement {})",
                mm(mt.min_running_clearance_mm),
                mm(mt.sleeve_liner_clearance_mm),
                mm(mt.adverse_movement_mm)
            ),
            level: clearance_level,
            from: scale(ret.sleeve_od_mm / 2.0, -PI / 2.0),
            to: scale(ret.liner_id_mm / 2.0, -PI / 2.0),
        },
    ];

    let mut dashed = Vec::new();
    let mut r = m.cup_od_mm / 2.0;
    if let Some(d) = diameter {
        let claim = d / 2.0;
        let over = results.housing.diameter_overshoot_mm;
        dashed.push(Dashed {
            points: (0..=64)
                .map(|i| scale(claim, 2.0 * PI * f64::from(i) / 64.0))
                .collect(),
            level: overshoot_level(over).filter(|l| *l == Level::Bad),
        });
        r = r.max(claim.abs());
    }
    // The arc the inner corners of face 1 sweep, from one corner's angle to the other's.
    dashed.push(Dashed {
        points: (0..=16)
            .map(|i| {
                let a = face_1 - corner_half + 2.0 * corner_half * f64::from(i) / 16.0;
                scale(m.inner_corner_radius_mm, a)
            })
            .collect(),
        level: None,
    });
    let r = r + 2.5;
    View {
        pieces,
        dashed,
        callouts,
        min: [-r, -r],
        max: [r, r],
    }
}

/// The level of a space-claim axis from its overshoot: red past the claim, amber when not a
/// number.
fn overshoot_level(over: f64) -> Option<Level> {
    if over.is_nan() {
        Some(Level::Caution)
    } else if over > 0.0 {
        Some(Level::Bad)
    } else {
        None
    }
}

/// One space-claim axis as a callout: the dimension it measures against its claim, or by how
/// much it passes it (the overshoot's result: housing.*), or unknown.
#[allow(clippy::too_many_arguments)] // one axis: its words, its two results, its numbers
fn claim_callout(
    tag: usize,
    axis: &str,
    dimension: &str,
    dimension_path: &'static str,
    overshoot_path: &'static str,
    value: f64,
    claim: f64,
    over: f64,
    line: (Mm, Mm),
) -> Callout {
    let level = overshoot_level(over);
    let (path, text) = match level {
        Some(Level::Bad) => (
            overshoot_path,
            format!(
                "{axis}: {dimension} {} is {} over the {} claim",
                mm(value),
                mm(over),
                mm(claim)
            ),
        ),
        Some(_) => (
            overshoot_path,
            format!("{axis}: unknown (a dimension or the claim is not a number)"),
        ),
        None => (
            dimension_path,
            format!("{axis}: {dimension} {} of {}", mm(value), mm(claim)),
        ),
    };
    Callout {
        tag,
        path,
        text,
        level,
        from: line.0,
        to: line.1,
    }
}

/// The side view: the upper half section through the flat centres, the axis along x from the
/// cap's front face, with the claim's lines that are drawn (`diameter`, [`drawn_claim`]; the
/// overall length and the bay are decided here, against the axial stack).
fn side_view(
    inputs: &DesignInputs,
    results: &DesignResults,
    diameter: Option<f64>,
    notes: &mut Vec<Note>,
) -> View {
    let (c, md, m) = (&inputs.coupling, &inputs.metal, &results.model);
    let (ret, h, mt) = (&results.retainers, &results.housing, &results.metal);
    let bore = c.bore_mm / 2.0;
    let cap = md.cap_axial_mm;
    let depth = h.cup_depth_mm;
    let web_end = cap + depth + md.web_mm;
    let boss_end = web_end + md.boss_length_mm;
    let rect = |part: Part, x0: f64, x1: f64, y0: f64, y1: f64| Piece {
        part,
        outline: Outline::Rect {
            min: [x0, y0],
            max: [x1, y1],
        },
    };
    // Inside the cavity every part is centred on it (the workbook gives no axial positions).
    let mid = cap + depth / 2.0;
    let centred = |part: Part, length: f64, y0: f64, y1: f64| {
        rect(part, mid - length / 2.0, mid + length / 2.0, y0, y1)
    };
    let pieces = vec![
        rect(
            Part::Body,
            cap,
            cap + depth,
            m.outer_back_apothem_mm + md.bond_outer_mm,
            m.cup_od_mm / 2.0,
        ),
        rect(Part::Body, cap + depth, web_end, bore, m.cup_od_mm / 2.0),
        rect(Part::Body, web_end, boss_end, bore, md.boss_od_mm / 2.0),
        rect(
            Part::Cap,
            0.0,
            cap,
            ret.liner_id_mm / 2.0,
            md.cap_od_mm / 2.0,
        ),
        centred(
            Part::Magnet { north: true },
            m.outer_length_mm,
            m.outer_face_apothem_mm,
            m.outer_back_apothem_mm,
        ),
        centred(
            Part::Retainer,
            h.retainer_span_mm,
            ret.liner_id_mm / 2.0,
            ret.liner_od_mm / 2.0,
        ),
        centred(
            Part::Retainer,
            h.retainer_span_mm,
            ret.sleeve_id_mm / 2.0,
            ret.sleeve_od_mm / 2.0,
        ),
        centred(
            Part::Magnet { north: false },
            m.inner_length_mm,
            c.inner_back_apothem_mm,
            m.inner_face_radius_mm,
        ),
        centred(
            Part::Body,
            h.hub_length_mm,
            bore,
            c.inner_back_apothem_mm - md.bond_inner_mm,
        ),
    ];
    // The pieces' extent: the axial stack and the body's top.
    let axial = boss_end.max(mt.axial_stack_mm);
    let body_top = body_od(inputs, results) / 2.0;
    let length = drawn_claim("overall length", md.max_overall_axial_mm, axial, notes);
    let bay = drawn_claim(
        "large-diameter bay",
        md.max_large_dia_axial_mm,
        axial,
        notes,
    );
    let radius = diameter.map(|d| d / 2.0);
    // The claim's lines reach its radius and its length where they are drawn and finite, else
    // the body's top and the axial stack.
    let claim_top = radius.filter(|r| r.is_finite()).unwrap_or(body_top);
    let claim_end = length.filter(|l| l.is_finite()).unwrap_or(axial);
    let red = |over: f64| overshoot_level(over).filter(|l| *l == Level::Bad);
    let right = claim_end.max(bay.unwrap_or(axial)).max(axial) + 2.0;
    let mut dashed = vec![
        Dashed {
            points: vec![[-1.0, 0.0], [right, 0.0]],
            level: None,
        },
        Dashed {
            points: vec![[0.0, 0.0], [0.0, claim_top]],
            level: None,
        },
    ];
    if let Some(radius) = radius {
        dashed.push(Dashed {
            points: vec![[0.0, radius], [claim_end, radius]],
            level: red(h.diameter_overshoot_mm),
        });
    }
    if let Some(length) = length {
        dashed.push(Dashed {
            points: vec![[length, 0.0], [length, claim_top]],
            level: red(h.length_overshoot_mm),
        });
    }
    if let Some(bay) = bay {
        dashed.push(Dashed {
            points: vec![[bay, 0.0], [bay, claim_top]],
            level: red(h.bay_overshoot_mm),
        });
    }
    let callouts = vec![
        claim_callout(
            4,
            "Overall length",
            "axial stack",
            "metal.axial_stack_mm",
            "housing.length_overshoot_mm",
            mt.axial_stack_mm,
            md.max_overall_axial_mm,
            h.length_overshoot_mm,
            ([0.0, -2.0], [mt.axial_stack_mm, -2.0]),
        ),
        claim_callout(
            5,
            "Large-diameter bay",
            "stack",
            "metal.large_dia_stack_mm",
            "housing.bay_overshoot_mm",
            mt.large_dia_stack_mm,
            md.max_large_dia_axial_mm,
            h.bay_overshoot_mm,
            ([0.0, -4.0], [mt.large_dia_stack_mm, -4.0]),
        ),
        claim_callout(
            6,
            "Diameter",
            "rotating OD",
            "metal.rotating_od_mm",
            "housing.diameter_overshoot_mm",
            mt.rotating_od_mm,
            md.max_diameter_mm,
            h.diameter_overshoot_mm,
            ([-2.0, 0.0], [-2.0, mt.rotating_od_mm / 2.0]),
        ),
    ];
    let top = radius.map_or(body_top, |r| body_top.max(r.abs())) + 2.0;
    View {
        pieces,
        dashed,
        callouts,
        min: [-4.0, -6.0],
        max: [right, top],
    }
}

/// The notes under the views: the autofit wall (Addendum A1, decision 27) and the two design
/// checks (decision 28, flagged only), each while it applies (decision M42-9).
fn notes(inputs: &DesignInputs, results: &DesignResults) -> Vec<Note> {
    let md = &inputs.metal;
    let mut notes = Vec::new();
    if let NumOrText::Num(wall) = results.materials.cup_wall_suggested_mm {
        notes.push(Note {
            text: format!(
                "{AUTOFIT}: the wall rule asks for at least {} at the pocket corners (input {})",
                mm(wall),
                mm(md.cup_wall_corner_mm)
            ),
            level: None,
        });
    }
    let cup_body = results.metal.cup_body_od_mm;
    if md.cap_thread_dia_mm < cup_body {
        notes.push(Note {
            text: format!(
                "{DESIGN_CHECK}: the cap thread diameter {} (Metal design C169) is below the cup body OD {} (C165)",
                mm(md.cap_thread_dia_mm),
                mm(cup_body)
            ),
            level: Some(Level::Caution),
        });
    }
    if md.boss_od_mm != inputs.clamps.boss_od_mm {
        notes.push(Note {
            text: format!(
                "{DESIGN_CHECK}: the boss OD is {} in Metal design (C127) and {} in Shaft clamps (C35)",
                mm(md.boss_od_mm),
                mm(inputs.clamps.boss_od_mm)
            ),
            level: Some(Level::Caution),
        });
    }
    notes
}

/// The note for `dropped` parts of the `name` view that hold a number that is not finite.
fn parts_not_drawn(dropped: usize, name: &str) -> String {
    if dropped == 1 {
        format!("{NOT_DRAWN}: 1 part of the {name} view holds a number that is not a number")
    } else {
        format!(
            "{NOT_DRAWN}: {dropped} parts of the {name} view hold a number that is not a number"
        )
    }
}

/// The geometry of the design shown: its inputs and the results computed from them.
pub fn geometry(inputs: &DesignInputs, results: &DesignResults) -> Geometry {
    let mut all_notes = Vec::new();
    // The diameter claim is decided once, for both views, against the body's largest OD.
    let diameter = drawn_claim(
        "diameter",
        inputs.metal.max_diameter_mm,
        body_od(inputs, results),
        &mut all_notes,
    );
    let mut end = end_view(inputs, results, diameter, &mut all_notes);
    let mut side = side_view(inputs, results, diameter, &mut all_notes);
    for (view, name) in [(&mut end, "end"), (&mut side, "side")] {
        let dropped = view.retain_finite();
        if !view.is_drawable() {
            all_notes.push(Note {
                text: format!("{NOT_DRAWN}: a dimension of the {name} view is not a number"),
                level: Some(Level::Caution),
            });
        } else if dropped > 0 {
            all_notes.push(Note {
                text: parts_not_drawn(dropped, name),
                level: Some(Level::Caution),
            });
        }
    }
    all_notes.extend(notes(inputs, results));
    Geometry {
        end,
        side,
        notes: all_notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;

    fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    fn of(inputs: &DesignInputs) -> Geometry {
        geometry(inputs, &compute_all(inputs))
    }

    fn callout(g: &Geometry, tag: usize) -> &Callout {
        g.callouts().find(|c| c.tag == tag).unwrap()
    }

    fn length(c: &Callout) -> f64 {
        ((c.to[0] - c.from[0]).powi(2) + (c.to[1] - c.from[1]).powi(2)).sqrt()
    }

    #[test]
    fn the_end_view_draws_every_part_from_the_results() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let g = geometry(&inputs, &r);
        let n = inputs.coupling.npole as usize;
        let magnets = g
            .end
            .pieces
            .iter()
            .filter(|p| matches!(p.part, Part::Magnet { .. }))
            .count();
        assert_eq!(magnets, 2 * n, "both rings");
        assert_eq!(
            g.end.pieces[0],
            Piece {
                part: Part::Body,
                outline: Outline::Disc {
                    r: r.model.cup_od_mm / 2.0
                }
            }
        );
        // The pocket is the N-gon whose corners sit at the pocket corner radius (C61).
        let Outline::Polygon(pocket) = &g.end.pieces[1].outline else {
            panic!("faceted pocket")
        };
        assert_eq!(pocket.len(), n);
        for p in pocket {
            let r_corner = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r_corner - r.model.pocket_corner_radius_mm).abs() < 1e-9);
        }
        let rings: Vec<&Outline> = g
            .end
            .pieces
            .iter()
            .filter(|p| p.part == Part::Retainer)
            .map(|p| &p.outline)
            .collect();
        assert_eq!(
            rings,
            [
                &Outline::Ring {
                    r_in: r.retainers.liner_id_mm / 2.0,
                    r_out: r.retainers.liner_od_mm / 2.0
                },
                &Outline::Ring {
                    r_in: r.retainers.sleeve_id_mm / 2.0,
                    r_out: r.retainers.sleeve_od_mm / 2.0
                },
            ]
        );
        assert!(
            g.end
                .pieces
                .iter()
                .any(|p| p.part == Part::Shaft && p.outline == Outline::Disc { r: 5.0 })
        );
        let key = g.end.pieces.iter().find(|p| p.part == Part::Key).unwrap();
        let Outline::Polygon(key) = &key.outline else {
            panic!("the key is a rectangle")
        };
        // From where its 4 mm sides cross the 10 mm bore to 1.7 mm past the bore.
        assert_eq!(key[1], [6.7, -2.0]);
        assert!((key[0][0] - 21.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn faceted_blocks_sit_on_the_flats_and_arcs_are_sectors() {
        let r = compute_all(&DesignInputs::default());
        let g = of(&DesignInputs::default());
        // The first outer block is face 0 at the top: its face at the outer face apothem, its
        // back at the outer back apothem, 6.35 mm wide.
        let first = g
            .end
            .pieces
            .iter()
            .find(|p| matches!(p.part, Part::Magnet { north: false }))
            .unwrap();
        let Outline::Polygon(q) = &first.outline else {
            panic!("a flat block")
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(q[0][1], r.model.outer_face_apothem_mm));
        assert!(close(q[2][1], r.model.outer_back_apothem_mm));
        assert!(close((q[1][0] - q[0][0]).abs(), r.model.outer_width_mm));
        let arcs = of(&design(|i| i.coupling.faceted = 0));
        let sector = arcs
            .end
            .pieces
            .iter()
            .find(|p| matches!(p.part, Part::Magnet { .. }))
            .unwrap();
        assert!(matches!(sector.outline, Outline::Sector { .. }));
        assert!(matches!(arcs.end.pieces[1].outline, Outline::Disc { .. }));
    }

    #[test]
    fn the_end_view_callouts_measure_the_gaps_they_name() {
        let r = compute_all(&DesignInputs::default());
        let g = of(&DesignInputs::default());
        let tags: Vec<usize> = g.callouts().map(|c| c.tag).collect();
        assert_eq!(tags, [1, 2, 3, 4, 5, 6]);
        let face = callout(&g, 1);
        assert_eq!(face.text, "Face gap 1.400 mm");
        assert_eq!(face.path, "model.face_gap_mm");
        assert!((length(face) - r.model.face_gap_mm).abs() < 1e-9);
        assert_eq!(face.level, None);
        let corner = callout(&g, 2);
        assert_eq!(corner.text, "Corner gap 1.027 mm");
        assert!((length(corner) - r.model.corner_gap_mm).abs() < 1e-9);
        // Along face 1's normal: from the circle the inner corners sweep, drawn dashed, to the
        // outer block's flat (both ends on drawn features).
        let normal = face_angle(1, DesignInputs::default().coupling.npole);
        let dot = |p: Mm, a: f64| p[0] * a.cos() + p[1] * a.sin();
        assert!((dot(corner.to, normal) - r.model.outer_face_apothem_mm).abs() < 1e-9);
        assert!(dot(corner.to, normal + PI / 2.0).abs() <= r.model.outer_width_mm / 2.0);
        let on_corner_circle =
            |p: &Mm| (p[0].hypot(p[1]) - r.model.inner_corner_radius_mm).abs() < 1e-9;
        assert!(on_corner_circle(&corner.from));
        assert!(
            g.end
                .dashed
                .iter()
                .any(|d| d.level.is_none() && d.points.iter().all(on_corner_circle))
        );
        // The default design's running clearance is below zero: red (decision M42-4).
        let run = callout(&g, 3);
        assert_eq!(
            run.text,
            "Running clearance -0.1032 mm (sleeve-to-liner gap 0.6768 mm less movement 0.7800 mm)"
        );
        assert_eq!(run.level, Some(Level::Bad));
        assert!((length(run) - r.metal.sleeve_liner_clearance_mm).abs() < 1e-9);
    }

    #[test]
    fn a_clearance_below_its_target_is_red_and_a_negative_corner_gap_too() {
        // Above zero but below the 0.2 mm target: the dashboard's check fails, so red.
        let small = design(|i| i.metal.magnet_position_mm = 0.0);
        let r = compute_all(&small);
        assert!(r.metal.min_running_clearance_mm > 0.0 && r.metal.min_running_clearance_mm < 0.2);
        assert_eq!(callout(&of(&small), 3).level, Some(Level::Bad));
        // Past the target: no colour.
        let clear = of(&design(|i| {
            i.metal.magnet_position_mm = 0.0;
            i.metal.shaft_displacement_mm = 0.0;
        }));
        assert_eq!(callout(&clear, 3).level, None);
        // Face gap 0.3 mm puts the inner corners past the outer faces.
        let tight = of(&design(|i| i.metal.face_gap_mm = 0.3));
        assert_eq!(callout(&tight, 2).text, "Corner gap -0.07317 mm");
        assert_eq!(callout(&tight, 2).level, Some(Level::Bad));
        assert_eq!(callout(&tight, 1).level, None);
    }

    #[test]
    fn the_side_view_uses_the_housing_in_effect() {
        let inputs = design(|i| i.coupling.magnets.axial_length_mm = Some(50.8));
        let r = compute_all(&inputs);
        let g = geometry(&inputs, &r);
        let rects: Vec<(Part, Mm, Mm)> = g
            .side
            .pieces
            .iter()
            .map(|p| match p.outline {
                Outline::Rect { min, max } => (p.part, min, max),
                ref other => panic!("{other:?}"),
            })
            .collect();
        let cap = inputs.metal.cap_axial_mm;
        // The cup wall spans the cavity in effect (53.6 mm, not the 15.5 mm input).
        assert_eq!(rects[0].1[0], cap);
        assert!((rects[0].2[0] - (cap + r.housing.cup_depth_mm)).abs() < 1e-9);
        assert!((r.housing.cup_depth_mm - 53.6).abs() < 1e-9);
        // The hub is the hub length in effect long, centred on the cavity.
        let hub = rects.last().unwrap();
        assert_eq!(hub.0, Part::Body);
        assert!((hub.2[0] - hub.1[0] - r.housing.hub_length_mm).abs() < 1e-9);
        let mid = cap + r.housing.cup_depth_mm / 2.0;
        assert!(((hub.1[0] + hub.2[0]) / 2.0 - mid).abs() < 1e-9);
        // The liner and the sleeve span the retainer span in effect, not the input's.
        assert_ne!(r.housing.retainer_span_mm, inputs.metal.retainer_span_mm);
        let retainers: Vec<&(Part, Mm, Mm)> = rects
            .iter()
            .filter(|(p, ..)| *p == Part::Retainer)
            .collect();
        assert_eq!(retainers.len(), 2);
        for (_, min, max) in retainers {
            assert!((max[0] - min[0] - r.housing.retainer_span_mm).abs() < 1e-9);
        }
    }

    #[test]
    fn the_space_claim_callouts_name_each_overshoot_in_red() {
        let inside = of(&DesignInputs::default());
        assert_eq!(
            callout(&inside, 4).text,
            "Overall length: axial stack 31.80 mm of 35.00 mm"
        );
        assert_eq!(callout(&inside, 4).path, "metal.axial_stack_mm");
        assert_eq!(callout(&inside, 4).level, None);
        assert_eq!(
            callout(&inside, 6).text,
            "Diameter: rotating OD 42.80 mm of 43.00 mm"
        );
        assert!(inside.side.dashed.iter().all(|d| d.level.is_none()));
        let long = of(&design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        let length = callout(&long, 4);
        assert_eq!(
            length.text,
            "Overall length: axial stack 69.90 mm is 34.90 mm over the 35.00 mm claim"
        );
        assert_eq!(length.path, "housing.length_overshoot_mm");
        assert_eq!(length.level, Some(Level::Bad));
        assert!(
            callout(&long, 5)
                .text
                .contains("36.90 mm over the 20.00 mm claim")
        );
        assert_eq!(callout(&long, 6).level, None);
        // The exceeded axes' dashed lines are red: the bay line and the right edge.
        let red = long
            .side
            .dashed
            .iter()
            .filter(|d| d.level == Some(Level::Bad))
            .count();
        assert_eq!(red, 2);
        // A diameter past the claim reddens the end view's claim circle.
        let wide = of(&design(|i| i.metal.max_diameter_mm = 40.0));
        assert_eq!(callout(&wide, 6).level, Some(Level::Bad));
        assert_eq!(wide.end.dashed[0].level, Some(Level::Bad));
        // Every axis at once: three red callouts, three red side lines and the red circle.
        let all = of(&design(|i| {
            i.coupling.magnets.axial_length_mm = Some(50.8);
            i.metal.max_diameter_mm = 40.0;
        }));
        for tag in [4, 5, 6] {
            assert_eq!(callout(&all, tag).level, Some(Level::Bad), "{tag}");
        }
        let red = all
            .side
            .dashed
            .iter()
            .filter(|d| d.level == Some(Level::Bad))
            .count();
        assert_eq!(red, 3);
        assert_eq!(all.end.dashed[0].level, Some(Level::Bad));
        // Exactly at the claim is inside; the next number below it is over (per axis, as the
        // dashboard's space-claim check reads it).
        let od = compute_all(&DesignInputs::default()).metal.rotating_od_mm;
        let at = of(&design(|i| i.metal.max_diameter_mm = od));
        assert_eq!(callout(&at, 6).level, None);
        assert_eq!(at.end.dashed[0].level, None);
        let below = of(&design(|i| i.metal.max_diameter_mm = od.next_down()));
        assert_eq!(callout(&below, 6).level, Some(Level::Bad));
        assert_eq!(
            of(&design(|i| i.metal.max_diameter_mm = f64::NAN))
                .callouts()
                .nth(5)
                .unwrap()
                .level,
            Some(Level::Caution)
        );
    }

    #[test]
    fn the_design_checks_show_while_each_inconsistency_holds() {
        let checks = |inputs: &DesignInputs| -> Vec<String> {
            of(inputs)
                .notes
                .into_iter()
                .filter(|n| n.text.starts_with(DESIGN_CHECK))
                .map(|n| n.text)
                .collect()
        };
        // The workbook's defaults hold both (decision 28).
        assert_eq!(
            checks(&DesignInputs::default()),
            [
                "Design check: the cap thread diameter 41.00 mm (Metal design C169) is below the cup body OD 41.33 mm (C165)",
                "Design check: the boss OD is 22.00 mm in Metal design (C127) and 25.00 mm in Shaft clamps (C35)",
            ]
        );
        let fixed = design(|i| {
            i.metal.cap_thread_dia_mm = 42.0;
            i.metal.boss_od_mm = 25.0;
        });
        assert!(checks(&fixed).is_empty());
        // At equality the thread is not below the cup body.
        let cup = compute_all(&DesignInputs::default()).metal.cup_body_od_mm;
        let equal = design(|i| i.metal.cap_thread_dia_mm = cup);
        assert_eq!(checks(&equal).len(), 1);
    }

    #[test]
    fn the_autofit_hint_shows_the_wall_rule_unless_there_is_no_back_iron() {
        let hint = |inputs: &DesignInputs| {
            of(inputs)
                .notes
                .into_iter()
                .find(|n| n.text.starts_with(AUTOFIT))
        };
        let note = hint(&DesignInputs::default()).expect("a steel circuit has a wall rule");
        assert_eq!(
            note.text,
            "Autofit: the wall rule asks for at least 2.000 mm at the pocket corners (input 1.800 mm)"
        );
        assert_eq!(note.level, None);
        assert_eq!(hint(&design(|i| i.coupling.backiron = 0)), None);
    }

    #[test]
    fn odd_pole_counts_from_a_file_never_draw_unbounded_blocks() {
        for npole in [0, 1, -4, MAX_DRAWN_POLES + 2, i64::MAX] {
            let g = of(&design(|i| i.coupling.npole = npole));
            assert!(
                g.end
                    .pieces
                    .iter()
                    .all(|p| !matches!(p.part, Part::Magnet { .. })),
                "{npole}"
            );
            assert!(
                g.notes.iter().any(|n| n.text.starts_with(BLOCKS_NOT_DRAWN)),
                "{npole}"
            );
        }
        let most = of(&design(|i| i.coupling.npole = MAX_DRAWN_POLES));
        let magnets = most
            .end
            .pieces
            .iter()
            .filter(|p| matches!(p.part, Part::Magnet { .. }))
            .count();
        assert_eq!(magnets, 2 * MAX_DRAWN_POLES as usize);
    }

    #[test]
    fn a_number_that_is_not_finite_draws_nothing_wrong_and_says_so() {
        // validate() refuses these at every boundary; a struct literal can still hold them.
        let g = of(&design(|i| i.coupling.bore_mm = f64::NAN));
        assert!(g.end.pieces.iter().all(|p| p.outline.is_finite()));
        assert!(g.side.pieces.iter().all(|p| p.outline.is_finite()));
        // The dropped parts are counted: the shaft and the key; the web, the boss and the hub.
        let texts: Vec<&str> = g.notes.iter().map(|n| n.text.as_str()).collect();
        assert!(
            texts
                .contains(&"Not drawn: 2 parts of the end view hold a number that is not a number"),
            "{texts:?}"
        );
        assert!(
            texts.contains(
                &"Not drawn: 3 parts of the side view hold a number that is not a number"
            ),
            "{texts:?}"
        );
        // A claim that is not a number: its circle and its line, one part per view.
        let nan_claim = of(&design(|i| i.metal.max_diameter_mm = f64::NAN));
        let notes: Vec<&Note> = nan_claim
            .notes
            .iter()
            .filter(|n| n.text.starts_with(NOT_DRAWN))
            .collect();
        assert_eq!(
            notes,
            [
                &Note {
                    text: "Not drawn: 1 part of the end view holds a number that is not a number"
                        .to_owned(),
                    level: Some(Level::Caution)
                },
                &Note {
                    text: "Not drawn: 1 part of the side view holds a number that is not a number"
                        .to_owned(),
                    level: Some(Level::Caution)
                },
            ]
        );
        // The cup's OD and the claim both NaN: the end view has no extent.
        let nan_cup = of(&design(|i| {
            i.metal.cup_wall_corner_mm = f64::NAN;
            i.metal.max_diameter_mm = f64::NAN;
        }));
        assert!(!nan_cup.end.is_drawable());
        assert!(nan_cup.side.is_drawable());
        assert!(
            nan_cup
                .notes
                .iter()
                .any(|n| n.text == "Not drawn: a dimension of the end view is not a number")
        );
        // The default design draws everything: no such note.
        assert!(
            of(&DesignInputs::default())
                .notes
                .iter()
                .all(|n| !n.text.starts_with(NOT_DRAWN))
        );
    }

    #[test]
    fn a_claim_far_past_the_pieces_is_left_off_the_drawing_and_says_so() {
        // InputSet::set checks no range: a design file can hold a finite but huge claim.
        let default = of(&DesignInputs::default());
        let off = |g: &Geometry| -> Vec<String> {
            g.notes
                .iter()
                .filter(|n| n.text.ends_with("is off the drawing"))
                .map(|n| {
                    assert_eq!(n.level, Some(Level::Caution));
                    n.text.clone()
                })
                .collect()
        };
        assert!(off(&default).is_empty());
        let wide = of(&design(|i| i.metal.max_diameter_mm = 1e300));
        // One note for both views; neither draws the claim, and each keeps its pieces' extent.
        assert_eq!(
            off(&wide),
            ["Not drawn: the diameter claim 1.000e300 mm is off the drawing"]
        );
        let r = compute_all(&DesignInputs::default());
        assert_eq!(wide.end.max[0], r.model.cup_od_mm / 2.0 + 2.5);
        assert_eq!(wide.end.dashed.len(), default.end.dashed.len() - 1);
        assert_eq!(wide.side.dashed.len(), default.side.dashed.len() - 1);
        assert!(wide.end.is_drawable() && wide.side.is_drawable());
        assert!(wide.side.max[1] < 30.0, "{:?}", wide.side.max);
        assert_eq!(
            callout(&wide, 6).level,
            None,
            "the callout still reads the claim"
        );
        // The overall length and the bay, either sign.
        let long = of(&design(|i| i.metal.max_overall_axial_mm = 1e39));
        assert_eq!(
            off(&long),
            ["Not drawn: the overall length claim 1.000e39 mm is off the drawing"]
        );
        assert_eq!(long.side.max[0], r.metal.axial_stack_mm + 2.0);
        let bay = of(&design(|i| i.metal.max_large_dia_axial_mm = -1e39));
        assert_eq!(
            off(&bay),
            ["Not drawn: the large-diameter bay claim -1.000e39 mm is off the drawing"]
        );
        assert_eq!(bay.side.max, default.side.max);
        // At the reach the claim is drawn; just past it, it is not.
        let reach = CLAIM_REACH * body_od(&DesignInputs::default(), &r);
        assert!(off(&of(&design(|i| i.metal.max_diameter_mm = reach))).is_empty());
        assert_eq!(
            off(&of(&design(|i| i.metal.max_diameter_mm = reach.next_up()))).len(),
            1
        );
    }
}
