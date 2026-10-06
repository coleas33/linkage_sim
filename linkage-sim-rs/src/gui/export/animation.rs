//! Animated HTML export (plan 2026-10-06 Part B, decisions H-1 to H-8): the
//! current mechanism through every solved sample of its sweep, drawn by a
//! small JS player in one self-contained page (`animation_template.html`).
//!
//! Poses are never solved again: each moving body's pose (the hidden cylinder
//! and rod of a mount-point actuator too) is rebuilt from its angle and the
//! world trace of its first attachment point, both of which the sweep records
//! (`SweepData::body_angles`, `coupler_traces`), so the page shows exactly the
//! samples the plots show. Samples with no solution are
//! left out (decision H-5). Forces come from the helpers the canvas and the
//! sweep use; lengths are millimetres, forces newtons (lbf in the labels,
//! decision H-4).

use nalgebra::{DVector, Vector2};
use serde::Serialize;

use crate::analysis::gravity_breakdown::{gravity_vector, weight_sources};
use crate::core::body::GeometryShape;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::{force_zone_application, ForceElement};
use crate::gui::state::{AppState, LengthUnit};
use crate::gui::sweep::{sweep_time, ShareBasis, SweepData};
use crate::io::{JointJson, MechanismJson};
use crate::solver::reactions::solve_reactions_with_actuator;

/// Newtons per pound-force (decision H-4: lbf beside every force).
const N_PER_LBF: f64 = 4.4482216152605;

/// The page, with `DATA_SLOT` where the JSON goes.
const TEMPLATE: &str = include_str!("animation_template.html");
/// The template's placeholder for the data (valid JS before the swap).
const DATA_SLOT: &str = "/*__ANIMATION_DATA__*/null";

/// Everything the page draws: millimetres in the world frame, newtons.
#[derive(Debug, Serialize)]
pub(crate) struct AnimationData {
    pub title: String,
    /// The driver axis, as the app's plots name it: "Driver angle (deg)" or "Actuator stroke (mm)".
    pub x_label: String,
    /// "deg" or "mm".
    pub x_unit: String,
    /// "Actuator force (N)" (an actuator's, or a linear driver's own), else
    /// "Driver torque (N m)" (decision H-2).
    pub chart_label: String,
    pub n_per_lbf: f64,
    /// The unit vector of gravity; weights hang along it.
    pub gravity: [f64; 2],
    /// The ground pivots of the joints (the hatch marks). An actuator's
    /// anchor is not one, so the view fits the linkage and the actuator runs
    /// off to its anchor, as the hand-built press page did.
    pub ground: Vec<[f64; 2]>,
    /// The driver's fixed pivot, for the driver-angle arc (angle sweeps only).
    pub driver_pivot: Option<[f64; 2]>,
    pub zones: Vec<ZoneBox>,
    /// One frame per solved sweep sample.
    pub frames: Vec<Frame>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ZoneBox {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

#[derive(Debug, Serialize)]
pub(crate) struct Frame {
    /// The driver value as the app shows it: degrees (display frame, BL-041)
    /// or millimetres of stroke.
    pub x: f64,
    /// The chart's value here; `None` where the sweep has none.
    pub chart: Option<f64>,
    /// The driver link's angle for the arc (angle sweeps only), degrees.
    pub driver_angle: Option<f64>,
    pub links: Vec<Link>,
    pub shapes: Vec<Shape>,
    pub joints: Vec<[f64; 2]>,
    pub actuators: Vec<Actuator>,
    pub zone_points: Vec<ZonePoint>,
    pub weights: Vec<Weight>,
    pub reactions: Vec<Reaction>,
    /// The readout panel's rows: (label, value) in the app's units, lbf beside forces.
    pub readouts: Vec<[String; 2]>,
}

/// A moving body: its attachment points in name order.
#[derive(Debug, Serialize)]
pub(crate) struct Link {
    pub name: String,
    pub points: Vec<[f64; 2]>,
    /// Three or more points draw as a plate.
    pub closed: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Shape {
    Polygon { points: Vec<[f64; 2]> },
    Circle { centre: [f64; 2], r: f64 },
}

#[derive(Debug, Serialize)]
pub(crate) struct Actuator {
    pub a: [f64; 2],
    pub b: [f64; 2],
    /// The sweep's force (N, positive = push): the first actuator element's
    /// required force, or in a stroke sweep the first linear driver's.
    pub force: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ZonePoint {
    pub point: [f64; 2],
    pub force: [f64; 2],
    /// The geometry overlaps the zone, so the force applies.
    pub active: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct Weight {
    pub name: String,
    pub point: [f64; 2],
    pub newtons: f64,
    /// A link's own weight (at its centre of gravity), not a payload.
    pub link_self_weight: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct Reaction {
    /// The joint's id; `name` is its label when it has one.
    pub id: String,
    pub name: String,
    pub point: [f64; 2],
    /// The reaction force in the world frame (N), as the canvas draws it.
    pub force: [f64; 2],
}

/// Whether the animation export can run: a mechanism and an angle or stroke
/// sweep. The File menu enables its item by this.
pub fn animation_export_available(state: &AppState) -> bool {
    state.mechanism.is_some() && state.sweep_data.as_ref().is_some_and(|s| !s.sweep_mode.is_trajectory())
}

/// The self-contained animated page of the current mechanism (decision H-6:
/// no network). Every `<` in the JSON is written `\u003c`, the same string to
/// JSON and to JS, so no name can end the script (`</script>`) or switch the
/// HTML parser into a script comment (`<!--` then `<script`).
pub fn generate_animation_html(state: &AppState) -> Result<String, String> {
    let data = animation_data(state)?;
    let json = serde_json::to_string(&data).map_err(|e| e.to_string())?;
    Ok(TEMPLATE.replacen(DATA_SLOT, &json.replace('<', "\\u003c"), 1))
}

/// The page's data: one frame per solved sample of the current sweep.
pub(crate) fn animation_data(state: &AppState) -> Result<AnimationData, String> {
    let (Some(mech), Some(bp)) = (state.mechanism.as_ref(), state.blueprint.as_ref()) else {
        return Err("No mechanism loaded".to_string());
    };
    let Some(sweep) = state.sweep_data.as_ref() else {
        return Err("No sweep computed".to_string());
    };
    if sweep.sweep_mode.is_trajectory() {
        return Err("The animation export needs an angle or stroke sweep".to_string());
    }
    let is_stroke = sweep.sweep_mode.is_stroke();
    // The chart (decision H-2): an actuator element's required force; in a
    // stroke sweep without one, the linear driver's own force (the sweep keeps
    // it in `driver_torques`, which the plots then call the actuator force);
    // else the driver torque.
    let (chart, chart_is_force) = match (&sweep.actuator_forces, is_stroke) {
        (Some(_), _) => (&sweep.actuator_forces, true),
        (None, true) => (&sweep.driver_torques, true),
        (None, false) => (&sweep.driver_torques, false),
    };
    let g = gravity_vector(mech);
    let g_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
    let gravity = if g_norm > 0.0 { [g[0] / g_norm, g[1] / g_norm] } else { [0.0, -1.0] };
    let driver_pivot =
        if is_stroke { None } else { state.driver_joint_id.as_ref().and_then(|id| ground_pivot(bp, id)).map(mm) };
    let ctx = FrameContext { state, mech, bp, sweep, is_stroke, chart, chart_is_force, g_norm, driver_pivot };

    let frames: Vec<Frame> = (0..sweep.angles_deg.len())
        .filter_map(|i| sample_q(mech, sweep, i).map(|q| frame(&ctx, i, &q)))
        .collect();
    if frames.is_empty() {
        return Err("The sweep has no solved sample to animate".to_string());
    }

    let mut joint_ids: Vec<&String> = bp.joints.keys().collect();
    joint_ids.sort();
    let mut ground: Vec<[f64; 2]> = Vec::new();
    for p in joint_ids.iter().filter_map(|id| ground_pivot(bp, id)).map(mm) {
        if !ground.contains(&p) {
            ground.push(p);
        }
    }
    let zones = mech
        .forces()
        .iter()
        .filter_map(|f| match f {
            ForceElement::ForceZone(z) => Some(ZoneBox {
                min: mm(xy(z.zone_min)),
                max: mm(xy(z.zone_max)),
            }),
            _ => None,
        })
        .collect();
    let (x_label, x_unit) =
        if is_stroke { ("Actuator stroke (mm)", "mm") } else { ("Driver angle (deg)", "deg") };
    let chart_label = if chart_is_force { "Actuator force (N)" } else { "Driver torque (N m)" };
    Ok(AnimationData {
        title: format!("Linkage animation: {} links, {} samples", frames[0].links.len(), frames.len()),
        x_label: x_label.to_string(),
        x_unit: x_unit.to_string(),
        chart_label: chart_label.to_string(),
        n_per_lbf: N_PER_LBF,
        gravity,
        ground,
        driver_pivot,
        zones,
        frames,
    })
}

/// What every frame reads.
struct FrameContext<'a> {
    state: &'a AppState,
    mech: &'a Mechanism,
    bp: &'a MechanismJson,
    sweep: &'a SweepData,
    is_stroke: bool,
    /// The chart's series, and whether it is a force (N) or a torque (N m).
    chart: &'a Option<Vec<f64>>,
    chart_is_force: bool,
    g_norm: f64,
    driver_pivot: Option<[f64; 2]>,
}

/// The generalized coordinates of sweep sample `i`, rebuilt from the sweep's
/// own records: each moving body's angle and the world trace of its first
/// attachment point (by name) give its origin. Every body of the built
/// mechanism counts, the blueprint's and the hidden actuator bodies alike.
/// `None` for a sample with no solution (its traces are NaN).
fn sample_q(mech: &Mechanism, sweep: &SweepData, i: usize) -> Option<DVector<f64>> {
    let state = mech.state();
    let mut q = state.make_q();
    for body_id in mech.body_order() {
        let theta = sweep.body_angles.get(body_id)?.get(i)?.to_radians();
        let (name, local) = mech.bodies().get(body_id)?.attachment_points.iter().min_by(|a, b| a.0.cmp(b.0))?;
        let traced = sweep.coupler_traces.get(&format!("{body_id}.{name}"))?.get(i)?;
        if !(theta.is_finite() && traced[0].is_finite() && traced[1].is_finite()) {
            return None;
        }
        let (sin_t, cos_t) = theta.sin_cos();
        let idx = state.get_index(body_id).ok()?;
        q[idx.x_idx()] = traced[0] - (cos_t * local.x - sin_t * local.y);
        q[idx.y_idx()] = traced[1] - (sin_t * local.x + cos_t * local.y);
        q[idx.theta_idx()] = theta;
    }
    Some(q)
}

/// One frame of sample `i` at its rebuilt coordinates `q`.
fn frame(ctx: &FrameContext, i: usize, q: &DVector<f64>) -> Frame {
    let FrameContext { state, mech, bp, sweep, .. } = *ctx;
    let raw_x = sweep.angles_deg[i];
    let x = if ctx.is_stroke { raw_x * 1e3 } else { raw_x + state.driver_display_offset.to_degrees() };
    let world = |body: &str, local: [f64; 2]| mech.state().body_point_global(body, &xy(local), q);

    // The blueprint's bodies only: an actuator's hidden bodies draw as the actuator.
    let links: Vec<Link> = mech
        .body_order()
        .iter()
        .filter_map(|body_id| {
            let body = bp.bodies.get(body_id)?;
            let mut names: Vec<&String> = body.attachment_points.keys().collect();
            names.sort();
            let points: Vec<[f64; 2]> =
                names.iter().map(|n| mm(world(body_id, body.attachment_points[*n]))).collect();
            Some(Link {
                name: body.label.clone().unwrap_or_else(|| body_id.clone()),
                closed: points.len() >= 3,
                points,
            })
        })
        .collect();

    let shapes: Vec<Shape> = mech
        .body_order()
        .iter()
        .filter_map(|body_id| {
            let geo = mech.bodies().get(body_id)?.geometry.as_ref()?;
            let (bx, by, th) = mech.state().get_pose(body_id, q);
            Some(match geo.shape {
                GeometryShape::Circle => Shape::Circle {
                    centre: mm(geo.centre_world(bx, by, th)),
                    r: geo.width / 2.0 * 1e3,
                },
                GeometryShape::Rectangle => Shape::Polygon {
                    points: geo.outline_world(bx, by, th).into_iter().map(mm).collect(),
                },
            })
        })
        .collect();

    let mut joint_ids: Vec<&String> = bp.joints.keys().collect();
    joint_ids.sort();
    let joint_point = |id: &str| match bp.joints.get(id)? {
        JointJson::Revolute { body_i, point_i, .. } | JointJson::Fixed { body_i, point_i, .. } => {
            let local = *bp.bodies.get(body_i)?.attachment_points.get(point_i)?;
            Some(world(body_i, local))
        }
        _ => None,
    };
    let joints: Vec<[f64; 2]> = joint_ids.iter().filter_map(|id| joint_point(id)).map(mm).collect();

    let mut actuators: Vec<Actuator> = Vec::new();
    for la in mech.forces().iter().filter_map(|f| match f {
        ForceElement::LinearActuator(la) => Some(la),
        _ => None,
    }) {
        let force = if actuators.is_empty() { at(&sweep.actuator_forces, i) } else { None };
        actuators.push(Actuator { a: mm(world(&la.body_a, la.point_a)), b: mm(world(&la.body_b, la.point_b)), force });
    }
    // A linear driver is the actuator the user sees; in a stroke sweep the
    // first one's force is the sweep's driver force.
    for (k, ld) in bp.linear_drivers.iter().enumerate() {
        let force = if k == 0 && ctx.is_stroke { at(&sweep.driver_torques, i) } else { None };
        actuators.push(Actuator { a: mm(world(&ld.body_a, ld.point_a)), b: mm(world(&ld.body_b, ld.point_b)), force });
    }

    let zone_points: Vec<ZonePoint> = mech
        .forces()
        .iter()
        .filter_map(|f| match f {
            ForceElement::ForceZone(fz) => {
                let geo = mech.bodies().get(&fz.body_id)?.geometry.as_ref()?;
                let app = force_zone_application(fz, geo, mech.state().get_pose(&fz.body_id, q));
                Some(ZonePoint { point: mm(app.point?), force: fz.force, active: app.active })
            }
            _ => None,
        })
        .collect();

    let weights: Vec<Weight> = weight_sources(bp)
        .into_iter()
        .map(|w| Weight {
            point: mm(world(&w.body_id, w.local_pos)),
            newtons: w.mass * ctx.g_norm,
            link_self_weight: w.is_link_self_weight,
            name: w.name,
        })
        .collect();

    let t = sweep_time(raw_x, ctx.is_stroke, state.driver_omega(), state.driver_theta_0());
    let mut reactions: Vec<Reaction> = match solve_reactions_with_actuator(mech, q, t, state.driver_omega()) {
        Ok(r) => r
            .reactions
            .iter()
            .filter_map(|jr| {
                let point = joint_point(&jr.joint_id)?;
                let label = match bp.joints.get(&jr.joint_id)? {
                    JointJson::Revolute { label, .. } | JointJson::Fixed { label, .. } => label.clone(),
                    _ => None,
                };
                Some(Reaction {
                    id: jr.joint_id.clone(),
                    name: label.unwrap_or_else(|| jr.joint_id.clone()),
                    point: mm(point),
                    force: jr.force_global,
                })
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    // By id: the same order in every export (the solver's follows a hash map).
    reactions.sort_by(|a, b| a.id.cmp(&b.id));

    let chart = at(ctx.chart, i);
    let readouts = readouts(ctx, i, x, &reactions);
    Frame {
        x,
        chart,
        driver_angle: ctx.driver_pivot.map(|_| x),
        links,
        shapes,
        joints,
        actuators,
        zone_points,
        weights,
        reactions,
        readouts,
    }
}

/// The readout rows of sample `i`.
fn readouts(ctx: &FrameContext, i: usize, x: f64, reactions: &[Reaction]) -> Vec<[String; 2]> {
    let sweep = ctx.sweep;
    let mut rows = Vec::new();
    rows.push(if ctx.is_stroke {
        ["Actuator stroke".to_string(), format!("{x:.1} mm")]
    } else {
        ["Driver angle".to_string(), format!("{x:.1} deg")]
    });
    if let Some(v) = at(ctx.chart, i) {
        rows.push(if ctx.chart_is_force {
            let way = if v < 0.0 { "pull" } else { "push" };
            ["Actuator force".to_string(), format!("{} ({way})", force_text(v))]
        } else {
            ["Driver torque".to_string(), format!("{v:.2} N m")]
        });
    }
    if let Some(len) = at(&sweep.actuator_lengths, i) {
        rows.push(["Actuator length".to_string(), length_text(len, ctx.state.display_units.length)]);
    }
    if let Some(ma) = sweep.mechanical_advantage.get(i).copied().filter(|v| v.is_finite()) {
        rows.push(["Mech. advantage".to_string(), format!("{ma:.3}")]);
    }
    if let Some(bd) = sweep.weight_breakdown.as_ref() {
        for (k, source) in bd.sources.iter().enumerate() {
            let Some(share) = bd.force_share.get(k).and_then(|s| s.get(i)).copied().filter(|v| v.is_finite())
            else {
                continue;
            };
            // A share of a force (an actuator's, or a linear driver's in a
            // stroke sweep) is newtons; of a crank's torque, newton metres.
            let value = if matches!(bd.basis, ShareBasis::ActuatorForce) || ctx.is_stroke {
                format!("{:+.0} lbf", share / N_PER_LBF)
            } else {
                format!("{share:+.2} N m")
            };
            rows.push([format!("{} share", source.name), value]);
        }
    }
    for r in reactions {
        let magnitude = (r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt();
        rows.push([format!("Reaction {}", r.name), force_text(magnitude)]);
    }
    rows
}

/// The ground side of joint `id` (metres; ground's frame is the world's),
/// when it is a revolute or fixed joint on ground: the driver's pivot, and
/// the hatch marks.
fn ground_pivot(bp: &MechanismJson, id: &str) -> Option<Vector2<f64>> {
    let (JointJson::Revolute { body_i, body_j, point_i, point_j, .. }
    | JointJson::Fixed { body_i, body_j, point_i, point_j, .. }) = bp.joints.get(id)?
    else {
        return None;
    };
    let point = if body_i == GROUND_ID {
        point_i
    } else if body_j == GROUND_ID {
        point_j
    } else {
        return None;
    };
    Some(xy(*bp.bodies.get(GROUND_ID)?.attachment_points.get(point)?))
}

/// A world point (metres) in millimetres, to the micrometre.
fn mm(p: Vector2<f64>) -> [f64; 2] {
    [(p.x * 1e6).round() / 1e3, (p.y * 1e6).round() / 1e3]
}

fn xy(p: [f64; 2]) -> Vector2<f64> {
    Vector2::new(p[0], p[1])
}

/// Sample `i` of an optional sweep series, if finite.
fn at(series: &Option<Vec<f64>>, i: usize) -> Option<f64> {
    series.as_ref()?.get(i).copied().filter(|v| v.is_finite())
}

/// A force's size with its lbf beside it (decision H-4): kN from 1 kN up,
/// newtons below, so a small mechanism's forces do not read as zero.
fn force_text(newtons: f64) -> String {
    let (n, lbf) = (newtons.abs(), newtons.abs() / N_PER_LBF);
    if n >= 1e3 {
        format!("{:.2} kN / {lbf:.0} lbf", n / 1e3)
    } else {
        format!("{n:.1} N / {lbf:.1} lbf")
    }
}

fn length_text(metres: f64, unit: LengthUnit) -> String {
    match unit {
        LengthUnit::Millimeters => format!("{:.1} mm", metres * 1e3),
        LengthUnit::Meters => format!("{metres:.4} m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::BodyGeometry;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::test_support::swept_lift;

    /// The sweep samples with a solution (every body angle finite).
    fn solved(sweep: &SweepData) -> Vec<usize> {
        (0..sweep.angles_deg.len())
            .filter(|&i| sweep.body_angles.values().all(|a| a[i].is_finite()))
            .collect()
    }

    /// Every frame is a solved sample of `state`'s sweep, with each link's
    /// points (name order) where the sweep traced them.
    fn assert_frames_follow_the_traces(state: &AppState) {
        let data = animation_data(state).unwrap();
        let (sweep, bp, mech) =
            (state.sweep_data.as_ref().unwrap(), state.blueprint.as_ref().unwrap(), state.mechanism.as_ref().unwrap());
        let samples = solved(sweep);
        assert!(samples.len() > 300, "only {} samples solved", samples.len());
        assert_eq!(data.frames.len(), samples.len());
        let drawn: Vec<&String> = mech.body_order().iter().filter(|id| bp.bodies.contains_key(*id)).collect();
        for (frame, &i) in data.frames.iter().zip(&samples) {
            assert_eq!(frame.links.len(), drawn.len());
            for (link, &body_id) in frame.links.iter().zip(&drawn) {
                let mut names: Vec<&String> = bp.bodies[body_id].attachment_points.keys().collect();
                names.sort();
                for (p, name) in link.points.iter().zip(names) {
                    let traced = sweep.coupler_traces[&format!("{body_id}.{name}")][i];
                    assert!(
                        (p[0] - traced[0] * 1e3).abs() < 2e-3 && (p[1] - traced[1] * 1e3).abs() < 2e-3,
                        "sample {i} {body_id}.{name}: {p:?} vs {traced:?} m"
                    );
                }
            }
        }
    }

    #[test]
    fn frames_are_the_sweeps_solved_samples_at_their_traced_positions() {
        assert_frames_follow_the_traces(&swept_lift());
    }

    #[test]
    fn a_body_whose_first_pin_is_off_its_origin_is_placed_by_its_trace() {
        // Every other sample puts each body's first pin (by name) at the body's
        // origin, which would hide a wrong origin in the pose rebuild.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::Strandbeest);
        state.compute_sweep();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(
            mech.body_order().iter().any(|id| {
                let (_, p) = mech.bodies()[id].attachment_points.iter().min_by(|a, b| a.0.cmp(b.0)).unwrap();
                p.norm() > 1e-3
            }),
            "the fixture has a body whose first pin is off its origin"
        );
        assert_frames_follow_the_traces(&state);
    }

    #[test]
    fn the_actuator_draws_pin_to_pin_and_its_hidden_bodies_are_not_links() {
        // The lift's actuator mounts on a mount point, so the built mechanism
        // carries its cylinder and rod as bodies the blueprint does not have.
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        let bp = state.blueprint.as_ref().unwrap();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(mech.body_order().iter().any(|id| !bp.bodies.contains_key(id)), "the fixture has hidden bodies");
        let ForceElement::LinearActuator(la) =
            mech.forces().iter().find(|f| matches!(f, ForceElement::LinearActuator(_))).unwrap()
        else {
            unreachable!()
        };
        // The traced attachment point at each end of the actuator.
        let pin_trace = |body: &str, local: [f64; 2]| {
            let (name, _) = mech.bodies()[body]
                .attachment_points
                .iter()
                .find(|(_, p)| (p.x - local[0]).abs() < 1e-12 && (p.y - local[1]).abs() < 1e-12)
                .unwrap_or_else(|| panic!("{body} has a pin at {local:?}"));
            &sweep.coupler_traces[&format!("{body}.{name}")]
        };
        let ends = [(pin_trace(&la.body_a, la.point_a), "a"), (pin_trace(&la.body_b, la.point_b), "b")];
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            let mut names: Vec<&str> = frame.links.iter().map(|l| l.name.as_str()).collect();
            names.sort();
            let mut want: Vec<String> = mech
                .body_order()
                .iter()
                .filter_map(|id| bp.bodies.get(id).map(|b| b.label.clone().unwrap_or_else(|| id.clone())))
                .collect();
            want.sort();
            assert_eq!(names, want);
            for (trace, end) in ends {
                let (got, traced) = (if end == "a" { frame.actuators[0].a } else { frame.actuators[0].b }, trace[i]);
                assert!(
                    (got[0] - traced[0] * 1e3).abs() < 2e-3 && (got[1] - traced[1] * 1e3).abs() < 2e-3,
                    "sample {i} end {end}: {got:?} vs {traced:?} m"
                );
            }
        }
    }

    #[test]
    fn ground_marks_are_the_joints_pivots_not_the_actuators_anchor() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let bp = state.blueprint.as_ref().unwrap();
        let mut want: Vec<[f64; 2]> = Vec::new();
        for joint in bp.joints.values() {
            let (JointJson::Revolute { body_i, body_j, point_i, point_j, .. }
            | JointJson::Fixed { body_i, body_j, point_i, point_j, .. }) = joint
            else {
                continue;
            };
            for (body, point) in [(body_i, point_i), (body_j, point_j)] {
                if body == GROUND_ID {
                    let p = bp.bodies[GROUND_ID].attachment_points[point];
                    want.push([p[0] * 1e3, p[1] * 1e3]);
                }
            }
        }
        assert!(!want.is_empty());
        assert_eq!(data.ground.len(), want.len(), "{:?} vs {want:?}", data.ground);
        for p in &want {
            assert!(data.ground.iter().any(|g| (g[0] - p[0]).abs() < 1e-6 && (g[1] - p[1]).abs() < 1e-6), "{p:?}");
        }
        // The actuator's ground anchor (its cylinder's base) is not a mark.
        let anchor = data.frames[0].actuators[0].a;
        assert!(data.ground.iter().all(|g| (g[0] - anchor[0]).abs() > 1.0 || (g[1] - anchor[1]).abs() > 1.0));
        // The driver arc sits on the driven joint's ground pivot.
        let pivot = data.driver_pivot.expect("an angle sweep");
        assert!(data.ground.contains(&pivot));
    }

    #[test]
    fn reactions_match_the_sweeps_joint_reaction_magnitudes() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        let mut checked = 0;
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            for r in &frame.reactions {
                let Some(series) = sweep.joint_reaction_magnitudes.get(&r.id) else { continue };
                let (got, want) = ((r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt(), series[i]);
                assert!((got - want).abs() <= 1e-6 * want.max(1.0), "{} at sample {i}: {got} vs {want}", r.id);
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} reactions compared");
    }

    #[test]
    fn the_chart_is_the_actuator_force_with_lbf_in_the_readouts() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert_eq!(data.chart_label, "Actuator force (N)");
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            assert_eq!(frame.chart, at(&sweep.actuator_forces, i));
            assert_eq!(frame.actuators[0].force, frame.chart);
            let row = frame.readouts.iter().find(|r| r[0] == "Actuator force");
            assert!(row.map_or(frame.chart.is_none(), |r| r[1].contains("lbf")), "{:?}", frame.readouts);
        }
    }

    #[test]
    fn without_an_actuator_the_chart_is_the_driver_torque() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let data = animation_data(&state).unwrap();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert_eq!(data.chart_label, "Driver torque (N m)");
        assert!(data.driver_pivot.is_some(), "an angle sweep has the driver arc");
        for (frame, &i) in data.frames.iter().zip(&solved(sweep)) {
            assert_eq!(frame.chart, at(&sweep.driver_torques, i));
            assert!(frame.actuators.is_empty());
        }
    }

    #[test]
    fn a_stroke_sweep_runs_in_millimetres_with_the_linear_driver_as_the_actuator() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        let act = state
            .mechanism
            .as_ref()
            .unwrap()
            .forces()
            .iter()
            .position(|f| matches!(f, ForceElement::LinearActuator(_)))
            .unwrap();
        state.convert_actuator_to_linear_driver(act);
        assert_eq!(state.add_point_mass("rocker", 50.0, [0.0, 0.0]).as_deref(), Some("W1"));
        state.compute_sweep();
        let sweep = state.sweep_data.as_ref().unwrap();
        assert!(sweep.sweep_mode.is_stroke() && sweep.actuator_forces.is_none());
        let data = animation_data(&state).unwrap();
        assert_eq!(
            (data.x_label.as_str(), data.x_unit.as_str(), data.chart_label.as_str()),
            ("Actuator stroke (mm)", "mm", "Actuator force (N)")
        );
        assert!(data.driver_pivot.is_none(), "no driver arc without a crank");
        let samples = solved(sweep);
        assert!(samples.len() > 10, "only {} samples solved", samples.len());
        assert_eq!(data.frames.len(), samples.len());
        for (frame, &i) in data.frames.iter().zip(&samples) {
            assert!((frame.x - sweep.angles_deg[i] * 1e3).abs() < 1e-9, "x is the stroke in mm");
            assert!(frame.driver_angle.is_none());
            assert_eq!(frame.chart, at(&sweep.driver_torques, i));
            // The linear driver draws pin to pin, as long as the stroke says.
            let [act] = frame.actuators.as_slice() else { panic!("{} actuators", frame.actuators.len()) };
            assert_eq!(act.force, frame.chart);
            let length = ((act.b[0] - act.a[0]).powi(2) + (act.b[1] - act.a[1]).powi(2)).sqrt();
            assert!((length - frame.x).abs() < 1e-2, "sample {i}: {length} mm long at stroke {} mm", frame.x);
            if frame.chart.is_some() {
                assert!(frame.readouts.iter().any(|r| r[0] == "Actuator force" && r[1].contains("lbf")));
            }
            let shares: Vec<&[String; 2]> = frame.readouts.iter().filter(|r| r[0].ends_with(" share")).collect();
            assert!(!shares.is_empty() && shares.iter().all(|r| r[1].ends_with("lbf")), "{:?}", frame.readouts);
        }
    }

    #[test]
    fn samples_without_a_solution_are_left_out() {
        let mut state = swept_lift();
        let sweep = state.sweep_data.as_mut().unwrap();
        for angles in sweep.body_angles.values_mut() {
            angles[3] = f64::NAN;
        }
        let gone = sweep.angles_deg[3] + state.driver_display_offset.to_degrees();
        let solved_count = solved(state.sweep_data.as_ref().unwrap()).len();
        let data = animation_data(&state).unwrap();
        assert_eq!(data.frames.len(), solved_count);
        assert!(data.frames.iter().all(|f| (f.x - gone).abs() > 1e-9), "sample 3 is left out");
    }

    /// The Parallelogram Press with a 40 mm wheel for its coupler's rectangle
    /// (same centre), the zone grown over the whole sweep and in contact mode.
    fn press_with_wheel() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);
        let bp = state.blueprint.as_mut().unwrap();
        let coupler = bp.bodies.get_mut("coupler").unwrap();
        let hub = coupler.geometry.as_ref().unwrap().offset;
        coupler.geometry = Some(BodyGeometry::circle(0.04, hub).unwrap());
        for force in &mut bp.forces {
            if let ForceElement::ForceZone(zone) = force {
                zone.zone_min = [-10.0, -10.0];
                zone.zone_max = [10.0, 10.0];
                zone.at_contact_point = true;
            }
        }
        state.rebuild();
        state.compute_sweep();
        state
    }

    #[test]
    fn a_wheel_s_contact_point_is_its_top_against_a_downward_force() {
        let state = press_with_wheel();
        let data = animation_data(&state).unwrap();
        assert!(!data.frames.is_empty());
        for frame in &data.frames {
            let Some(Shape::Circle { centre, r }) = frame.shapes.first() else { panic!("the wheel") };
            assert!((r - 20.0).abs() < 1e-9);
            let zone = &frame.zone_points[0];
            assert!(zone.active);
            assert!((zone.point[0] - centre[0]).abs() < 2e-3 && (zone.point[1] - (centre[1] + 20.0)).abs() < 2e-3);
        }
    }

    #[test]
    fn weights_carry_their_names_and_weight() {
        let state = swept_lift();
        let data = animation_data(&state).unwrap();
        let g = gravity_vector(state.mechanism.as_ref().unwrap());
        let g = (g[0] * g[0] + g[1] * g[1]).sqrt();
        let frame = &data.frames[0];
        for (name, kg) in [("W1", 50.0), ("W2", 20.0)] {
            let w = frame.weights.iter().find(|w| w.name == name).unwrap_or_else(|| panic!("{name}"));
            assert!((w.newtons - kg * g).abs() < 1e-9, "{name}: {}", w.newtons);
            assert!(!w.link_self_weight, "{name} is a payload");
        }
        let bp = state.blueprint.as_ref().unwrap();
        let links: Vec<&Weight> = frame.weights.iter().filter(|w| w.link_self_weight).collect();
        let massive = bp.bodies.iter().filter(|(id, b)| id.as_str() != GROUND_ID && b.mass > 0.0).count();
        assert!(massive > 0 && links.len() == massive, "one self-weight per link with mass");
        assert!(frame.readouts.iter().any(|r| r[0] == "W1 share" && r[1].ends_with("lbf")), "{:?}", frame.readouts);
    }

    #[test]
    fn forces_read_in_kn_or_n_with_lbf_beside_them() {
        assert_eq!(force_text(21_506.0), "21.51 kN / 4835 lbf");
        assert_eq!(force_text(-21_506.0), "21.51 kN / 4835 lbf", "the size; push or pull is said apart");
        assert_eq!(force_text(1_000.0), "1.00 kN / 225 lbf");
        assert_eq!(force_text(12.0), "12.0 N / 2.7 lbf");
    }

    #[test]
    fn reactions_read_with_lbf_beside_them() {
        let state = swept_lift();
        let frame = &animation_data(&state).unwrap().frames[0];
        let rows: Vec<&[String; 2]> = frame.readouts.iter().filter(|r| r[0].starts_with("Reaction ")).collect();
        assert_eq!(rows.len(), frame.reactions.len());
        for (row, r) in rows.iter().zip(&frame.reactions) {
            assert_eq!(row[0], format!("Reaction {}", r.name));
            assert_eq!(row[1], force_text((r.force[0] * r.force[0] + r.force[1] * r.force[1]).sqrt()));
        }
    }

    #[test]
    fn no_sweep_means_no_animation() {
        assert!(animation_data(&AppState::default()).is_err());
        assert!(!animation_export_available(&AppState::default()));
        let mut state = swept_lift();
        assert!(animation_export_available(&state));
        state.sweep_data = None;
        assert!(!animation_export_available(&state));
        assert_eq!(animation_data(&state).unwrap_err(), "No sweep computed");
    }

    #[test]
    fn a_trajectory_sweep_cannot_be_animated() {
        use crate::gui::state::{MotionProfile, Trajectory, TrajectoryProfile};
        use crate::gui::sweep::SweepMode;
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};
        let mut state = swept_lift();
        let profile =
            TrajectoryProfile { shape: MotionProfile::ConstantSpeed, start_value: 0.0, end_value: 1.0, duration: 2.0 };
        state.sweep_data.as_mut().unwrap().sweep_mode = SweepMode::Trajectory {
            target: ControlTarget::Angle { body_id: "crank".to_string() },
            trajectory: Trajectory::Profile(profile),
            severity: Severity::Analysis,
            n_samples: 10,
        };
        assert!(!animation_export_available(&state));
        assert_eq!(animation_data(&state).unwrap_err(), "The animation export needs an angle or stroke sweep");
    }

    #[test]
    fn the_template_has_one_data_slot() {
        assert_eq!(TEMPLATE.matches(DATA_SLOT).count(), 1);
    }

    /// The byte range of the JSON the page embeds after `const DATA = `.
    fn data_span(html: &str) -> (usize, usize) {
        let start = html.find("const DATA = ").unwrap() + "const DATA = ".len();
        (start, start + html[start..].find(";\n").unwrap())
    }

    #[test]
    fn the_page_is_self_contained_and_embeds_the_data() {
        let state = swept_lift();
        let html = generate_animation_html(&state).unwrap();
        assert!(!html.contains("https://") && !html.contains("<script src") && !html.contains("<link"));
        assert_eq!(
            html.matches("http://").count(),
            html.matches("http://www.w3.org/2000/svg").count(),
            "the only http:// is the SVG namespace"
        );
        assert!(!html.contains(DATA_SLOT));
        let (start, end) = data_span(&html);
        let v: serde_json::Value = serde_json::from_str(&html[start..end]).unwrap();
        assert_eq!(v["frames"].as_array().unwrap().len(), animation_data(&state).unwrap().frames.len());
    }

    #[test]
    fn a_name_cannot_end_the_script_or_open_a_comment_in_it() {
        let mut state = swept_lift();
        let bp = state.blueprint.as_mut().unwrap();
        let weight = bp.bodies.values_mut().flat_map(|b| b.point_masses.iter_mut()).find(|w| w.id == "W1").unwrap();
        weight.label = Some("</script><!--<script><b>".to_string());
        let html = generate_animation_html(&state).unwrap();
        // The embedded data holds no `<` at all, so the page has only the template's tags.
        let (start, end) = data_span(&html);
        assert!(!html[start..end].contains('<'), "a raw < in the data");
        // The name still reads back whole.
        let v: serde_json::Value = serde_json::from_str(&html[start..end]).unwrap();
        let names: Vec<&str> =
            v["frames"][0]["weights"].as_array().unwrap().iter().map(|w| w["name"].as_str().unwrap()).collect();
        assert!(names.iter().any(|n| n.starts_with("</script><!--<script><b>")), "{names:?}");
    }

    /// The sorted keys of a JSON object.
    fn keys(v: &serde_json::Value) -> Vec<&str> {
        let mut keys: Vec<&str> = v.as_object().expect("an object").keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }

    #[test]
    fn the_data_has_every_field_the_player_reads() {
        // animation_template.html reads exactly these names; renaming one in
        // Rust would break the page without failing any other test.
        let lift = serde_json::to_value(animation_data(&swept_lift()).unwrap()).unwrap();
        assert_eq!(
            keys(&lift),
            ["chart_label", "driver_pivot", "frames", "gravity", "ground", "n_per_lbf", "title", "x_label", "x_unit", "zones"]
        );
        let frame = &lift["frames"][0];
        assert_eq!(
            keys(frame),
            [
                "actuators", "chart", "driver_angle", "joints", "links", "reactions", "readouts", "shapes", "weights", "x",
                "zone_points"
            ]
        );
        assert_eq!(keys(&frame["links"][0]), ["closed", "name", "points"]);
        assert_eq!(keys(&frame["actuators"][0]), ["a", "b", "force"]);
        assert_eq!(keys(&frame["weights"][0]), ["link_self_weight", "name", "newtons", "point"]);
        assert_eq!(keys(&frame["reactions"][0]), ["force", "id", "name", "point"]);

        let wheel = serde_json::to_value(animation_data(&press_with_wheel()).unwrap()).unwrap();
        assert_eq!(keys(&wheel["zones"][0]), ["max", "min"]);
        let frame = &wheel["frames"][0];
        assert_eq!(keys(&frame["zone_points"][0]), ["active", "force", "point"]);
        assert_eq!(frame["shapes"][0]["kind"], "circle");
        assert_eq!(keys(&frame["shapes"][0]), ["centre", "kind", "r"]);

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);
        state.compute_sweep();
        let press = serde_json::to_value(animation_data(&state).unwrap()).unwrap();
        assert_eq!(press["frames"][0]["shapes"][0]["kind"], "polygon");
        assert_eq!(keys(&press["frames"][0]["shapes"][0]), ["kind", "points"]);
    }
}
