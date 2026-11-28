//! Force element rendering: draws all force element visualizations on the canvas.
//!
//! Handles springs, dampers, external forces/torques, gas springs, motors,
//! linear actuators, bearing friction, joint limits, gravity, and force zones.
//! Also contains load-path visualization helpers (color-coding links by
//! reaction force magnitude).

use eframe::egui::{self, Color32, FontId, Pos2, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::State;
use crate::forces::elements::*;
use crate::gui::canvas::colors::*;
use crate::gui::state::{ActuatorLabelForce, AppState, ViewTransform};

use super::primitives::*;

/// A magnitude (`value` >= 0) in `unit` for a canvas label: one decimal in
/// k<unit> from 999.5 ("1.2 kN"), whole units from 9.995 ("875 N"), two
/// decimals below ("0.35 N").
pub(crate) fn format_magnitude(value: f64, unit: &str) -> String {
    if value >= 999.5 {
        format!("{:.1} k{unit}", value / 1000.0)
    } else if value >= 9.995 {
        format!("{value:.0} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

/// A force (N) whose size is below this shows as "0.00 N" and gets no
/// push/pull word.
pub(crate) const SHOWN_AS_ZERO_N: f64 = 0.005;

/// Canvas actuator label: force magnitude, push or pull, motoring or
/// braking, e.g. "1.2 kN push, braking".
///
/// Push/pull follows the element's sign convention (positive = extension,
/// `forces/elements/evaluation.rs`); a force that shows as zero gets no
/// direction. `power_w` is the required actuator power at this pose
/// (`AppState::actuator_label_power`): motoring above `brake_tol_w`,
/// braking below `-brake_tol_w` (the braking bands' rule), no word in
/// between (the actuator is momentarily still) or when it is `None` or not
/// finite. A non-finite force shows as "-".
pub fn format_actuator_label(force_n: f64, power_w: Option<f64>, brake_tol_w: f64) -> String {
    if !force_n.is_finite() {
        return "-".to_string();
    }
    let mut label = format_magnitude(force_n.abs(), "N");
    if force_n >= SHOWN_AS_ZERO_N {
        label.push_str(" push");
    } else if force_n <= -SHOWN_AS_ZERO_N {
        label.push_str(" pull");
    }
    match power_w {
        Some(p) if p > brake_tol_w => label.push_str(", motoring"),
        Some(p) if p < -brake_tol_w => label.push_str(", braking"),
        _ => {}
    }
    label
}

/// Text of the canvas label of actuator `la`: the statics force the
/// Actuator Force plot draws at this pose (`AppState::actuator_label_force`)
/// with motoring/braking from `AppState::actuator_label_power`; without a
/// usable sweep sample, the element's stored force marked "(stored)".
pub(super) fn actuator_label_text(state: &AppState, la: &LinearActuatorElement) -> String {
    match state.actuator_label_force(la) {
        ActuatorLabelForce::Computed(force) => match state.actuator_label_power() {
            Some((power, tol)) => format_actuator_label(force, Some(power), tol),
            None => format_actuator_label(force, None, 0.0),
        },
        ActuatorLabelForce::Stored(force) => format!("{} (stored)", format_actuator_label(force, None, 0.0)),
    }
}

/// Compute the current world-space application point of a `ForceZoneElement`
/// (`force_zone_application`): the contact point or the locked point at the
/// body's current pose, else the overlap centroid.
///
/// Returns `None` when the target body is missing, has no geometry, or the
/// zone uses the centroid and there is no overlap.
pub fn force_zone_app_point_world(
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let geo = mech.bodies().get(&fz.body_id)?.geometry.as_ref()?;
    force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q)).point
}

// ── Force element rendering ──────────────────────────────────────────────────

/// Draw visual representations of all force elements (springs, dampers, external
/// forces/torques) on the canvas.
///
/// Called when `state.show_forces` is true. Skipped when there is no mechanism.
pub(super) fn draw_force_elements(
    painter: &egui::Painter,
    state: &AppState,
    view: &ViewTransform,
) {
    let mech = match state.mechanism.as_ref() {
        Some(m) => m,
        None => return,
    };
    let mech_state = mech.state();
    let q = &state.q;

    for elem in mech.forces() {
        match elem {
            ForceElement::Gravity(_) => {
                // Already shown via the "g" indicator; skip.
            }
            ForceElement::LinearSpring(s) => {
                let pt_a = mech_state.body_point_global(
                    &s.body_a,
                    &nalgebra::Vector2::new(s.point_a[0], s.point_a[1]),
                    q,
                );
                let pt_b = mech_state.body_point_global(
                    &s.body_b,
                    &nalgebra::Vector2::new(s.point_b[0], s.point_b[1]),
                    q,
                );
                let start_sp = view.world_to_screen(pt_a.x, pt_a.y);
                let end_sp = view.world_to_screen(pt_b.x, pt_b.y);
                let start = Pos2::new(start_sp[0], start_sp[1]);
                let end = Pos2::new(end_sp[0], end_sp[1]);
                draw_spring_zigzag(painter, start, end, SPRING_COLOR);
            }
            ForceElement::LinearDamper(d) => {
                let pt_a = mech_state.body_point_global(
                    &d.body_a,
                    &nalgebra::Vector2::new(d.point_a[0], d.point_a[1]),
                    q,
                );
                let pt_b = mech_state.body_point_global(
                    &d.body_b,
                    &nalgebra::Vector2::new(d.point_b[0], d.point_b[1]),
                    q,
                );
                let start_sp = view.world_to_screen(pt_a.x, pt_a.y);
                let end_sp = view.world_to_screen(pt_b.x, pt_b.y);
                let start = Pos2::new(start_sp[0], start_sp[1]);
                let end = Pos2::new(end_sp[0], end_sp[1]);
                draw_damper_symbol(painter, start, end, DAMPER_COLOR);
            }
            ForceElement::ExternalForce(f) => {
                let pt = mech_state.body_point_global(
                    &f.body_id,
                    &nalgebra::Vector2::new(f.local_point[0], f.local_point[1]),
                    q,
                );
                let sp = view.world_to_screen(pt.x, pt.y);
                let origin = Pos2::new(sp[0], sp[1]);
                draw_external_force_arrow(
                    painter,
                    origin,
                    f.force[0] as f32,
                    f.force[1] as f32,
                    view,
                );
            }
            ForceElement::ExternalTorque(t) => {
                let (bx, by, _) = mech_state.get_pose(&t.body_id, q);
                let sp = view.world_to_screen(bx, by);
                let center = Pos2::new(sp[0], sp[1]);
                draw_torque_arc(painter, center, t.torque as f32, EXT_FORCE_COLOR);
            }
            ForceElement::TorsionSpring(s) => {
                let (xi, yi, _) = mech_state.get_pose(&s.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&s.body_j, q);
                let spi = view.world_to_screen(xi, yi);
                let spj = view.world_to_screen(xj, yj);
                let pi = Pos2::new(spi[0], spi[1]);
                let pj = Pos2::new(spj[0], spj[1]);
                draw_rotary_badge(painter, pi, pj, "k", SPRING_COLOR);
                let mid = Pos2::new((pi.x + pj.x) * 0.5, (pi.y + pj.y) * 0.5);
                painter.text(
                    Pos2::new(mid.x, mid.y - 14.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("k={:.1}", s.stiffness),
                    FontId::proportional(11.0),
                    SPRING_COLOR,
                );
                draw_torque_arc(painter, mid, 1.0, SPRING_COLOR);
            }
            ForceElement::RotaryDamper(d) => {
                let (xi, yi, _) = mech_state.get_pose(&d.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&d.body_j, q);
                let spi = view.world_to_screen(xi, yi);
                let spj = view.world_to_screen(xj, yj);
                let pi = Pos2::new(spi[0], spi[1]);
                let pj = Pos2::new(spj[0], spj[1]);
                draw_rotary_badge(painter, pi, pj, "c", DAMPER_COLOR);
                let mid = Pos2::new((pi.x + pj.x) * 0.5, (pi.y + pj.y) * 0.5);
                painter.text(
                    Pos2::new(mid.x, mid.y - 14.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("c={:.1}", d.damping),
                    FontId::proportional(11.0),
                    DAMPER_COLOR,
                );
                draw_torque_arc(painter, mid, 1.0, DAMPER_COLOR);
            }
            ForceElement::GasSpring(gs) => {
                let pt_a = mech_state.body_point_global(
                    &gs.body_a,
                    &nalgebra::Vector2::new(gs.point_a[0], gs.point_a[1]),
                    q,
                );
                let pt_b = mech_state.body_point_global(
                    &gs.body_b,
                    &nalgebra::Vector2::new(gs.point_b[0], gs.point_b[1]),
                    q,
                );
                let start_sp = view.world_to_screen(pt_a.x, pt_a.y);
                let end_sp = view.world_to_screen(pt_b.x, pt_b.y);
                let start = Pos2::new(start_sp[0], start_sp[1]);
                let end = Pos2::new(end_sp[0], end_sp[1]);
                draw_spring_zigzag(painter, start, end, GAS_SPRING_COLOR);
            }
            ForceElement::LinearActuator(la) => {
                let pt_a = mech_state.body_point_global(
                    &la.body_a,
                    &nalgebra::Vector2::new(la.point_a[0], la.point_a[1]),
                    q,
                );
                let pt_b = mech_state.body_point_global(
                    &la.body_b,
                    &nalgebra::Vector2::new(la.point_b[0], la.point_b[1]),
                    q,
                );
                let start_sp = view.world_to_screen(pt_a.x, pt_a.y);
                let end_sp = view.world_to_screen(pt_b.x, pt_b.y);
                let start = Pos2::new(start_sp[0], start_sp[1]);
                let end = Pos2::new(end_sp[0], end_sp[1]);
                // Draw line of action and an arrow from A toward B.
                let delta = end - start;
                let length = delta.length();
                if length > 2.0 {
                    let dir = delta / length;
                    let stroke = Stroke::new(2.0, ACTUATOR_COLOR);
                    painter.line_segment([start, end], stroke);
                    // Arrowhead at midpoint pointing A -> B.
                    let mid = start + delta * 0.5;
                    draw_arrowhead(painter, mid, dir, ARROW_HEAD_LEN_PX, stroke);
                    // Force label: magnitude, push/pull, motoring/braking
                    // (`actuator_label_text`, unit-tested).
                    let label = actuator_label_text(state, la);
                    painter.text(
                        Pos2::new(mid.x, mid.y - 10.0),
                        egui::Align2::CENTER_BOTTOM,
                        label,
                        FontId::proportional(11.0),
                        ACTUATOR_COLOR,
                    );

                    // Draw stroke limit tick marks
                    let limits_active = la.stroke_max > 0.0 && la.stroke_max > la.stroke_min;
                    if limits_active {
                        let perp = Vec2::new(-dir.y, dir.x);
                        let tick_half = 6.0_f32;
                        let tick_stroke = Stroke::new(1.5, ACTUATOR_COLOR);
                        let scale = view.scale;

                        if la.stroke_min > 0.0 {
                            let min_frac = (la.stroke_min as f32 * scale) / length;
                            if min_frac > 0.0 && min_frac < 1.0 {
                                let tick_pos = Pos2::new(
                                    start.x + delta.x * min_frac,
                                    start.y + delta.y * min_frac,
                                );
                                painter.line_segment(
                                    [
                                        Pos2::new(tick_pos.x + perp.x * tick_half, tick_pos.y + perp.y * tick_half),
                                        Pos2::new(tick_pos.x - perp.x * tick_half, tick_pos.y - perp.y * tick_half),
                                    ],
                                    tick_stroke,
                                );
                            }
                        }

                        if la.stroke_max > 0.0 {
                            let max_frac = (la.stroke_max as f32 * scale) / length;
                            if max_frac > 0.0 && max_frac < 1.0 {
                                let tick_pos = Pos2::new(
                                    start.x + delta.x * max_frac,
                                    start.y + delta.y * max_frac,
                                );
                                painter.line_segment(
                                    [
                                        Pos2::new(tick_pos.x + perp.x * tick_half, tick_pos.y + perp.y * tick_half),
                                        Pos2::new(tick_pos.x - perp.x * tick_half, tick_pos.y - perp.y * tick_half),
                                    ],
                                    tick_stroke,
                                );
                            }
                        }
                    }
                }
            }
            ForceElement::BearingFriction(bf) => {
                let (xi, yi, _) = mech_state.get_pose(&bf.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&bf.body_j, q);
                let spi = view.world_to_screen(xi, yi);
                let spj = view.world_to_screen(xj, yj);
                let pi = Pos2::new(spi[0], spi[1]);
                let pj = Pos2::new(spj[0], spj[1]);
                draw_rotary_badge(painter, pi, pj, "f", BEARING_COLOR);
                let mid = Pos2::new((pi.x + pj.x) * 0.5, (pi.y + pj.y) * 0.5);
                draw_torque_arc(painter, mid, 1.0, BEARING_COLOR);
            }
            ForceElement::JointLimit(jl) => {
                let (xi, yi, _) = mech_state.get_pose(&jl.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&jl.body_j, q);
                let spi = view.world_to_screen(xi, yi);
                let spj = view.world_to_screen(xj, yj);
                let pi = Pos2::new(spi[0], spi[1]);
                let pj = Pos2::new(spj[0], spj[1]);
                draw_rotary_badge(painter, pi, pj, "[", JOINT_LIMIT_COLOR);
                let mid = Pos2::new((pi.x + pj.x) * 0.5, (pi.y + pj.y) * 0.5);
                painter.text(
                    Pos2::new(mid.x, mid.y - 14.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("[{:.1},{:.1}]", jl.angle_min, jl.angle_max),
                    FontId::proportional(11.0),
                    JOINT_LIMIT_COLOR,
                );
                draw_torque_arc(painter, mid, 1.0, JOINT_LIMIT_COLOR);
            }
            ForceElement::Motor(m) => {
                let (xi, yi, _) = mech_state.get_pose(&m.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&m.body_j, q);
                let spi = view.world_to_screen(xi, yi);
                let spj = view.world_to_screen(xj, yj);
                let pi = Pos2::new(spi[0], spi[1]);
                let pj = Pos2::new(spj[0], spj[1]);
                draw_rotary_badge(painter, pi, pj, "M", MOTOR_COLOR);
                let mid = Pos2::new((pi.x + pj.x) * 0.5, (pi.y + pj.y) * 0.5);
                painter.text(
                    Pos2::new(mid.x, mid.y - 14.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("M {:.1}Nm", m.stall_torque),
                    FontId::proportional(11.0),
                    MOTOR_COLOR,
                );
                draw_torque_arc(painter, mid, m.direction as f32, MOTOR_COLOR);
            }
            ForceElement::ForceZone(fz) => {
                draw_force_zone(painter, fz, mech, mech_state, q, view);
            }
        }
    }
}

/// Draw a force zone on the canvas: dashed red border, faint red fill, direction
/// arrows, label, and (if the target body has geometry) a yellow overlap highlight.
fn draw_force_zone(
    painter: &egui::Painter,
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
    view: &ViewTransform,
) {
    let zone_stroke = Stroke::new(2.0, FORCE_ZONE_COLOR);
    let zone_fill = Color32::from_rgba_premultiplied(255, 80, 80, 30);

    // Transform all four world corners; the mounted rectangle is not screen-aligned.
    let corners = [
        [fz.zone_min[0], fz.zone_max[1]],
        [fz.zone_max[0], fz.zone_max[1]],
        [fz.zone_max[0], fz.zone_min[1]],
        [fz.zone_min[0], fz.zone_min[1]],
    ].map(|[x, y]| {
        let sp = view.world_to_screen(x, y);
        Pos2::new(sp[0], sp[1])
    });
    let [tl, tr, br, bl] = corners;
    let s_min_x = corners.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let s_min_y = corners.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let s_max_x = corners.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);

    // 1. Faint red fill.
    painter.add(egui::epaint::PathShape::convex_polygon(
        corners.to_vec(),
        zone_fill,
        Stroke::NONE,
    ));

    // 2. Dashed red border (4 edges).
    let dash = 6.0_f32;
    let gap = 4.0_f32;
    draw_dashed_line(painter, tl, tr, zone_stroke, dash, gap);
    draw_dashed_line(painter, tr, br, zone_stroke, dash, gap);
    draw_dashed_line(painter, br, bl, zone_stroke, dash, gap);
    draw_dashed_line(painter, bl, tl, zone_stroke, dash, gap);

    // 3. Force direction arrows: 3 evenly spaced inside the zone.
    let fx = fz.force[0] as f32;
    let fy = fz.force[1] as f32;
    let fmag = (fx * fx + fy * fy).sqrt();
    if fmag > 1e-9 {
        let direction = world_direction_to_screen(view, fx / fmag, fy / fmag);
        let zone_x = tr - tl;
        let zone_y = bl - tl;
        let zone_w = zone_x.length();
        let zone_h = zone_y.length();
        let arrow_len = (zone_w.min(zone_h) * 0.35).clamp(10.0, 40.0);
        let head_len = 6.0_f32;

        for i in 0..3 {
            let frac = (i as f32 + 1.0) / 4.0;
            let center = tl + zone_x * frac + zone_y * 0.5;
            let half = arrow_len * 0.5;
            let start = center - direction * half;
            let tip = center + direction * half;

            // Arrow shaft and a small arrowhead at the tip.
            let stroke = Stroke::new(1.5, FORCE_ZONE_COLOR);
            painter.line_segment([start, tip], stroke);
            draw_arrowhead(painter, tip, direction, head_len, stroke);
        }
    }

    // 4. Label above the zone.
    let label_text = fz.label.as_deref().unwrap_or("Force Zone");
    painter.text(
        Pos2::new((s_min_x + s_max_x) * 0.5, s_min_y - 4.0),
        egui::Align2::CENTER_BOTTOM,
        label_text,
        FontId::monospace(10.0),
        FORCE_ZONE_COLOR,
    );

    // 5. Active overlap highlight + application-point marker.
    if let Some(body) = mech.bodies().get(&fz.body_id)
        && let Some(ref geo) = body.geometry {
            let zone = force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q));
            if zone.active {
                let screen_verts: Vec<Pos2> = zone
                    .overlap
                    .iter()
                    .map(|v| {
                        let sp = view.world_to_screen(v.x, v.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_verts,
                    FORCE_ZONE_OVERLAP_FILL,
                    Stroke::new(1.0, FORCE_ZONE_OVERLAP_STROKE),
                ));
            }

            // Application-point marker. Drawn whenever the point exists: a
            // locked or contact point even before the body reaches the zone,
            // so the user sees where the force will act; the centroid only
            // with overlap. Shown as a crosshair + dot. A locked or contact
            // point uses a stronger color; the auto-centroid a softer one.
            let (color, label) = match zone.mode {
                ZoneAppMode::Centroid => (Color32::from_rgb(255, 215, 120), "F"),
                ZoneAppMode::Locked => (Color32::from_rgb(255, 165, 80), "F (locked)"),
                ZoneAppMode::Contact => (Color32::from_rgb(255, 165, 80), "F (contact)"),
            };
            if let Some(app_w) = zone.point {
                let sp = view.world_to_screen(app_w.x, app_w.y);
                let center = Pos2::new(sp[0], sp[1]);
                let stroke = Stroke::new(1.5, color);
                let half = 8.0_f32;
                // Crosshair.
                painter.line_segment(
                    [
                        Pos2::new(center.x - half, center.y),
                        Pos2::new(center.x + half, center.y),
                    ],
                    stroke,
                );
                painter.line_segment(
                    [
                        Pos2::new(center.x, center.y - half),
                        Pos2::new(center.x, center.y + half),
                    ],
                    stroke,
                );
                // Filled center dot.
                painter.circle_filled(center, 3.5, color);
                // Keep the marker's semantic color while a measured dark
                // plate makes the small label readable over yellow overlap.
                let galley = painter.layout_no_wrap(label.to_string(), FontId::monospace(10.0), color);
                let text_pos = Pos2::new(center.x + half + 3.0, center.y - galley.size().y * 0.5);
                let label_rect = egui::Rect::from_min_size(text_pos, galley.size()).expand2(egui::vec2(3.0, 2.0));
                painter.rect_filled(label_rect, 3.0, Color32::from_rgb(20, 22, 30));
                painter.galley(text_pos, galley, color);
            }
        }
}


// ── Load path visualization helpers ─────────────────────────────────────────

/// Map a normalized value [0, 1] to a blue-cyan-green-yellow-red heat gradient.
///
/// Used by the load path visualization to color-code links by force magnitude.
pub fn heat_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    // 5-stop gradient: blue -> cyan -> green -> yellow -> red
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)                         // blue -> cyan
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)                   // cyan -> green
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)                         // green -> yellow
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)                   // yellow -> red
    };
    Color32::from_rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

/// Compute the load path color for a body based on the maximum joint reaction
/// force magnitude at its attachment points.
///
/// Returns `None` if load path visualization is disabled or there are no
/// reaction forces available for this body's joints.
pub(super) fn load_path_color_for_body(
    body_id: &str,
    mech: &Mechanism,
    state: &AppState,
) -> Option<Color32> {
    if !state.show_load_path {
        return None;
    }
    if state.force_results.joint_reactions.is_empty() {
        return None;
    }

    // Find the maximum force magnitude across ALL joints (for normalization).
    let global_max = state
        .force_results
        .joint_reactions
        .values()
        .map(|(fx, fy)| (fx * fx + fy * fy).sqrt())
        .fold(0.0_f64, f64::max);

    if global_max < 1e-12 {
        return None;
    }

    // Find the maximum force magnitude at joints connected to this body.
    let mut body_max = 0.0_f64;
    for joint in mech.joints() {
        if (joint.body_i_id() == body_id || joint.body_j_id() == body_id)
            && let Some(&(fx, fy)) = state.force_results.joint_reactions.get(joint.id()) {
                let mag = (fx * fx + fy * fy).sqrt();
                body_max = body_max.max(mag);
            }
    }

    let t = (body_max / global_max) as f32;
    Some(heat_color(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};
    use crate::gui::test_support::{set_actuator_stored_force, swept_lift};

    // ── BL-046: application labels remain legible over overlap geometry ──

    fn bl046_zone_label_frame(shape: crate::core::body::GeometryShape, mode: ZoneAppMode, active: bool) -> egui::FullOutput {
        use crate::core::body::{Body, BodyGeometry, GeometryShape};
        let mut mech = Mechanism::new();
        let mut body = Body::new("body");
        body.geometry = Some(match shape {
            GeometryShape::Rectangle => BodyGeometry::new(2.0, 2.0, nalgebra::Vector2::zeros()).unwrap(),
            GeometryShape::Circle => BodyGeometry::circle(2.0, nalgebra::Vector2::zeros()).unwrap(),
        });
        mech.add_body(body).unwrap();
        mech.build().unwrap();
        let q = mech.state().make_q();
        let zone = ForceZoneElement {
            body_id: "body".into(),
            zone_min: if active { [-2.0, -2.0] } else { [3.0, 3.0] },
            zone_max: if active { [2.0, 2.0] } else { [4.0, 4.0] },
            force: [0.0, 5.0], label: Some("Label contrast fixture".into()),
            body_local_app_point: if mode == ZoneAppMode::Centroid { None } else { Some([0.0, 0.0]) },
            at_contact_point: mode == ZoneAppMode::Contact,
        };
        let view = bl049_view(0.0);
        crate::gui::test_support::central_panel_frame(&egui::Context::default(), Vec::new(), |ui| {
            draw_force_zone(ui.painter(), &zone, &mech, mech.state(), &q, &view);
        })
    }

    fn bl046_assert_plate_and_existing_marker(output: &egui::FullOutput, mode: ZoneAppMode, active: bool) {
        let (label, color) = match mode {
            ZoneAppMode::Centroid => ("F", Color32::from_rgb(255, 215, 120)),
            ZoneAppMode::Locked => ("F (locked)", Color32::from_rgb(255, 165, 80)),
            ZoneAppMode::Contact => ("F (contact)", Color32::from_rgb(255, 165, 80)),
        };
        let center = Pos2::new(200.0, if mode == ZoneAppMode::Contact { 400.0 } else { 300.0 });
        let mut text = None;
        let mut rectangles = Vec::new();
        let mut overlap_order = None;
        let mut marker_dot = false;
        let mut order = 0usize;
        crate::gui::test_support::visit_shapes(output, |shape| {
            match shape {
                egui::Shape::Text(drawn) if drawn.galley.text() == label => {
                    text = Some((order, drawn.clone()));
                }
                egui::Shape::Rect(rect) => rectangles.push((order, rect.clone())),
                egui::Shape::Path(path) if path.fill == FORCE_ZONE_OVERLAP_FILL => overlap_order = Some(order),
                egui::Shape::Circle(circle) if circle.center.distance(center) < 0.001
                    && (circle.radius - 3.5).abs() < 0.001 && circle.fill == color => marker_dot = true,
                _ => {}
            }
            order += 1;
        });
        let (text_order, text) = text.expect("the actual application label is painted");
        assert!((text.pos.x - 211.0).abs() < 0.001, "keep the existing LEFT_CENTER label anchor x");
        assert!((text.pos.y + text.galley.size().y * 0.5 - center.y).abs() < 0.001,
            "keep the existing label center y");
        assert!(!text.galley.job.sections.is_empty());
        for section in &text.galley.job.sections {
            assert_eq!(section.format.font_id, FontId::monospace(10.0), "preserve the label font");
            assert_eq!(section.format.color, color, "preserve the semantic label color");
        }
        let text_rect = egui::Rect::from_min_size(text.pos, text.galley.size());
        let (plate_order, _) = rectangles.iter().find(|(_, plate)| {
            plate.fill.a() == 255 && plate.fill.r() <= 35 && plate.fill.g() <= 35 && plate.fill.b() <= 35
                && plate.rect.contains_rect(text_rect)
                && plate.rect.width() > text_rect.width()
                && plate.rect.height() > text_rect.height()
                && plate.rect.width() <= text_rect.width() + 12.0
                && plate.rect.height() <= text_rect.height() + 8.0
        }).expect("a measured, opaque dark backing plate must contain the complete application label");
        assert!(*plate_order < text_order, "plate must be painted before its text");
        if active {
            assert!(overlap_order.expect("literal body and zone overlap") < *plate_order,
                "plate must cover the overlap highlight rather than sit below it");
        } else {
            assert!(overlap_order.is_none(), "the no-overlap control must actually be inactive");
        }
        assert!(marker_dot, "preserve the application point's orange/yellow dot");
        let crosshair = bl049_lines(output, color, 1.5);
        bl049_assert_line(&crosshair, Pos2::new(center.x - 8.0, center.y), Pos2::new(center.x + 8.0, center.y));
        bl049_assert_line(&crosshair, Pos2::new(center.x, center.y - 8.0), Pos2::new(center.x, center.y + 8.0));
    }

    #[test]
    fn bl046_zone_application_labels_have_a_plate_above_yellow_overlap() {
        use crate::core::body::GeometryShape;
        for shape in [GeometryShape::Rectangle, GeometryShape::Circle] {
            for mode in [ZoneAppMode::Centroid, ZoneAppMode::Locked, ZoneAppMode::Contact] {
                let output = bl046_zone_label_frame(shape, mode, true);
                bl046_assert_plate_and_existing_marker(&output, mode, true);
            }
        }
    }

    #[test]
    fn bl046_locked_and_contact_label_plates_remain_visible_without_overlap() {
        use crate::core::body::GeometryShape;
        for shape in [GeometryShape::Rectangle, GeometryShape::Circle] {
            for mode in [ZoneAppMode::Locked, ZoneAppMode::Contact] {
                let output = bl046_zone_label_frame(shape, mode, false);
                bl046_assert_plate_and_existing_marker(&output, mode, false);
            }
            let centroid = bl046_zone_label_frame(shape, ZoneAppMode::Centroid, false);
            assert!(!crate::gui::test_support::drew_text(&centroid, "F"),
                "inactive automatic centroid has no application point to label");
        }
    }

    fn bl049_view(angle: f64) -> ViewTransform {
        ViewTransform { offset: [200.0, 300.0], scale: 100.0, mounting_angle: angle }
    }

    fn bl049_lines(output: &egui::FullOutput, color: Color32, width: f32) -> Vec<[Pos2; 2]> {
        let mut lines = Vec::new();
        crate::gui::test_support::visit_shapes(output, |shape| {
            if let egui::Shape::LineSegment { points, stroke } = shape
                && stroke.color == color && (stroke.width - width).abs() < 1e-6
            {
                lines.push(*points);
            }
        });
        lines
    }

    fn bl049_assert_line(lines: &[[Pos2; 2]], from: Pos2, to: Pos2) {
        assert!(lines.iter().any(|line| line[0].distance(from) < 0.002 && line[1].distance(to) < 0.002),
            "missing screen segment {from:?} -> {to:?}; got {lines:?}");
    }

    fn bl049_overlay(angle: f64, components: bool, external: bool) -> egui::FullOutput {
        let mut state = AppState {
            view: bl049_view(angle),
            mounting_angle: angle,
            show_forces: true,
            show_force_components: components,
            gravity_magnitude: 0.0,
            ..Default::default()
        };
        let joint_targets = if external {
            let mut mech = Mechanism::new();
            mech.add_body(crate::core::body::Body::new("body")).unwrap();
            mech.add_force(ForceElement::ExternalForce(ExternalForceElement {
                body_id: "body".into(), local_point: [0.0, 0.0], local_point_name: None,
                force: [1.0, 0.0], modulation: TimeModulation::default(),
            }));
            mech.build().unwrap();
            state.q = mech.state().make_q();
            state.mechanism = Some(mech);
            Vec::new()
        } else {
            state.force_results.joint_reactions.insert("J1".into(), (1.0, if components { -1.0 } else { 0.0 }));
            vec![(Pos2::new(200.0, 300.0), "J1".into())]
        };
        crate::gui::test_support::central_panel_frame(&egui::Context::default(), Vec::new(), |ui| {
            let painter = ui.painter().clone();
            let canvas_rect = ui.max_rect();
            super::super::render_overlays(ui, &painter, canvas_rect, &state,
                &joint_targets, &[], &[], true, 0.0, 0, false);
        })
    }

    #[test]
    fn bl049_mounted_reaction_arrow_uses_literal_screen_direction() {
        // One newton gives a 30 px shaft. Expected coordinates are literal
        // cos(30)=sqrt(3)/2, sin(30)=1/2, followed by screen-Y reversal.
        for (angle, tip) in [(0.0, Pos2::new(230.0, 300.0)),
            (std::f64::consts::PI / 6.0, Pos2::new(225.98076, 285.0))]
        {
            let output = bl049_overlay(angle, false, false);
            bl049_assert_line(&bl049_lines(&output, FORCE_ARROW_COLOR, FORCE_ARROW_WIDTH),
                Pos2::new(200.0, 300.0), tip);
            assert!(crate::gui::test_support::drew_text(&output, "1 N"));
        }
    }

    #[test]
    fn bl049_mounted_component_arrows_keep_world_components_and_styles() {
        let output = bl049_overlay(std::f64::consts::PI / 6.0, true, false);
        let lines = bl049_lines(&output, FORCE_ARROW_COLOR, FORCE_ARROW_WIDTH);
        // Fx=+1 is solid along mounted world X. Fy=-1 is dashed along
        // negative mounted world Y; its first dash is exactly 5 px.
        bl049_assert_line(&lines, Pos2::new(200.0, 300.0), Pos2::new(225.98076, 285.0));
        bl049_assert_line(&lines, Pos2::new(200.0, 300.0), Pos2::new(202.5, 304.33014));
        let fy_tip = Pos2::new(215.0, 325.98077);
        assert_eq!(lines.iter().filter(|line| line[0].distance(fy_tip) < 0.002).count(), 2,
            "the solid two-line Fy head stays at the mounted tip");
        assert!(crate::gui::test_support::drew_text(&output, "Fx = 1 N"));
        assert!(crate::gui::test_support::drew_text(&output, "Fy = -1 N"));
    }

    #[test]
    fn bl049_mounted_external_arrow_keeps_its_tip_at_the_application_point() {
        for (angle, tail) in [(0.0, Pos2::new(170.0, 300.0)),
            (std::f64::consts::PI / 6.0, Pos2::new(174.01924, 315.0))]
        {
            let output = bl049_overlay(angle, false, true);
            bl049_assert_line(&bl049_lines(&output, EXT_FORCE_COLOR, FORCE_ARROW_WIDTH),
                tail, Pos2::new(200.0, 300.0));
            assert!(crate::gui::test_support::drew_text(&output, "1 N"));
        }
    }

    fn bl049_zone_frame() -> egui::FullOutput {
        let mech = Mechanism::new();
        let zone = ForceZoneElement {
            body_id: "absent".into(), zone_min: [0.0, 0.0], zone_max: [2.0, 1.0],
            force: [1.0, 0.0], label: Some("Mounted zone".into()),
            body_local_app_point: None, at_contact_point: false,
        };
        let view = bl049_view(std::f64::consts::PI / 6.0);
        crate::gui::test_support::central_panel_frame(&egui::Context::default(), Vec::new(), |ui| {
            draw_force_zone(ui.painter(), &zone, &mech, mech.state(), &nalgebra::DVector::zeros(0), &view);
        })
    }

    #[test]
    fn bl049_mounted_zone_draws_its_four_literal_corners_and_dashed_edges() {
        let output = bl049_zone_frame();
        // World corners (0,1), (2,1), (2,0), (0,0), mounted at 30 deg,
        // scale 100 px/m and translated to (200,300).
        let corners = [Pos2::new(150.0, 213.39746), Pos2::new(323.20508, 113.39746),
            Pos2::new(373.20508, 200.0), Pos2::new(200.0, 300.0)];
        let fill = Color32::from_rgba_premultiplied(255, 80, 80, 30);
        let mut filled_corners = None;
        crate::gui::test_support::visit_shapes(&output, |shape| {
            if let egui::Shape::Path(path) = shape
                && path.fill == fill
            {
                filled_corners = Some(path.points.clone());
            }
        });
        let filled = filled_corners.expect("the zone fill is its four-corner polygon");
        assert_eq!(filled.len(), 4);
        for corner in corners {
            assert!(filled.iter().any(|point| point.distance(corner) < 0.002), "missing literal corner {corner:?}: {filled:?}");
        }
        let lines = bl049_lines(&output, FORCE_ZONE_COLOR, 2.0);
        for (corner, dash_end) in corners.into_iter().zip([
            Pos2::new(155.19615, 210.39746), Pos2::new(326.20508, 118.59361),
            Pos2::new(368.00894, 203.0), Pos2::new(197.0, 294.80386),
        ]) {
            bl049_assert_line(&lines, corner, dash_end);
        }
        assert!(crate::gui::test_support::drew_text(&output, "Mounted zone"));
    }

    #[test]
    fn bl049_mounted_zone_arrows_use_rotated_centres_and_literal_direction() {
        let output = bl049_zone_frame();
        let shafts: Vec<_> = bl049_lines(&output, FORCE_ZONE_COLOR, 1.5).into_iter()
            .filter(|line| line[0].distance(line[1]) > 20.0).collect();
        assert_eq!(shafts.len(), 3);
        // The original 35 px shafts retain their length. Their centres lie
        // at world (.5,.5), (1,.5), (1.5,.5), not the screen bounding box.
        for (tail, tip) in [
            (Pos2::new(203.14583, 240.44873), Pos2::new(233.45671, 222.94873)),
            (Pos2::new(246.4471, 215.44873), Pos2::new(276.758, 197.94873)),
            (Pos2::new(289.74835, 190.44873), Pos2::new(320.05927, 172.94873)),
        ] {
            bl049_assert_line(&shafts, tail, tip);
        }
    }

    /// Every force sign (push, pull, shown as zero) against every power
    /// case (motoring, braking, inside the tolerance, unknown).
    #[test]
    fn format_actuator_label_words_for_every_sign_combination() {
        let tol = 1.0;
        let cases: [(f64, Option<f64>, &str); 16] = [
            (1234.0, Some(500.0), "1.2 kN push, motoring"),
            (1234.0, Some(-500.0), "1.2 kN push, braking"),
            (-1234.0, Some(500.0), "1.2 kN pull, motoring"),
            (-1234.0, Some(-500.0), "1.2 kN pull, braking"),
            (875.0, Some(0.5), "875 N push"),
            (-875.0, Some(-0.5), "875 N pull"),
            (875.0, Some(1.0), "875 N push"),
            (875.0, Some(-1.0), "875 N push"),
            (875.0, None, "875 N push"),
            (-875.0, None, "875 N pull"),
            (875.0, Some(f64::NAN), "875 N push"),
            (0.0, Some(500.0), "0.00 N, motoring"),
            (0.004, Some(-500.0), "0.00 N, braking"),
            (-0.004, None, "0.00 N"),
            (-0.0, Some(0.0), "0.00 N"),
            (0.005, None, "0.01 N push"),
        ];
        for (force, power, want) in cases {
            assert_eq!(format_actuator_label(force, power, tol), want, "force {force}, power {power:?}");
        }
    }

    #[test]
    fn format_actuator_label_switches_units_at_the_rounding_edges() {
        let label = |f: f64| format_actuator_label(f, None, 0.0);
        assert_eq!(label(0.35), "0.35 N push");
        assert_eq!(label(9.994), "9.99 N push");
        assert_eq!(label(9.995), "10 N push");
        assert_eq!(label(999.4), "999 N push");
        assert_eq!(label(999.5), "1.0 kN push");
        assert_eq!(label(-12_345.0), "12.3 kN pull");
    }

    #[test]
    fn format_actuator_label_shows_a_dash_for_a_non_finite_force() {
        assert_eq!(format_actuator_label(f64::NAN, Some(5.0), 1.0), "-");
        assert_eq!(format_actuator_label(f64::INFINITY, None, 0.0), "-");
    }

    fn first_actuator(state: &AppState) -> LinearActuatorElement {
        state
            .mechanism
            .as_ref()
            .unwrap()
            .forces()
            .iter()
            .find_map(|f| match f {
                ForceElement::LinearActuator(la) => Some(la.clone()),
                _ => None,
            })
            .expect("the sample has a LinearActuator")
    }

    /// Since BL-026 the Actuator Force and Actuator Power plots draw the
    /// REQUIRED force and power whatever the element's stored force, and the
    /// label's words follow them: push/pull from the plotted force's sign,
    /// motoring/braking from the plotted power, i.e. the braking bands
    /// (`WeightBreakdown::braking`). Checked at every sample in sizing mode
    /// (stored force 0), with the sample's stored force, and with a stored
    /// force above every required force, where a label that read
    /// F_required - F_stored would say "pull" at every push sample.
    #[test]
    fn actuator_label_words_follow_the_plotted_required_force_and_power_in_every_mode() {
        let mut state = swept_lift();
        let shipped = first_actuator(&state).force;
        assert!(shipped > 0.0, "fixture: the sample stores a force");
        set_actuator_stored_force(&mut state, 0.0);
        state.compute_sweep();
        let required = state.sweep_data.as_ref().unwrap().actuator_forces.clone().unwrap();
        let force_tol = 1e-9 * max_abs_finite(&required);

        for stored in [0.0, shipped, 2.0 * max_abs_finite(&required)] {
            set_actuator_stored_force(&mut state, stored);
            state.compute_sweep();
            let la = first_actuator(&state);
            let sweep = state.sweep_data.clone().unwrap();
            let forces = sweep.actuator_forces.clone().unwrap();
            let power = sweep.actuator_power.clone().unwrap();
            let breakdown = sweep.weight_breakdown.clone().unwrap();
            let power_tol = BRAKE_TOL_REL * max_abs_finite(&breakdown.total_power);
            let (mut push, mut motoring, mut braking) = (0, 0, 0);
            for (k, &deg) in sweep.angles_deg.iter().enumerate() {
                if deg >= 360.0 || !forces[k].is_finite() {
                    continue; // 360 is the pose of 0; failed samples show the stored force
                }
                let what = format!("stored {stored} N, {deg} deg");
                assert!((forces[k] - required[k]).abs() <= force_tol, "{what}: plotted {} N, required {} N", forces[k], required[k]);
                state.driver_angle = deg.to_radians();
                let label = actuator_label_text(&state, &la);
                let what = format!("{what}: {label:?}");
                assert_eq!(label.contains(" push"), forces[k] >= SHOWN_AS_ZERO_N, "{what}");
                assert_eq!(label.contains(" pull"), forces[k] <= -SHOWN_AS_ZERO_N, "{what}");
                assert_eq!(label.ends_with(", braking"), breakdown.braking[k], "{what}: the braking bands");
                assert_eq!(label.ends_with(", motoring"), breakdown.total_power[k] > power_tol, "{what}");
                if power[k].is_finite() && power[k].abs() > 2.0 * power_tol {
                    assert_eq!(label.ends_with(", motoring"), power[k] > 0.0, "{what}: plotted power {} W", power[k]);
                    assert_eq!(label.ends_with(", braking"), power[k] < 0.0, "{what}: plotted power {} W", power[k]);
                }
                push += usize::from(label.contains(" push"));
                motoring += usize::from(label.ends_with(", motoring"));
                braking += usize::from(label.ends_with(", braking"));
            }
            assert!(
                push > 10 && motoring > 10 && braking > 10,
                "fixture, stored {stored} N: the lift pushes ({push}), motors ({motoring}) and brakes ({braking})"
            );
        }
    }

    #[test]
    fn actuator_label_falls_back_to_the_stored_force_without_a_sweep() {
        let mut state = swept_lift();
        let la = first_actuator(&state);
        state.sweep_data = None;
        assert_eq!(actuator_label_text(&state, &la), format!("{} (stored)", format_actuator_label(la.force, None, 0.0)));
        assert_eq!(actuator_label_text(&state, &la), "50 N push (stored)", "the sample stores 50 N");
    }
}
