//! Force element rendering: draws all force element visualizations on the canvas.
//!
//! Handles springs, dampers, external forces/torques, gas springs, motors,
//! linear actuators, bearing friction, joint limits, gravity, and force zones.
//! Also contains load-path visualization helpers (color-coding links by
//! reaction force magnitude).

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::State;
use crate::forces::elements::*;
use crate::gui::canvas::colors::*;
use crate::gui::state::{AppState, ViewTransform};

use super::primitives::*;

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
                    painter.line_segment(
                        [start, end],
                        Stroke::new(2.0, ACTUATOR_COLOR),
                    );
                    // Arrowhead at midpoint pointing A -> B.
                    let mid = Pos2::new(
                        start.x + delta.x * 0.5,
                        start.y + delta.y * 0.5,
                    );
                    let head_len = 8.0_f32;
                    let head_angle = 0.44_f32;
                    let back_dx = -dir.x;
                    let back_dy = -dir.y;
                    for sign in [-1.0_f32, 1.0] {
                        let cos_a = head_angle.cos();
                        let sin_a = head_angle.sin() * sign;
                        let hx = back_dx * cos_a - back_dy * sin_a;
                        let hy = back_dx * sin_a + back_dy * cos_a;
                        let head_end = Pos2::new(mid.x + hx * head_len, mid.y + hy * head_len);
                        painter.line_segment(
                            [mid, head_end],
                            Stroke::new(2.0, ACTUATOR_COLOR),
                        );
                    }
                    // Force magnitude label. If the actuator's stored force
                    // is zero (sizing mode — the user wants the solver to
                    // back-calculate what force is needed), look up the
                    // computed actuator force at the current driver angle
                    // from the sweep data instead of showing "0 N".
                    let display_force = if la.force.abs() < 1e-12 {
                        state.sweep_data.as_ref().and_then(|sweep| {
                            let forces = sweep.actuator_forces.as_ref()?;
                            if forces.is_empty() || sweep.angles_deg.is_empty() {
                                return None;
                            }
                            // Find nearest angle index to the current driver angle
                            let current_deg = state.driver_angle.to_degrees().rem_euclid(360.0);
                            let idx = sweep.angles_deg
                                .iter()
                                .enumerate()
                                .min_by(|(_, a), (_, b)| {
                                    (*a - current_deg).abs()
                                        .partial_cmp(&(*b - current_deg).abs())
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i)?;
                            let f = forces.get(idx).copied()?;
                            if f.is_finite() { Some(f) } else { None }
                        })
                    } else {
                        None
                    };

                    let label = match display_force {
                        Some(f) => format!("{:.0} N (computed)", f),
                        None => format!("{:.0} N", la.force),
                    };
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
                        let scale = view.scale as f32;

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
                draw_force_zone(painter, &fz, mech, mech_state, q, view);
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

    // Convert zone corners to screen space.
    let min_sp = view.world_to_screen(fz.zone_min[0], fz.zone_min[1]);
    let max_sp = view.world_to_screen(fz.zone_max[0], fz.zone_max[1]);
    let s_min_x = min_sp[0].min(max_sp[0]);
    let s_min_y = min_sp[1].min(max_sp[1]);
    let s_max_x = min_sp[0].max(max_sp[0]);
    let s_max_y = min_sp[1].max(max_sp[1]);

    let tl = Pos2::new(s_min_x, s_min_y);
    let tr = Pos2::new(s_max_x, s_min_y);
    let br = Pos2::new(s_max_x, s_max_y);
    let bl = Pos2::new(s_min_x, s_max_y);

    // 1. Faint red fill.
    painter.rect_filled(
        Rect::from_min_max(tl, br),
        0.0,
        zone_fill,
    );

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
        // Unit direction in screen space (flip Y because screen Y is down).
        let dir_x = fx / fmag;
        let dir_y = -fy / fmag;

        let zone_w = s_max_x - s_min_x;
        let zone_h = s_max_y - s_min_y;
        let arrow_len = (zone_w.min(zone_h) * 0.35).clamp(10.0, 40.0);
        let head_len = 6.0_f32;
        let head_angle = 0.44_f32;

        for i in 0..3 {
            let frac = (i as f32 + 1.0) / 4.0;
            let cx = s_min_x + zone_w * frac;
            let cy = s_min_y + zone_h * 0.5;

            let half = arrow_len * 0.5;
            let start = Pos2::new(cx - dir_x * half, cy - dir_y * half);
            let tip = Pos2::new(cx + dir_x * half, cy + dir_y * half);

            // Arrow shaft.
            painter.line_segment([start, tip], Stroke::new(1.5, FORCE_ZONE_COLOR));

            // Arrowhead: two small lines forming a V at the tip.
            let back_x = -dir_x;
            let back_y = -dir_y;
            for sign in [-1.0_f32, 1.0] {
                let cos_a = head_angle.cos();
                let sin_a = head_angle.sin() * sign;
                let hx = back_x * cos_a - back_y * sin_a;
                let hy = back_x * sin_a + back_y * cos_a;
                let head_pt = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
                painter.line_segment([tip, head_pt], Stroke::new(1.5, FORCE_ZONE_COLOR));
            }
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

    // 5. Active overlap highlight: body geometry clipped to zone AABB.
    if let Some(body) = mech.bodies().get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            let (bx, by, btheta) = mech_state.get_pose(&fz.body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                bx, by, btheta, geo.width, geo.height, &geo.offset,
            );
            let zone_min_v = nalgebra::Vector2::new(fz.zone_min[0], fz.zone_min[1]);
            let zone_max_v = nalgebra::Vector2::new(fz.zone_max[0], fz.zone_max[1]);
            let clipped = crate::geometry::clip_polygon_to_aabb(
                &corners, &zone_min_v, &zone_max_v,
            );
            if clipped.len() >= 3 {
                let screen_verts: Vec<Pos2> = clipped
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
        if joint.body_i_id() == body_id || joint.body_j_id() == body_id {
            if let Some(&(fx, fy)) = state.force_results.joint_reactions.get(joint.id()) {
                let mag = (fx * fx + fy * fy).sqrt();
                body_max = body_max.max(mag);
            }
        }
    }

    let t = (body_max / global_max) as f32;
    Some(heat_color(t))
}
