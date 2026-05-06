//! Drawing primitive helpers for canvas rendering.
//!
//! Reusable low-level shapes: springs, dampers, arrows, arcs, markers,
//! and alignment guides. No mechanism/state knowledge — pure painter ops.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::forces::elements::*;
use crate::gui::canvas::colors::*;
use crate::gui::state::AppState;

// ── Drawing primitives ───────────────────────────────────────────────────────

/// Draw a zigzag spring symbol between two screen-space points.
///
/// The spring is rendered as: short straight lead-in, N zigzag segments, short
/// straight lead-out. The zigzag amplitude is fixed at 6 pixels perpendicular
/// to the line of action.
pub(super) fn draw_spring_zigzag(
    painter: &egui::Painter,
    start: Pos2,
    end: Pos2,
    color: Color32,
) {
    let total = end - start;
    let length = total.length();
    if length < 2.0 {
        return;
    }

    let dir = total / length;
    let perp = Vec2::new(-dir.y, dir.x);
    let amplitude = 6.0_f32;
    let n_zags: usize = 8;
    let stroke = Stroke::new(2.0, color);

    // Divide the total length into: lead_in + n_zags segments + lead_out.
    let lead_frac = 0.1; // 10% lead-in and lead-out
    let lead_len = length * lead_frac;
    let zag_region = length - 2.0 * lead_len;
    let seg_len = if n_zags > 0 { zag_region / n_zags as f32 } else { 0.0 };

    // Lead-in straight segment.
    let lead_in_end = start + dir * lead_len;
    painter.line_segment([start, lead_in_end], stroke);

    // Zigzag segments.
    let mut prev = lead_in_end;
    for i in 0..n_zags {
        let t = lead_len + (i as f32 + 0.5) * seg_len;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mid_point = start + dir * t + perp * amplitude * sign;

        painter.line_segment([prev, mid_point], stroke);
        prev = mid_point;
    }

    // Connect last zigzag to lead-out start.
    let lead_out_start = start + dir * (length - lead_len);
    painter.line_segment([prev, lead_out_start], stroke);

    // Lead-out straight segment.
    painter.line_segment([lead_out_start, end], stroke);
}

/// Draw a dashpot (damper) symbol between two screen-space points.
///
/// Rendered as: line from start to 40% mark, a small rectangle (the cylinder)
/// centered at the midpoint, and a line from 60% to end. A piston line runs
/// through the center of the rectangle.
pub(super) fn draw_damper_symbol(
    painter: &egui::Painter,
    start: Pos2,
    end: Pos2,
    color: Color32,
) {
    let total = end - start;
    let length = total.length();
    if length < 2.0 {
        return;
    }

    let dir = total / length;
    let perp = Vec2::new(-dir.y, dir.x);
    let stroke = Stroke::new(2.0, color);
    let rect_half_w = 5.0_f32; // half-width perpendicular
    let mid = start + total * 0.5;
    let rect_start_frac = 0.4;
    let rect_end_frac = 0.6;

    // Line from start to rectangle leading edge.
    let p1 = start + dir * (length * rect_start_frac);
    painter.line_segment([start, p1], stroke);

    // Line from rectangle trailing edge to end (piston rod).
    let p2 = start + dir * (length * rect_end_frac);
    painter.line_segment([p2, end], stroke);

    // Piston line through center of rectangle (from ~35% to midpoint).
    let piston_start = start + dir * (length * 0.35);
    painter.line_segment([piston_start, mid], stroke);

    // Rectangle (dashpot cylinder): four corners.
    let r_start = start + dir * (length * rect_start_frac);
    let r_end = start + dir * (length * rect_end_frac);
    let c1 = r_start + perp * rect_half_w;
    let c2 = r_start - perp * rect_half_w;
    let c3 = r_end - perp * rect_half_w;
    let c4 = r_end + perp * rect_half_w;
    painter.line_segment([c1, c2], stroke);
    painter.line_segment([c2, c3], stroke);
    painter.line_segment([c3, c4], stroke);
    painter.line_segment([c4, c1], stroke);

    // Cap at the piston entry side (perpendicular bar at ~35%).
    let cap = start + dir * (length * 0.35);
    painter.line_segment(
        [cap + perp * rect_half_w, cap - perp * rect_half_w],
        stroke,
    );
}

/// Draw an external force arrow (orange) at a point on the canvas.
///
/// Similar to `draw_force_arrow` but uses the external force color and shows
/// the prescribed force magnitude rather than a computed reaction.
pub(super) fn draw_external_force_arrow(
    painter: &egui::Painter,
    origin: Pos2,
    fx: f32,
    fy: f32,
) {
    let mag = (fx * fx + fy * fy).sqrt();
    if mag < 1e-12 {
        return;
    }

    let px_len = (mag * FORCE_ARROW_SCALE).clamp(FORCE_ARROW_MIN_PX, FORCE_ARROW_MAX_PX);

    // Unit direction in screen coords (flip Y).
    let dx = fx / mag;
    let dy = -fy / mag;

    // Arrow points INTO the body: shaft starts away from origin, tip at origin.
    let tail = Pos2::new(origin.x - dx * px_len, origin.y - dy * px_len);
    let tip = origin;

    // Shaft.
    painter.line_segment(
        [tail, tip],
        Stroke::new(FORCE_ARROW_WIDTH, EXT_FORCE_COLOR),
    );

    // Arrowhead.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44;
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment(
            [tip, head_end],
            Stroke::new(FORCE_ARROW_WIDTH, EXT_FORCE_COLOR),
        );
    }

    // Magnitude label.
    painter.text(
        Pos2::new(tail.x - 4.0, tail.y - 4.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("{:.0} N", mag),
        FontId::proportional(11.0),
        EXT_FORCE_COLOR,
    );
}

/// Draw a curved torque arc with an arrowhead at a point on the canvas.
///
/// Positive torque draws counterclockwise; negative draws clockwise.
/// The arc spans approximately 270 degrees and has a small arrowhead at the tip.
pub(super) fn draw_torque_arc(
    painter: &egui::Painter,
    center: Pos2,
    torque: f32,
    color: Color32,
) {
    let radius = 12.0_f32;
    let n_segments = 20;
    let arc_span = std::f32::consts::PI * 1.5; // 270 degrees
    let direction = if torque >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let stroke = Stroke::new(1.5, color);

    let start_angle = 0.0_f32;
    let mut prev = Pos2::new(
        center.x + radius * start_angle.cos(),
        center.y - radius * start_angle.sin(), // screen Y is flipped
    );

    for i in 1..=n_segments {
        let frac = i as f32 / n_segments as f32;
        let angle = start_angle + direction * arc_span * frac;
        let pt = Pos2::new(
            center.x + radius * angle.cos(),
            center.y - radius * angle.sin(),
        );
        painter.line_segment([prev, pt], stroke);
        prev = pt;
    }

    // Arrowhead at the end of the arc.
    let end_angle = start_angle + direction * arc_span;
    let tip = prev;
    // Tangent direction at the tip (perpendicular to radius, in the arc direction).
    let tangent_x = -direction * end_angle.sin();
    let tangent_y = -direction * (-end_angle.cos()); // flipped Y
    let head_len = 5.0_f32;
    let head_angle_offset = 0.5_f32;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle_offset.cos();
        let sin_a = head_angle_offset.sin() * sign;
        let hx = -tangent_x * cos_a - (-tangent_y) * sin_a;
        let hy = -tangent_x * sin_a + (-tangent_y) * cos_a;
        let head_pt = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment([tip, head_pt], stroke);
    }
}

/// Draw alignment guide lines across the canvas for active snap guides.
///
/// Renders a dashed cyan line spanning the full visible canvas for each
/// alignment guide, plus a small label at the guide's intersection with the
/// nearest canvas edge.
pub(super) fn draw_alignment_guides(
    painter: &egui::Painter,
    canvas_rect: Rect,
    state: &AppState,
) {
    use crate::gui::state::AlignmentAxis;

    if state.alignment_guides.is_empty() {
        return;
    }

    let guide_color = Color32::from_rgba_premultiplied(0, 200, 255, 120);
    let guide_stroke = Stroke::new(1.0, guide_color);
    let dash = 6.0_f32;
    let gap = 4.0_f32;
    let label_color = Color32::from_rgba_premultiplied(0, 200, 255, 180);

    let view = &state.view;

    for guide in &state.alignment_guides {
        match guide.axis {
            AlignmentAxis::Horizontal => {
                // Horizontal guide: same y-value, line spans full canvas width.
                let left_world = view.screen_to_world(canvas_rect.left(), 0.0);
                let right_world = view.screen_to_world(canvas_rect.right(), 0.0);
                let left_sp = view.world_to_screen(left_world[0], guide.world_value);
                let right_sp = view.world_to_screen(right_world[0], guide.world_value);
                let left_pos = Pos2::new(canvas_rect.left(), left_sp[1]);
                let right_pos = Pos2::new(canvas_rect.right(), right_sp[1]);
                draw_dashed_line(painter, left_pos, right_pos, guide_stroke, dash, gap);

                // Label near the right edge.
                painter.text(
                    Pos2::new(canvas_rect.right() - 4.0, left_sp[1] - 2.0),
                    egui::Align2::RIGHT_BOTTOM,
                    &guide.label,
                    FontId::proportional(10.0),
                    label_color,
                );
            }
            AlignmentAxis::Vertical => {
                // Vertical guide: same x-value, line spans full canvas height.
                let top_world = view.screen_to_world(0.0, canvas_rect.top());
                let bot_world = view.screen_to_world(0.0, canvas_rect.bottom());
                let top_sp = view.world_to_screen(guide.world_value, top_world[1]);
                let bot_sp = view.world_to_screen(guide.world_value, bot_world[1]);
                let top_pos = Pos2::new(top_sp[0], canvas_rect.top());
                let bot_pos = Pos2::new(bot_sp[0], canvas_rect.bottom());
                draw_dashed_line(painter, top_pos, bot_pos, guide_stroke, dash, gap);

                // Label near the top edge.
                painter.text(
                    Pos2::new(top_sp[0] + 4.0, canvas_rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    &guide.label,
                    FontId::proportional(10.0),
                    label_color,
                );
            }
        }
    }
}

/// Draw a dashed line between two screen-space points.
///
/// `dash_len` and `gap_len` control the dash pattern in pixels.
pub fn draw_dashed_line(
    painter: &egui::Painter,
    start: Pos2,
    end: Pos2,
    stroke: Stroke,
    dash_len: f32,
    gap_len: f32,
) {
    let delta = end - start;
    let length = delta.length();
    if length < 1.0 {
        return;
    }
    let dir = delta / length;
    let mut t = 0.0_f32;
    while t < length {
        let seg_start = Pos2::new(start.x + dir.x * t, start.y + dir.y * t);
        let seg_end_t = (t + dash_len).min(length);
        let seg_end = Pos2::new(start.x + dir.x * seg_end_t, start.y + dir.y * seg_end_t);
        painter.line_segment([seg_start, seg_end], stroke);
        t += dash_len + gap_len;
    }
}

/// Draw a rotary force element badge: a dashed line from body_i CG to body_j CG,
/// with a circled letter at the midpoint.
///
/// `body_i_screen` / `body_j_screen` are the screen-space CG positions.
/// `letter` is the single-character badge (e.g. "M", "k", "c", "f", "[").
/// `color` is the element's semantic color.
pub(super) fn draw_rotary_badge(
    painter: &egui::Painter,
    body_i_screen: Pos2,
    body_j_screen: Pos2,
    letter: &str,
    color: Color32,
) {
    // Dashed line connecting the two body CGs (semi-transparent).
    let dash_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 200);
    draw_dashed_line(
        painter,
        body_i_screen,
        body_j_screen,
        Stroke::new(1.5, dash_color),
        6.0,
        4.0,
    );

    // Midpoint badge.
    let mid = Pos2::new(
        (body_i_screen.x + body_j_screen.x) * 0.5,
        (body_i_screen.y + body_j_screen.y) * 0.5,
    );
    let badge_radius = 9.0_f32;

    // Filled circle background.
    painter.circle_filled(mid, badge_radius, Color32::from_rgb(30, 32, 42));
    // Circle outline in element color.
    painter.circle_stroke(mid, badge_radius, Stroke::new(1.5, color));
    // Letter centered inside.
    painter.text(
        mid,
        egui::Align2::CENTER_CENTER,
        letter,
        FontId::proportional(11.0),
        color,
    );
}

/// Draw a force arrow at a joint location.
///
/// `fx`, `fy` are force components in Newtons (world frame). The arrow is
/// scaled, clamped, and drawn with an arrowhead. Screen Y is flipped relative
/// to world Y.
pub(super) fn draw_force_arrow(painter: &egui::Painter, origin: Pos2, fx: f32, fy: f32) {
    let mag = (fx * fx + fy * fy).sqrt();
    if mag < 1e-12 {
        return; // negligible force
    }

    // Scale force magnitude to pixel length, then clamp.
    let px_len = (mag * FORCE_ARROW_SCALE).clamp(FORCE_ARROW_MIN_PX, FORCE_ARROW_MAX_PX);

    // Unit direction in screen coords (flip Y for screen space).
    let dx = fx / mag;
    let dy = -fy / mag; // flip Y: world up is screen down

    let tip = Pos2::new(origin.x + dx * px_len, origin.y + dy * px_len);

    // Shaft line.
    painter.line_segment(
        [origin, tip],
        Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR),
    );

    // Arrowhead: two lines at +/-25 degrees from the shaft, 8px long.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44; // ~25 degrees in radians
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment(
            [tip, head_end],
            Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR),
        );
    }

    // Magnitude label near the tip.
    painter.text(
        Pos2::new(tip.x + 4.0, tip.y - 4.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{:.0} N", mag),
        FontId::proportional(11.0),
        FORCE_ARROW_COLOR,
    );
}

/// Draw reaction force as separate Fx (solid) and Fy (dashed) component arrows.
///
/// `fx`, `fy` are world-frame components in Newtons. The Fx arrow runs
/// horizontally in screen space (sign-preserving: negative fx points left);
/// the Fy arrow runs vertically (positive fy points up — screen Y is flipped).
/// Both share `FORCE_ARROW_COLOR` and `FORCE_ARROW_SCALE` with the resultant
/// renderer so the toggle is purely a style/decomposition switch.
///
/// Components below the noise floor (|F| < 1e-12) are skipped to avoid
/// drawing zero-length arrows. A component is also skipped if the magnitude
/// alone is below the noise floor — so a purely horizontal force draws Fx
/// only and Fy is suppressed.
pub(super) fn draw_force_arrow_components(
    painter: &egui::Painter,
    origin: Pos2,
    fx: f32,
    fy: f32,
) {
    // Solid Fx along world-X (screen-x is +right, no flip).
    draw_axis_component(painter, origin, fx, AxisDirection::X, "Fx");
    // Dashed Fy along world-Y (screen-y is flipped: +world_y → -screen_y).
    draw_axis_component(painter, origin, fy, AxisDirection::Y, "Fy");
}

#[derive(Clone, Copy)]
enum AxisDirection { X, Y }

/// Render one axis-aligned component arrow. Solid for X, dashed shaft for Y.
/// Arrowhead is always solid so the arrow direction reads cleanly.
fn draw_axis_component(
    painter: &egui::Painter,
    origin: Pos2,
    component: f32,
    axis: AxisDirection,
    label_prefix: &str,
) {
    if component.abs() < 1e-12 {
        return;
    }
    let mag = component.abs();
    let px_len = (mag * FORCE_ARROW_SCALE).clamp(FORCE_ARROW_MIN_PX, FORCE_ARROW_MAX_PX);

    // Unit direction in screen space, accounting for the sign of the component
    // and the screen-Y flip on the Y axis.
    let (dx, dy) = match axis {
        AxisDirection::X => (component.signum(), 0.0_f32),
        AxisDirection::Y => (0.0_f32, -component.signum()),
    };
    let tip = Pos2::new(origin.x + dx * px_len, origin.y + dy * px_len);

    // Shaft: solid for X, dashed for Y.
    let stroke = Stroke::new(FORCE_ARROW_WIDTH, FORCE_ARROW_COLOR);
    match axis {
        AxisDirection::X => {
            painter.line_segment([origin, tip], stroke);
        }
        AxisDirection::Y => {
            draw_dashed_line(painter, origin, tip, stroke, 5.0, 3.0);
        }
    }

    // Solid arrowhead in both modes — keeps the arrow legible regardless
    // of dash phase at the tip.
    let head_len: f32 = 8.0;
    let head_angle: f32 = 0.44;
    let back_dx = -dx;
    let back_dy = -dy;
    for sign in [-1.0_f32, 1.0] {
        let cos_a = head_angle.cos();
        let sin_a = head_angle.sin() * sign;
        let hx = back_dx * cos_a - back_dy * sin_a;
        let hy = back_dx * sin_a + back_dy * cos_a;
        let head_end = Pos2::new(tip.x + hx * head_len, tip.y + hy * head_len);
        painter.line_segment([tip, head_end], stroke);
    }

    // Signed component label near the tip — sign carries direction, so
    // the label is unambiguous even when the arrow is short.
    painter.text(
        Pos2::new(tip.x + 4.0, tip.y - 4.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{} = {:.0} N", label_prefix, component),
        FontId::proportional(11.0),
        FORCE_ARROW_COLOR,
    );
}

/// Fill a force element template with actual body IDs and point coordinates.
pub fn fill_force_template(
    template: &ForceElement,
    body_a: &str, point_a: [f64; 2], point_a_name: Option<String>,
    body_b: &str, point_b: [f64; 2], point_b_name: Option<String>,
) -> ForceElement {
    match template {
        ForceElement::LinearSpring(s) => ForceElement::LinearSpring(LinearSpringElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..s.clone()
        }),
        ForceElement::LinearDamper(d) => ForceElement::LinearDamper(LinearDamperElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..d.clone()
        }),
        ForceElement::GasSpring(g) => ForceElement::GasSpring(GasSpringElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..g.clone()
        }),
        ForceElement::LinearActuator(a) => ForceElement::LinearActuator(LinearActuatorElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..a.clone()
        }),
        _ => unreachable!("fill_force_template called with non-two-point force template"),
    }
}

/// Draw a ground-fixed marker: an inverted triangle with hatch lines below.
pub fn draw_ground_marker(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let half = size / 2.0;

    // Triangle: center point at top, two corners at bottom-left and bottom-right.
    let top = center;
    let bl = Pos2::new(center.x - half, center.y + size);
    let br = Pos2::new(center.x + half, center.y + size);

    // Filled triangle with semi-transparent fill for better visibility.
    let fill = Color32::from_rgba_premultiplied(
        color.r() / 3,
        color.g() / 3,
        color.b() / 3,
        80,
    );
    painter.add(egui::Shape::convex_polygon(
        vec![top, bl, br],
        fill,
        Stroke::new(1.5, color),
    ));

    // Hatch lines below the triangle base.
    let hatch_y = center.y + size;
    let n_hatches = 5;
    let hatch_len = size * 0.4;
    let spacing = size / n_hatches as f32;
    for i in 0..=n_hatches {
        let x = center.x - half + spacing * i as f32;
        painter.line_segment(
            [
                Pos2::new(x, hatch_y),
                Pos2::new(x - hatch_len * 0.5, hatch_y + hatch_len),
            ],
            Stroke::new(1.0, color),
        );
    }
}

pub fn draw_diamond_marker(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    let points = vec![
        Pos2::new(center.x, center.y - radius),
        Pos2::new(center.x + radius, center.y),
        Pos2::new(center.x, center.y + radius),
        Pos2::new(center.x - radius, center.y),
    ];
    painter.add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::new(1.5, Color32::WHITE),
    ));
}

