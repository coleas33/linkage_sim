//! Canvas rendering: bodies, joints, force elements, drawing primitives, tooltips.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::body::Body;
use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::{State, GROUND_ID};
use crate::forces::elements::*;
use crate::gui::state::{AppState, SelectedEntity, ViewTransform, GridSettings};

use super::colors::*;
use super::hit_testing::{AttachmentHit, BodySegment};


// Submodules
pub mod primitives;
mod force_render;

// Re-export from submodules so external callers do not need to know the layout.
pub use primitives::{draw_dashed_line, draw_ground_marker, draw_diamond_marker, fill_force_template};
pub use force_render::{force_zone_app_point_world, heat_color};

use primitives::{
    draw_alignment_guides, draw_force_arrow, draw_force_arrow_components, draw_rotary_badge,
};
use force_render::{draw_force_elements, load_path_color_for_body};

// ── Main rendering pass (immutable) ──────────────────────────────────────────

/// Render the mechanism (bodies, joints, coupler traces, overlays) and collect
/// hit-test data. This borrows `state` immutably except for the hit-target vecs
/// which are populated for use by the interaction layer.
///
/// Returns grounded revolute joint IDs (needed by context menus).
pub fn render_mechanism(
    painter: &egui::Painter,
    canvas_rect: Rect,
    state: &AppState,
    joint_hit_targets: &mut Vec<(Pos2, String)>,
    attachment_hit_targets: &mut Vec<AttachmentHit>,
    body_segments: &mut Vec<BodySegment>,
) -> Vec<String> {
    let show_debug = state.show_debug_overlay;
    let current_driver_joint = state.driver_joint_id.clone();
    let highlight_joint = state.highlight_joint.clone();
    let creating_joint_first = state.creating_joint.clone();

    let mech = state.mechanism.as_ref().unwrap();
    let grounded_revolute_ids = mech.grounded_revolute_joint_ids();
    let mech_state = mech.state();
    let bodies = mech.bodies();
    let joints = mech.joints();
    let q = &state.q;
    let view = &state.view;
    let selected = &state.selected;

    // ── Motion ribbon: ghost poses along the back-solved trajectory ──
    // Renders BEFORE everything else (except grid + bg image, which are
    // already painted by draw_canvas) so the live mechanism draws on
    // top. Trajectory mode only.
    if state.show_motion_ribbon
        && state.sweep_mode.is_trajectory()
    {
        if let Some(sweep_data) = state.sweep_data.as_ref() {
            draw_motion_ribbon(painter, state, sweep_data);
        }
    }

    // Nathan Mode: optional grayscale color transform for canvas elements.
    let gc = |c: Color32| -> Color32 {
        if state.nathan_mode { to_grayscale(c) } else { c }
    };

    // ── Ground line (y=0) — spans full visible viewport ────────────
    {
        let world_left = view.screen_to_world(canvas_rect.left(), 0.0);
        let world_right = view.screen_to_world(canvas_rect.right(), 0.0);
        let left = view.world_to_screen(world_left[0], 0.0);
        let right = view.world_to_screen(world_right[0], 0.0);
        painter.line_segment(
            [Pos2::new(left[0], left[1]), Pos2::new(right[0], right[1])],
            Stroke::new(1.0, gc(GROUND_LINE_COLOR)),
        );
    }

    // ── Draw coupler traces from sweep data ──────────────────────────
    if let Some(sweep) = &state.sweep_data {
        let trace_colors = [
            Color32::from_rgba_premultiplied(80, 180, 255, 100),
            Color32::from_rgba_premultiplied(255, 140, 60, 100),
            Color32::from_rgba_premultiplied(100, 210, 100, 100),
            Color32::from_rgba_premultiplied(255, 90, 90, 100),
            Color32::from_rgba_premultiplied(180, 130, 255, 100),
            Color32::from_rgba_premultiplied(255, 210, 80, 100),
        ];
        let mut color_idx = 0;
        let mut keys: Vec<&String> = sweep.coupler_traces.keys().collect();
        keys.sort();

        for key in keys {
            let trace = &sweep.coupler_traces[key];
            if trace.len() < 2 {
                continue;
            }
            let color = gc(trace_colors[color_idx % trace_colors.len()]);
            color_idx += 1;

            let screen_pts: Vec<Pos2> = trace
                .iter()
                .map(|[wx, wy]| {
                    let sp = view.world_to_screen(*wx, *wy);
                    Pos2::new(sp[0], sp[1])
                })
                .collect();

            // Draw as dashed line — scale with zoom so dashes stay
            // proportional in world units (~5 mm dash, ~3 mm gap),
            // clamped to reasonable pixel sizes.
            let dash_len = (0.005_f32 * view.scale).clamp(3.0, 15.0);
            let gap_len  = (0.003_f32 * view.scale).clamp(2.0, 10.0);
            for pair in screen_pts.windows(2) {
                let a = pair[0];
                let b = pair[1];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let seg_len = (dx * dx + dy * dy).sqrt();
                if seg_len < 0.5 { continue; }
                let ux = dx / seg_len;
                let uy = dy / seg_len;
                let mut t = 0.0f32;
                while t < seg_len {
                    let t_end = (t + dash_len).min(seg_len);
                    let p0 = Pos2::new(a.x + ux * t, a.y + uy * t);
                    let p1 = Pos2::new(a.x + ux * t_end, a.y + uy * t_end);
                    painter.line_segment([p0, p1], Stroke::new(1.5, color));
                    t += dash_len + gap_len;
                }
            }
        }
    }

    // ── Draw bodies ─────────────────────────────────────────────────
    for (body_id, body) in bodies.iter() {
        if body_id == GROUND_ID {
            continue;
        }

        let is_selected =
            matches!(selected, Some(SelectedEntity::Body(s)) if s == body_id)
            || state.multi_selected.iter().any(|e| matches!(e, SelectedEntity::Body(s) if s == body_id));
        let color = if is_selected {
            gc(BODY_SELECTED_COLOR)
        } else if let Some(lp_color) = load_path_color_for_body(body_id, mech, state) {
            gc(lp_color)
        } else {
            gc(BODY_COLOR)
        };

        // Collect attachment point positions, sorted by name for consistency.
        let mut point_names: Vec<&String> = body.attachment_points.keys().collect();
        point_names.sort();

        let point_positions: Vec<(Pos2, [f64; 2])> = point_names
            .iter()
            .map(|name| {
                let local = &body.attachment_points[*name];
                let global = mech_state.body_point_global(body_id, local, q);
                let sp = view.world_to_screen(global.x, global.y);
                (Pos2::new(sp[0], sp[1]), [global.x, global.y])
            })
            .collect();
        let screen_points: Vec<Pos2> = point_positions.iter().map(|(sp, _)| *sp).collect();

        // Draw body geometry rectangle (behind the link bar).
        if let Some(ref geo) = body.geometry {
            let pos = mech_state.get_position(body_id, q);
            let theta = mech_state.get_angle(body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                pos.x, pos.y, theta, geo.width, geo.height, &geo.offset,
            );
            let screen_corners: Vec<Pos2> = corners
                .iter()
                .map(|c| {
                    let sp = view.world_to_screen(c.x, c.y);
                    Pos2::new(sp[0], sp[1])
                })
                .collect();
            let geo_fill = Color32::from_rgba_premultiplied(64, 42, 0, 64);
            let geo_stroke = Stroke::new(2.0, Color32::from_rgb(255, 165, 0));
            painter.add(egui::epaint::PathShape::convex_polygon(
                screen_corners,
                geo_fill,
                geo_stroke,
            ));
        }

        // Draw links as rounded rectangles (bars) for visibility and click targets.
        if screen_points.len() >= 2 {
            let fill_alpha = if is_selected { 80u8 } else { 40u8 };
            let fill_color = egui::Color32::from_rgba_unmultiplied(
                color.r(), color.g(), color.b(), fill_alpha,
            );
            let stroke_color = color;

            let draw_link_bar = |p: &egui::Painter, a: Pos2, b: Pos2| {
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1.0 { return; }
                // Normal perpendicular to the link direction
                let nx = -dy / len * LINK_HALF_WIDTH;
                let ny = dx / len * LINK_HALF_WIDTH;
                let corners = [
                    Pos2::new(a.x + nx, a.y + ny),
                    Pos2::new(b.x + nx, b.y + ny),
                    Pos2::new(b.x - nx, b.y - ny),
                    Pos2::new(a.x - nx, a.y - ny),
                ];
                let shape = egui::epaint::PathShape::convex_polygon(
                    corners.to_vec(),
                    fill_color,
                    Stroke::new(BODY_STROKE_WIDTH, stroke_color),
                );
                p.add(shape);
            };

            for pair in screen_points.windows(2) {
                draw_link_bar(painter, pair[0], pair[1]);
            }
            if screen_points.len() >= 3 {
                draw_link_bar(painter, *screen_points.last().unwrap(), screen_points[0]);
            }
        } else if screen_points.len() == 1 {
            painter.circle_filled(screen_points[0], 4.0, color);
        }

        // Draw small dots at each attachment point for visual clarity.
        for sp in &screen_points {
            painter.circle_filled(*sp, ATTACHMENT_DOT_RADIUS, ATTACHMENT_DOT_COLOR);
        }

        // Draw mount points as diamonds AND register as hit targets
        for (name, local) in &body.mount_points {
            let global = mech_state.body_point_global(body_id, local, q);
            let sp = view.world_to_screen(global.x, global.y);
            let screen_pos = Pos2::new(sp[0], sp[1]);
            draw_diamond_marker(painter, screen_pos, MOUNT_POINT_RADIUS, MOUNT_POINT_COLOR);

            attachment_hit_targets.push(AttachmentHit {
                screen_pos,
                world_pos: [global.x, global.y],
                body_id: body_id.clone(),
                point_name: name.clone(),
            });
        }

        // Dimension labels: show link segment lengths at midpoints.
        if state.show_dimensions && point_positions.len() >= 2 {
            let segments: Vec<(usize, usize)> = if point_positions.len() >= 3 {
                // Closed polygon: all consecutive pairs + last->first
                (0..point_positions.len())
                    .map(|i| (i, (i + 1) % point_positions.len()))
                    .collect()
            } else {
                // Binary link: single segment
                vec![(0, 1)]
            };
            for (i, j) in segments {
                let (sp_a, wp_a) = &point_positions[i];
                let (sp_b, wp_b) = &point_positions[j];
                let dx = wp_b[0] - wp_a[0];
                let dy = wp_b[1] - wp_a[1];
                let dist_m = (dx * dx + dy * dy).sqrt();
                let angle_rad = dy.atan2(dx);
                let label = format!(
                    "{:.1}{}  {:.1}{}",
                    state.display_units.length(dist_m),
                    state.display_units.length_suffix(),
                    state.display_units.angle(angle_rad),
                    state.display_units.angle_suffix()
                );
                let mid = Pos2::new(
                    (sp_a.x + sp_b.x) * 0.5,
                    (sp_a.y + sp_b.y) * 0.5,
                );
                // Offset perpendicular to the segment so the label doesn't overlap the link.
                let seg_dx = sp_b.x - sp_a.x;
                let seg_dy = sp_b.y - sp_a.y;
                let seg_len = (seg_dx * seg_dx + seg_dy * seg_dy).sqrt().max(1.0);
                let nx = -seg_dy / seg_len * 10.0;
                let ny = seg_dx / seg_len * 10.0;
                painter.text(
                    Pos2::new(mid.x + nx, mid.y + ny),
                    egui::Align2::CENTER_CENTER,
                    &label,
                    FontId::proportional(11.0),
                    DIM_LABEL_COLOR,
                );
            }
        }

        // Debug overlay: body ID at CG, attachment point labels.
        if show_debug {
            let cg_global = mech_state.body_point_global(body_id, &body.cg_local, q);
            let cg_screen = view.world_to_screen(cg_global.x, cg_global.y);
            painter.text(
                Pos2::new(cg_screen[0], cg_screen[1] - 12.0),
                egui::Align2::CENTER_BOTTOM,
                body_id,
                FontId::proportional(11.0),
                DEBUG_TEXT_COLOR,
            );

            for (i, name) in point_names.iter().enumerate() {
                painter.text(
                    Pos2::new(screen_points[i].x + 6.0, screen_points[i].y - 6.0),
                    egui::Align2::LEFT_BOTTOM,
                    *name,
                    FontId::proportional(9.0),
                    DEBUG_DIM_COLOR,
                );
            }
        }

        // Store attachment hit targets with world positions for precise snapping.
        for (i, (sp, wp)) in point_positions.iter().enumerate() {
            attachment_hit_targets.push(AttachmentHit {
                screen_pos: *sp,
                world_pos: *wp,
                body_id: body_id.clone(),
                point_name: point_names[i].clone(),
            });
        }

        // Collect segments for hit testing.
        if point_positions.len() >= 2 {
            for i in 0..point_positions.len() - 1 {
                body_segments.push(BodySegment {
                    screen_a: point_positions[i].0,
                    screen_b: point_positions[i + 1].0,
                    world_a: point_positions[i].1,
                    world_b: point_positions[i + 1].1,
                    body_id: body_id.clone(),
                    point_a_name: point_names[i].clone(),
                    point_b_name: point_names[i + 1].clone(),
                });
            }
            // Close polygon for 3+ points.
            if point_positions.len() >= 3 {
                body_segments.push(BodySegment {
                    screen_a: point_positions.last().unwrap().0,
                    screen_b: point_positions[0].0,
                    world_a: point_positions.last().unwrap().1,
                    world_b: point_positions[0].1,
                    body_id: body_id.clone(),
                    point_a_name: point_names.last().unwrap().to_string(),
                    point_b_name: point_names[0].clone(),
                });
            }
        }
    }

    // ── Draw point masses from blueprint ───────────────────────────
    if let Some(bp) = &state.blueprint {
        let point_mass_color = gc(Color32::from_rgb(255, 200, 50));
        for (body_id, bp_body) in &bp.bodies {
            if body_id == GROUND_ID {
                continue;
            }
            for pm in &bp_body.point_masses {
                let local = nalgebra::Vector2::new(pm.local_pos[0], pm.local_pos[1]);
                let global = mech_state.body_point_global(body_id, &local, q);
                let sp = view.world_to_screen(global.x, global.y);
                let screen_pos = Pos2::new(sp[0], sp[1]);
                painter.circle_filled(screen_pos, 5.0, point_mass_color);
                painter.text(
                    screen_pos + Vec2::new(8.0, -8.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{:.2} kg", pm.mass),
                    FontId::proportional(9.0),
                    point_mass_color,
                );
            }
        }
    }

    // ── Draw ground markers and collect ground hit targets ──────────
    if let Some(ground) = bodies.get(GROUND_ID) {
        let mut ground_point_names: Vec<&String> =
            ground.attachment_points.keys().collect();
        ground_point_names.sort();

        for name in &ground_point_names {
            let local = &ground.attachment_points[*name];
            let global = mech_state.body_point_global(GROUND_ID, local, q);
            let sp = view.world_to_screen(global.x, global.y);
            let center = Pos2::new(sp[0], sp[1]);
            draw_ground_marker(painter, center, GROUND_MARKER_SIZE, GROUND_MARKER_COLOR);

            // Ground points are also draggable hit targets.
            attachment_hit_targets.push(AttachmentHit {
                screen_pos: center,
                world_pos: [global.x, global.y],
                body_id: GROUND_ID.to_string(),
                point_name: (*name).clone(),
            });

            if show_debug {
                painter.text(
                    Pos2::new(center.x + 8.0, center.y + GROUND_MARKER_SIZE + 4.0),
                    egui::Align2::LEFT_TOP,
                    *name,
                    FontId::proportional(9.0),
                    DEBUG_DIM_COLOR,
                );
            }
        }
    }

    // ── Draw joints ─────────────────────────────────────────────────
    for joint in joints {
        let global =
            mech_state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
        let sp = view.world_to_screen(global.x, global.y);
        let center = Pos2::new(sp[0], sp[1]);

        let is_selected =
            matches!(selected, Some(SelectedEntity::Joint(s)) if s == joint.id())
            || state.multi_selected.iter().any(|e| matches!(e, SelectedEntity::Joint(s) if s == joint.id()));
        let is_driver =
            current_driver_joint.as_deref() == Some(joint.id());
        let is_highlighted =
            highlight_joint.as_deref() == Some(joint.id());
        let color = if is_selected {
            gc(JOINT_SELECTED_COLOR)
        } else if is_driver {
            gc(DRIVER_JOINT_COLOR)
        } else {
            gc(JOINT_COLOR)
        };

        if joint.is_revolute() {
            // Glow ring for panel hover highlight
            if is_highlighted {
                painter.circle_filled(
                    center, JOINT_RADIUS + 6.0,
                    Color32::from_rgba_premultiplied(100, 200, 255, 60),
                );
                painter.circle_stroke(
                    center, JOINT_RADIUS + 6.0,
                    Stroke::new(1.5, JOINT_HOVER_HIGHLIGHT),
                );
            }
            // Glow ring for driver joint
            if is_driver {
                painter.circle_filled(
                    center, JOINT_RADIUS + 4.0,
                    Color32::from_rgba_premultiplied(80, 220, 130, 40),
                );
            }
            // Selection glow
            if is_selected {
                painter.circle_filled(
                    center, JOINT_RADIUS + 3.0,
                    Color32::from_rgba_premultiplied(255, 180, 40, 50),
                );
            }
            // Dark fill + colored ring
            painter.circle_filled(center, JOINT_RADIUS, Color32::from_rgb(28, 30, 38));
            painter.circle_stroke(
                center,
                JOINT_RADIUS,
                Stroke::new(JOINT_STROKE_WIDTH, color),
            );
            // Inner dot for driver joint
            if is_driver {
                painter.circle_filled(center, 3.0, gc(DRIVER_JOINT_COLOR));
            }
        } else if joint.is_prismatic() {
            let half = JOINT_RADIUS;
            let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
            if is_highlighted {
                let glow = rect.expand(6.0);
                painter.rect_filled(glow, 3.0, Color32::from_rgba_premultiplied(100, 200, 255, 60));
                painter.rect_stroke(glow, 3.0, Stroke::new(1.5, JOINT_HOVER_HIGHLIGHT), egui::StrokeKind::Middle);
            }
            if is_selected {
                let glow = rect.expand(3.0);
                painter.rect_filled(glow, 2.0, Color32::from_rgba_premultiplied(255, 180, 40, 50));
            }
            painter.rect_filled(rect, 2.0, Color32::from_rgb(28, 30, 38));
            painter.rect_stroke(rect, 2.0, Stroke::new(JOINT_STROKE_WIDTH, color), egui::StrokeKind::Middle);
        } else if joint.is_fixed() {
            // X marker for fixed joints
            let r = JOINT_RADIUS * 0.7;
            painter.line_segment(
                [Pos2::new(center.x - r, center.y - r), Pos2::new(center.x + r, center.y + r)],
                Stroke::new(2.5, color),
            );
            painter.line_segment(
                [Pos2::new(center.x + r, center.y - r), Pos2::new(center.x - r, center.y + r)],
                Stroke::new(2.5, color),
            );
        }

        if show_debug {
            painter.text(
                Pos2::new(center.x, center.y - JOINT_RADIUS - 5.0),
                egui::Align2::CENTER_BOTTOM,
                joint.id(),
                FontId::proportional(10.0),
                DEBUG_TEXT_COLOR,
            );
        }

        // Store joint hit targets.
        joint_hit_targets.push((center, joint.id().to_string()));
    }

    // ── Labels pass: body and joint labels ─────────────────────────
    if state.show_labels {
        // Body labels: rendered near each non-ground body's CG.
        for (body_id, body) in bodies.iter() {
            if body_id == GROUND_ID {
                continue;
            }
            let cg_global = mech_state.body_point_global(body_id, &body.cg_local, q);
            let cg_screen = view.world_to_screen(cg_global.x, cg_global.y);
            painter.text(
                Pos2::new(cg_screen[0], cg_screen[1] - 12.0),
                egui::Align2::CENTER_BOTTOM,
                &body.label,
                FontId::monospace(10.0),
                LABEL_COLOR,
            );
        }

        // Joint labels: use explicit blueprint label if set, otherwise
        // use the joint's actual ID (e.g. "J1", "J2") so the canvas
        // matches the IDs shown in plot legends, diagnostics, and JSON.
        let bp_joints = state.blueprint.as_ref().map(|bp| &bp.joints);
        for joint in joints {
            // Look up the blueprint label for this joint.
            let bp_label = bp_joints
                .and_then(|bj| bj.get(joint.id()))
                .and_then(|jj| match jj {
                    crate::io::JointJson::Revolute { label, .. }
                    | crate::io::JointJson::Fixed { label, .. }
                    | crate::io::JointJson::Prismatic { label, .. }
                    | crate::io::JointJson::CamFollower { label, .. }
                    | crate::io::JointJson::RevoluteDriver { label, .. } => {
                        label.as_deref()
                    }
                });

            let display_label = bp_label.unwrap_or_else(|| joint.id());

            let global =
                mech_state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
            let sp = view.world_to_screen(global.x, global.y);
            painter.text(
                Pos2::new(sp[0], sp[1] - JOINT_RADIUS - 4.0),
                egui::Align2::CENTER_BOTTOM,
                display_label,
                FontId::monospace(10.0),
                LABEL_COLOR,
            );
        }
    }

    // ── Joint creation mode: highlight first selected point ─────────
    if let Some((ref cj_body, ref cj_point, _)) = creating_joint_first {
        // Find the screen position of the first-click attachment point.
        for hit in attachment_hit_targets.iter() {
            if hit.body_id == *cj_body && hit.point_name == *cj_point {
                painter.circle_stroke(
                    hit.screen_pos,
                    JOINT_RADIUS + 4.0,
                    Stroke::new(2.0, JOINT_CREATE_HIGHLIGHT),
                );
                break;
            }
        }
    }

    grounded_revolute_ids
}

// ── Post-immutable-borrow rendering (force arrows, overlays, hints) ──────────

/// Draw force reaction arrows at joints, force element visuals, add-body preview,
/// gravity indicator, debug overlay, tool hints, and solver failure banner.
///
/// This is called after the immutable borrow scope ends, because it needs
/// mutable access to state for tool previews.
pub fn render_overlays(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    canvas_rect: Rect,
    state: &AppState,
    joint_hit_targets: &[(Pos2, String)],
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
    solver_converged: bool,
    solver_residual: f64,
    solver_iterations: usize,
    show_debug: bool,
) {
    use crate::gui::state::EditorTool;

    // ── Force arrows ────────────────────────────────────────────────────
    if state.show_forces && solver_converged {
        for (screen_pos, joint_id) in joint_hit_targets {
            if let Some(&(fx, fy)) = state.force_results.joint_reactions.get(joint_id) {
                if state.show_force_components {
                    draw_force_arrow_components(painter, *screen_pos, fx as f32, fy as f32);
                } else {
                    draw_force_arrow(painter, *screen_pos, fx as f32, fy as f32);
                }
            }
        }
    }

    // ── Force element visuals ────────────────────────────────────────
    if state.show_forces {
        draw_force_elements(painter, state, &state.view);
    }

    // ── Equation overlay (View ▸ Show equations) ─────────────────────
    if state.show_equation_overlay {
        draw_equation_overlay(painter, canvas_rect, state, joint_hit_targets);
    }

    // ── Alignment guides ────────────────────────────────────────────
    draw_alignment_guides(painter, canvas_rect, state);

    // ── Add Body mode: render placed points and preview ─────────────
    if let Some(ref abs) = state.add_body_state {
        let placed_points: Vec<Pos2> = abs
            .points
            .iter()
            .map(|(_, [wx, wy])| {
                let sp = state.view.world_to_screen(*wx, *wy);
                Pos2::new(sp[0], sp[1])
            })
            .collect();

        // Draw connecting lines between placed points.
        if placed_points.len() >= 2 {
            for pair in placed_points.windows(2) {
                painter.line_segment(
                    [pair[0], pair[1]],
                    Stroke::new(2.0, JOINT_CREATE_HIGHLIGHT),
                );
            }
            // Close preview polygon for 3+ points.
            if placed_points.len() >= 3 {
                let dimmer_green = Color32::from_rgba_premultiplied(60, 230, 100, 80);
                painter.line_segment(
                    [*placed_points.last().unwrap(), placed_points[0]],
                    Stroke::new(1.5, dimmer_green),
                );
            }
        }

        // Draw green dots at each placed point.
        for sp in &placed_points {
            painter.circle_filled(*sp, JOINT_RADIUS, JOINT_CREATE_HIGHLIGHT);
        }

        // Ghost dot at cursor position with connecting line from last placed point.
        if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
            if canvas_rect.contains(hover_pos) {
                let [gwx, gwy] = state.view.screen_to_world(hover_pos.x, hover_pos.y);
                let (sx, sy) = state.grid.snap_point(gwx, gwy);
                let ghost_screen = state.view.world_to_screen(sx, sy);
                let ghost_pos = Pos2::new(ghost_screen[0], ghost_screen[1]);

                let ghost_color = Color32::from_rgba_premultiplied(60, 230, 100, 120);
                painter.circle_filled(ghost_pos, JOINT_RADIUS * 0.7, ghost_color);

                if let Some(last) = placed_points.last() {
                    painter.line_segment(
                        [*last, ghost_pos],
                        Stroke::new(1.5, ghost_color),
                    );
                }
            }
        }
    }

    // ── Crank-angle indicator ────────────────────────────────────────
    // Visualises the driver constraint's prescribed angle (θⱼ − θᵢ) as
    // an arc at the driver pivot so users can see exactly what the
    // "Crank Angle" slider controls.
    draw_crank_angle_indicator(painter, state);

    // ── Gravity indicator ─────────────────────────────────────────────
    if state.gravity_magnitude > 0.0 {
        let indicator_x = canvas_rect.left() + 20.0;
        let indicator_y = canvas_rect.top() + 18.0;
        let arrow_len = 14.0;
        let arrow_tip_y = indicator_y + arrow_len;
        let indicator_color = Color32::from_rgb(160, 160, 180);

        // "g" label
        painter.text(
            Pos2::new(indicator_x, indicator_y - 2.0),
            egui::Align2::CENTER_BOTTOM,
            "g",
            FontId::proportional(12.0),
            indicator_color,
        );

        // Downward arrow shaft
        painter.line_segment(
            [
                Pos2::new(indicator_x, indicator_y),
                Pos2::new(indicator_x, arrow_tip_y),
            ],
            Stroke::new(1.5, indicator_color),
        );

        // Arrowhead
        let head_size = 4.0;
        painter.line_segment(
            [
                Pos2::new(indicator_x - head_size, arrow_tip_y - head_size),
                Pos2::new(indicator_x, arrow_tip_y),
            ],
            Stroke::new(1.5, indicator_color),
        );
        painter.line_segment(
            [
                Pos2::new(indicator_x + head_size, arrow_tip_y - head_size),
                Pos2::new(indicator_x, arrow_tip_y),
            ],
            Stroke::new(1.5, indicator_color),
        );
    }

    // ── Debug overlay: solver status indicator ──────────────────────────
    if show_debug {
        let dot_center = Pos2::new(canvas_rect.right() - 15.0, canvas_rect.top() + 15.0);
        let dot_color = if solver_converged {
            Color32::from_rgb(80, 200, 80)
        } else {
            Color32::from_rgb(220, 60, 60)
        };
        painter.circle_filled(dot_center, 5.0, dot_color);

        let status_text = if solver_converged {
            format!("OK (r={:.1e}, {}it)", solver_residual, solver_iterations)
        } else {
            format!("FAIL (r={:.1e}, {}it)", solver_residual, solver_iterations)
        };
        painter.text(
            Pos2::new(dot_center.x - 12.0, dot_center.y),
            egui::Align2::RIGHT_CENTER,
            status_text,
            FontId::proportional(9.0),
            DEBUG_DIM_COLOR,
        );
    }

    // ── Tool mode hint text ─────────────────────────────────────────────
    let hint_text: Option<String> = match state.active_tool {
        EditorTool::DrawLink => {
            if state.draw_link_start.is_some() {
                Some("Drag to set link length and direction, release to place (Esc to cancel)".to_string())
            } else {
                Some("Click an existing point to start drawing a link \u{2014} use +Ground to place anchors first (Esc to cancel)".to_string())
            }
        }
        EditorTool::AddBody => {
            if let Some(ref abs) = state.add_body_state {
                let n = abs.points.len();
                if n < 2 {
                    Some(format!("Click to add more points \u{2014} need at least 2 ({} placed). Esc to cancel", n))
                } else {
                    Some(format!("Click to add more, double-click or Enter to finish ({} points). Esc to cancel", n))
                }
            } else {
                Some("Click to place first point of new body (Esc to cancel)".to_string())
            }
        }
        EditorTool::AddGroundPivot => {
            Some("Click on canvas to place a ground pivot (Esc to cancel)".to_string())
        }
        EditorTool::PlaceForce => {
            if state.place_force_state.as_ref().and_then(|s| s.start.as_ref()).is_some() {
                Some("Click a second point to place the force element (Esc to cancel)".to_string())
            } else {
                Some("Click a point to set the first attachment of the force element (Esc to cancel)".to_string())
            }
        }
        EditorTool::CreateForceZone => {
            if state.creating_force_zone.is_some() {
                Some("Drag to define the force zone rectangle, release to create (Esc to cancel)".to_string())
            } else {
                Some("Click and drag on the canvas to define a force zone rectangle (Esc to cancel)".to_string())
            }
        }
        EditorTool::PlaceMass => {
            if let Some(ref body_id) = state.place_mass_body {
                Some(format!("Click anywhere to place a point mass on '{}' (Esc to cancel)", body_id))
            } else {
                Some("Click a link to select the attachment body (Esc to cancel)".to_string())
            }
        }
        EditorTool::DrawBodyGeometry => {
            Some("Draw body geometry mode (Esc to cancel)".to_string())
        }
        EditorTool::Select => {
            if let Some(ref bid) = state.adding_joint_point {
                Some(format!("Click to place a new joint point on '{}' (Esc to cancel)", bid))
            } else if state.reassigning_point_mass.is_some() {
                Some("Click a link to move the point mass to that body (Esc to cancel)".to_string())
            } else if state.repositioning_point_mass.is_some() {
                Some("Click anywhere to reposition the point mass (Esc to cancel)".to_string())
            } else {
                None
            }
        }
    };
    if let Some(ref hint) = hint_text {
        painter.text(
            Pos2::new(canvas_rect.center().x, canvas_rect.top() + 20.0),
            egui::Align2::CENTER_TOP,
            hint,
            FontId::proportional(13.0),
            JOINT_CREATE_HIGHLIGHT,
        );
    }

    // ── Solver failure overlay banner ─────────────────────────────────
    if !solver_converged && state.mechanism.is_some() {
        let banner_text = "Solver failed to converge \u{2014} mechanism may be over-constrained or at a toggle point";
        let banner_pos = Pos2::new(canvas_rect.center().x, canvas_rect.top() + 30.0);
        let galley = painter.layout_no_wrap(
            banner_text.to_string(),
            FontId::proportional(13.0),
            Color32::from_rgb(255, 100, 100),
        );
        let text_rect = egui::Rect::from_min_size(
            Pos2::new(banner_pos.x - galley.size().x / 2.0, banner_pos.y),
            galley.size(),
        )
        .expand2(egui::vec2(8.0, 4.0));
        painter.rect_filled(text_rect, 4.0, Color32::from_rgba_premultiplied(40, 10, 10, 220));
        painter.galley(
            Pos2::new(banner_pos.x - galley.size().x / 2.0, banner_pos.y),
            galley,
            Color32::PLACEHOLDER,
        );
    }

    // ── Hover tooltips ────────────────────────────────────────────────
    render_hover_tooltips(ui, canvas_rect, state, joint_hit_targets, attachment_hit_targets, body_segments);
}

/// Draw an arc + reference tick at the driver's pivot showing the
/// current crank angle (the driver constraint's θⱼ − θᵢ) measured
/// from the partner body's local +X axis. For ground-mounted drivers
/// (the common case) the reference is world +X.
///
/// No-op if the mechanism has no driver, the driver isn't revolute,
/// or the connecting revolute joint can't be identified. The arc is
/// rendered in a distinct orange so it doesn't blend with joint
/// glyphs or force arrows.
fn draw_crank_angle_indicator(painter: &egui::Painter, state: &AppState) {
    use crate::core::constraint::JointConstraint;

    let Some(mech) = state.mechanism.as_ref() else { return };
    let Some((partner, driver)) = mech.driver_body_pair() else { return };

    // Locate the revolute joint that connects driver and partner so we
    // know where to anchor the arc in world coordinates.
    let mut partner_anchor: Option<(String, nalgebra::Vector2<f64>)> = None;
    for joint in mech.joints() {
        if let JointConstraint::Revolute(rev) = joint {
            let bi = rev.body_i_id();
            let bj = rev.body_j_id();
            if bi == partner && bj == driver {
                partner_anchor = Some((bi.to_string(), *rev.point_i_local()));
                break;
            }
            if bi == driver && bj == partner {
                partner_anchor = Some((bj.to_string(), *rev.point_j_local()));
                break;
            }
        }
    }
    let Some((partner_body, partner_local)) = partner_anchor else { return };

    let mech_state = mech.state();
    let pivot_world = mech_state.body_point_global(&partner_body, &partner_local, &state.q);
    let partner_theta = mech_state.get_angle(&partner_body, &state.q);
    let driver_theta = mech_state.get_angle(driver, &state.q);
    // The arc sweeps in display frame so its endpoint points along the
    // visible bar direction, not the body's internal +X. See
    // `AppState::driver_display_offset` docs. Partner side stays at the
    // body-frame reference because we don't compute an offset for it.
    let crank_angle = (driver_theta + state.driver_display_offset) - partner_theta;

    let pivot_screen = state.view.world_to_screen(pivot_world.x, pivot_world.y);
    let center = Pos2::new(pivot_screen[0], pivot_screen[1]);

    // Screen Y is inverted (+Y points down), so we negate sin terms.
    let point_at = |radius: f32, theta: f64| -> Pos2 {
        Pos2::new(
            center.x + radius * (theta.cos() as f32),
            center.y - radius * (theta.sin() as f32),
        )
    };

    let color = state.nc(Color32::from_rgb(255, 190, 80));
    let faint = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 140);
    let radius: f32 = 34.0;

    // Reference tick in the partner frame's +X direction (world +X when
    // partner is ground). Small dashed stub so it reads as "zero".
    painter.line_segment(
        [center, point_at(radius, partner_theta)],
        Stroke::new(1.0, faint),
    );

    // Small tick mark perpendicular at the tip of the reference line
    // to emphasise it as the zero reference.
    let tick_tip = point_at(radius, partner_theta);
    let perp_dir = partner_theta + std::f64::consts::FRAC_PI_2;
    let perp_half = 4.0_f32;
    painter.line_segment(
        [
            Pos2::new(
                tick_tip.x + perp_half * (perp_dir.cos() as f32),
                tick_tip.y - perp_half * (perp_dir.sin() as f32),
            ),
            Pos2::new(
                tick_tip.x - perp_half * (perp_dir.cos() as f32),
                tick_tip.y + perp_half * (perp_dir.sin() as f32),
            ),
        ],
        Stroke::new(1.0, faint),
    );

    // Arc sweeping from reference (partner_theta) to driver orientation,
    // spanning crank_angle radians. Segment count scales with magnitude
    // so large angles still look smooth.
    let segments = ((crank_angle.abs() / 0.1).ceil() as usize).clamp(8, 96);
    let stroke = Stroke::new(1.5, color);
    let mut prev = point_at(radius, partner_theta);
    for i in 1..=segments {
        let frac = i as f64 / segments as f64;
        let a = partner_theta + crank_angle * frac;
        let pt = point_at(radius, a);
        painter.line_segment([prev, pt], stroke);
        prev = pt;
    }

    // Angle label at the midpoint of the arc, in display units.
    let mid_theta = partner_theta + crank_angle * 0.5;
    let label_pos = point_at(radius + 14.0, mid_theta);
    let units = &state.display_units;
    let label = format!("{:.1}{}", units.angle(crank_angle), units.angle_suffix());
    painter.text(
        label_pos,
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(11.0),
        color,
    );
}

/// Draw the equation-overlay labels: per-joint Φ tag + per-body q vector.
///
/// Active only when `state.show_equation_overlay` is true. Uses the existing
/// `joint_hit_targets` (which were populated during `render_mechanism`) for
/// joint anchor positions to avoid recomputing world transforms. Bodies are
/// labeled at their CG. Drivers get an orange highlight; other constraint
/// kinds use kind-specific colors so different joint types are scannable.
///
/// Performance: `joint_hit_targets` already iterated all joints; this pass
/// iterates them again (cheap). Body iteration is one pass over
/// `mech.bodies()`. Each label is one egui text-paint call. Total cost on a
/// 6-bar mechanism is well under 1 ms.
fn draw_equation_overlay(
    painter: &egui::Painter,
    canvas_rect: Rect,
    state: &AppState,
    joint_hit_targets: &[(Pos2, String)],
) {
    use crate::gui::eq_rendering::{kind_at, EqKind};

    let Some(mech) = state.mechanism.as_ref() else { return };
    let view = &state.view;
    let mech_state = mech.state();
    let q = &state.q;

    // ── Per-joint constraint tags ─────────────────────────────────────
    // joint_hit_targets contains (screen_pos, joint_id) pairs in mech.joints()
    // order, which lines up with the first n_joints constraint indices.
    for (i, (screen_pos, joint_id)) in joint_hit_targets.iter().enumerate() {
        if !canvas_rect.contains(*screen_pos) {
            continue;
        }
        let Some(kind) = kind_at(mech, i) else { continue };
        let label = format!("{}: {}", joint_id, kind.short_label());
        let color = kind_overlay_color(state, kind);
        // Place label below-right of the joint glyph so the joint marker
        // stays readable.
        let pos = Pos2::new(screen_pos.x + 8.0, screen_pos.y + 8.0);
        draw_pill_label(painter, pos, &label, color, egui::Align2::LEFT_TOP);
    }

    // ── Driver tag at the driver pivot ────────────────────────────────
    // Drivers don't have hit targets in joint_hit_targets, so render their
    // tag at the partner-pivot world location (same anchor that
    // draw_crank_angle_indicator uses). For linear drivers, anchor between
    // the two attachment points.
    let n_joints = mech.joints().len();
    for (drv_idx, drv) in mech.drivers().iter().enumerate() {
        let constr_idx = n_joints + drv_idx;
        let id_label = drv.id();
        // Find the joint that connects the driver's two bodies (best-effort).
        let mut anchor: Option<Pos2> = None;
        for joint in mech.joints() {
            if joint.is_revolute() {
                let bi = joint.body_i_id();
                let bj = joint.body_j_id();
                if (bi == drv.body_i_id() && bj == drv.body_j_id())
                    || (bi == drv.body_j_id() && bj == drv.body_i_id())
                {
                    let global = mech_state.body_point_global(
                        joint.body_i_id(),
                        &joint.point_i_local(),
                        q,
                    );
                    let sp = view.world_to_screen(global.x, global.y);
                    anchor = Some(Pos2::new(sp[0], sp[1]));
                    break;
                }
            }
        }
        if let Some(anchor) = anchor {
            if canvas_rect.contains(anchor) {
                let Some(kind) = kind_at(mech, constr_idx) else { continue };
                let label = render_driver_overlay_label(id_label, drv.meta());
                let color = kind_overlay_color(state, kind);
                let pos = Pos2::new(anchor.x - 8.0, anchor.y - 24.0);
                draw_pill_label(painter, pos, &label, color, egui::Align2::RIGHT_TOP);
            }
        }
    }

    // Linear drivers: label at the midpoint of P_a–P_b in world space.
    let n_rev_drivers = mech.drivers().len();
    for (drv_idx, drv) in mech.linear_drivers().iter().enumerate() {
        let constr_idx = n_joints + n_rev_drivers + drv_idx;
        let pa_local = nalgebra::Vector2::new(drv.point_a()[0], drv.point_a()[1]);
        let pb_local = nalgebra::Vector2::new(drv.point_b()[0], drv.point_b()[1]);
        let pa = mech_state.body_point_global(drv.body_i_id(), &pa_local, q);
        let pb = mech_state.body_point_global(drv.body_j_id(), &pb_local, q);
        let mid_x = (pa.x + pb.x) * 0.5;
        let mid_y = (pa.y + pb.y) * 0.5;
        let sp = view.world_to_screen(mid_x, mid_y);
        let anchor = Pos2::new(sp[0], sp[1]);
        if !canvas_rect.contains(anchor) {
            continue;
        }
        let Some(kind) = kind_at(mech, constr_idx) else { continue };
        let label = render_driver_overlay_label(drv.id(), drv.meta());
        let color = kind_overlay_color(state, kind);
        draw_pill_label(painter, anchor, &label, color, egui::Align2::CENTER_BOTTOM);
    }

    // ── Per-body q labels ─────────────────────────────────────────────
    for (body_id, body) in mech.bodies().iter() {
        if body_id == GROUND_ID {
            continue;
        }
        let (x, y, theta) = mech_state.get_pose(body_id, q);
        let cg_global = mech_state.body_point_global(body_id, &body.cg_local, q);
        let cg_screen = view.world_to_screen(cg_global.x, cg_global.y);
        let anchor = Pos2::new(cg_screen[0], cg_screen[1] + 14.0);
        if !canvas_rect.contains(anchor) {
            continue;
        }
        let units = &state.display_units;
        let label = format!(
            "q_{} = ({:.3}, {:.3}, {:.1}{})",
            body_id,
            units.length(x),
            units.length(y),
            units.angle(theta),
            units.angle_suffix(),
        );
        let color = state.nc(Color32::from_rgb(180, 200, 230));
        draw_pill_label(
            painter,
            anchor,
            &label,
            color,
            egui::Align2::CENTER_TOP,
        );
    }
}

/// Color for an `EqKind` overlay tag. Routed through `state.nc()` for
/// Nathan-Mode grayscale support.
fn kind_overlay_color(state: &AppState, kind: crate::gui::eq_rendering::EqKind) -> Color32 {
    use crate::gui::eq_rendering::EqKind;
    let raw = match kind {
        EqKind::Revolute => Color32::from_rgb(80, 160, 255),
        EqKind::Prismatic => Color32::from_rgb(180, 130, 255),
        EqKind::Fixed => Color32::from_rgb(220, 220, 220),
        EqKind::CamFollower => Color32::from_rgb(100, 220, 140),
        EqKind::RevoluteDriver | EqKind::LinearDriver => {
            Color32::from_rgb(255, 190, 80)
        }
    };
    state.nc(raw)
}

/// Build the driver's overlay label including a short parameterization hint.
fn render_driver_overlay_label(
    id: &str,
    meta: Option<&crate::core::driver::DriverMeta>,
) -> String {
    use crate::core::driver::DriverMeta;
    match meta {
        Some(DriverMeta::ConstantSpeed { omega, theta_0 }) => {
            format!("{}: θ = {:.2} + {:.2}·t", id, theta_0, omega)
        }
        Some(DriverMeta::Expression { expr, .. }) => {
            format!("{}: f(t) = {}", id, expr)
        }
        Some(DriverMeta::LinearLength { velocity, length_0 }) => {
            format!("{}: L = {:.3} + {:.3}·t", id, length_0, velocity)
        }
        Some(DriverMeta::CosineStroke { .. }) => format!("{}: cosine stroke", id),
        None => format!("{}: f(t)", id),
    }
}

/// Draw a small text label with a faint background pill so it stays legible
/// over geometry. `align` controls where `pos` sits relative to the text rect.
fn draw_pill_label(
    painter: &egui::Painter,
    pos: Pos2,
    text: &str,
    color: Color32,
    align: egui::Align2,
) {
    let font = FontId::proportional(10.0);
    let galley = painter.layout_no_wrap(text.to_string(), font.clone(), color);
    let size = galley.size();
    let text_min = match align {
        egui::Align2::LEFT_TOP => pos,
        egui::Align2::RIGHT_TOP => Pos2::new(pos.x - size.x, pos.y),
        egui::Align2::LEFT_BOTTOM => Pos2::new(pos.x, pos.y - size.y),
        egui::Align2::RIGHT_BOTTOM => Pos2::new(pos.x - size.x, pos.y - size.y),
        egui::Align2::CENTER_TOP => Pos2::new(pos.x - size.x * 0.5, pos.y),
        egui::Align2::CENTER_BOTTOM => Pos2::new(pos.x - size.x * 0.5, pos.y - size.y),
        _ => Pos2::new(pos.x - size.x * 0.5, pos.y - size.y * 0.5),
    };
    let pad = egui::vec2(3.0, 1.0);
    let rect = Rect::from_min_size(text_min, size).expand2(pad);
    let bg = Color32::from_rgba_unmultiplied(20, 22, 30, 200);
    painter.rect_filled(rect, 3.0, bg);
    painter.galley(text_min, galley, Color32::PLACEHOLDER);
}

/// Show rich tooltips when the mouse hovers over a body, joint, or force zone.
fn render_hover_tooltips(
    ui: &mut egui::Ui,
    canvas_rect: Rect,
    state: &AppState,
    joint_hit_targets: &[(Pos2, String)],
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
) {
    use crate::gui::state::EditorTool;
    use super::hit_testing::find_nearest_body_segment;

    if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
        if canvas_rect.contains(hover_pos) && state.active_tool == EditorTool::Select {
            let mut shown_tooltip = false;

            // Check joints first (they're drawn on top)
            if !shown_tooltip {
                for (jpos, jid) in joint_hit_targets {
                    if jpos.distance(hover_pos) < HIT_RADIUS {
                        if let Some(mech) = &state.mechanism {
                            if let Some(joint) = mech.joints().iter().find(|j| j.id() == jid) {
                                let joint_type_str = if joint.is_revolute() {
                                    "Revolute"
                                } else if joint.is_prismatic() {
                                    "Prismatic"
                                } else if joint.is_fixed() {
                                    "Fixed"
                                } else {
                                    "Cam Follower"
                                };

                                // Try to get a label from the blueprint.
                                let bp_label = state.blueprint.as_ref()
                                    .and_then(|bp| bp.joints.get(jid.as_str()))
                                    .and_then(|jj| match jj {
                                        crate::io::JointJson::Revolute { label, .. }
                                        | crate::io::JointJson::Fixed { label, .. }
                                        | crate::io::JointJson::Prismatic { label, .. }
                                        | crate::io::JointJson::CamFollower { label, .. }
                                        | crate::io::JointJson::RevoluteDriver { label, .. } => {
                                            label.as_deref()
                                        }
                                    });
                                let display_label = bp_label.unwrap_or(jid.as_str());

                                egui::Tooltip::always_open(
                                    ui.ctx().clone(),
                                    ui.layer_id(),
                                    egui::Id::new("joint_tooltip"),
                                    egui::PopupAnchor::Pointer,
                                ).show(|ui: &mut egui::Ui| {
                                    ui.label(egui::RichText::new(display_label).strong());
                                    ui.label(format!("Type: {}", joint_type_str));
                                    ui.label(format!("{} \u{2194} {}", joint.body_i_id(), joint.body_j_id()));
                                    // Show reaction forces if available.
                                    if let Some(&(fx, fy)) = state.force_results.joint_reactions.get(jid) {
                                        ui.label(format!("Reaction: ({:.0}, {:.0}) N", fx, fy));
                                    }
                                });
                                shown_tooltip = true;
                            }
                        }
                        break;
                    }
                }
            }

            // Then check attachment points / body areas
            if !shown_tooltip {
                for hit in attachment_hit_targets {
                    if hit.screen_pos.distance(hover_pos) < HIT_RADIUS {
                        if let Some(mech) = &state.mechanism {
                            if let Some(body) = mech.bodies().get(&hit.body_id) {
                                if hit.body_id == GROUND_ID {
                                    egui::Tooltip::always_open(
                                        ui.ctx().clone(),
                                        ui.layer_id(),
                                        egui::Id::new("body_tooltip"),
                                        egui::PopupAnchor::Pointer,
                                    ).show(|ui: &mut egui::Ui| {
                                        ui.label(egui::RichText::new("Ground").strong());
                                        ui.label(format!("Point: {}", hit.point_name));
                                    });
                                } else {
                                    show_body_tooltip(ui, body, &hit.body_id);
                                }
                                shown_tooltip = true;
                            }
                        }
                        break;
                    }
                }
            }

            // Then check body segments (link lines) -- wider radius for easier hover
            if !shown_tooltip {
                if let Some(seg_hit) = find_nearest_body_segment(hover_pos, body_segments, LINK_HALF_WIDTH + 4.0) {
                    if let Some(mech) = &state.mechanism {
                        if let Some(body) = mech.bodies().get(&seg_hit.body_id) {
                            show_body_tooltip(ui, body, &seg_hit.body_id);
                            shown_tooltip = true;
                        }
                    }
                }
            }

            // Then check force zones
            if !shown_tooltip {
                if let Some(mech) = &state.mechanism {
                    for elem in mech.forces() {
                        if let ForceElement::ForceZone(fz) = elem {
                            let min_sp = state.view.world_to_screen(fz.zone_min[0], fz.zone_min[1]);
                            let max_sp = state.view.world_to_screen(fz.zone_max[0], fz.zone_max[1]);
                            let s_min_x = min_sp[0].min(max_sp[0]);
                            let s_min_y = min_sp[1].min(max_sp[1]);
                            let s_max_x = min_sp[0].max(max_sp[0]);
                            let s_max_y = min_sp[1].max(max_sp[1]);
                            let zone_rect = Rect::from_min_max(
                                Pos2::new(s_min_x, s_min_y),
                                Pos2::new(s_max_x, s_max_y),
                            );
                            if zone_rect.contains(hover_pos) {
                                let label = fz.label.as_deref().unwrap_or("Force Zone");
                                egui::Tooltip::always_open(
                                    ui.ctx().clone(),
                                    ui.layer_id(),
                                    egui::Id::new("fz_tooltip"),
                                    egui::PopupAnchor::Pointer,
                                ).show(|ui: &mut egui::Ui| {
                                    ui.label(egui::RichText::new(label).strong());
                                    ui.label(format!("Body: {}", fz.body_id));
                                    ui.label(format!("Force: ({:.1}, {:.1}) N", fz.force[0], fz.force[1]));
                                    ui.label(format!(
                                        "Zone: ({:.1}, {:.1}) to ({:.1}, {:.1}) mm",
                                        fz.zone_min[0] * 1e3, fz.zone_min[1] * 1e3,
                                        fz.zone_max[0] * 1e3, fz.zone_max[1] * 1e3
                                    ));
                                });
                                shown_tooltip = true;
                                break;
                            }
                        }
                    }
                }
            }

            let _ = shown_tooltip; // suppress unused warning
        }
    }
}

// ── Background image rendering ──────────────────────────────────────────────

/// Draw the background image overlay on the canvas (behind mechanism, above grid).
///
/// The image is positioned in world coordinates using the `BackgroundImage`
/// offset and scale settings, and drawn with the configured opacity.
pub fn draw_background_image(
    painter: &egui::Painter,
    canvas_rect: Rect,
    state: &AppState,
) {
    let Some(ref bg) = state.background_image else {
        return;
    };
    if bg.opacity <= 0.0 {
        return;
    }

    let view = &state.view;

    // Compute image world-space dimensions from pixel size and scale.
    let img_w_world = bg.size_px[0] as f64 / bg.scale_px_per_m;
    let img_h_world = bg.size_px[1] as f64 / bg.scale_px_per_m;

    // Image corners in world space (centered at offset).
    let left = bg.world_offset[0] - img_w_world / 2.0;
    let top = bg.world_offset[1] + img_h_world / 2.0;
    let right = bg.world_offset[0] + img_w_world / 2.0;
    let bottom = bg.world_offset[1] - img_h_world / 2.0;

    // Convert to screen coordinates.
    let tl = view.world_to_screen(left, top);
    let br = view.world_to_screen(right, bottom);
    let screen_rect = Rect::from_min_max(
        Pos2::new(tl[0], tl[1]),
        Pos2::new(br[0], br[1]),
    );

    // Skip if entirely off-screen.
    if !canvas_rect.intersects(screen_rect) {
        return;
    }

    // Draw the image with opacity via tint alpha.
    let alpha = (bg.opacity * 255.0).clamp(0.0, 255.0) as u8;
    let tint = Color32::from_rgba_unmultiplied(255, 255, 255, alpha);
    painter.image(
        bg.texture.id(),
        screen_rect,
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
        tint,
    );
}


pub fn draw_grid(
    painter: &egui::Painter,
    rect: Rect,
    view: &ViewTransform,
    grid: &GridSettings,
    nathan_mode: bool,
) {
    if !grid.show_grid {
        return;
    }

    let gc = |c: Color32| -> Color32 {
        if nathan_mode { to_grayscale(c) } else { c }
    };

    let spacing = grid.spacing_m;
    if spacing <= 0.0 {
        return;
    }

    // Convert screen bounds to world coordinates.
    let [world_left, world_top] = view.screen_to_world(rect.left(), rect.top());
    let [world_right, world_bottom] = view.screen_to_world(rect.right(), rect.bottom());

    // Spacing is already zoom-adapted (set in draw_canvas each frame).
    // Safety cap: if somehow too many lines, bail.
    let x_count = (world_right / spacing).ceil() as i64 - (world_left / spacing).floor() as i64;
    let y_count = (world_top / spacing).ceil() as i64 - (world_bottom / spacing).floor() as i64;
    if x_count + y_count > 500 {
        return;
    }

    let x_min_i = (world_left / spacing).floor() as i64;
    let x_max_i = (world_right / spacing).ceil() as i64;
    let y_min_i = (world_bottom / spacing).floor() as i64;
    let y_max_i = (world_top / spacing).ceil() as i64;

    let minor_stroke = Stroke::new(0.5, gc(GRID_COLOR));
    let major_stroke = Stroke::new(1.0, gc(GRID_MAJOR_COLOR));

    let label_color = Color32::from_rgba_premultiplied(120, 125, 140, 180);
    let label_font = FontId::proportional(9.0);

    // Vertical lines (every 5th is major).
    for i in x_min_i..=x_max_i {
        let wx = i as f64 * spacing;
        let top = view.world_to_screen(wx, world_top);
        let bottom = view.world_to_screen(wx, world_bottom);
        let stroke = if i % 5 == 0 { major_stroke } else { minor_stroke };
        painter.line_segment(
            [Pos2::new(top[0], top[1]), Pos2::new(bottom[0], bottom[1])],
            stroke,
        );
        // Distance label at major grid lines along the X axis.
        if i % 5 == 0 && i != 0 {
            let mm = wx * 1000.0;
            let label = if spacing < 0.001 {
                format!("{:.1}", mm)  // sub-mm: show 0.1mm precision
            } else {
                format!("{:.0}", mm)  // mm or coarser: whole numbers
            };
            // Place label near the X axis (y=0), clamped to viewport.
            let origin_screen = view.world_to_screen(wx, 0.0);
            let label_y = origin_screen[1].clamp(rect.top() + 2.0, rect.bottom() - 12.0);
            painter.text(
                Pos2::new(origin_screen[0], label_y),
                egui::Align2::CENTER_TOP,
                &label,
                label_font.clone(),
                label_color,
            );
        }
    }

    // Horizontal lines (every 5th is major).
    for i in y_min_i..=y_max_i {
        let wy = i as f64 * spacing;
        let left = view.world_to_screen(world_left, wy);
        let right = view.world_to_screen(world_right, wy);
        let stroke = if i % 5 == 0 { major_stroke } else { minor_stroke };
        painter.line_segment(
            [Pos2::new(left[0], left[1]), Pos2::new(right[0], right[1])],
            stroke,
        );
        // Distance label at major grid lines along the Y axis.
        if i % 5 == 0 && i != 0 {
            let mm = wy * 1000.0;
            let label = format!("{:.0}", mm);
            // Place label near the Y axis (x=0), clamped to viewport.
            let origin_screen = view.world_to_screen(0.0, wy);
            let label_x = origin_screen[0].clamp(rect.left() + 2.0, rect.right() - 24.0);
            painter.text(
                Pos2::new(label_x, origin_screen[1]),
                egui::Align2::LEFT_CENTER,
                &label,
                label_font.clone(),
                label_color,
            );
        }
    }

    // Origin crosshair (subtle red/green axis lines like CAD tools).
    let origin = view.world_to_screen(0.0, 0.0);
    let origin_pos = Pos2::new(origin[0], origin[1]);
    if rect.contains(origin_pos) {
        // X-axis (red, horizontal)
        painter.line_segment(
            [Pos2::new(rect.left(), origin[1]), Pos2::new(rect.right(), origin[1])],
            Stroke::new(1.0, Color32::from_rgba_premultiplied(180, 60, 60, 100)),
        );
        // Y-axis (green, vertical)
        painter.line_segment(
            [Pos2::new(origin[0], rect.top()), Pos2::new(origin[0], rect.bottom())],
            Stroke::new(1.0, Color32::from_rgba_premultiplied(60, 180, 60, 100)),
        );
    }
}

/// Rich tooltip for a body element, showing label, mass, inertia, geometry,
/// and computed link length.
pub fn show_body_tooltip(ui: &mut egui::Ui, body: &Body, body_id: &str) {
    egui::Tooltip::always_open(
        ui.ctx().clone(),
        ui.layer_id(),
        egui::Id::new("body_tooltip"),
        egui::PopupAnchor::Pointer,
    ).show(|ui: &mut egui::Ui| {
        let display = if body.label.is_empty() { body_id } else { &body.label };
        ui.label(egui::RichText::new(display).strong());
        ui.label(format!("Mass: {:.3} kg", body.mass));
        ui.label(format!("Izz: {:.6} kg\u{00b7}m\u{00b2}", body.izz_cg));
        if let Some(ref geo) = body.geometry {
            ui.label(format!(
                "Geometry: {:.1} \u{00d7} {:.1} mm",
                geo.width * 1e3,
                geo.height * 1e3
            ));
        }
        if body.attachment_points.len() == 2 {
            let pts: Vec<_> = body.attachment_points.values().collect();
            let length = (pts[0] - pts[1]).norm();
            ui.label(format!("Length: {:.1} mm", length * 1e3));
        }
    });
}

// ── Motion ribbon (ghost poses along trajectory) ────────────────────────────

/// Render N evenly-spaced ghost poses of the mechanism along the
/// back-solved trajectory `q(t)` on the canvas, faded behind the live
/// pose. Visualises the swept path without animating.
///
/// Each ghost is re-solved by mapping the recorded `u_k` (the
/// back-solved input parameter) to the equivalent kinematic time and
/// invoking `solve_position`. The rendered geometry is a simple
/// link-bar outline (no joints, dimensions, or labels) so the silhouette
/// is unobtrusive.
///
/// Skipped silently when `sweep_data.u_values` is `None` (sweep
/// hasn't completed) or `n_ghosts < 2` / `total_samples < 2`.
fn draw_motion_ribbon(
    painter: &egui::Painter,
    state: &AppState,
    sweep_data: &crate::gui::sweep::SweepData,
) {
    use crate::solver::kinematics::solve_position;

    let n = state.motion_ribbon_n_ghosts;
    let total_samples = sweep_data.u_values.as_ref().map(|v| v.len()).unwrap_or(0);
    if total_samples < 2 || n < 2 {
        return;
    }

    let Some(mech) = state.mechanism.as_ref() else {
        return;
    };
    let nominal_rate = state.driver_omega();
    if nominal_rate.abs() < 1e-12 {
        return;
    }
    let u_0 = state.driver_theta_0();

    let stride = (total_samples - 1) as f64 / (n - 1) as f64;
    let u_values = sweep_data.u_values.as_ref().unwrap();

    // Warm-start each ghost solve from the previous successful pose so
    // we stay on the same assembly branch as the live mechanism.
    let mut q_seed = state.last_good_q.clone();

    for i in 0..n {
        let idx = ((i as f64 * stride).round() as usize).min(total_samples - 1);
        let u_k = u_values[idx];
        let t_mech = (u_k - u_0) / nominal_rate;

        match solve_position(mech, &q_seed, t_mech, 1e-10, 50) {
            Ok(res) if res.converged => {
                // Alpha ramps 40 → 240 across ghosts so older poses are
                // dimmer than later ones (gives a visible ordering).
                let alpha =
                    ((i as f64) / (n.saturating_sub(1).max(1) as f64) * 200.0 + 40.0)
                        .min(255.0) as u8;
                draw_mechanism_ghost(painter, state, mech, &res.q, alpha);
                q_seed = res.q;
            }
            _ => {
                // Skip this ghost; keep the previous q_seed for the next attempt.
            }
        }
    }
}

/// Render a faded silhouette of the mechanism at a given pose `q`.
///
/// Draws only the link bars (rounded-rect outlines) for each
/// non-ground body. No joints, no attachment dots, no labels — just
/// enough to convey the mechanism's pose silhouette. The fill and
/// stroke alpha are both scaled by `alpha`.
fn draw_mechanism_ghost(
    painter: &egui::Painter,
    state: &AppState,
    mech: &Mechanism,
    q: &nalgebra::DVector<f64>,
    alpha: u8,
) {
    let mech_state = mech.state();
    let view = &state.view;
    let bodies = mech.bodies();

    // Use BODY_COLOR with alpha override for the stroke; faded fill at half alpha.
    let base = if state.nathan_mode {
        to_grayscale(BODY_COLOR)
    } else {
        BODY_COLOR
    };
    let fill_alpha = alpha.saturating_div(3).max(8);
    let stroke_color = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha);
    let fill_color = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), fill_alpha);

    for (body_id, body) in bodies.iter() {
        if body_id == GROUND_ID {
            continue;
        }

        let mut point_names: Vec<&String> = body.attachment_points.keys().collect();
        point_names.sort();
        let screen_points: Vec<Pos2> = point_names
            .iter()
            .map(|name| {
                let local = &body.attachment_points[*name];
                let global = mech_state.body_point_global(body_id, local, q);
                let sp = view.world_to_screen(global.x, global.y);
                Pos2::new(sp[0], sp[1])
            })
            .collect();

        if screen_points.len() < 2 {
            continue;
        }

        // Reuse the bar geometry from the live render path.
        let draw_bar = |a: Pos2, b: Pos2| {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1.0 {
                return;
            }
            let nx = -dy / len * LINK_HALF_WIDTH;
            let ny = dx / len * LINK_HALF_WIDTH;
            let corners = vec![
                Pos2::new(a.x + nx, a.y + ny),
                Pos2::new(b.x + nx, b.y + ny),
                Pos2::new(b.x - nx, b.y - ny),
                Pos2::new(a.x - nx, a.y - ny),
            ];
            let shape = egui::epaint::PathShape::convex_polygon(
                corners,
                fill_color,
                Stroke::new(1.5, stroke_color),
            );
            painter.add(shape);
        };
        for pair in screen_points.windows(2) {
            draw_bar(pair[0], pair[1]);
        }
        if screen_points.len() >= 3 {
            draw_bar(*screen_points.last().unwrap(), screen_points[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_color_blue_at_zero() {
        let c = heat_color(0.0);
        assert_eq!(c, Color32::from_rgb(0, 0, 255));
    }

    #[test]
    fn heat_color_red_at_one() {
        let c = heat_color(1.0);
        assert_eq!(c, Color32::from_rgb(255, 0, 0));
    }

    #[test]
    fn heat_color_green_at_half() {
        let c = heat_color(0.5);
        assert_eq!(c, Color32::from_rgb(0, 255, 0));
    }

    #[test]
    fn heat_color_clamps_out_of_range() {
        let below = heat_color(-1.0);
        let above = heat_color(2.0);
        assert_eq!(below, Color32::from_rgb(0, 0, 255));
        assert_eq!(above, Color32::from_rgb(255, 0, 0));
    }

    #[test]
    fn heat_color_transitions_are_smooth() {
        // Check that adjacent samples don't have huge jumps.
        let steps = 100;
        for i in 0..steps {
            let t1 = i as f32 / steps as f32;
            let t2 = (i + 1) as f32 / steps as f32;
            let c1 = heat_color(t1);
            let c2 = heat_color(t2);
            let dr = (c1.r() as i32 - c2.r() as i32).unsigned_abs();
            let dg = (c1.g() as i32 - c2.g() as i32).unsigned_abs();
            let db = (c1.b() as i32 - c2.b() as i32).unsigned_abs();
            assert!(
                dr <= 12 && dg <= 12 && db <= 12,
                "Large color jump at t={:.3}: {:?} -> {:?}",
                t1, c1, c2
            );
        }
    }
}
