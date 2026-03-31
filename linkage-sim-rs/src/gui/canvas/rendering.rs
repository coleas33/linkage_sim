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
            matches!(selected, Some(SelectedEntity::Body(s)) if s == body_id);
        let color = if is_selected {
            gc(BODY_SELECTED_COLOR)
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
            matches!(selected, Some(SelectedEntity::Joint(s)) if s == joint.id());
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

        // Joint labels: auto-generated from type prefix + index, or from
        // the blueprint label if one exists.
        let bp_joints = state.blueprint.as_ref().map(|bp| &bp.joints);
        let mut rev_idx = 0usize;
        let mut pris_idx = 0usize;
        let mut fix_idx = 0usize;
        let mut cam_idx = 0usize;
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

            let auto_label: String;
            let display_label = if let Some(lbl) = bp_label {
                lbl
            } else {
                auto_label = if joint.is_revolute() {
                    rev_idx += 1;
                    format!("R{}", rev_idx)
                } else if joint.is_prismatic() {
                    pris_idx += 1;
                    format!("P{}", pris_idx)
                } else if joint.is_fixed() {
                    fix_idx += 1;
                    format!("F{}", fix_idx)
                } else {
                    cam_idx += 1;
                    format!("C{}", cam_idx)
                };
                &auto_label
            };

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
                draw_force_arrow(painter, *screen_pos, fx as f32, fy as f32);
            }
        }
    }

    // ── Force element visuals ────────────────────────────────────────
    if state.show_forces {
        draw_force_elements(painter, state, &state.view);
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
        EditorTool::Select => None,
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
                                        ui.label(format!("Reaction: ({:.1}, {:.1}) N", fx, fy));
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

// ── Force element rendering ──────────────────────────────────────────────────

/// Draw visual representations of all force elements (springs, dampers, external
/// forces/torques) on the canvas.
///
/// Called when `state.show_forces` is true. Skipped when there is no mechanism.
fn draw_force_elements(
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
                    // Force magnitude label.
                    painter.text(
                        Pos2::new(mid.x, mid.y - 10.0),
                        egui::Align2::CENTER_BOTTOM,
                        format!("{:.0} N", la.force),
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

// ── Drawing primitives ───────────────────────────────────────────────────────

/// Draw a zigzag spring symbol between two screen-space points.
///
/// The spring is rendered as: short straight lead-in, N zigzag segments, short
/// straight lead-out. The zigzag amplitude is fixed at 6 pixels perpendicular
/// to the line of action.
fn draw_spring_zigzag(
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
fn draw_damper_symbol(
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
fn draw_external_force_arrow(
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
        format!("{:.1} N", mag),
        FontId::proportional(11.0),
        EXT_FORCE_COLOR,
    );
}

/// Draw a curved torque arc with an arrowhead at a point on the canvas.
///
/// Positive torque draws counterclockwise; negative draws clockwise.
/// The arc spans approximately 270 degrees and has a small arrowhead at the tip.
fn draw_torque_arc(
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
fn draw_alignment_guides(
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
fn draw_rotary_badge(
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
fn draw_force_arrow(painter: &egui::Painter, origin: Pos2, fx: f32, fy: f32) {
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
        format!("{:.2} N", mag),
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

/// Draw a grid of lines on the canvas behind the mechanism.
///
/// Lines are drawn at multiples of `grid.spacing_m` in world coordinates.
/// If the viewport is zoomed out so far that more than 200 lines would be
/// drawn, the spacing is recursively doubled until the line count fits.
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
