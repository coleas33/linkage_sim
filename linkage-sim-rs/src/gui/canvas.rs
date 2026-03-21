//! 2D mechanism canvas: rendering, pan/zoom, hit testing, drag, context menus.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::gui::state::{
    AddBodyState, AppState, ContextMenuTarget, EditorTool, GridSettings,
    SelectedEntity, ViewTransform,
};

// ── Colors — CAD-inspired dark palette ───────────────────────────────────────

// Canvas background: subtle gradient-like dark with slight blue tint
const BG_COLOR: Color32 = Color32::from_rgb(22, 24, 32);
const GRID_COLOR: Color32 = Color32::from_rgba_premultiplied(45, 50, 65, 50);
const GRID_MAJOR_COLOR: Color32 = Color32::from_rgba_premultiplied(55, 60, 80, 80);
const GROUND_LINE_COLOR: Color32 = Color32::from_rgb(80, 85, 100);

// Bodies: clean blue with warm orange selection (SolidWorks-style)
const BODY_COLOR: Color32 = Color32::from_rgb(70, 150, 240);
const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 180, 40);

// Joints: bright with clear hierarchy
const JOINT_COLOR: Color32 = Color32::from_rgb(220, 225, 240);
const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 180, 40);
const DRIVER_JOINT_COLOR: Color32 = Color32::from_rgb(80, 220, 130);
const GROUND_MARKER_COLOR: Color32 = Color32::from_rgb(160, 145, 110);
const ATTACHMENT_DOT_COLOR: Color32 = Color32::from_rgb(170, 185, 210);

// Labels
const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(150, 160, 180);
const DEBUG_DIM_COLOR: Color32 = Color32::from_rgb(85, 90, 105);
const NO_MECH_TEXT_COLOR: Color32 = Color32::from_rgb(90, 95, 115);
const JOINT_CREATE_HIGHLIGHT: Color32 = Color32::from_rgb(50, 230, 100);
const DIM_LABEL_COLOR: Color32 = Color32::from_rgb(170, 195, 130);

// Force elements: semantic color coding
const FORCE_ARROW_COLOR: Color32 = Color32::from_rgb(255, 80, 80);
const SPRING_COLOR: Color32 = Color32::from_rgb(50, 200, 110);
const DAMPER_COLOR: Color32 = Color32::from_rgb(90, 145, 255);
const EXT_FORCE_COLOR: Color32 = Color32::from_rgb(255, 160, 30);
const GAS_SPRING_COLOR: Color32 = Color32::from_rgb(170, 95, 255);
const ACTUATOR_COLOR: Color32 = Color32::from_rgb(255, 115, 55);
const BEARING_COLOR: Color32 = Color32::from_rgb(190, 175, 95);
const JOINT_LIMIT_COLOR: Color32 = Color32::from_rgb(215, 75, 75);
const MOTOR_COLOR: Color32 = Color32::from_rgb(80, 220, 130);

// ── Sizing ──────────────────────────────────────────────────────────────────

const FORCE_ARROW_WIDTH: f32 = 2.5;
const FORCE_ARROW_MIN_PX: f32 = 3.0;
const FORCE_ARROW_MAX_PX: f32 = 80.0;
const FORCE_ARROW_SCALE: f32 = 30.0;

const BODY_STROKE_WIDTH: f32 = 3.5;
const LINK_HALF_WIDTH: f32 = 8.0;
const JOINT_RADIUS: f32 = 7.0;
const JOINT_STROKE_WIDTH: f32 = 2.0;
const GROUND_MARKER_SIZE: f32 = 14.0;
const HIT_RADIUS: f32 = 12.0;
const ATTACHMENT_DOT_RADIUS: f32 = 3.5;
const ZOOM_FACTOR: f32 = 1.12;
const MIN_SCALE: f32 = 100.0;
const MAX_SCALE: f32 = 100_000.0;

/// An attachment point hit target: screen + world position, body ID, point name.
#[derive(Clone)]
struct AttachmentHit {
    screen_pos: Pos2,
    /// World coordinates of this attachment point (for precise snapping).
    world_pos: [f64; 2],
    body_id: String,
    point_name: String,
}

/// Result of a body-segment hit test.
#[allow(dead_code)]
struct SegmentHit {
    body_id: String,
    world_pos: [f64; 2],
    screen_pos: Pos2,
    point_a_name: String,
    point_b_name: String,
}

/// Collected body segment for hit testing.
struct BodySegment {
    screen_a: Pos2,
    screen_b: Pos2,
    world_a: [f64; 2],
    world_b: [f64; 2],
    body_id: String,
    point_a_name: String,
    point_b_name: String,
}

/// Project a point onto a line segment. Returns projected point and distance,
/// or None if projection falls outside the segment.
fn project_onto_segment(point: Pos2, seg_a: Pos2, seg_b: Pos2) -> Option<(Pos2, f32)> {
    let ab = seg_b - seg_a;
    let ap = point - seg_a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-10 {
        return None;
    }
    let t = ab.dot(ap) / len_sq;
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let proj = seg_a + ab * t;
    let dist = point.distance(proj);
    Some((proj, dist))
}

/// Find the nearest body line segment to a screen point.
fn find_nearest_body_segment(
    point: Pos2,
    segments: &[BodySegment],
    max_distance: f32,
) -> Option<SegmentHit> {
    let mut best: Option<(f32, Pos2, [f64; 2], String, String, String)> = None;

    for seg in segments {
        if let Some((proj_screen, dist)) = project_onto_segment(point, seg.screen_a, seg.screen_b) {
            if dist <= max_distance {
                if best.as_ref().map_or(true, |(d, _, _, _, _, _)| dist < *d) {
                    let ab_screen = seg.screen_b - seg.screen_a;
                    let ap_screen = proj_screen - seg.screen_a;
                    let t = if ab_screen.length_sq() > 1e-10 {
                        ap_screen.length() / ab_screen.length()
                    } else {
                        0.0
                    };
                    let world_x = seg.world_a[0] + t as f64 * (seg.world_b[0] - seg.world_a[0]);
                    let world_y = seg.world_a[1] + t as f64 * (seg.world_b[1] - seg.world_a[1]);

                    best = Some((dist, proj_screen, [world_x, world_y], seg.body_id.clone(),
                                 seg.point_a_name.clone(), seg.point_b_name.clone()));
                }
            }
        }
    }

    best.map(|(_, screen_pos, world_pos, body_id, point_a_name, point_b_name)| SegmentHit {
        body_id, world_pos, screen_pos, point_a_name, point_b_name,
    })
}

/// Draw the 2D mechanism canvas with interaction.
pub fn draw_canvas(ui: &mut egui::Ui, state: &mut AppState) {
    let (response, painter) =
        ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
    let canvas_rect = response.rect;

    // Fill background.
    painter.rect_filled(canvas_rect, 0.0, BG_COLOR);

    // ── No mechanism message ────────────────────────────────────────────
    if state.mechanism.is_none() {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "No mechanism loaded",
            FontId::proportional(18.0),
            NO_MECH_TEXT_COLOR,
        );
        return;
    }

    // We know mechanism is Some here. We need to carefully manage borrows so
    // that rendering (immutable borrows of state fields via mechanism) finishes
    // before interaction code (which mutates state.view and state.selected).
    //
    // Strategy: collect all hit-test data into local Vecs during the immutable
    // pass, then do all mutation afterwards with no outstanding borrows.

    let show_debug = state.show_debug_overlay;
    let solver_converged = state.solver_status.converged;
    let solver_residual = state.solver_status.residual_norm;
    let solver_iterations = state.solver_status.iterations;

    // Copy driver state before the immutable scope (lives on AppState, not Mechanism).
    let current_driver_joint = state.driver_joint_id.clone();

    // Copy joint-creation state for rendering highlights.
    let creating_joint_first = state.creating_joint.clone();

    // Collect joint screen positions and IDs for hit testing later.
    let mut joint_hit_targets: Vec<(Pos2, String)> = Vec::new();
    // Collect attachment point hit targets (screen pos, body_id, point_name).
    let mut attachment_hit_targets: Vec<AttachmentHit> = Vec::new();
    // Collect body segments for Draw Link segment snap.
    let mut body_segments: Vec<BodySegment> = Vec::new();
    // Grounded revolute joint IDs — candidates for driver reassignment.
    // Collected inside the immutable scope from the mechanism.
    let grounded_revolute_ids: Vec<String>;

    // ── Draw grid behind everything ──────────────────────────────────────
    draw_grid(&painter, canvas_rect, &state.view, &state.grid);

    // Scoped immutable borrow for rendering.
    {
        let mech = state.mechanism.as_ref().unwrap();
        grounded_revolute_ids = mech.grounded_revolute_joint_ids();
        let mech_state = mech.state();
        let bodies = mech.bodies();
        let joints = mech.joints();
        let q = &state.q;
        let view = &state.view;
        let selected = &state.selected;

        // ── Ground line (y=0) — spans full visible viewport ────────────
        {
            let world_left = view.screen_to_world(canvas_rect.left(), 0.0);
            let world_right = view.screen_to_world(canvas_rect.right(), 0.0);
            let left = view.world_to_screen(world_left[0], 0.0);
            let right = view.world_to_screen(world_right[0], 0.0);
            painter.line_segment(
                [Pos2::new(left[0], left[1]), Pos2::new(right[0], right[1])],
                Stroke::new(1.0, GROUND_LINE_COLOR),
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
                let color = trace_colors[color_idx % trace_colors.len()];
                color_idx += 1;

                let screen_pts: Vec<Pos2> = trace
                    .iter()
                    .map(|[wx, wy]| {
                        let sp = view.world_to_screen(*wx, *wy);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();

                // Draw as dashed line
                let dash_len = 6.0f32;
                let gap_len = 4.0f32;
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
                BODY_SELECTED_COLOR
            } else {
                BODY_COLOR
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
                    draw_link_bar(&painter, pair[0], pair[1]);
                }
                if screen_points.len() >= 3 {
                    draw_link_bar(&painter, *screen_points.last().unwrap(), screen_points[0]);
                }
            } else if screen_points.len() == 1 {
                painter.circle_filled(screen_points[0], 4.0, color);
            }

            // Draw small dots at each attachment point for visual clarity.
            for sp in &screen_points {
                painter.circle_filled(*sp, ATTACHMENT_DOT_RADIUS, ATTACHMENT_DOT_COLOR);
            }

            // Dimension labels: show link segment lengths at midpoints.
            if state.show_dimensions && point_positions.len() >= 2 {
                let segments: Vec<(usize, usize)> = if point_positions.len() >= 3 {
                    // Closed polygon: all consecutive pairs + last→first
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
                    let label = format!(
                        "{:.1}{}",
                        state.display_units.length(dist_m),
                        state.display_units.length_suffix()
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
                        FontId::proportional(10.0),
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
                draw_ground_marker(&painter, center, GROUND_MARKER_SIZE, GROUND_MARKER_COLOR);

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
            let color = if is_selected {
                JOINT_SELECTED_COLOR
            } else if is_driver {
                DRIVER_JOINT_COLOR
            } else {
                JOINT_COLOR
            };

            if joint.is_revolute() {
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
                    painter.circle_filled(center, 3.0, DRIVER_JOINT_COLOR);
                }
            } else if joint.is_prismatic() {
                let half = JOINT_RADIUS;
                let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
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

        // ── Joint creation mode: highlight first selected point ─────────
        if let Some((ref cj_body, ref cj_point, _)) = creating_joint_first {
            // Find the screen position of the first-click attachment point.
            for hit in &attachment_hit_targets {
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
    }
    // Immutable borrows of state.mechanism (and its sub-borrows) are now dropped.

    // ── Force arrows ────────────────────────────────────────────────────
    if state.show_forces && solver_converged {
        for (screen_pos, joint_id) in &joint_hit_targets {
            if let Some(&(fx, fy)) = state.force_results.joint_reactions.get(joint_id) {
                draw_force_arrow(&painter, *screen_pos, fx as f32, fy as f32);
            }
        }
    }

    // ── Force element visuals ────────────────────────────────────────
    if state.show_forces {
        draw_force_elements(&painter, state, &state.view);
    }

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
    let hint_text: Option<&str> = match state.active_tool {
        EditorTool::DrawLink => {
            if state.draw_link_start.is_some() {
                Some("Drag to set link length and direction, release to place (Esc to cancel)")
            } else {
                Some("Click an existing point to start drawing a link — use +Ground to place anchors first (Esc to cancel)")
            }
        }
        EditorTool::AddBody => {
            if state.add_body_state.is_some() {
                Some("Click to add points, double-click or Enter to finish (Esc to cancel)")
            } else {
                Some("Click to place first point of new body (Esc to cancel)")
            }
        }
        EditorTool::AddGroundPivot => {
            Some("Click on canvas to place a ground pivot (Esc to cancel)")
        }
        EditorTool::Select => None,
    };
    if let Some(hint) = hint_text {
        painter.text(
            Pos2::new(canvas_rect.center().x, canvas_rect.top() + 20.0),
            egui::Align2::CENTER_TOP,
            hint,
            FontId::proportional(13.0),
            JOINT_CREATE_HIGHLIGHT,
        );
    }

    // ── Hover tooltips ────────────────────────────────────────────────
    // Show a tooltip when the mouse hovers over a body or joint.
    if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
        if canvas_rect.contains(hover_pos) && state.active_tool == EditorTool::Select {
            let mut tooltip_text: Option<String> = None;

            // Check joints first (they're drawn on top)
            for (jpos, jid) in &joint_hit_targets {
                if jpos.distance(hover_pos) < HIT_RADIUS {
                    if let Some(mech) = &state.mechanism {
                        if let Some(joint) = mech.joints().iter().find(|j| j.id() == jid) {
                            let jtype = if joint.is_revolute() {
                                "Revolute"
                            } else if joint.is_prismatic() {
                                "Prismatic"
                            } else {
                                "Fixed"
                            };
                            tooltip_text = Some(format!(
                                "{} ({}) \u{2014} {} \u{2194} {}",
                                jid,
                                jtype,
                                joint.body_i_id(),
                                joint.body_j_id()
                            ));
                        }
                    }
                    break;
                }
            }

            // Then check attachment points / body areas
            if tooltip_text.is_none() {
                for hit in &attachment_hit_targets {
                    if hit.screen_pos.distance(hover_pos) < HIT_RADIUS {
                        if let Some(mech) = &state.mechanism {
                            if let Some(body) = mech.bodies().get(&hit.body_id) {
                                if hit.body_id == GROUND_ID {
                                    tooltip_text = Some(format!(
                                        "Ground: {}",
                                        hit.point_name
                                    ));
                                } else {
                                    tooltip_text = Some(format!(
                                        "{}: {} \u{2014} {:.3} kg",
                                        hit.body_id,
                                        hit.point_name,
                                        body.mass
                                    ));
                                }
                            }
                        }
                        break;
                    }
                }
            }

            // Then check body segments (link lines) -- wider radius for easier hover
            if tooltip_text.is_none() {
                if let Some(seg_hit) = find_nearest_body_segment(hover_pos, &body_segments, LINK_HALF_WIDTH + 4.0) {
                    if let Some(mech) = &state.mechanism {
                        if let Some(body) = mech.bodies().get(&seg_hit.body_id) {
                            tooltip_text = Some(format!(
                                "{} \u{2014} {:.3} kg \u{2014} click to select",
                                seg_hit.body_id,
                                body.mass
                            ));
                        }
                    }
                }
            }

            if let Some(text) = tooltip_text {
                // Paint tooltip as a text label near the cursor with a background box.
                let tip_pos = Pos2::new(hover_pos.x + 14.0, hover_pos.y - 18.0);
                let galley = painter.layout_no_wrap(
                    text,
                    FontId::proportional(11.0),
                    Color32::from_rgb(220, 225, 235),
                );
                let text_rect = egui::Align2::LEFT_BOTTOM
                    .anchor_size(tip_pos, galley.size());
                let bg_rect = text_rect.expand(3.0);
                painter.rect_filled(bg_rect, 3.0, Color32::from_rgba_premultiplied(30, 32, 40, 220));
                painter.rect_stroke(bg_rect, 3.0, Stroke::new(1.0, Color32::from_rgb(60, 65, 80)), egui::StrokeKind::Outside);
                painter.galley(text_rect.min, galley, Color32::PLACEHOLDER);
            }
        }
    }

    // ── Interaction: drag attachment points ─────────────────────────────
    // Primary drag near an attachment point (Select mode) moves the point.
    // Primary drag on empty space (Select mode) pans the view.
    // Middle-click drag always pans.

    let is_shift = ui.input(|i| i.modifiers.shift);
    let mut is_panning = false;

    // Helper: find nearest attachment point within hit radius.
    let find_nearest_attachment = |pos: Pos2| -> Option<&AttachmentHit> {
        attachment_hit_targets
            .iter()
            .filter(|h| pos.distance(h.screen_pos) <= HIT_RADIUS)
            .min_by(|a, b| {
                pos.distance(a.screen_pos)
                    .partial_cmp(&pos.distance(b.screen_pos))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    };

    // Canvas click-to-select removed — Link Editor uses dropdown instead.

    // Primary drag on empty space (Select mode) pans the view.
    if response.dragged_by(egui::PointerButton::Primary) && !is_shift {
        if state.active_tool == EditorTool::Select
            && state.draw_link_start.is_none()
        {
            is_panning = true;
        }
    }

    // ── Interaction: pan ────────────────────────────────────────────────
    if response.dragged() {
        let is_middle = response.dragged_by(egui::PointerButton::Middle);
        let is_shift_primary =
            is_shift && response.dragged_by(egui::PointerButton::Primary);
        if is_middle || is_shift_primary || is_panning {
            let delta = response.drag_delta();
            state.view.offset[0] += delta.x;
            state.view.offset[1] += delta.y;
        }
    }

    // ── Interaction: zoom toward mouse ──────────────────────────────────
    if response.hovered() {
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta.abs() > 0.0 {
            let factor = if scroll_delta > 0.0 {
                ZOOM_FACTOR
            } else {
                1.0 / ZOOM_FACTOR
            };
            if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                let old_scale = state.view.scale;
                let new_scale = (old_scale * factor).clamp(MIN_SCALE, MAX_SCALE);
                let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);
                state.view.scale = new_scale;
                let new_screen = state.view.world_to_screen(wx, wy);
                state.view.offset[0] += pointer_pos.x - new_screen[0];
                state.view.offset[1] += pointer_pos.y - new_screen[1];
            }
        }
    }

    // ── Interaction: Escape cancels active tool ─────────────────────────
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.creating_joint = None;
        state.draw_link_start = None;
        state.add_body_state = None;
        state.active_tool = EditorTool::Select;
    }

    // ── Interaction: Draw Link tool ─────────────────────────────────────
    // Click-and-drag creates a bar. If the start or end point lands on an
    // existing attachment point, the body endpoint snaps to its EXACT world
    // position and a revolute joint is auto-created. If on empty space, a
    // new ground pivot is created first.
    if state.active_tool == EditorTool::DrawLink {
        use crate::gui::state::DrawLinkStart;

        // Drag start: record start point, snapping to existing point if near one.
        if response.drag_started_by(egui::PointerButton::Primary) && !is_shift {
            if let Some(pos) = response.interact_pointer_pos() {
                let snap_hit = find_nearest_attachment(pos);
                if let Some(hit) = snap_hit {
                    // Snap to existing attachment point.
                    state.draw_link_start = Some(DrawLinkStart {
                        world_pos: hit.world_pos,
                        attachment: Some((hit.body_id.clone(), hit.point_name.clone())),
                    });
                } else if let Some(seg_hit) = find_nearest_body_segment(pos, &body_segments, 8.0) {
                    // Snap to body segment — create new pivot on that body.
                    let name = state.next_attachment_point_name(&seg_hit.body_id);
                    let [lx, ly] = state.world_to_body_local(&seg_hit.body_id, seg_hit.world_pos[0], seg_hit.world_pos[1]);
                    state.add_attachment_point_local_raw(&seg_hit.body_id, &name, lx, ly);
                    state.draw_link_start = Some(DrawLinkStart {
                        world_pos: seg_hit.world_pos,
                        attachment: Some((seg_hit.body_id.clone(), name)),
                    });
                }
                // If clicking on empty space, do NOT start — user must use +Ground tool first.
            }
        }

        // Preview line while dragging — snap end to existing points or body segments.
        if let Some(ref start) = state.draw_link_start {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let [sx, sy] = start.world_pos;
                let snap_end = find_nearest_attachment(pos);
                let (ex, ey, end_snapped) = if let Some(hit) = snap_end {
                    (hit.world_pos[0], hit.world_pos[1], true)
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    let (gx, gy) = state.grid.snap_point(wx, wy);
                    (gx, gy, false)
                };

                // Check for segment snap when no point snap is active.
                let segment_snap = if !end_snapped {
                    find_nearest_body_segment(pos, &body_segments, 8.0)
                } else {
                    None
                };

                let start_screen = state.view.world_to_screen(sx, sy);
                let end_screen = state.view.world_to_screen(ex, ey);

                let end_color = if end_snapped || segment_snap.is_some() {
                    JOINT_CREATE_HIGHLIGHT
                } else {
                    BODY_COLOR
                };

                painter.line_segment(
                    [
                        Pos2::new(start_screen[0], start_screen[1]),
                        Pos2::new(end_screen[0], end_screen[1]),
                    ],
                    Stroke::new(BODY_STROKE_WIDTH, JOINT_CREATE_HIGHLIGHT),
                );
                painter.circle_filled(
                    Pos2::new(start_screen[0], start_screen[1]),
                    JOINT_RADIUS,
                    JOINT_CREATE_HIGHLIGHT,
                );
                painter.circle_filled(
                    Pos2::new(end_screen[0], end_screen[1]),
                    JOINT_RADIUS,
                    end_color,
                );

                // Diamond indicator for segment snap.
                if let Some(ref seg_hit) = segment_snap {
                    let center = seg_hit.screen_pos;
                    let size = 5.0_f32;
                    let diamond = vec![
                        Pos2::new(center.x, center.y - size),
                        Pos2::new(center.x + size, center.y),
                        Pos2::new(center.x, center.y + size),
                        Pos2::new(center.x - size, center.y),
                    ];
                    painter.add(egui::Shape::convex_polygon(
                        diamond,
                        JOINT_CREATE_HIGHLIGHT,
                        Stroke::NONE,
                    ));
                }
            }
        }

        // Drag released → create body + auto-joints with exact snap positions.
        if state.draw_link_start.is_some() && response.drag_stopped() {
            let start = state.draw_link_start.take().unwrap();
            if let Some(pos) = response.interact_pointer_pos() {
                let [sx, sy] = start.world_pos;

                // Snap end to existing point, body segment, or grid.
                let snap_end = find_nearest_attachment(pos);
                let (ex, ey, end_attach) = if let Some(hit) = snap_end {
                    // Priority 1: snap to existing attachment point.
                    (hit.world_pos[0], hit.world_pos[1],
                     Some((hit.body_id.clone(), hit.point_name.clone())))
                } else if let Some(seg_hit) = find_nearest_body_segment(pos, &body_segments, 8.0) {
                    // Priority 2: snap to body segment — create new pivot.
                    let name = state.next_attachment_point_name(&seg_hit.body_id);
                    let [lx, ly] = state.world_to_body_local(&seg_hit.body_id, seg_hit.world_pos[0], seg_hit.world_pos[1]);
                    state.add_attachment_point_local_raw(&seg_hit.body_id, &name, lx, ly);
                    (seg_hit.world_pos[0], seg_hit.world_pos[1],
                     Some((seg_hit.body_id.clone(), name)))
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    let (gx, gy) = state.grid.snap_point(wx, wy);
                    (gx, gy, None)
                };

                // Minimum drag distance: 10px.
                let dist_px = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt()
                    * state.view.scale as f64;

                if dist_px > 10.0 {
                    // Single undo snapshot for the entire compound operation.
                    state.push_undo();

                    // Start connection must be an existing point (no auto-ground).
                    let start_attach = match start.attachment.clone() {
                        Some(a) => a,
                        None => {
                            // Should not happen since we guard on drag start,
                            // but be safe.
                            return;
                        }
                    };

                    // Create the body with endpoints at exact snap positions.
                    let body_id = state.next_body_id();
                    let points = vec![
                        ("A".to_string(), [sx, sy]),
                        ("B".to_string(), [ex, ey]),
                    ];
                    state.add_body_with_points_raw(&body_id, &points);

                    // Joint at start.
                    state.add_revolute_joint_raw(
                        &start_attach.0,
                        &start_attach.1,
                        &body_id,
                        "A",
                    );

                    // Joint at end (only if snapped to existing point).
                    if let Some((end_body, end_point)) = end_attach {
                        state.add_revolute_joint_raw(
                            &end_body,
                            &end_point,
                            &body_id,
                            "B",
                        );
                    }

                    // Single rebuild after all mutations.
                    state.rebuild();
                }
            }
            // Stay in DrawLink tool for chaining.
        }
    }

    // ── Interaction: Add Body tool ──────────────────────────────────────
    if state.active_tool == EditorTool::AddBody {
        // Helper: check if placed points are ready to finalize (>= 2 points).
        let can_finish = state
            .add_body_state
            .as_ref()
            .map_or(false, |abs| abs.points.len() >= 2);

        // Enter key finishes the body.
        if ui.input(|i| i.key_pressed(egui::Key::Enter)) && can_finish {
            if let Some(abs) = state.add_body_state.take() {
                state.add_body_with_points(&abs.points);
            }
        }

        // Double-click finishes (same as Enter -- do NOT place a new point).
        // Guard with state.add_body_state.is_some() in case Enter already consumed it
        // on the same frame.
        if response.double_clicked() {
            if can_finish {
                if let Some(abs) = state.add_body_state.take() {
                    state.add_body_with_points(&abs.points);
                }
            }
        } else if response.clicked() {
            // Single click: place a point.
            if let Some(pos) = response.interact_pointer_pos() {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                let (sx, sy) = state.grid.snap_point(wx, wy);

                if let Some(ref mut abs) = state.add_body_state {
                    let n = abs.points.len();
                    let name = if n < 26 {
                        String::from((b'A' + n as u8) as char)
                    } else {
                        let hi = (n - 26) / 26;
                        let lo = (n - 26) % 26;
                        format!(
                            "{}{}",
                            (b'A' + hi as u8) as char,
                            (b'A' + lo as u8) as char
                        )
                    };
                    abs.points.push((name, [sx, sy]));
                } else {
                    state.add_body_state = Some(AddBodyState {
                        points: vec![("A".to_string(), [sx, sy])],
                    });
                }
            }
        }
    }

    // ── Interaction: Create Joint two-click flow ────────────────────────
    // Must fire BEFORE the selection handler to consume the click.
    if state.creating_joint.is_some() && response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let second_hit = find_nearest_attachment(pos);
            if let Some(hit) = second_hit {
                let (first_body, first_point, joint_type) = state.creating_joint.clone().unwrap();
                let second_body = hit.body_id.clone();
                let second_point = hit.point_name.clone();

                // Validate: not same body, not ground-ground
                if first_body == second_body {
                    // Invalid: same body — ignore, stay in creating_joint mode
                } else if first_body == GROUND_ID && second_body == GROUND_ID {
                    // Invalid: ground-ground — ignore, stay in creating_joint mode
                } else {
                    use super::state::PendingJointType;
                    match joint_type {
                        PendingJointType::Revolute => {
                            state.add_revolute_joint(
                                &first_body, &first_point,
                                &second_body, &second_point,
                            );
                        }
                        PendingJointType::Prismatic => {
                            state.add_prismatic_joint(
                                &first_body, &first_point,
                                &second_body, &second_point,
                            );
                        }
                        PendingJointType::Fixed => {
                            state.add_fixed_joint(
                                &first_body, &first_point,
                                &second_body, &second_point,
                            );
                        }
                    }
                    state.creating_joint = None;
                }
            }
        }
    }

    // ── Interaction: click for selection / ground pivot ──────────────────
    if state.draw_link_start.is_none()
        && state.creating_joint.is_none()
        && state.active_tool != EditorTool::DrawLink
        && state.active_tool != EditorTool::AddBody
        && response.clicked()
    {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);

            match state.active_tool {
                EditorTool::AddGroundPivot => {
                    let (sx, sy) = state.grid.snap_point(wx, wy);
                    let name = state.next_ground_pivot_name();
                    state.add_ground_pivot(&name, sx, sy);
                    // Stay in AddGroundPivot tool for placing multiple pivots.
                }
                EditorTool::DrawLink => {
                    // Handled by drag section above.
                }
                EditorTool::AddBody => {
                    // Handled by Add Body interaction section above.
                }
                EditorTool::Select => {
                    let mut hit: Option<SelectedEntity> = None;

                    for (joint_screen, joint_id) in &joint_hit_targets {
                        if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                            hit = Some(SelectedEntity::Joint(joint_id.clone()));
                            break;
                        }
                    }

                    if hit.is_none() {
                        for ah in &attachment_hit_targets {
                            if pointer_pos.distance(ah.screen_pos) <= HIT_RADIUS {
                                hit = Some(SelectedEntity::Body(ah.body_id.clone()));
                                break;
                            }
                        }
                    }

                    state.selected = hit;
                }
            }
        }
    }

    // ── Interaction: right-click context menu ────────────────────────────
    // On the frame the right-click occurs, capture what was under the cursor
    // and store it in AppState. The context_menu closure runs every frame
    // while the popup is open, so it reads from stored state.
    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let joint_id = joint_hit_targets
                .iter()
                .find(|(screen_pos, _)| pos.distance(*screen_pos) <= HIT_RADIUS)
                .map(|(_, id)| id.clone());

            // Attachment points first (priority over body area)
            let attachment_point = attachment_hit_targets
                .iter()
                .find(|h| h.body_id != GROUND_ID && pos.distance(h.screen_pos) <= HIT_RADIUS)
                .map(|h| (h.body_id.clone(), h.point_name.clone()));

            // Body area: only if no attachment point matched
            let body_area = if attachment_point.is_none() {
                find_nearest_body_segment(pos, &body_segments, HIT_RADIUS)
                    .map(|hit| hit.body_id)
            } else {
                None
            };

            let world_pos = Some(state.view.screen_to_world(pos.x, pos.y));

            state.context_menu_target = ContextMenuTarget {
                joint_id,
                attachment_point,
                body_area,
                world_pos,
            };
        }
    }

    // Show context menu using egui's built-in context_menu.
    // Reads from state.context_menu_target which persists across frames.
    let ctx_target = state.context_menu_target.clone();
    response.context_menu(|ui| {
        if let Some(ref joint_id) = ctx_target.joint_id {
            // ── Joint context menu ──────────────────────────────────────
            ui.label(format!("Joint: {}", joint_id));
            ui.separator();

            let is_grounded_revolute = grounded_revolute_ids.contains(joint_id);
            let is_current_driver =
                current_driver_joint.as_deref() == Some(joint_id.as_str());

            if is_grounded_revolute {
                let label = if is_current_driver {
                    "Set as Driver (current)"
                } else {
                    "Set as Driver"
                };
                if ui
                    .add_enabled(!is_current_driver, egui::Button::new(label))
                    .clicked()
                {
                    state.pending_driver_reassignment = Some(joint_id.clone());
                    ui.close();
                }
            }

            if ui.button("Delete Joint").clicked() {
                state.remove_joint(joint_id);
                ui.close();
            }
        } else if let Some((ref body_id, ref point_name)) = ctx_target.attachment_point {
            // ── Attachment point context menu ────────────────────────────
            ui.label(format!("Point: {}.{}", body_id, point_name));
            ui.separator();

            ui.menu_button("Create Joint", |ui| {
                use super::state::PendingJointType;
                if ui.button("Revolute").clicked() {
                    state.creating_joint = Some((body_id.clone(), point_name.clone(), PendingJointType::Revolute));
                    ui.close();
                }
                if ui.button("Prismatic").clicked() {
                    state.creating_joint = Some((body_id.clone(), point_name.clone(), PendingJointType::Prismatic));
                    ui.close();
                }
                if ui.button("Fixed").clicked() {
                    state.creating_joint = Some((body_id.clone(), point_name.clone(), PendingJointType::Fixed));
                    ui.close();
                }
            });

            if ui.button("Delete Pivot").clicked() {
                state.remove_attachment_point(body_id, point_name);
                ui.close();
            }

            // Set as Driver: only if this body belongs to a grounded revolute joint.
            if let Some(mech) = &state.mechanism {
                let grounded = mech.grounded_revolute_joint_ids();
                for joint in mech.joints() {
                    if joint.is_revolute()
                        && grounded.contains(&joint.id().to_string())
                        && ((joint.body_i_id() == *body_id) || (joint.body_j_id() == *body_id))
                        && current_driver_joint.as_deref() != Some(joint.id())
                    {
                        if ui.button("Set as Driver").clicked() {
                            state.pending_driver_reassignment = Some(joint.id().to_string());
                            ui.close();
                        }
                        break;
                    }
                }
            }
        } else if let Some(ref body_id) = ctx_target.body_area {
            // ── Body area context menu ──────────────────────────────────
            ui.label(format!("Body: {}", body_id));
            ui.separator();

            if let Some([wx, wy]) = ctx_target.world_pos {
                if ui.button("Add Pivot Here").clicked() {
                    let name = state.next_attachment_point_name(body_id);
                    let (sx, sy) = state.grid.snap_point(wx, wy);
                    state.add_attachment_point_to_body(body_id, &name, sx, sy);
                    ui.close();
                }
            }

            if ui.button("Delete Body").clicked() {
                state.remove_body(body_id);
                state.selected = None;
                ui.close();
            }
        } else {
            // ── Empty canvas context menu ───────────────────────────────
            if let Some([wx, wy]) = ctx_target.world_pos {
                if ui.button("Add Ground Pivot Here").clicked() {
                    let name = state.next_ground_pivot_name();
                    state.add_ground_pivot(&name, wx, wy);
                    ui.close();
                }

                if ui.button("Draw Link").clicked() {
                    state.active_tool = EditorTool::DrawLink;
                    ui.close();
                }

                if ui.button("Start Body Here").clicked() {
                    let (sx, sy) = state.grid.snap_point(wx, wy);
                    let name = state.next_attachment_point_name("__pending__");
                    state.active_tool = EditorTool::AddBody;
                    state.add_body_state = Some(AddBodyState {
                        points: vec![(name, [sx, sy])],
                    });
                    ui.close();
                }
            }
        }
    });
}

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
                // Draw a small label at the midpoint between the two bodies.
                let (xi, yi, _) = mech_state.get_pose(&s.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&s.body_j, q);
                let mid_x = (xi + xj) / 2.0;
                let mid_y = (yi + yj) / 2.0;
                let sp = view.world_to_screen(mid_x, mid_y);
                let center = Pos2::new(sp[0], sp[1]);
                painter.text(
                    Pos2::new(center.x, center.y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("k={:.1}", s.stiffness),
                    FontId::proportional(9.0),
                    SPRING_COLOR,
                );
                draw_torque_arc(painter, center, 1.0, SPRING_COLOR);
            }
            ForceElement::RotaryDamper(d) => {
                // Draw a small label at the midpoint between the two bodies.
                let (xi, yi, _) = mech_state.get_pose(&d.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&d.body_j, q);
                let mid_x = (xi + xj) / 2.0;
                let mid_y = (yi + yj) / 2.0;
                let sp = view.world_to_screen(mid_x, mid_y);
                let center = Pos2::new(sp[0], sp[1]);
                painter.text(
                    Pos2::new(center.x, center.y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("c={:.1}", d.damping),
                    FontId::proportional(9.0),
                    DAMPER_COLOR,
                );
                draw_torque_arc(painter, center, 1.0, DAMPER_COLOR);
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
                        FontId::proportional(9.0),
                        ACTUATOR_COLOR,
                    );
                }
            }
            ForceElement::BearingFriction(bf) => {
                let (xi, yi, _) = mech_state.get_pose(&bf.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&bf.body_j, q);
                let mid_x = (xi + xj) / 2.0;
                let mid_y = (yi + yj) / 2.0;
                let sp = view.world_to_screen(mid_x, mid_y);
                let center = Pos2::new(sp[0], sp[1]);
                painter.text(
                    Pos2::new(center.x, center.y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    "Brg",
                    FontId::proportional(9.0),
                    BEARING_COLOR,
                );
                draw_torque_arc(painter, center, 1.0, BEARING_COLOR);
            }
            ForceElement::JointLimit(jl) => {
                let (xi, yi, _) = mech_state.get_pose(&jl.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&jl.body_j, q);
                let mid_x = (xi + xj) / 2.0;
                let mid_y = (yi + yj) / 2.0;
                let sp = view.world_to_screen(mid_x, mid_y);
                let center = Pos2::new(sp[0], sp[1]);
                painter.text(
                    Pos2::new(center.x, center.y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("[{:.1},{:.1}]", jl.angle_min, jl.angle_max),
                    FontId::proportional(9.0),
                    JOINT_LIMIT_COLOR,
                );
                draw_torque_arc(painter, center, 1.0, JOINT_LIMIT_COLOR);
            }
            ForceElement::Motor(m) => {
                let (xi, yi, _) = mech_state.get_pose(&m.body_i, q);
                let (xj, yj, _) = mech_state.get_pose(&m.body_j, q);
                let mid_x = (xi + xj) / 2.0;
                let mid_y = (yi + yj) / 2.0;
                let sp = view.world_to_screen(mid_x, mid_y);
                let center = Pos2::new(sp[0], sp[1]);
                painter.text(
                    Pos2::new(center.x, center.y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("M {:.1}Nm", m.stall_torque),
                    FontId::proportional(9.0),
                    MOTOR_COLOR,
                );
                draw_torque_arc(painter, center, m.direction as f32, MOTOR_COLOR);
            }
        }
    }
}

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
        FontId::proportional(9.0),
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
        FontId::proportional(9.0),
        FORCE_ARROW_COLOR,
    );
}

/// Draw a ground-fixed marker: an inverted triangle with hatch lines below.
fn draw_ground_marker(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
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

/// Draw a grid of lines on the canvas behind the mechanism.
///
/// Lines are drawn at multiples of `grid.spacing_m` in world coordinates.
/// If the viewport is zoomed out so far that more than 200 lines would be
/// drawn, the grid is suppressed to avoid visual clutter and performance cost.
fn draw_grid(
    painter: &egui::Painter,
    rect: Rect,
    view: &ViewTransform,
    grid: &GridSettings,
) {
    if !grid.show_grid {
        return;
    }

    let spacing = grid.spacing_m;
    if spacing <= 0.0 {
        return;
    }

    // Convert screen bounds to world coordinates.
    let [world_left, world_top] = view.screen_to_world(rect.left(), rect.top());
    let [world_right, world_bottom] = view.screen_to_world(rect.right(), rect.bottom());

    // world_top > world_bottom because screen Y is flipped.
    let x_min_i = (world_left / spacing).floor() as i64;
    let x_max_i = (world_right / spacing).ceil() as i64;
    let y_min_i = (world_bottom / spacing).floor() as i64;
    let y_max_i = (world_top / spacing).ceil() as i64;

    // Bail out if too many lines (zoomed out too far for this spacing).
    let n_lines = (x_max_i - x_min_i) + (y_max_i - y_min_i);
    if n_lines > 200 {
        return;
    }

    let minor_stroke = Stroke::new(0.5, GRID_COLOR);
    let major_stroke = Stroke::new(1.0, GRID_MAJOR_COLOR);

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
