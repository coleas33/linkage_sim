//! 2D mechanism canvas: rendering, pan/zoom, hit testing, drag, context menus.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::gui::state::{
    AppState, ContextMenuTarget, DragTarget, EditorTool, GridSettings, SelectedEntity,
    ViewTransform,
};

// ── Colors ──────────────────────────────────────────────────────────────────

const BG_COLOR: Color32 = Color32::from_rgb(18, 20, 28);
const GRID_COLOR: Color32 = Color32::from_rgba_premultiplied(50, 55, 70, 60);
const GROUND_LINE_COLOR: Color32 = Color32::from_rgb(70, 75, 90);
const BODY_COLOR: Color32 = Color32::from_rgb(65, 160, 255);
const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 185, 40);
const JOINT_COLOR: Color32 = Color32::from_rgb(230, 235, 245);
const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 185, 40);
const DRIVER_JOINT_COLOR: Color32 = Color32::from_rgb(100, 220, 140);
const GROUND_MARKER_COLOR: Color32 = Color32::from_rgb(170, 155, 120);
const ATTACHMENT_DOT_COLOR: Color32 = Color32::from_rgb(180, 195, 220);
const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(160, 170, 190);
const DEBUG_DIM_COLOR: Color32 = Color32::from_rgb(90, 95, 110);
const NO_MECH_TEXT_COLOR: Color32 = Color32::from_rgb(100, 105, 120);
const JOINT_CREATE_HIGHLIGHT: Color32 = Color32::from_rgb(60, 230, 100);
const FORCE_ARROW_COLOR: Color32 = Color32::from_rgb(255, 85, 85);
const SPRING_COLOR: Color32 = Color32::from_rgb(60, 200, 120);
const DAMPER_COLOR: Color32 = Color32::from_rgb(100, 150, 255);
const EXT_FORCE_COLOR: Color32 = Color32::from_rgb(255, 165, 0);

// ── Sizing ──────────────────────────────────────────────────────────────────

const FORCE_ARROW_WIDTH: f32 = 2.0;
/// Minimum arrow length in pixels (below this, arrows are not drawn).
const FORCE_ARROW_MIN_PX: f32 = 3.0;
/// Maximum arrow length in pixels (clamp very large forces).
const FORCE_ARROW_MAX_PX: f32 = 80.0;
/// Scale factor: pixels per Newton. Adjustable; 1 N = 30 px is a reasonable default.
const FORCE_ARROW_SCALE: f32 = 30.0;

const BODY_STROKE_WIDTH: f32 = 3.5;
const JOINT_RADIUS: f32 = 6.0;
const JOINT_STROKE_WIDTH: f32 = 2.0;
const GROUND_MARKER_SIZE: f32 = 12.0;
const HIT_RADIUS: f32 = 12.0;
const ATTACHMENT_DOT_RADIUS: f32 = 3.0;
const ZOOM_FACTOR: f32 = 1.04;
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

        // ── Ground line (y=0) ───────────────────────────────────────────
        {
            let left = view.world_to_screen(-10.0, 0.0);
            let right = view.world_to_screen(10.0, 0.0);
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

                for pair in screen_pts.windows(2) {
                    painter.line_segment([pair[0], pair[1]], Stroke::new(1.5, color));
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

            // Draw lines between consecutive attachment points.
            if screen_points.len() >= 2 {
                for pair in screen_points.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        Stroke::new(BODY_STROKE_WIDTH, color),
                    );
                }
            } else if screen_points.len() == 1 {
                painter.circle_filled(screen_points[0], 4.0, color);
            }

            // Draw small dots at each attachment point for visual clarity.
            for sp in &screen_points {
                painter.circle_filled(*sp, ATTACHMENT_DOT_RADIUS, ATTACHMENT_DOT_COLOR);
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
                // Filled circle with contrasting stroke for better visibility.
                painter.circle_filled(center, JOINT_RADIUS, BG_COLOR);
                painter.circle_stroke(
                    center,
                    JOINT_RADIUS,
                    Stroke::new(JOINT_STROKE_WIDTH, color),
                );
                // Inner dot for driver joint to make it extra visible.
                if is_driver {
                    painter.circle_filled(center, 2.5, DRIVER_JOINT_COLOR);
                }
            } else if joint.is_prismatic() {
                let half = JOINT_RADIUS;
                let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
                painter.rect_filled(rect, 0.0, BG_COLOR);
                painter.rect_stroke(
                    rect,
                    0.0,
                    Stroke::new(JOINT_STROKE_WIDTH, color),
                    egui::StrokeKind::Middle,
                );
            } else if joint.is_fixed() {
                painter.circle_filled(center, JOINT_RADIUS, color);
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
        if let Some((ref cj_body, ref cj_point)) = creating_joint_first {
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

    // ── Gravity indicator ─────────────────────────────────────────────
    if state.enable_gravity {
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
                Some("Click a point or empty space, then drag to draw a link (Esc to cancel)")
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

    // Drag start (Select mode): attachment drag or pan.
    if response.drag_started_by(egui::PointerButton::Primary)
        && !is_shift
        && state.active_tool == EditorTool::Select
    {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            if let Some(hit) = find_nearest_attachment(pointer_pos) {
                state.drag_target = Some(DragTarget {
                    body_id: hit.body_id.clone(),
                    point_name: hit.point_name.clone(),
                    started: false,
                });
            }
        }
    }

    // Drag in progress (Select mode): move point or pan.
    if response.dragged_by(egui::PointerButton::Primary) && !is_shift {
        if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
            let drag_info = state.drag_target.as_ref().map(|d| {
                (d.body_id.clone(), d.point_name.clone(), d.started)
            });

            if let Some((body_id, point_name, started)) = drag_info {
                if !started {
                    state.push_undo();
                    if let Some(ref mut drag) = state.drag_target {
                        drag.started = true;
                    }
                }
                let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);
                let (snap_x, snap_y) = state.grid.snap_point(wx, wy);
                state.move_attachment_point(&body_id, &point_name, snap_x, snap_y);
            } else if state.active_tool == EditorTool::Select
                && state.draw_link_start.is_none()
            {
                is_panning = true;
            }
        }
    }

    if state.drag_target.is_some() && !response.dragged() {
        state.drag_target = None;
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
                let (world_pos, attachment) = if let Some(hit) = snap_hit {
                    // Snap to exact world position of existing point.
                    (hit.world_pos, Some((hit.body_id.clone(), hit.point_name.clone())))
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    let (sx, sy) = state.grid.snap_point(wx, wy);
                    ([sx, sy], None)
                };

                state.draw_link_start = Some(DrawLinkStart {
                    world_pos,
                    attachment,
                });
            }
        }

        // Preview line while dragging — snap end to existing points.
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

                let start_screen = state.view.world_to_screen(sx, sy);
                let end_screen = state.view.world_to_screen(ex, ey);

                let end_color = if end_snapped {
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
            }
        }

        // Drag released → create body + auto-joints with exact snap positions.
        if state.draw_link_start.is_some() && response.drag_stopped() {
            let start = state.draw_link_start.take().unwrap();
            if let Some(pos) = response.interact_pointer_pos() {
                let [sx, sy] = start.world_pos;

                // Snap end to existing point or grid.
                let snap_end = find_nearest_attachment(pos);
                let (ex, ey, end_attach) = if let Some(hit) = snap_end {
                    (hit.world_pos[0], hit.world_pos[1],
                     Some((hit.body_id.clone(), hit.point_name.clone())))
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    let (gx, gy) = state.grid.snap_point(wx, wy);
                    (gx, gy, None)
                };

                // Minimum drag distance: 10px.
                let dist_px = ((ex - sx).powi(2) + (ey - sy).powi(2)).sqrt()
                    * state.view.scale as f64;

                if dist_px > 10.0 {
                    // Start connection: existing point or new ground pivot.
                    let start_attach = start.attachment.clone().unwrap_or_else(|| {
                        let name = state.next_ground_pivot_name();
                        state.add_ground_pivot(&name, sx, sy);
                        (GROUND_ID.to_string(), name)
                    });

                    // Create the body with endpoints at exact snap positions.
                    let body_id = state.next_body_id();
                    state.add_body(&body_id, ("A", sx, sy), ("B", ex, ey));

                    // Joint at start.
                    state.add_revolute_joint(
                        &start_attach.0,
                        &start_attach.1,
                        &body_id,
                        "A",
                    );

                    // Joint at end (only if snapped to existing point).
                    if let Some((end_body, end_point)) = end_attach {
                        state.add_revolute_joint(
                            &end_body,
                            &end_point,
                            &body_id,
                            "B",
                        );
                    }
                }
            }
            // Stay in DrawLink tool for chaining.
        }
    }

    // ── Interaction: click for selection / ground pivot ──────────────────
    if state.drag_target.is_none()
        && state.draw_link_start.is_none()
        && state.active_tool != EditorTool::DrawLink
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

            let body = attachment_hit_targets
                .iter()
                .find(|h| h.body_id != GROUND_ID && pos.distance(h.screen_pos) <= HIT_RADIUS)
                .map(|h| (h.body_id.clone(), h.point_name.clone()));

            let world_pos = Some(state.view.screen_to_world(pos.x, pos.y));

            state.context_menu_target = ContextMenuTarget {
                joint_id,
                body,
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
        } else if let Some((ref body_id, ref _point_name)) = ctx_target.body {
            // ── Body context menu ───────────────────────────────────────
            ui.label(format!("Body: {}", body_id));
            ui.separator();

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

    let grid_stroke = Stroke::new(0.5, GRID_COLOR);

    // Vertical lines.
    for i in x_min_i..=x_max_i {
        let wx = i as f64 * spacing;
        let top = view.world_to_screen(wx, world_top);
        let bottom = view.world_to_screen(wx, world_bottom);
        painter.line_segment(
            [
                Pos2::new(top[0], top[1]),
                Pos2::new(bottom[0], bottom[1]),
            ],
            grid_stroke,
        );
    }

    // Horizontal lines.
    for i in y_min_i..=y_max_i {
        let wy = i as f64 * spacing;
        let left = view.world_to_screen(world_left, wy);
        let right = view.world_to_screen(world_right, wy);
        painter.line_segment(
            [
                Pos2::new(left[0], left[1]),
                Pos2::new(right[0], right[1]),
            ],
            grid_stroke,
        );
    }
}
