//! Canvas interaction: drag, pan, zoom, keyboard shortcuts, tool modes.

use eframe::egui::{self, Color32, FontId, Pos2, Stroke};

use crate::core::state::GROUND_ID;
use crate::forces::elements::*;
use crate::gui::state::{
    AddBodyState, AppState, EditorTool, ForceZoneDragState, SelectedEntity,
};

use super::alignment::compute_alignment_guides;
use super::colors::*;
use super::hit_testing::{find_nearest_body_segment, AttachmentHit, BodySegment};
use super::rendering::{draw_dashed_line, fill_force_template};

/// Find the nearest attachment point within hit radius of a screen position.
fn find_nearest_attachment<'a>(
    pos: Pos2,
    attachment_hit_targets: &'a [AttachmentHit],
) -> Option<&'a AttachmentHit> {
    attachment_hit_targets
        .iter()
        .filter(|h| pos.distance(h.screen_pos) <= HIT_RADIUS)
        .min_by(|a, b| {
            pos.distance(a.screen_pos)
                .partial_cmp(&pos.distance(b.screen_pos))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Handle all canvas interaction: drag, pan, zoom, keyboard shortcuts, tool modes,
/// and click-to-select. Returns whether a right-drag just ended (used to suppress
/// context menu).
pub fn handle_interaction(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    canvas_rect: egui::Rect,
    response: &egui::Response,
    state: &mut AppState,
    joint_hit_targets: &[(Pos2, String)],
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
) -> bool {
    let is_shift = ui.input(|i| i.modifiers.shift);
    let mut is_panning = false;

    // Clear alignment guides when no drag is active.
    if state.dragging_ground_pivot.is_none() {
        state.alignment_guides.clear();
    }

    // ── Interaction: ground pivot drag ─────────────────────────────────
    // Start drag when pointer is near a ground attachment point in Select mode.
    if state.active_tool == EditorTool::Select
        && response.drag_started_by(egui::PointerButton::Primary)
        && !is_shift
    {
        if let Some(pos) = response.interact_pointer_pos() {
            // Only consider ground body attachment points.
            let ground_hit = attachment_hit_targets
                .iter()
                .filter(|h| h.body_id == GROUND_ID)
                .filter(|h| pos.distance(h.screen_pos) <= HIT_RADIUS)
                .min_by(|a, b| {
                    pos.distance(a.screen_pos)
                        .partial_cmp(&pos.distance(b.screen_pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(hit) = ground_hit {
                state.dragging_ground_pivot =
                    Some((hit.point_name.clone(), hit.world_pos));
            }
        }
    }

    // During drag: draw a ghost marker at the cursor position with alignment snapping.
    if state.dragging_ground_pivot.is_some() {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (gx, gy) = state.grid.snap_point(wx, wy);

            // Compute alignment guides and snap to aligned points.
            let snap_threshold = 10.0 / state.view.scale as f64;
            let exclude_name = state.dragging_ground_pivot.as_ref().map(|(n, _)| n.as_str());
            let mut guides = Vec::new();
            let (sx, sy) = compute_alignment_guides(
                state, gx, gy, exclude_name, snap_threshold, &mut guides,
            );
            state.alignment_guides = guides;

            let ghost_screen = state.view.world_to_screen(sx, sy);
            let ghost_pos = Pos2::new(ghost_screen[0], ghost_screen[1]);
            let half = GROUND_MARKER_SIZE * 0.5;
            // Draw ghost X marker.
            let ghost_color = GROUND_MARKER_COLOR.linear_multiply(0.6);
            painter.line_segment(
                [
                    Pos2::new(ghost_pos.x - half, ghost_pos.y - half),
                    Pos2::new(ghost_pos.x + half, ghost_pos.y + half),
                ],
                Stroke::new(2.5, ghost_color),
            );
            painter.line_segment(
                [
                    Pos2::new(ghost_pos.x + half, ghost_pos.y - half),
                    Pos2::new(ghost_pos.x - half, ghost_pos.y + half),
                ],
                Stroke::new(2.5, ghost_color),
            );
        }
    }

    // On drag end: apply the ground pivot move and rebuild.
    if state.dragging_ground_pivot.is_some()
        && response.drag_stopped_by(egui::PointerButton::Primary)
    {
        if let Some((pivot_name, _start_pos)) = state.dragging_ground_pivot.take() {
            if let Some(pos) = response.interact_pointer_pos() {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                let (gx, gy) = state.grid.snap_point(wx, wy);

                // Apply alignment snapping to the final position.
                let snap_threshold = 10.0 / state.view.scale as f64;
                let mut guides = Vec::new();
                let (sx, sy) = compute_alignment_guides(
                    state, gx, gy, Some(&pivot_name), snap_threshold, &mut guides,
                );

                state.update_ground_pivot_position(&pivot_name, sx, sy);
            }
        }
        state.alignment_guides.clear();
    }

    let is_dragging_ground = state.dragging_ground_pivot.is_some();

    // Primary drag on empty space (Select mode) pans the view.
    // Suppress panning when dragging a ground pivot.
    if response.dragged_by(egui::PointerButton::Primary) && !is_shift && !is_dragging_ground {
        if state.active_tool == EditorTool::Select
            && state.draw_link_start.is_none()
        {
            is_panning = true;
        }
    }

    // ── Interaction: pan ────────────────────────────────────────────────
    // Right-drag also pans (in addition to middle-click and shift+primary).
    if response.dragged() {
        let is_middle = response.dragged_by(egui::PointerButton::Middle);
        let is_secondary = response.dragged_by(egui::PointerButton::Secondary);
        let is_shift_primary =
            is_shift && response.dragged_by(egui::PointerButton::Primary);
        if is_middle || is_secondary || is_shift_primary || is_panning {
            let delta = response.drag_delta();
            state.view.offset[0] += delta.x;
            state.view.offset[1] += delta.y;
        }
    }

    // Track whether a right-drag just ended this frame. If so, suppress
    // the context menu so that a right-drag-to-pan doesn't accidentally
    // open it on release.
    let right_drag_ended = response.drag_stopped_by(egui::PointerButton::Secondary);

    // ── Interaction: zoom toward mouse ──────────────────────────────────
    if response.hovered() {
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta.abs() > 0.0 {
            // Normalize: apply ZOOM_FACTOR once per ~50px of scroll delta
            // so trackpads and mouse wheels feel consistent.
            let ticks = (scroll_delta / 50.0).clamp(-3.0, 3.0);
            let factor = ZOOM_FACTOR.powf(ticks);
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

    // ── Interaction: Arrow key nudge for selected entity ────────────────
    if state.selected.is_some() && state.blueprint.is_some() {
        let shift = ui.input(|i| i.modifiers.shift);
        let base_step = state.grid.spacing_m;
        let step = if shift { base_step * 10.0 } else { base_step };

        let mut dx = 0.0_f64;
        let mut dy = 0.0_f64;

        if ui.input(|i| i.key_pressed(egui::Key::ArrowLeft))  { dx = -step; }
        if ui.input(|i| i.key_pressed(egui::Key::ArrowRight)) { dx = step; }
        if ui.input(|i| i.key_pressed(egui::Key::ArrowUp))    { dy = step; }
        if ui.input(|i| i.key_pressed(egui::Key::ArrowDown))  { dy = -step; }

        if dx != 0.0 || dy != 0.0 {
            match &state.selected.clone() {
                Some(SelectedEntity::Body(body_id)) => {
                    state.nudge_body(body_id, dx, dy);
                }
                Some(SelectedEntity::Joint(joint_id)) => {
                    state.nudge_joint(joint_id, dx, dy);
                }
                _ => {}
            }
        }
    }

    // ── Interaction: Fit to View (F key) ────────────────────────────────
    if ui.input(|i| i.key_pressed(egui::Key::F)) {
        state.fit_to_view(canvas_rect.width(), canvas_rect.height());
    }

    // ── Interaction: Escape cancels active tool ─────────────────────────
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.creating_joint = None;
        state.draw_link_start = None;
        state.add_body_state = None;
        state.place_force_state = None;
        state.creating_force_zone = None;
        state.dragging_ground_pivot = None;
        state.active_tool = EditorTool::Select;
    }

    // ── Interaction: Draw Link tool ─────────────────────────────────────
    if state.active_tool == EditorTool::DrawLink {
        handle_draw_link(ui, painter, response, state, is_shift,
                         attachment_hit_targets, body_segments);
    }

    // ── Interaction: Place Force tool ───────────────────────────────────
    if state.active_tool == EditorTool::PlaceForce {
        handle_place_force(ui, painter, response, state,
                           attachment_hit_targets, body_segments);
    }

    // ── Interaction: Create Force Zone tool ─────────────────────────────
    if state.active_tool == EditorTool::CreateForceZone {
        handle_create_force_zone(ui, painter, canvas_rect, response, state, is_shift);
    }

    // ── Interaction: Add Body tool ──────────────────────────────────────
    if state.active_tool == EditorTool::AddBody {
        handle_add_body(ui, response, state);
    }

    // ── Interaction: Create Joint two-click flow ────────────────────────
    if state.creating_joint.is_some() && response.clicked() {
        handle_create_joint(response, state, attachment_hit_targets);
    }

    // ── Interaction: click for selection / ground pivot ──────────────────
    if state.draw_link_start.is_none()
        && state.creating_joint.is_none()
        && state.active_tool != EditorTool::DrawLink
        && state.active_tool != EditorTool::AddBody
        && state.active_tool != EditorTool::PlaceForce
        && state.active_tool != EditorTool::CreateForceZone
        && response.clicked()
    {
        handle_click_selection(response, state, canvas_rect, joint_hit_targets, attachment_hit_targets);
    }

    right_drag_ended
}

// ── Draw Link tool ───────────────────────────────────────────────────────────

fn handle_draw_link(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &mut AppState,
    is_shift: bool,
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
) {
    use crate::gui::state::DrawLinkStart;

    // Drag start: record start point, snapping to existing point if near one.
    if response.drag_started_by(egui::PointerButton::Primary) && !is_shift {
        if let Some(pos) = response.interact_pointer_pos() {
            let snap_hit = find_nearest_attachment(pos, attachment_hit_targets);
            if let Some(hit) = snap_hit {
                // Snap to existing attachment point.
                state.draw_link_start = Some(DrawLinkStart {
                    world_pos: hit.world_pos,
                    attachment: Some((hit.body_id.clone(), hit.point_name.clone())),
                });
            } else if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, 8.0) {
                // Snap to body segment -- create new pivot on that body.
                let name = state.next_attachment_point_name(&seg_hit.body_id);
                let [lx, ly] = state.world_to_body_local(&seg_hit.body_id, seg_hit.world_pos[0], seg_hit.world_pos[1]);
                state.add_attachment_point_local_raw(&seg_hit.body_id, &name, lx, ly);
                state.draw_link_start = Some(DrawLinkStart {
                    world_pos: seg_hit.world_pos,
                    attachment: Some((seg_hit.body_id.clone(), name)),
                });
            }
            // If clicking on empty space, do NOT start -- user must use +Ground tool first.
        }
    }

    // Preview line while dragging -- snap end to existing points or body segments.
    if let Some(ref start) = state.draw_link_start {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            let [sx, sy] = start.world_pos;
            let snap_end = find_nearest_attachment(pos, attachment_hit_targets);
            let (ex, ey, end_snapped) = if let Some(hit) = snap_end {
                (hit.world_pos[0], hit.world_pos[1], true)
            } else {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                let (gx, gy) = state.grid.snap_point(wx, wy);
                (gx, gy, false)
            };

            // Check for segment snap when no point snap is active.
            let segment_snap = if !end_snapped {
                find_nearest_body_segment(pos, body_segments, 8.0)
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

            // Live readout: length and angle near cursor while dragging.
            let world_dx = ex - sx;
            let world_dy = ey - sy;
            let length_m = (world_dx * world_dx + world_dy * world_dy).sqrt();
            let angle_rad = world_dy.atan2(world_dx);
            let units = &state.display_units;
            let readout = format!(
                "{:.1}{}  {:.1}{}",
                units.length(length_m), units.length_suffix(),
                units.angle(angle_rad), units.angle_suffix()
            );
            painter.text(
                Pos2::new(end_screen[0] + 15.0, end_screen[1] - 15.0),
                egui::Align2::LEFT_BOTTOM,
                &readout,
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
    }

    // Drag released -> create body + auto-joints with exact snap positions.
    if state.draw_link_start.is_some() && response.drag_stopped() {
        let start = state.draw_link_start.take().unwrap();
        if let Some(pos) = response.interact_pointer_pos() {
            let [sx, sy] = start.world_pos;

            // Snap end to existing point, body segment, or grid.
            let snap_end = find_nearest_attachment(pos, attachment_hit_targets);
            let (ex, ey, end_attach) = if let Some(hit) = snap_end {
                // Priority 1: snap to existing attachment point.
                (hit.world_pos[0], hit.world_pos[1],
                 Some((hit.body_id.clone(), hit.point_name.clone())))
            } else if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, 8.0) {
                // Priority 2: snap to body segment -- create new pivot.
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

// ── Place Force tool ─────────────────────────────────────────────────────────

fn handle_place_force(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    response: &egui::Response,
    state: &mut AppState,
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
) {
    use crate::gui::state::PlaceForceStart;

    // Highlight all snap targets while in placement mode.
    for hit in attachment_hit_targets {
        painter.circle_stroke(
            hit.screen_pos,
            HIT_RADIUS,
            Stroke::new(1.0, JOINT_CREATE_HIGHLIGHT.linear_multiply(0.3)),
        );
    }

    // Preview line from start to cursor after first click.
    if let Some(ref pf_state) = state.place_force_state {
        if let Some(ref start) = pf_state.start {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let [sx, sy] = start.world_pos;
                let snap_end = find_nearest_attachment(pos, attachment_hit_targets);
                let (ex, ey, end_snapped) = if let Some(hit) = snap_end {
                    (hit.world_pos[0], hit.world_pos[1], true)
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    (wx, wy, false)
                };

                let start_screen = state.view.world_to_screen(sx, sy);
                let end_screen = state.view.world_to_screen(ex, ey);

                let force_preview_color = Color32::from_rgb(255, 165, 80);
                painter.line_segment(
                    [
                        Pos2::new(start_screen[0], start_screen[1]),
                        Pos2::new(end_screen[0], end_screen[1]),
                    ],
                    Stroke::new(2.0, force_preview_color),
                );
                painter.circle_filled(
                    Pos2::new(start_screen[0], start_screen[1]),
                    JOINT_RADIUS,
                    force_preview_color,
                );
                let end_color = if end_snapped {
                    JOINT_CREATE_HIGHLIGHT
                } else {
                    force_preview_color
                };
                painter.circle_filled(
                    Pos2::new(end_screen[0], end_screen[1]),
                    JOINT_RADIUS,
                    end_color,
                );
            }
        }
    }

    // Handle clicks.
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let snap_hit = find_nearest_attachment(pos, attachment_hit_targets);

            let (world_pos, body_id, point_name) = if let Some(hit) = snap_hit {
                (hit.world_pos, hit.body_id.clone(), Some(hit.point_name.clone()))
            } else {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                if let Some(seg_hit) = find_nearest_body_segment(pos, body_segments, 8.0) {
                    (seg_hit.world_pos, seg_hit.body_id.clone(), None)
                } else {
                    ([wx, wy], GROUND_ID.to_string(), None)
                }
            };

            let has_start = state.place_force_state.as_ref()
                .map(|s| s.start.is_some())
                .unwrap_or(false);

            if !has_start {
                // First click -- record point A.
                if let Some(ref mut pf_state) = state.place_force_state {
                    pf_state.start = Some(PlaceForceStart {
                        world_pos,
                        body_id,
                        point_name,
                    });
                }
            } else {
                // Second click -- create the force and exit tool.
                if let Some(pf_state) = state.place_force_state.take() {
                    let start = pf_state.start.unwrap();

                    let [la_x, la_y] = state.world_to_body_local(
                        &start.body_id, start.world_pos[0], start.world_pos[1],
                    );
                    let [lb_x, lb_y] = state.world_to_body_local(
                        &body_id, world_pos[0], world_pos[1],
                    );

                    let force = fill_force_template(
                        &pf_state.force_template,
                        &start.body_id, [la_x, la_y], start.point_name,
                        &body_id, [lb_x, lb_y], point_name,
                    );

                    // add_force_element() calls push_undo() internally.
                    state.add_force_element(force);
                    state.active_tool = EditorTool::Select;
                }
            }
        }
    }
}

// ── Create Force Zone tool ───────────────────────────────────────────────────

fn handle_create_force_zone(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    _canvas_rect: egui::Rect,
    response: &egui::Response,
    state: &mut AppState,
    is_shift: bool,
) {
    // On drag start: record the starting world position.
    if response.drag_started_by(egui::PointerButton::Primary) && !is_shift {
        if let Some(pos) = response.interact_pointer_pos() {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (gx, gy) = state.grid.snap_point(wx, wy);
            state.creating_force_zone = Some(ForceZoneDragState {
                start_world: [gx, gy],
            });
        }
    }

    // Preview rectangle while dragging.
    if let Some(ref fz_drag) = state.creating_force_zone {
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            let [sx, sy] = fz_drag.start_world;
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (gx, gy) = state.grid.snap_point(wx, wy);

            let start_sp = state.view.world_to_screen(sx, sy);
            let end_sp = state.view.world_to_screen(gx, gy);

            let s_min_x = start_sp[0].min(end_sp[0]);
            let s_min_y = start_sp[1].min(end_sp[1]);
            let s_max_x = start_sp[0].max(end_sp[0]);
            let s_max_y = start_sp[1].max(end_sp[1]);

            let tl = Pos2::new(s_min_x, s_min_y);
            let tr = Pos2::new(s_max_x, s_min_y);
            let br = Pos2::new(s_max_x, s_max_y);
            let bl = Pos2::new(s_min_x, s_max_y);

            // Faint red fill preview.
            let preview_fill = Color32::from_rgba_premultiplied(255, 80, 80, 20);
            painter.rect_filled(
                egui::Rect::from_min_max(tl, br),
                0.0,
                preview_fill,
            );

            // Dashed red border preview.
            let preview_stroke = Stroke::new(2.0, FORCE_ZONE_COLOR);
            let dash = 6.0_f32;
            let gap = 4.0_f32;
            draw_dashed_line(painter, tl, tr, preview_stroke, dash, gap);
            draw_dashed_line(painter, tr, br, preview_stroke, dash, gap);
            draw_dashed_line(painter, br, bl, preview_stroke, dash, gap);
            draw_dashed_line(painter, bl, tl, preview_stroke, dash, gap);
        }
    }

    // On drag release: create the force zone element.
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        if let Some(fz_drag) = state.creating_force_zone.take() {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                let (gx, gy) = state.grid.snap_point(wx, wy);

                let [sx, sy] = fz_drag.start_world;

                // Normalize so min < max.
                let zone_min = [sx.min(gx), sy.min(gy)];
                let zone_max = [sx.max(gx), sy.max(gy)];

                // Skip degenerate (zero-area) zones.
                let w = (zone_max[0] - zone_min[0]).abs();
                let h = (zone_max[1] - zone_min[1]).abs();
                if w > 1e-6 && h > 1e-6 {
                    // Pick the first body that has geometry, or fall back to empty string.
                    let target_body = state.blueprint.as_ref()
                        .and_then(|bp| {
                            bp.bodies.iter()
                                .find(|(id, b)| b.geometry.is_some() && id.as_str() != "ground")
                                .map(|(id, _)| id.clone())
                        })
                        .unwrap_or_default();

                    let fz = ForceElement::ForceZone(ForceZoneElement {
                        body_id: target_body,
                        zone_min,
                        zone_max,
                        force: [0.0, -100.0],
                        label: None,
                    });

                    state.add_force_element(fz);
                }

                state.active_tool = EditorTool::Select;
            }
        }
    }
}

// ── Add Body tool ────────────────────────────────────────────────────────────

fn handle_add_body(
    ui: &mut egui::Ui,
    response: &egui::Response,
    state: &mut AppState,
) {
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

// ── Create Joint two-click flow ──────────────────────────────────────────────

fn handle_create_joint(
    response: &egui::Response,
    state: &mut AppState,
    attachment_hit_targets: &[AttachmentHit],
) {
    if let Some(pos) = response.interact_pointer_pos() {
        let second_hit = find_nearest_attachment(pos, attachment_hit_targets);
        if let Some(hit) = second_hit {
            let (first_body, first_point, joint_type) = state.creating_joint.clone().unwrap();
            let second_body = hit.body_id.clone();
            let second_point = hit.point_name.clone();

            // Validate: not same body, not ground-ground
            if first_body == second_body {
                // Invalid: same body -- ignore, stay in creating_joint mode
            } else if first_body == GROUND_ID && second_body == GROUND_ID {
                // Invalid: ground-ground -- ignore, stay in creating_joint mode
            } else {
                use crate::gui::state::PendingJointType;
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

// ── Click selection / ground pivot ───────────────────────────────────────────

fn handle_click_selection(
    response: &egui::Response,
    state: &mut AppState,
    canvas_rect: egui::Rect,
    joint_hit_targets: &[(Pos2, String)],
    attachment_hit_targets: &[AttachmentHit],
) {
    if let Some(pointer_pos) = response.interact_pointer_pos() {
        let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);

        match state.active_tool {
            EditorTool::AddGroundPivot => {
                let (sx, sy) = state.grid.snap_point(wx, wy);
                let name = state.next_ground_pivot_name();
                state.add_ground_pivot(&name, sx, sy);
                // Stay in AddGroundPivot tool for placing multiple pivots.

                // Tutorial auto-zoom: after placing the first ground pivot,
                // zoom out so ~120mm is visible, giving room for the second.
                if state.tutorial.active && state.tutorial.step == 1 {
                    let count = state.blueprint.as_ref()
                        .and_then(|bp| bp.bodies.get("ground"))
                        .map(|g| g.attachment_points.len())
                        .unwrap_or(0);
                    if count == 1 {
                        // 0.12 m visible across the canvas width.
                        state.view.scale = canvas_rect.width() / 0.12;
                    }
                }
            }
            EditorTool::DrawLink => {
                // Handled by drag section above.
            }
            EditorTool::AddBody => {
                // Handled by Add Body interaction section above.
            }
            EditorTool::PlaceForce => {
                // Handled by PlaceForce interaction section above.
            }
            EditorTool::CreateForceZone => {
                // Handled by CreateForceZone drag interaction section above.
            }
            EditorTool::Select => {
                let mut hit: Option<SelectedEntity> = None;

                for (joint_screen, joint_id) in joint_hit_targets {
                    if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                        hit = Some(SelectedEntity::Joint(joint_id.clone()));
                        break;
                    }
                }

                if hit.is_none() {
                    for ah in attachment_hit_targets {
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
