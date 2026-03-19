//! 2D mechanism canvas: rendering, pan/zoom, hit testing, drag, context menus.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::gui::state::{AppState, DragTarget, GridSettings, SelectedEntity, ViewTransform};

// ── Colors ──────────────────────────────────────────────────────────────────

const BG_COLOR: Color32 = Color32::from_rgb(30, 30, 35);
const GRID_COLOR: Color32 = Color32::from_rgba_premultiplied(60, 60, 70, 40);
const GROUND_LINE_COLOR: Color32 = Color32::from_rgb(60, 60, 65);
const BODY_COLOR: Color32 = Color32::from_rgb(140, 180, 220);
const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 200, 80);
const JOINT_COLOR: Color32 = Color32::from_rgb(200, 200, 200);
const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 200, 80);
const GROUND_MARKER_COLOR: Color32 = Color32::from_rgb(120, 120, 100);
const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(180, 180, 180);
const DEBUG_DIM_COLOR: Color32 = Color32::from_rgb(100, 100, 100);
const NO_MECH_TEXT_COLOR: Color32 = Color32::from_rgb(120, 120, 120);
const JOINT_CREATE_HIGHLIGHT: Color32 = Color32::from_rgb(80, 255, 80);
const FORCE_ARROW_COLOR: Color32 = Color32::from_rgb(255, 80, 80);

// ── Sizing ──────────────────────────────────────────────────────────────────

const FORCE_ARROW_WIDTH: f32 = 2.0;
/// Minimum arrow length in pixels (below this, arrows are not drawn).
const FORCE_ARROW_MIN_PX: f32 = 3.0;
/// Maximum arrow length in pixels (clamp very large forces).
const FORCE_ARROW_MAX_PX: f32 = 80.0;
/// Scale factor: pixels per Newton. Adjustable; 1 N = 30 px is a reasonable default.
const FORCE_ARROW_SCALE: f32 = 30.0;

const BODY_STROKE_WIDTH: f32 = 3.0;
const JOINT_RADIUS: f32 = 5.0;
const JOINT_STROKE_WIDTH: f32 = 2.0;
const GROUND_MARKER_SIZE: f32 = 10.0;
const HIT_RADIUS: f32 = 10.0;
const ZOOM_FACTOR: f32 = 1.1;
const MIN_SCALE: f32 = 100.0;
const MAX_SCALE: f32 = 100_000.0;

/// An attachment point hit target: screen position, body ID, point name.
#[derive(Clone)]
struct AttachmentHit {
    screen_pos: Pos2,
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
                Color32::from_rgba_premultiplied(100, 200, 255, 60),
                Color32::from_rgba_premultiplied(255, 150, 80, 60),
                Color32::from_rgba_premultiplied(120, 220, 120, 60),
                Color32::from_rgba_premultiplied(255, 100, 100, 60),
                Color32::from_rgba_premultiplied(200, 150, 255, 60),
                Color32::from_rgba_premultiplied(255, 220, 100, 60),
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
                    painter.line_segment([pair[0], pair[1]], Stroke::new(1.0, color));
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

            let screen_points: Vec<Pos2> = point_names
                .iter()
                .map(|name| {
                    let local = &body.attachment_points[*name];
                    let global = mech_state.body_point_global(body_id, local, q);
                    let sp = view.world_to_screen(global.x, global.y);
                    Pos2::new(sp[0], sp[1])
                })
                .collect();

            // Draw lines between consecutive attachment points.
            if screen_points.len() >= 2 {
                for pair in screen_points.windows(2) {
                    painter.line_segment(
                        [pair[0], pair[1]],
                        Stroke::new(BODY_STROKE_WIDTH, color),
                    );
                }
            } else if screen_points.len() == 1 {
                // Single-point body: draw a small dot.
                painter.circle_filled(screen_points[0], 3.0, color);
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

            // Store attachment hit targets (screen positions of attachment points).
            for (i, sp) in screen_points.iter().enumerate() {
                attachment_hit_targets.push(AttachmentHit {
                    screen_pos: *sp,
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
            let color = if is_selected {
                JOINT_SELECTED_COLOR
            } else {
                JOINT_COLOR
            };

            if joint.is_revolute() {
                painter.circle_stroke(
                    center,
                    JOINT_RADIUS,
                    Stroke::new(JOINT_STROKE_WIDTH, color),
                );
            } else if joint.is_prismatic() {
                let half = JOINT_RADIUS;
                let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
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
                    Pos2::new(center.x, center.y - JOINT_RADIUS - 4.0),
                    egui::Align2::CENTER_BOTTOM,
                    joint.id(),
                    FontId::proportional(9.0),
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

    // ── Joint creation mode hint text ──────────────────────────────────
    if state.creating_joint.is_some() {
        painter.text(
            Pos2::new(canvas_rect.center().x, canvas_rect.top() + 20.0),
            egui::Align2::CENTER_TOP,
            "Click a second attachment point to create joint (Esc to cancel)",
            FontId::proportional(13.0),
            JOINT_CREATE_HIGHLIGHT,
        );
    }

    // ── Interaction: drag attachment points ─────────────────────────────
    // Primary drag (not shift, not middle) near an attachment point initiates
    // a drag operation. During drag, the attachment point tracks the mouse
    // position in world coords. Undo is pushed once at drag start.

    let is_shift = ui.input(|i| i.modifiers.shift);

    // Drag start: on primary button press near an attachment point.
    if response.drag_started_by(egui::PointerButton::Primary) && !is_shift {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            // Find nearest attachment point within hit radius.
            let nearest = attachment_hit_targets
                .iter()
                .filter(|h| pointer_pos.distance(h.screen_pos) <= HIT_RADIUS)
                .min_by(|a, b| {
                    pointer_pos
                        .distance(a.screen_pos)
                        .partial_cmp(&pointer_pos.distance(b.screen_pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(hit) = nearest {
                state.drag_target = Some(DragTarget {
                    body_id: hit.body_id.clone(),
                    point_name: hit.point_name.clone(),
                    started: false,
                });
            }
        }
    }

    // Drag in progress: update attachment point position.
    // Extract drag info first to avoid overlapping borrows with &mut self methods.
    if response.dragged_by(egui::PointerButton::Primary) && !is_shift {
        if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
            let drag_info = state.drag_target.as_ref().map(|d| {
                (d.body_id.clone(), d.point_name.clone(), d.started)
            });

            if let Some((body_id, point_name, started)) = drag_info {
                // Push undo once at the start of the drag.
                if !started {
                    state.push_undo();
                    if let Some(ref mut drag) = state.drag_target {
                        drag.started = true;
                    }
                }

                // Convert screen position to world coordinates, snapping to grid.
                let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);
                let (snap_x, snap_y) = state.grid.snap_point(wx, wy);
                state.move_attachment_point(&body_id, &point_name, snap_x, snap_y);
            }
        }
    }

    // Drag end: clear drag target when the pointer is no longer dragging.
    // response.dragged() is false when the button is released.
    if state.drag_target.is_some() && !response.dragged() {
        state.drag_target = None;
    }

    // ── Interaction: pan ────────────────────────────────────────────────
    // Middle-click drag OR shift+primary drag.
    if response.dragged() {
        let is_middle = response.dragged_by(egui::PointerButton::Middle);
        let is_shift_primary =
            is_shift && response.dragged_by(egui::PointerButton::Primary);
        if is_middle || is_shift_primary {
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

            // Zoom centered on mouse position.
            if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                let old_scale = state.view.scale;
                let new_scale = (old_scale * factor).clamp(MIN_SCALE, MAX_SCALE);

                // Adjust offset so the world point under the cursor stays fixed.
                // Use screen_to_world/world_to_screen to correctly handle the
                // Y-axis flip in the view transform.
                let [wx, wy] = state.view.screen_to_world(pointer_pos.x, pointer_pos.y);
                state.view.scale = new_scale;
                let new_screen = state.view.world_to_screen(wx, wy);
                state.view.offset[0] += pointer_pos.x - new_screen[0];
                state.view.offset[1] += pointer_pos.y - new_screen[1];
            }
        }
    }

    // ── Interaction: Escape key cancels joint creation mode ─────────────
    if state.creating_joint.is_some() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.creating_joint = None;
    }

    // ── Interaction: hit testing for selection / joint creation ─────────
    // Only process clicks when no drag is in progress.
    if state.drag_target.is_none() && response.clicked() {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            // In joint-creation mode, the second click creates the joint.
            if let Some((first_body, first_point)) = state.creating_joint.take() {
                // Find the attachment point under the cursor.
                let second_hit = attachment_hit_targets
                    .iter()
                    .find(|h| pointer_pos.distance(h.screen_pos) <= HIT_RADIUS);

                if let Some(hit) = second_hit {
                    // Don't create a joint between the same point on the same body.
                    if hit.body_id != first_body || hit.point_name != first_point {
                        state.add_revolute_joint(
                            &first_body,
                            &first_point,
                            &hit.body_id,
                            &hit.point_name,
                        );
                    }
                }
                // If no hit or same point, joint creation is simply cancelled.
            } else {
                // Normal selection mode.
                let mut hit: Option<SelectedEntity> = None;

                // Check joints first (smaller targets get priority).
                for (joint_screen, joint_id) in &joint_hit_targets {
                    if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                        hit = Some(SelectedEntity::Joint(joint_id.clone()));
                        break;
                    }
                }

                // If no joint hit, check attachment points.
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

    // ── Interaction: right-click context menu ────────────────────────────
    // Determine what entity (if any) is under the right-click position.
    let right_click_pos: Option<Pos2> = response
        .secondary_clicked()
        .then(|| response.interact_pointer_pos())
        .flatten();

    let right_click_joint: Option<String> = right_click_pos.and_then(|pos| {
        joint_hit_targets
            .iter()
            .find(|(screen_pos, _)| pos.distance(*screen_pos) <= HIT_RADIUS)
            .map(|(_, id)| id.clone())
    });

    let right_click_body: Option<(String, String)> = right_click_pos.and_then(|pos| {
        attachment_hit_targets
            .iter()
            .find(|h| h.body_id != GROUND_ID && pos.distance(h.screen_pos) <= HIT_RADIUS)
            .map(|h| (h.body_id.clone(), h.point_name.clone()))
    });

    // Cache the world coordinates of the right-click for "Add" operations.
    let right_click_world: Option<[f64; 2]> = right_click_pos.map(|pos| {
        state.view.screen_to_world(pos.x, pos.y)
    });

    // Show context menu using egui's built-in context_menu.
    response.context_menu(|ui| {
        if let Some(ref joint_id) = right_click_joint {
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
        } else if let Some((ref body_id, ref point_name)) = right_click_body {
            // ── Body context menu ───────────────────────────────────────
            ui.label(format!("Body: {}", body_id));
            ui.separator();

            if ui.button("Delete Body").clicked() {
                state.remove_body(body_id);
                state.selected = None;
                ui.close();
            }

            if ui.button("Add Revolute Joint...").clicked() {
                // Enter joint-creation mode with first click = this point.
                state.creating_joint = Some((body_id.clone(), point_name.clone()));
                ui.close();
            }
        } else {
            // ── Empty canvas context menu ───────────────────────────────
            if let Some([wx, wy]) = right_click_world {
                if ui.button("Add Ground Pivot Here").clicked() {
                    let name = state.next_ground_pivot_name();
                    state.add_ground_pivot(&name, wx, wy);
                    ui.close();
                }

                if ui.button("Add Body Here").clicked() {
                    let body_id = state.next_body_id();
                    // Create a binary body with two points: one at click, one offset.
                    let offset = 0.02; // 2cm offset in world coords
                    state.add_body(
                        &body_id,
                        ("A", wx, wy),
                        ("B", wx + offset, wy),
                    );
                    ui.close();
                }

                if ui.button("Add Revolute Joint...").clicked() {
                    // Enter joint-creation mode: user clicks two attachment points.
                    // First, find the nearest attachment point to the click.
                    if let Some(rcp) = right_click_pos {
                        let nearest = attachment_hit_targets
                            .iter()
                            .filter(|h| rcp.distance(h.screen_pos) <= HIT_RADIUS * 2.0)
                            .min_by(|a, b| {
                                rcp.distance(a.screen_pos)
                                    .partial_cmp(&rcp.distance(b.screen_pos))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                        if let Some(hit) = nearest {
                            state.creating_joint =
                                Some((hit.body_id.clone(), hit.point_name.clone()));
                        }
                    }
                    ui.close();
                }
            }
        }
    });
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

    let stroke = Stroke::new(1.5, color);
    painter.line_segment([top, bl], stroke);
    painter.line_segment([bl, br], stroke);
    painter.line_segment([br, top], stroke);

    // Hatch lines below the triangle base.
    let hatch_y = center.y + size;
    let n_hatches = 4;
    let hatch_len = size * 0.4;
    let spacing = size / n_hatches as f32;
    for i in 0..n_hatches {
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
