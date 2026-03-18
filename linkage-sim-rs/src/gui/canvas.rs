//! 2D mechanism canvas: rendering, pan/zoom, hit testing, debug overlay.

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::gui::state::{AppState, SelectedEntity};

// ── Colors ──────────────────────────────────────────────────────────────────

const BG_COLOR: Color32 = Color32::from_rgb(30, 30, 35);
const GROUND_LINE_COLOR: Color32 = Color32::from_rgb(60, 60, 65);
const BODY_COLOR: Color32 = Color32::from_rgb(140, 180, 220);
const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 200, 80);
const JOINT_COLOR: Color32 = Color32::from_rgb(200, 200, 200);
const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 200, 80);
const GROUND_MARKER_COLOR: Color32 = Color32::from_rgb(120, 120, 100);
const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(180, 180, 180);
const DEBUG_DIM_COLOR: Color32 = Color32::from_rgb(100, 100, 100);
const NO_MECH_TEXT_COLOR: Color32 = Color32::from_rgb(120, 120, 120);

// ── Sizing ──────────────────────────────────────────────────────────────────

const BODY_STROKE_WIDTH: f32 = 3.0;
const JOINT_RADIUS: f32 = 5.0;
const JOINT_STROKE_WIDTH: f32 = 2.0;
const GROUND_MARKER_SIZE: f32 = 10.0;
const HIT_RADIUS: f32 = 10.0;
const ZOOM_FACTOR: f32 = 1.1;
const MIN_SCALE: f32 = 100.0;
const MAX_SCALE: f32 = 100_000.0;

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

    // Collect joint screen positions and IDs for hit testing later.
    let mut joint_hit_targets: Vec<(Pos2, String)> = Vec::new();
    // Collect body attachment point screen positions and body IDs for hit testing.
    let mut body_hit_targets: Vec<(Pos2, String)> = Vec::new();

    // Scoped immutable borrow for rendering.
    {
        let mech = state.mechanism.as_ref().unwrap();
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

            // Store body hit targets (screen positions of attachment points).
            for sp in &screen_points {
                body_hit_targets.push((*sp, body_id.clone()));
            }
        }

        // ── Draw ground markers ─────────────────────────────────────────
        if let Some(ground) = bodies.get(GROUND_ID) {
            for (_name, local) in &ground.attachment_points {
                let global = mech_state.body_point_global(GROUND_ID, local, q);
                let sp = view.world_to_screen(global.x, global.y);
                let center = Pos2::new(sp[0], sp[1]);
                draw_ground_marker(&painter, center, GROUND_MARKER_SIZE, GROUND_MARKER_COLOR);
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
    }
    // Immutable borrows of state.mechanism (and its sub-borrows) are now dropped.

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

    // ── Interaction: pan ────────────────────────────────────────────────
    // Middle-click drag OR shift+primary drag.
    if response.dragged() {
        let is_middle = response.dragged_by(egui::PointerButton::Middle);
        let is_shift_primary = ui.input(|i| i.modifiers.shift)
            && response.dragged_by(egui::PointerButton::Primary);
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

    // ── Interaction: hit testing for selection ──────────────────────────
    if response.clicked() {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            let mut hit: Option<SelectedEntity> = None;

            // Check joints first (smaller targets get priority).
            for (joint_screen, joint_id) in &joint_hit_targets {
                if pointer_pos.distance(*joint_screen) <= HIT_RADIUS {
                    hit = Some(SelectedEntity::Joint(joint_id.clone()));
                    break;
                }
            }

            // If no joint hit, check body attachment points.
            if hit.is_none() {
                for (pt_screen, body_id) in &body_hit_targets {
                    if pointer_pos.distance(*pt_screen) <= HIT_RADIUS {
                        hit = Some(SelectedEntity::Body(body_id.clone()));
                        break;
                    }
                }
            }

            state.selected = hit;
        }
    }
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
