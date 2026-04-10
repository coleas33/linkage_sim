//! Canvas right-click context menus for joints, attachment points, bodies, and
//! empty canvas.

use eframe::egui::{self, Pos2};

use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::gui::state::{
    AddBodyState, AppState, ContextMenuTarget, EditorTool,
};

use super::colors::HIT_RADIUS;
use super::hit_testing::{find_nearest_body_segment, AttachmentHit, BodySegment};

/// Capture the right-click target and show the context menu popup.
pub fn handle_context_menu(
    response: &egui::Response,
    state: &mut AppState,
    joint_hit_targets: &[(Pos2, String)],
    attachment_hit_targets: &[AttachmentHit],
    body_segments: &[BodySegment],
    grounded_revolute_ids: &[String],
    current_driver_joint: &Option<String>,
    right_drag_ended: bool,
) {
    // On the frame the right-click occurs, capture what was under the cursor
    // and store it in AppState. The context_menu closure runs every frame
    // while the popup is open, so it reads from stored state.
    // Only trigger on a true click (not after a right-drag pan).
    if response.secondary_clicked() && !right_drag_ended {
        if let Some(pos) = response.interact_pointer_pos() {
            let joint_id = joint_hit_targets
                .iter()
                .find(|(screen_pos, _)| pos.distance(*screen_pos) <= HIT_RADIUS)
                .map(|(_, id)| id.clone());

            // Attachment points first (priority over body area);
            // includes ground pivots so they get context-menu actions.
            let attachment_point = attachment_hit_targets
                .iter()
                .find(|h| pos.distance(h.screen_pos) <= HIT_RADIUS)
                .map(|h| (h.body_id.clone(), h.point_name.clone()));

            // Body area: only if no attachment point matched
            let body_area = if attachment_point.is_none() {
                find_nearest_body_segment(pos, body_segments, HIT_RADIUS)
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

    // Show context menu using a manual Popup so we can suppress it after
    // a right-drag pan.  Popup::context_menu() unconditionally opens on
    // secondary_clicked(); we replicate its logic but gate on
    // `!right_drag_ended`.
    let ctx_target = state.context_menu_target.clone();
    let should_open = response.secondary_clicked() && !right_drag_ended;
    let should_close = response.clicked(); // primary click closes menu
    let open_cmd = if should_open {
        Some(egui::SetOpenCommand::Bool(true))
    } else if should_close {
        Some(egui::SetOpenCommand::Bool(false))
    } else {
        None
    };
    egui::Popup::menu(response)
        .open_memory(open_cmd)
        .at_pointer_fixed()
        .show(|ui: &mut egui::Ui| {
        if let Some(ref joint_id) = ctx_target.joint_id {
            // ── Joint context menu ──────────────────────────────────────
            show_joint_menu(ui, state, joint_id, grounded_revolute_ids, current_driver_joint);
        } else if let Some((ref body_id, ref point_name)) = ctx_target.attachment_point {
            // ── Attachment point context menu ────────────────────────────
            show_attachment_menu(ui, state, body_id, point_name, current_driver_joint);
        } else if let Some(ref body_id) = ctx_target.body_area {
            // ── Body area context menu ──────────────────────────────────
            show_body_area_menu(ui, state, body_id, &ctx_target);
        } else {
            // ── Empty canvas context menu ───────────────────────────────
            show_canvas_menu(ui, state, &ctx_target);
        }
    });
}

// ── Joint context menu ───────────────────────────────────────────────────────

fn show_joint_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    joint_id: &str,
    grounded_revolute_ids: &[String],
    current_driver_joint: &Option<String>,
) {
    ui.label(format!("Joint: {}", joint_id));
    ui.separator();

    let is_grounded_revolute = grounded_revolute_ids.contains(&joint_id.to_string());
    let is_current_driver =
        current_driver_joint.as_deref() == Some(joint_id);

    if is_grounded_revolute {
        let label = if is_current_driver {
            "Set as Driver (current)"
        } else {
            "Set as Driver"
        };
        if ui
            .add_enabled(!is_current_driver, egui::Button::new(label))
            .on_hover_text("Make this grounded revolute joint the driven input")
            .clicked()
        {
            state.pending_driver_reassignment = Some(joint_id.to_string());
            ui.close();
        }
    } else {
        // Show a disabled option so users see that Set as Driver exists
        // and understand why it's not available for this joint.
        ui.add_enabled(false, egui::Button::new("Set as Driver"))
            .on_hover_text(
                "Only grounded revolute joints can be drivers. Use the + Ground tool to ground a link, then right-click the grounded joint and select Set as Driver."
            );
    }

    if ui.button("Delete Joint").on_hover_text("Remove this joint and disconnect the bodies").clicked() {
        state.remove_joint(joint_id);
        ui.close();
    }
}

// ── Attachment point context menu ────────────────────────────────────────────

fn show_attachment_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    body_id: &str,
    point_name: &str,
    current_driver_joint: &Option<String>,
) {
    if body_id == GROUND_ID {
        ui.label(format!("Ground Pivot: {}", point_name));
    } else {
        ui.label(format!("Point: {}.{}", body_id, point_name));
    }
    ui.separator();

    ui.menu_button("Create Joint", |ui| {
        use crate::gui::state::PendingJointType;
        if ui.button("Revolute").on_hover_text("Create a pin joint allowing relative rotation").clicked() {
            state.creating_joint = Some((body_id.to_string(), point_name.to_string(), PendingJointType::Revolute));
            ui.close();
        }
        if ui.button("Prismatic").on_hover_text("Create a slider joint allowing linear translation").clicked() {
            state.creating_joint = Some((body_id.to_string(), point_name.to_string(), PendingJointType::Prismatic));
            ui.close();
        }
        if ui.button("Fixed").on_hover_text("Create a rigid joint locking both bodies together").clicked() {
            state.creating_joint = Some((body_id.to_string(), point_name.to_string(), PendingJointType::Fixed));
            ui.close();
        }
        ui.separator();
        ui.add_enabled(false, egui::Button::new("Cam Follower [WIP]"))
            .on_hover_text("Cam-follower joint (available via JSON import only)");
    });

    let delete_label = if body_id == GROUND_ID { "Delete Ground Pivot" } else { "Delete Pivot" };
    if ui.button(delete_label).on_hover_text("Remove this attachment point and any connected joints").clicked() {
        state.remove_attachment_point(body_id, point_name);
        ui.close();
    }

    // Set as Driver: show if this body belongs to a grounded revolute joint.
    // Otherwise show a disabled version with guidance so the user knows
    // the feature exists and how to enable it.
    if body_id != GROUND_ID {
        let mut found_grounded_joint: Option<String> = None;
        if let Some(mech) = &state.mechanism {
            let grounded = mech.grounded_revolute_joint_ids();
            for joint in mech.joints() {
                if joint.is_revolute()
                    && grounded.contains(&joint.id().to_string())
                    && ((joint.body_i_id() == body_id) || (joint.body_j_id() == body_id))
                    && current_driver_joint.as_deref() != Some(joint.id())
                {
                    found_grounded_joint = Some(joint.id().to_string());
                    break;
                }
            }
        }

        if let Some(joint_id) = found_grounded_joint {
            if ui.button("Set as Driver").on_hover_text("Make this joint the driven input for kinematic analysis").clicked() {
                state.pending_driver_reassignment = Some(joint_id);
                ui.close();
            }
        } else {
            ui.add_enabled(false, egui::Button::new("Set as Driver"))
                .on_hover_text(
                    "This body has no grounded revolute joint. Use the + Ground tool to click a free endpoint of a link to ground it, then come back and right-click to set the driver."
                );
        }
    }
}

// ── Body area context menu ───────────────────────────────────────────────────

fn show_body_area_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    body_id: &str,
    ctx_target: &ContextMenuTarget,
) {
    ui.label(format!("Body: {}", body_id));
    ui.separator();

    if let Some([wx, wy]) = ctx_target.world_pos {
        if ui.button("Add Pivot Here").on_hover_text("Add a new attachment point on this link at the clicked location").clicked() {
            let name = state.next_attachment_point_name(body_id);
            let (sx, sy) = state.grid.snap_point(wx, wy);
            state.add_attachment_point_to_body(body_id, &name, sx, sy);
            ui.close();
        }
    }

    if ui.button("Delete Body").on_hover_text("Remove this body and all its joints from the mechanism").clicked() {
        state.remove_body(body_id);
        state.selected = None;
        ui.close();
    }
}

// ── Empty canvas context menu ────────────────────────────────────────────────

fn show_canvas_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    ctx_target: &ContextMenuTarget,
) {
    if let Some([wx, wy]) = ctx_target.world_pos {
        if ui.button("Add Ground Pivot Here").on_hover_text("Place a fixed ground pivot at this canvas location").clicked() {
            let name = state.next_ground_pivot_name();
            state.add_ground_pivot(&name, wx, wy);
            ui.close();
        }

        if ui.button("Draw Link").on_hover_text("Switch to Draw Link tool to create a new link").clicked() {
            state.active_tool = EditorTool::DrawLink;
            ui.close();
        }

        if ui.button("Start Body Here").on_hover_text("Begin placing a new multi-point body at this location").clicked() {
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
