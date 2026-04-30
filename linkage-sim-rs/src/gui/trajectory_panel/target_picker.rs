//! Target picker UI: ControlTarget selector + body/point/axis fields.

use eframe::egui;

use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::gui::state::{AppState, PendingCanvasPickKind};
use crate::gui::sweep::SweepMode;
use crate::solver::inverse_kinematics::ControlTarget;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    // Compute body list and capture mech-ref before we mutably borrow
    // sweep_mode (split-borrow: sweep_mode and pending_canvas_pick / mechanism
    // are disjoint fields on AppState, but the borrow checker only sees the
    // structural lifetime, so we order the destructures carefully below).
    let body_ids: Vec<String> = state
        .mechanism
        .as_ref()
        .map(|m| {
            m.bodies()
                .iter()
                .filter_map(|(id, _)| {
                    if id == "ground" {
                        None
                    } else {
                        Some(id.to_string())
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let mech_ref = state.mechanism.as_ref();

    // Borrow pending_canvas_pick first (disjoint field from sweep_mode), then
    // destructure sweep_mode for the variant editor below.
    let pending = &mut state.pending_canvas_pick;
    let SweepMode::Trajectory { target, .. } = &mut state.sweep_mode else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    let current_kind = target_kind_label(target);
    let mut new_kind = current_kind;
    egui::ComboBox::from_label("Observable")
        .selected_text(new_kind)
        .show_ui(ui, |ui| {
            for kind in &["Angle", "WorldX", "WorldY", "Projection", "Distance"] {
                ui.selectable_value(&mut new_kind, kind, *kind);
            }
        });

    if new_kind != current_kind {
        let body = body_ids.first().cloned().unwrap_or_default();
        if !body.is_empty() {
            *target = match new_kind {
                "Angle" => ControlTarget::angle(body),
                "WorldX" => ControlTarget::world_x(body, [0.0, 0.0]),
                "WorldY" => ControlTarget::world_y(body, [0.0, 0.0]),
                "Projection" => {
                    ControlTarget::projection(body, [0.0, 0.0], [0.0, 0.0], [1.0, 0.0])
                }
                "Distance" => ControlTarget::distance(body, [0.0, 0.0], [0.0, 0.0]),
                _ => target.clone(),
            };
        }
    }

    // Body picker (common to all variants)
    draw_body_picker(target, &body_ids, ui);

    // Extract body_id once so we can pass it to the joint helper without
    // colliding with the variant-specific destructure below.
    let body_id_for_target = match target {
        ControlTarget::Angle { body_id }
        | ControlTarget::WorldX { body_id, .. }
        | ControlTarget::WorldY { body_id, .. }
        | ControlTarget::Projection { body_id, .. }
        | ControlTarget::Distance { body_id, .. } => body_id.clone(),
    };

    // Variant-specific fields
    match target {
        ControlTarget::Angle { .. } => {}
        ControlTarget::WorldX { local_pt, .. } | ControlTarget::WorldY { local_pt, .. } => {
            ui.horizontal(|ui| {
                draw_point_input_with_joint_helper(
                    local_pt,
                    &body_id_for_target,
                    mech_ref,
                    ui,
                    "Local point (m)",
                );
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::LocalPt, ui);
            });
        }
        ControlTarget::Projection {
            local_pt,
            axis_origin,
            axis_dir,
            ..
        } => {
            ui.horizontal(|ui| {
                draw_point_input_with_joint_helper(
                    local_pt,
                    &body_id_for_target,
                    mech_ref,
                    ui,
                    "Local point (m)",
                );
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::LocalPt, ui);
            });
            ui.horizontal(|ui| {
                ui.label("Axis origin (m):");
                draw_point_input(axis_origin, ui);
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::AxisOrigin, ui);
            });
            ui.horizontal(|ui| {
                ui.label("Axis direction (will be normalized):");
                draw_point_input(axis_dir, ui);
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::AxisDir, ui);
            });
        }
        ControlTarget::Distance {
            local_pt, ref_pt, ..
        } => {
            ui.horizontal(|ui| {
                draw_point_input_with_joint_helper(
                    local_pt,
                    &body_id_for_target,
                    mech_ref,
                    ui,
                    "Local point (m)",
                );
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::LocalPt, ui);
            });
            ui.horizontal(|ui| {
                ui.label("Reference point (m):");
                draw_point_input(ref_pt, ui);
                draw_pick_on_canvas_button(pending, PendingCanvasPickKind::RefPt, ui);
            });
        }
    }

    // Live readout — clone target out of the mutable borrow so we can call
    // evaluate without holding a mut borrow to state.sweep_mode.
    let target_clone = target.clone();
    if let Some(mech) = mech_ref {
        let g_now = target_clone.evaluate(mech, &state.q);
        ui.label(format!(
            "Current g(q) = {:.4} {}",
            g_now,
            target_clone.unit_label()
        ));
    }
}

/// Renders a "📍 Pick" button. When clicked, sets `*pending` to `kind` (or
/// clears it if already active for the same kind, providing a toggle to cancel).
/// Label changes to "📍 Click canvas..." while pick mode is active for this field.
fn draw_pick_on_canvas_button(
    pending: &mut Option<PendingCanvasPickKind>,
    kind: PendingCanvasPickKind,
    ui: &mut egui::Ui,
) {
    let active = *pending == Some(kind);
    let label = if active { "📍 Click canvas..." } else { "📍 Pick" };
    let resp = ui.button(label).on_hover_text(
        "Click to enter pick mode, then click anywhere on the canvas to set this point's coordinates.",
    );
    if resp.clicked() {
        *pending = if active { None } else { Some(kind) };
    }
}

fn target_kind_label(t: &ControlTarget) -> &'static str {
    match t {
        ControlTarget::Angle { .. } => "Angle",
        ControlTarget::WorldX { .. } => "WorldX",
        ControlTarget::WorldY { .. } => "WorldY",
        ControlTarget::Projection { .. } => "Projection",
        ControlTarget::Distance { .. } => "Distance",
    }
}

fn draw_body_picker(target: &mut ControlTarget, body_ids: &[String], ui: &mut egui::Ui) {
    let current = match target {
        ControlTarget::Angle { body_id }
        | ControlTarget::WorldX { body_id, .. }
        | ControlTarget::WorldY { body_id, .. }
        | ControlTarget::Projection { body_id, .. }
        | ControlTarget::Distance { body_id, .. } => body_id.clone(),
    };
    let mut new = current.clone();
    egui::ComboBox::from_label("Body")
        .selected_text(&current)
        .show_ui(ui, |ui| {
            for id in body_ids {
                ui.selectable_value(&mut new, id.clone(), id);
            }
        });
    if new != current && !new.is_empty() {
        match target {
            ControlTarget::Angle { body_id }
            | ControlTarget::WorldX { body_id, .. }
            | ControlTarget::WorldY { body_id, .. }
            | ControlTarget::Projection { body_id, .. }
            | ControlTarget::Distance { body_id, .. } => *body_id = new,
        }
    }
}

fn draw_point_input(pt: &mut [f64; 2], ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("x:");
        ui.add(egui::DragValue::new(&mut pt[0]).speed(0.001).suffix(" m"));
        ui.label("y:");
        ui.add(egui::DragValue::new(&mut pt[1]).speed(0.001).suffix(" m"));
    });
}

/// Like `draw_point_input` but adds a "From joint" dropdown that lists every
/// joint attached to `body_id`; selecting one populates `pt` with that joint's
/// body-local coordinates on `body_id`.
///
/// `mech` is the optional active mechanism — when `None` (or when no joints
/// touch the body), the dropdown is omitted.
fn draw_point_input_with_joint_helper(
    pt: &mut [f64; 2],
    body_id: &str,
    mech: Option<&Mechanism>,
    ui: &mut egui::Ui,
    label: &str,
) {
    ui.horizontal(|ui| {
        ui.label(format!("{}:", label));
        ui.label("x:");
        ui.add(egui::DragValue::new(&mut pt[0]).speed(0.001).suffix(" m"));
        ui.label("y:");
        ui.add(egui::DragValue::new(&mut pt[1]).speed(0.001).suffix(" m"));

        let joints_on_body: Vec<(String, [f64; 2])> = mech
            .map(|m| {
                m.joints()
                    .iter()
                    .filter_map(|j| {
                        if j.body_i_id() == body_id {
                            let p = j.point_i_local();
                            Some((j.id().to_string(), [p.x, p.y]))
                        } else if j.body_j_id() == body_id {
                            let p = j.point_j_local();
                            Some((j.id().to_string(), [p.x, p.y]))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        if !joints_on_body.is_empty() {
            egui::ComboBox::from_id_salt(format!("from-joint-{}-{}", label, body_id))
                .selected_text("From joint")
                .show_ui(ui, |ui| {
                    for (jid, jpt) in &joints_on_body {
                        if ui.selectable_label(false, jid).clicked() {
                            *pt = *jpt;
                        }
                    }
                });
        }
    });
}
