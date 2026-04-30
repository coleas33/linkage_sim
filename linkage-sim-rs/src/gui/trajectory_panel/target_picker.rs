//! Target picker UI: ControlTarget selector + body/point/axis fields.

use eframe::egui;

use crate::gui::state::AppState;
use crate::gui::sweep::SweepMode;
use crate::solver::inverse_kinematics::ControlTarget;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let SweepMode::Trajectory { target, .. } = &mut state.sweep_mode else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    // Variant selector — show current kind, allow change
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

    // Variant-specific fields
    match target {
        ControlTarget::Angle { .. } => {}
        ControlTarget::WorldX { local_pt, .. } | ControlTarget::WorldY { local_pt, .. } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
        }
        ControlTarget::Projection {
            local_pt,
            axis_origin,
            axis_dir,
            ..
        } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
            ui.label("Axis origin (m):");
            draw_point_input(axis_origin, ui);
            ui.label("Axis direction (will be normalized):");
            draw_point_input(axis_dir, ui);
        }
        ControlTarget::Distance {
            local_pt, ref_pt, ..
        } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
            ui.label("Reference point (m):");
            draw_point_input(ref_pt, ui);
        }
    }

    // Live readout — clone target out of the mutable borrow so we can call
    // evaluate without holding a mut borrow to state.sweep_mode.
    let target_clone = target.clone();
    if let Some(mech) = &state.mechanism {
        let g_now = target_clone.evaluate(mech, &state.q);
        ui.label(format!(
            "Current g(q) = {:.4} {}",
            g_now,
            target_clone.unit_label()
        ));
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
