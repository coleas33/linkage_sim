//! Read-only property panel for the selected entity.

use eframe::egui;
use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use super::state::{AppState, SelectedEntity};

/// Draw the property panel showing info about the selected entity.
pub fn draw_property_panel(ui: &mut egui::Ui, state: &AppState) {
    ui.heading("Properties");

    let Some(mech) = &state.mechanism else {
        ui.label("No mechanism loaded.");
        return;
    };

    let Some(selected) = &state.selected else {
        ui.label("Click a body or joint to inspect.");
        return;
    };

    match selected {
        SelectedEntity::Body(body_id) => {
            if let Some(body) = mech.bodies().get(body_id) {
                ui.strong(format!("Body: {}", body_id));
                ui.separator();

                let mech_state = mech.state();
                let q = &state.q;

                if body_id != GROUND_ID {
                    let (x, y, theta) = mech_state.get_pose(body_id, q);
                    ui.label(format!("Position: ({:.4}, {:.4}) m", x, y));
                    ui.label(format!("Angle: {:.2}°", theta.to_degrees()));
                } else {
                    ui.label("Position: (0, 0) — fixed");
                    ui.label("Angle: 0° — fixed");
                }

                ui.separator();
                ui.label(format!("Mass: {:.4} kg", body.mass));
                ui.label(format!("Izz_cg: {:.6} kg·m²", body.izz_cg));
                ui.label(format!("CG local: ({:.4}, {:.4})", body.cg_local.x, body.cg_local.y));

                ui.separator();
                ui.strong("Attachment points:");
                let mut pts: Vec<_> = body.attachment_points.iter().collect();
                pts.sort_by_key(|(name, _)| name.as_str());
                for (name, local) in pts {
                    let global = mech_state.body_point_global(body_id, local, q);
                    ui.label(format!(
                        "  {} — local: ({:.4}, {:.4}), global: ({:.4}, {:.4})",
                        name, local.x, local.y, global.x, global.y
                    ));
                }
            }
        }
        SelectedEntity::Joint(joint_id) => {
            if let Some(joint) = mech.joints().iter().find(|j| j.id() == joint_id) {
                let joint_type = if joint.is_revolute() {
                    "Revolute"
                } else if joint.is_prismatic() {
                    "Prismatic"
                } else {
                    "Fixed"
                };

                ui.strong(format!("Joint: {}", joint_id));
                ui.label(format!("Type: {}", joint_type));
                ui.separator();

                ui.label(format!("Body i: {}", joint.body_i_id()));
                ui.label(format!("Body j: {}", joint.body_j_id()));
                ui.label(format!("DOF removed: {}", joint.dof_removed()));
                ui.label(format!("Equations: {}", joint.n_equations()));

                ui.separator();
                let mech_state = mech.state();
                let q = &state.q;
                let global_pos = mech_state.body_point_global(
                    joint.body_i_id(),
                    &joint.point_i_local(),
                    q,
                );
                ui.label(format!("Position: ({:.4}, {:.4}) m", global_pos.x, global_pos.y));
            }
        }
        SelectedEntity::Driver(_driver_id) => {
            ui.label("Driver properties not yet available.");
        }
    }
}
