//! Property panel for the selected entity.
//!
//! Mass and inertia properties are editable for non-ground bodies when a
//! blueprint is available. Edits are applied via `AppState::set_body_mass`
//! and `AppState::set_body_izz`, which mutate the blueprint and rebuild.

use eframe::egui;
use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use super::state::{AppState, SelectedEntity};

/// Pending edit collected during UI rendering, applied after all reads
/// are done to avoid borrow conflicts.
enum PendingPropertyEdit {
    Mass { body_id: String, value: f64 },
    Izz { body_id: String, value: f64 },
}

/// Draw the property panel showing info about the selected entity.
///
/// Mass and inertia fields are editable via `DragValue` widgets when a
/// blueprint is present and the selected body is not ground.
pub fn draw_property_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Properties");

    // Collect any pending edits during rendering, then apply them after
    // we're done reading from state (avoids &/&mut borrow overlap).
    let mut pending: Option<PendingPropertyEdit> = None;

    // --- Immutable-borrow block: read mechanism, blueprint, and draw UI ---
    {
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
                let body_id = body_id.clone();
                if let Some(body) = mech.bodies().get(&body_id) {
                    ui.strong(format!("Body: {}", body_id));
                    ui.separator();

                    let mech_state = mech.state();
                    let q = &state.q;

                    if body_id != GROUND_ID {
                        let (x, y, theta) = mech_state.get_pose(&body_id, q);
                        ui.label(format!("Position: ({:.4}, {:.4}) m", x, y));
                        ui.label(format!("Angle: {:.2}\u{00b0}", theta.to_degrees()));
                    } else {
                        ui.label("Position: (0, 0) \u{2014} fixed");
                        ui.label("Angle: 0\u{00b0} \u{2014} fixed");
                    }

                    ui.separator();

                    // Editable mass properties for non-ground bodies with a blueprint
                    if body_id != GROUND_ID {
                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                let mut mass = bp_body.mass;
                                let mut izz = bp_body.izz_cg;

                                ui.horizontal(|ui| {
                                    ui.label("Mass:");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut mass)
                                                .speed(0.01)
                                                .range(0.0..=f64::MAX)
                                                .suffix(" kg"),
                                        )
                                        .changed()
                                    {
                                        pending = Some(PendingPropertyEdit::Mass {
                                            body_id: body_id.clone(),
                                            value: mass,
                                        });
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Izz_cg:");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut izz)
                                                .speed(0.001)
                                                .range(0.0..=f64::MAX)
                                                .suffix(" kg\u{00b7}m\u{00b2}"),
                                        )
                                        .changed()
                                    {
                                        pending = Some(PendingPropertyEdit::Izz {
                                            body_id: body_id.clone(),
                                            value: izz,
                                        });
                                    }
                                });
                            } else {
                                // Blueprint exists but body not found (shouldn't happen)
                                ui.label(format!("Mass: {:.4} kg", body.mass));
                                ui.label(format!(
                                    "Izz_cg: {:.6} kg\u{00b7}m\u{00b2}",
                                    body.izz_cg
                                ));
                            }
                        } else {
                            // No blueprint -- read-only display from mechanism
                            ui.label(format!("Mass: {:.4} kg", body.mass));
                            ui.label(format!(
                                "Izz_cg: {:.6} kg\u{00b7}m\u{00b2}",
                                body.izz_cg
                            ));
                        }
                    } else {
                        ui.label(format!("Mass: {:.4} kg", body.mass));
                        ui.label(format!(
                            "Izz_cg: {:.6} kg\u{00b7}m\u{00b2}",
                            body.izz_cg
                        ));
                    }

                    ui.label(format!(
                        "CG local: ({:.4}, {:.4})",
                        body.cg_local.x, body.cg_local.y
                    ));

                    ui.separator();
                    ui.strong("Attachment points:");
                    let mut pts: Vec<_> = body.attachment_points.iter().collect();
                    pts.sort_by_key(|(name, _)| name.as_str());
                    for (name, local) in pts {
                        let global = mech_state.body_point_global(&body_id, local, q);
                        ui.label(format!(
                            "  {} \u{2014} local: ({:.4}, {:.4}), global: ({:.4}, {:.4})",
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
                    ui.label(format!(
                        "Position: ({:.4}, {:.4}) m",
                        global_pos.x, global_pos.y
                    ));
                }
            }
            SelectedEntity::Driver(_driver_id) => {
                ui.label("Driver properties not yet available.");
            }
        }
    } // end immutable borrow block

    // --- Apply any pending edits (mutable borrow now safe) ---
    if let Some(edit) = pending {
        match edit {
            PendingPropertyEdit::Mass { body_id, value } => {
                state.set_body_mass(&body_id, value);
            }
            PendingPropertyEdit::Izz { body_id, value } => {
                state.set_body_izz(&body_id, value);
            }
        }
    }
}
