//! Force element toolbar ribbon with categorized dropdown menus.

use eframe::egui;
use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::forces::elements::*;
use super::state::{AppState, SelectedEntity};

/// Pending force addition from the toolbar.
pub enum PendingForceAdd {
    Add(ForceElement),
}

/// Draw the force toolbar ribbon. Returns a pending force addition if clicked.
pub fn draw_force_toolbar(ui: &mut egui::Ui, state: &AppState) -> Option<PendingForceAdd> {
    let mut pending: Option<PendingForceAdd> = None;
    let (selected_body, connected_body) = resolve_target_bodies(state);

    ui.horizontal(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(10.0, 5.0);

        // Show the current target body so the user knows where forces will be added
        if let Some(ref body_id) = selected_body {
            ui.colored_label(egui::Color32::from_rgb(120, 180, 255), format!("Target: {}", body_id));
        } else {
            ui.colored_label(egui::Color32::GRAY, "Target: (select a body)");
        }
        ui.separator();

        let torque_color = egui::Color32::from_rgb(100, 220, 140);
        let force_color = egui::Color32::from_rgb(255, 165, 80);

        // -- Joint Torques dropdown --
        ui.menu_button(
            egui::RichText::new("\u{2699} Joint Torques \u{25BC}").color(torque_color),
            |ui| {
            if let Some((ref a, ref b)) = two_bodies(&selected_body, &connected_body) {
                if ui.button("Motor").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::Motor(MotorElement {
                        body_i: a.clone(), body_j: b.clone(),
                        stall_torque: 10.0, no_load_speed: 10.0, direction: 1.0,
                    })));
                    ui.close();
                }
                if ui.button("Torsion Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::TorsionSpring(TorsionSpringElement {
                        body_i: a.clone(), body_j: b.clone(),
                        stiffness: 10.0, free_angle: 0.0,
                    })));
                    ui.close();
                }
                if ui.button("Rotary Damper").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::RotaryDamper(RotaryDamperElement {
                        body_i: a.clone(), body_j: b.clone(), damping: 5.0,
                    })));
                    ui.close();
                }
                if ui.button("Bearing Friction").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::BearingFriction(BearingFrictionElement {
                        body_i: a.clone(), body_j: b.clone(),
                        constant_drag: 0.1, viscous_coeff: 0.01, coulomb_coeff: 0.0,
                        pin_radius: 0.0, radial_load: 0.0, v_threshold: 0.01,
                    })));
                    ui.close();
                }
                if ui.button("Joint Limit").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::JointLimit(JointLimitElement {
                        body_i: a.clone(), body_j: b.clone(),
                        angle_min: -std::f64::consts::FRAC_PI_2,
                        angle_max: std::f64::consts::FRAC_PI_2,
                        stiffness: 100.0, damping: 0.0, restitution: 0.5,
                    })));
                    ui.close();
                }
            } else {
                ui.label("Select a body first");
            }
        });

        // -- Link Forces dropdown --
        ui.menu_button(
            egui::RichText::new("\u{2B06} Link Forces \u{25BC}").color(force_color),
            |ui| {
            // Single-body elements
            if let Some(ref body_id) = selected_body {
                if ui.button("External Force").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::ExternalForce(ExternalForceElement {
                        body_id: body_id.clone(), local_point: [0.0, 0.0],
                        force: [0.0, -10.0], modulation: TimeModulation::Constant,
                    })));
                    ui.close();
                }
                if ui.button("External Torque").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::ExternalTorque(ExternalTorqueElement {
                        body_id: body_id.clone(), torque: 1.0,
                        modulation: TimeModulation::Constant,
                    })));
                    ui.close();
                }
            } else {
                ui.label("Select a body first");
            }

            ui.separator();

            // Two-body elements
            if let Some((ref a, ref b)) = two_bodies(&selected_body, &connected_body) {
                if ui.button("Linear Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearSpring(LinearSpringElement {
                        body_a: a.clone(), point_a: [0.0, 0.0],
                        body_b: b.clone(), point_b: [0.0, 0.0],
                        stiffness: 100.0, free_length: 0.1,
                    })));
                    ui.close();
                }
                if ui.button("Linear Damper").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearDamper(LinearDamperElement {
                        body_a: a.clone(), point_a: [0.0, 0.0],
                        body_b: b.clone(), point_b: [0.0, 0.0], damping: 10.0,
                    })));
                    ui.close();
                }
                if ui.button("Gas Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::GasSpring(GasSpringElement {
                        body_a: a.clone(), point_a: [0.0, 0.0],
                        body_b: b.clone(), point_b: [0.0, 0.0],
                        initial_force: 100.0, extended_length: 0.5, stroke: 0.2,
                        damping: 0.0, polytropic_exp: 1.0,
                    })));
                    ui.close();
                }
                if ui.button("Linear Actuator").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearActuator(LinearActuatorElement {
                        body_a: a.clone(), point_a: [0.0, 0.0],
                        body_b: b.clone(), point_b: [0.0, 0.0],
                        force: 100.0, speed_limit: 0.0,
                    })));
                    ui.close();
                }
            } else if selected_body.is_none() {
                ui.separator();
                ui.label("Select a body for 2-body elements");
            }
        });
    });

    pending
}

/// Determine target bodies from current Link Editor selection.
pub(crate) fn resolve_target_bodies(state: &AppState) -> (Option<String>, Option<String>) {
    // Use link_editor_body (dropdown) as the primary source
    let selected_body = state.link_editor_body.clone()
        .filter(|id| id != GROUND_ID)
        // Fall back to canvas selection if Link Editor has nothing
        .or_else(|| match &state.selected {
            Some(SelectedEntity::Body(id)) if id != GROUND_ID => Some(id.clone()),
            _ => None,
        });

    let connected_body = if let (Some(sel_id), Some(mech)) = (&selected_body, &state.mechanism) {
        mech.joints()
            .iter()
            .find_map(|j| {
                if j.body_i_id() == sel_id {
                    Some(j.body_j_id().to_string())
                } else if j.body_j_id() == sel_id {
                    Some(j.body_i_id().to_string())
                } else {
                    None
                }
            })
    } else {
        None
    };

    (selected_body, connected_body)
}

fn two_bodies(selected: &Option<String>, connected: &Option<String>) -> Option<(String, String)> {
    match (selected, connected) {
        (Some(a), Some(b)) => Some((a.clone(), b.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::AppState;
    use crate::gui::samples::SampleMechanism;

    #[test]
    fn resolve_target_bodies_with_selected_body() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = Some(SelectedEntity::Body("crank".to_string()));
        let (sel, conn) = resolve_target_bodies(&state);
        assert_eq!(sel, Some("crank".to_string()));
        assert!(conn.is_some(), "Connected body should be found via joints");
    }

    #[test]
    fn resolve_target_bodies_with_no_selection() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = None;
        let (sel, conn) = resolve_target_bodies(&state);
        assert!(sel.is_none());
        assert!(conn.is_none());
    }

    #[test]
    fn resolve_target_bodies_ground_selection_excluded() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = Some(SelectedEntity::Body("ground".to_string()));
        let (sel, _conn) = resolve_target_bodies(&state);
        assert!(sel.is_none(), "Ground should not be a valid force target");
    }
}
