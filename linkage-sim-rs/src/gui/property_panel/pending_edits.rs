//! Pending property edit enum and application logic.
//!
//! Edits are collected during UI rendering and applied after all reads
//! are done to avoid borrow conflicts.

use nalgebra::Vector2;
use crate::core::body::BodyGeometry;
use crate::forces::elements::ForceElement;
use super::force_editor::draw_force_elements_panel;
use crate::gui::state::{AppState, SelectedEntity};

/// Pending edit collected during UI rendering, applied after all reads
/// are done to avoid borrow conflicts.
#[allow(dead_code)] // RenameMountPoint is wired but no UI triggers it yet
pub(super) enum PendingPropertyEdit {
    Mass { body_id: String, value: f64 },
    Izz { body_id: String, value: f64 },
    RemoveForce(usize),
    UpdateForce { index: usize, force: ForceElement },
    LinkLength { body_id: String, point_a: String, point_b: String, length: f64 },
    LinkOrientation { body_id: String, point_a: String, point_b: String, angle_rad: f64 },
    SetEditorBody(String),
    AddMountPoint { body_id: String, name: String, position: [f64; 2] },
    DeleteMountPoint { body_id: String, name: String },
    RenameMountPoint { body_id: String, old_name: String, new_name: String },
    UpdateMountPointPosition { body_id: String, name: String, position: [f64; 2] },
    AddGeometry { body_id: String, width: f64, height: f64 },
    UpdateGeometryWidth { body_id: String, width: f64 },
    UpdateGeometryHeight { body_id: String, height: f64 },
    UpdateGeometryOffsetX { body_id: String, offset_x: f64 },
    UpdateGeometryOffsetY { body_id: String, offset_y: f64 },
    RemoveGeometry { body_id: String },
    UpdateLabel { body_id: String, label: String },
    UpdateGroundPivot { name: String, x: f64, y: f64 },
    /// Commit a typed / drag-stopped weight mass (kg).
    SetPointMassMass { body_id: String, weight_id: String, mass: f64 },
    /// Commit a typed / drag-stopped body-local X or Y of a weight.
    SetPointMassPosition { body_id: String, weight_id: String, local_pos: [f64; 2] },
    /// Commit a typed weight name (trimmed; blank clears it).
    SetPointMassLabel { body_id: String, weight_id: String, label: Option<String> },
    /// Add a weight of `AppState::last_point_mass_kg` at `local_pos` on
    /// `body_id` and select it (the link editor's Add weight button).
    AddPointMass { body_id: String, local_pos: [f64; 2] },
    /// Delete a weight; a selection of it is cleared.
    RemovePointMass { body_id: String, weight_id: String },
    /// Enter mode to reassign a point mass to a different body.
    ReassignPointMass { body_id: String, weight_id: String },
    /// Enter mode to reposition a point mass via mouse click.
    RepositionPointMass { body_id: String, weight_id: String },
    /// Uniformly scale the entire mechanism by a factor.
    ScaleMechanism { factor: f64 },
    /// Undo the last action.
    Undo,
    /// Redo the last undone action.
    Redo,
    /// Enter canvas placement mode to add a new attachment point to a body.
    AddJointPoint { body_id: String },
    /// Enter draw-body-geometry mode: user drags on canvas to define geometry.
    EnterDrawGeometryMode { body_id: String },
    /// Delete a body and all joints connected to it.
    DeleteBody { body_id: String },
    /// Convert a `LinearActuator` force element at the given index into
    /// a `LinearDriver` constraint. Removes the original force element,
    /// strips any revolute drivers, and adds a new linear driver
    /// targeting the same body/point pair so stroke-driven analysis
    /// becomes available.
    ConvertActuatorToLinearDriver { index: usize },
}

/// Draw the force elements collapsible section.
pub(super) fn draw_force_elements_inner(
    ui: &mut eframe::egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    ui.separator();
    let force_color = state.nc(eframe::egui::Color32::from_rgb(255, 140, 60));
    eframe::egui::CollapsingHeader::new(
        eframe::egui::RichText::new("Force Elements").color(force_color),
    )
        .id_salt("force_elements_section")
        .default_open(true)
        .show(ui, |ui| {
            draw_force_elements_panel(ui, state, pending);
        });
}

/// Apply a pending property edit.
pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
    if let Some(edit) = pending {
        match edit {
            PendingPropertyEdit::Mass { body_id, value } => {
                state.set_body_mass(&body_id, value);
            }
            PendingPropertyEdit::Izz { body_id, value } => {
                state.set_body_izz(&body_id, value);
            }
            PendingPropertyEdit::RemoveForce(idx) => {
                state.remove_force_element(idx);
            }
            PendingPropertyEdit::UpdateForce { index, force } => {
                state.update_force_element(index, force);
            }
            PendingPropertyEdit::LinkLength { body_id, point_a, point_b, length } => {
                state.set_link_length(&body_id, &point_a, &point_b, length);
            }
            PendingPropertyEdit::LinkOrientation { body_id, point_a, point_b, angle_rad } => {
                state.set_link_orientation(&body_id, &point_a, &point_b, angle_rad);
            }
            PendingPropertyEdit::SetEditorBody(body_id) => {
                state.link_editor_body = Some(body_id);
            }
            PendingPropertyEdit::AddMountPoint { body_id, name, position } => {
                state.add_mount_point(&body_id, &name, position);
            }
            PendingPropertyEdit::DeleteMountPoint { body_id, name } => {
                let cleared = state.delete_mount_point(&body_id, &name);
                if cleared > 0 {
                    log::warn!(
                        "Mount point '{}' removed — {} force ref(s) reverted to fixed coordinates",
                        name, cleared
                    );
                }
            }
            PendingPropertyEdit::RenameMountPoint { body_id, old_name, new_name } => {
                state.rename_mount_point(&body_id, &old_name, &new_name);
            }
            PendingPropertyEdit::UpdateMountPointPosition { body_id, name, position } => {
                state.update_mount_point_position(&body_id, &name, position);
            }
            PendingPropertyEdit::AddGeometry { body_id, width, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                        });
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                        });
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateLabel { body_id, label } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.label = if label.is_empty() { None } else { Some(label.clone()) };
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.label = label;
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGroundPivot { name, x, y } => {
                state.update_ground_pivot_position(&name, x, y);
            }
            PendingPropertyEdit::SetPointMassMass { body_id, weight_id, mass } => {
                state.set_point_mass_mass(&body_id, &weight_id, mass);
            }
            PendingPropertyEdit::SetPointMassPosition { body_id, weight_id, local_pos } => {
                state.move_point_mass(&body_id, &weight_id, &body_id, local_pos);
            }
            PendingPropertyEdit::SetPointMassLabel { body_id, weight_id, label } => {
                state.set_point_mass_label(&body_id, &weight_id, label);
            }
            PendingPropertyEdit::AddPointMass { body_id, local_pos } => {
                if let Some(weight_id) = state.add_point_mass(&body_id, state.last_point_mass_kg, local_pos) {
                    state.multi_selected.clear();
                    state.selected = Some(SelectedEntity::Weight { body_id, weight_id });
                }
            }
            PendingPropertyEdit::RemovePointMass { body_id, weight_id } => {
                if state.remove_point_mass_by_id(&body_id, &weight_id) {
                    let removed = SelectedEntity::Weight { body_id, weight_id };
                    if state.selected.as_ref() == Some(&removed) {
                        state.selected = None;
                    }
                    state.multi_selected.retain(|e| *e != removed);
                }
            }
            PendingPropertyEdit::ReassignPointMass { body_id, weight_id } => {
                state.reassigning_point_mass = Some((body_id, weight_id));
                state.repositioning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::RepositionPointMass { body_id, weight_id } => {
                state.repositioning_point_mass = Some((body_id, weight_id));
                state.reassigning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::ScaleMechanism { factor } => {
                state.scale_mechanism(factor);
            }
            PendingPropertyEdit::Undo => {
                state.undo();
            }
            PendingPropertyEdit::Redo => {
                state.redo();
            }
            PendingPropertyEdit::AddJointPoint { body_id } => {
                state.adding_joint_point = Some(body_id);
                state.reassigning_point_mass = None;
                state.repositioning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::EnterDrawGeometryMode { body_id } => {
                state.drawing_body_geometry = Some(
                    crate::gui::state::DrawBodyGeometryState {
                        body_id,
                        start_world: None,
                    },
                );
                state.active_tool = crate::gui::state::EditorTool::DrawBodyGeometry;
                state.status_message = Some("Drag on canvas to draw body geometry".to_string());
            }
            PendingPropertyEdit::DeleteBody { body_id } => {
                // Mirror the canvas context menu: clear selection + Link
                // Editor focus so the UI doesn't point at a missing body.
                state.remove_body(&body_id);
                state.selected = None;
                if state.link_editor_body.as_deref() == Some(body_id.as_str()) {
                    state.link_editor_body = None;
                }
            }
            PendingPropertyEdit::ConvertActuatorToLinearDriver { index } => {
                state.convert_actuator_to_linear_driver(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::GROUND_ID;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::EditorTool;
    use crate::gui::test_support::sorted_link_ids;

    /// Four-bar with one 2 kg weight on the first (sorted) link. Returns the
    /// state, that link's id, a second link's id and the weight id.
    fn four_bar_with_weight() -> (AppState, String, String, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let links = sorted_link_ids(&state);
        let weight = state.add_point_mass(&links[0], 2.0, [0.03, 0.02]).unwrap();
        (state, links[0].clone(), links[1].clone(), weight)
    }

    #[test]
    fn weight_edits_apply_by_id_as_one_undo_step_each() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassMass {
                body_id: body.clone(),
                weight_id: w.clone(),
                mass: 5.0,
            }),
        );
        assert_eq!(state.find_point_mass(&body, &w).unwrap().mass, 5.0);
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassPosition {
                body_id: body.clone(),
                weight_id: w.clone(),
                local_pos: [-0.01, 0.04],
            }),
        );
        assert_eq!(state.find_point_mass(&body, &w).unwrap().local_pos, [-0.01, 0.04]);
        assert_eq!(state.undo_history.undo_count(), depth + 2);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::RemovePointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert!(state.find_point_mass(&body, &w).is_none());
        assert_eq!(state.undo_history.undo_count(), depth + 3);
    }

    #[test]
    fn reassign_and_reposition_arm_canvas_modes_by_weight_id() {
        let (mut state, body, _, w) = four_bar_with_weight();
        state.active_tool = EditorTool::PlaceMass;

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::ReassignPointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert_eq!(state.reassigning_point_mass, Some((body.clone(), w.clone())));
        assert_eq!(state.repositioning_point_mass, None);
        assert_eq!(state.active_tool, EditorTool::Select);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::RepositionPointMass { body_id: body.clone(), weight_id: w.clone() }),
        );
        assert_eq!(state.repositioning_point_mass, Some((body, w)));
        assert_eq!(state.reassigning_point_mass, None);
    }

    #[test]
    fn add_point_mass_uses_the_last_mass_and_selects_the_new_weight() {
        let (mut state, body, other, w) = four_bar_with_weight();
        state.last_point_mass_kg = 4.5;
        state.multi_selected = vec![SelectedEntity::Weight { body_id: body.clone(), weight_id: w }];
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::AddPointMass { body_id: other.clone(), local_pos: [0.01, -0.02] }),
        );

        let pm = state.find_point_mass(&other, "W2").expect("the next id");
        assert_eq!(pm.mass, 4.5);
        assert_eq!(pm.local_pos, [0.01, -0.02]);
        assert_eq!(state.selected, Some(SelectedEntity::Weight { body_id: other, weight_id: "W2".to_string() }));
        assert!(state.multi_selected.is_empty());
        assert_eq!(state.undo_history.undo_count(), depth + 1);
    }

    #[test]
    fn add_point_mass_on_ground_changes_nothing() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let selected = Some(SelectedEntity::Weight { body_id: body, weight_id: w });
        state.selected = selected.clone();
        let depth = state.undo_history.undo_count();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::AddPointMass { body_id: GROUND_ID.to_string(), local_pos: [0.0, 0.0] }),
        );

        assert_eq!(state.selected, selected, "the selection stays");
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn set_point_mass_label_trims_and_clears_as_one_undo_step_each() {
        let (mut state, body, _, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();
        let label = |state: &AppState| state.find_point_mass(&body, &w).unwrap().label.clone();

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassLabel {
                body_id: body.clone(),
                weight_id: w.clone(),
                label: Some("  Robot torso ".to_string()),
            }),
        );
        assert_eq!(label(&state).as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassLabel {
                body_id: body.clone(),
                weight_id: w.clone(),
                label: Some("   ".to_string()),
            }),
        );
        assert_eq!(label(&state), None, "a blank name shows the id again");
        assert_eq!(state.undo_history.undo_count(), depth + 2);
    }

    #[test]
    fn removing_the_selected_weight_clears_it_from_the_selection() {
        let (mut state, body, other, w) = four_bar_with_weight();
        let removed = SelectedEntity::Weight { body_id: body.clone(), weight_id: w.clone() };
        let kept = SelectedEntity::Body(other);
        state.selected = Some(removed.clone());
        state.multi_selected = vec![removed, kept.clone()];

        apply_pending(&mut state, Some(PendingPropertyEdit::RemovePointMass { body_id: body, weight_id: w }));

        assert_eq!(state.selected, None);
        assert_eq!(state.multi_selected, vec![kept]);
    }

    #[test]
    fn stale_weight_edit_is_a_no_op() {
        let (mut state, body, other, w) = four_bar_with_weight();
        let depth = state.undo_history.undo_count();
        // The weight moved away (e.g. by an earlier edit) before this one applied.
        assert!(state.move_point_mass(&body, &w, &other, [0.0, 0.0]));
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetPointMassMass { body_id: body, weight_id: w.clone(), mass: 9.0 }),
        );
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }
}
