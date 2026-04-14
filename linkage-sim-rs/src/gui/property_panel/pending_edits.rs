//! Pending property edit enum and application logic.
//!
//! Edits are collected during UI rendering and applied after all reads
//! are done to avoid borrow conflicts.

use nalgebra::Vector2;
use crate::core::body::BodyGeometry;
use crate::forces::elements::ForceElement;
use super::force_editor::draw_force_elements_panel;
use crate::gui::state::AppState;

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
    UpdatePointMass { body_id: String, index: usize, mass: f64, local_pos: [f64; 2] },
    RemovePointMass { body_id: String, index: usize },
    /// Enter mode to reassign a point mass to a different body.
    ReassignPointMass { body_id: String, index: usize },
    /// Enter mode to reposition a point mass via mouse click.
    RepositionPointMass { body_id: String, index: usize },
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
            PendingPropertyEdit::UpdatePointMass { body_id, index, mass, local_pos } => {
                state.update_point_mass(&body_id, index, mass, local_pos);
            }
            PendingPropertyEdit::RemovePointMass { body_id, index } => {
                state.remove_point_mass(&body_id, index);
            }
            PendingPropertyEdit::ReassignPointMass { body_id, index } => {
                state.reassigning_point_mass = Some((body_id, index));
                state.repositioning_point_mass = None;
                state.active_tool = crate::gui::state::EditorTool::Select;
            }
            PendingPropertyEdit::RepositionPointMass { body_id, index } => {
                state.repositioning_point_mass = Some((body_id, index));
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
        }
    }
}
