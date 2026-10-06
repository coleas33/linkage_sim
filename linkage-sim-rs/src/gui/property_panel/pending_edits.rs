//! Pending property edit enum and application logic.
//!
//! Edits are collected during UI rendering and applied after all reads
//! are done to avoid borrow conflicts.

use nalgebra::Vector2;
use crate::core::body::{BodyGeometry, GeometryShape};
use crate::forces::elements::ForceElement;
use super::force_editor::draw_force_elements_panel;
use crate::gui::state::{AppState, SelectedEntity};

/// Pending edit collected during UI rendering, applied after all reads
/// are done to avoid borrow conflicts.
#[allow(dead_code)] // RenameMountPoint is wired but no UI triggers it yet; nor is EnterDrawGeometryMode (no UI produces it since the Draw Geometry button was replaced; BL-044)
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
    /// Give a link without geometry a `shape` sized from the link (decision R-2).
    AddGeometry { body_id: String, shape: GeometryShape },
    /// Switch a link's geometry to `shape`, keeping its width; the height becomes
    /// equal to it (a square, or a circle of that diameter).
    SetGeometryShape { body_id: String, shape: GeometryShape },
    /// Set a circle's diameter (width and height).
    UpdateGeometryDiameter { body_id: String, diameter: f64 },
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

/// Apply `edit` to `body_id`'s geometry in the blueprint and in the built
/// mechanism (the two copies the GUI keeps), then mark the sweep dirty. Each
/// geometry edit is one undo step and marks the file changed (`push_undo`
/// sets `dirty`, which the web autosave needs).
fn edit_geometry(state: &mut AppState, body_id: &str, edit: impl Fn(&mut Option<BodyGeometry>)) {
    state.push_undo();
    if let Some(body) = state.blueprint.as_mut().and_then(|bp| bp.bodies.get_mut(body_id)) {
        edit(&mut body.geometry);
    }
    if let Some(body) = state.mechanism.as_mut().and_then(|mech| mech.body_mut(body_id)) {
        edit(&mut body.geometry);
    }
    state.mark_sweep_dirty();
}

/// A new `shape` for `body` (decision R-2): centred on its attachment points'
/// centroid (body-local) and sized from their span L, the largest distance
/// between two of them (0.1 m when not positive and finite, e.g. fewer than two points): a rectangle L x L/4, a
/// circle of diameter L/2.
fn default_geometry(body: &crate::io::BodyJson, shape: GeometryShape) -> BodyGeometry {
    let points: Vec<Vector2<f64>> =
        body.attachment_points.values().map(|p| Vector2::new(p[0], p[1])).collect();
    let centre = if points.is_empty() {
        Vector2::zeros()
    } else {
        points.iter().sum::<Vector2<f64>>() / points.len() as f64
    };
    let span = points
        .iter()
        .flat_map(|a| points.iter().map(move |b| (a - b).norm()))
        .fold(0.0, f64::max);
    let span = if span > 0.0 && span.is_finite() { span } else { 0.1 };
    match shape {
        GeometryShape::Rectangle => BodyGeometry::new(span, span / 4.0, centre),
        GeometryShape::Circle => BodyGeometry::circle(span / 2.0, centre),
    }
    .expect("a positive span gives a valid shape")
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
            PendingPropertyEdit::AddGeometry { body_id, shape } => {
                let new = state
                    .blueprint
                    .as_ref()
                    .and_then(|bp| bp.bodies.get(&body_id))
                    .map(|body| default_geometry(body, shape));
                if let Some(new) = new {
                    edit_geometry(state, &body_id, |geo| *geo = Some(new.clone()));
                }
            }
            PendingPropertyEdit::SetGeometryShape { body_id, shape } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.shape = shape;
                        g.height = g.width;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryDiameter { body_id, diameter } => {
                if diameter > 0.0 && diameter.is_finite() {
                    edit_geometry(state, &body_id, |geo| {
                        if let Some(g) = geo {
                            g.width = diameter;
                            g.height = diameter;
                        }
                    });
                }
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                if width > 0.0 && width.is_finite() {
                    edit_geometry(state, &body_id, |geo| {
                        if let Some(g) = geo {
                            g.width = width;
                        }
                    });
                }
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                if height > 0.0 && height.is_finite() {
                    edit_geometry(state, &body_id, |geo| {
                        if let Some(g) = geo {
                            g.height = height;
                        }
                    });
                }
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.x = offset_x;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.y = offset_y;
                    }
                });
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                edit_geometry(state, &body_id, |geo| *geo = None);
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

    // ── Geometry shapes (decision R-2) ──────────────────────────────────

    /// `body`'s geometry in the blueprint and in the built mechanism.
    fn geometry_copies(state: &AppState, body: &str) -> [BodyGeometry; 2] {
        let bp = state.blueprint.as_ref().unwrap().bodies[body].geometry.clone();
        let mech = state.mechanism.as_ref().unwrap().bodies()[body].geometry.clone();
        [bp.expect("blueprint geometry"), mech.expect("mechanism geometry")]
    }

    /// The Four-Bar sample's coupler: its length and its midpoint (body-local).
    fn coupler_span(state: &AppState) -> (f64, [f64; 2]) {
        let points: Vec<[f64; 2]> =
            state.blueprint.as_ref().unwrap().bodies["coupler"].attachment_points.values().cloned().collect();
        assert_eq!(points.len(), 2, "the sample coupler is a two-point bar");
        let (a, b) = (points[0], points[1]);
        let span = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        (span, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
    }

    fn add_geometry(state: &mut AppState, shape: GeometryShape) {
        apply_pending(state, Some(PendingPropertyEdit::AddGeometry { body_id: "coupler".into(), shape }));
    }

    #[test]
    fn add_circle_centres_a_wheel_half_the_link_long_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span / 2.0).abs() < 1e-12 && (geo.height - span / 2.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn add_rectangle_spans_the_link_a_quarter_as_deep() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span / 4.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn switching_shape_keeps_the_width_and_squares_the_height() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, _) = coupler_span(&state);
        let set = |state: &mut AppState, shape| {
            apply_pending(state, Some(PendingPropertyEdit::SetGeometryShape { body_id: "coupler".into(), shape }));
        };
        set(&mut state, GeometryShape::Circle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
        set(&mut state, GeometryShape::Rectangle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
    }

    #[test]
    fn a_diameter_edit_sets_width_and_height_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::UpdateGeometryDiameter { body_id: "coupler".into(), diameter: 0.05 }),
        );
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!((geo.width, geo.height), (0.05, 0.05));
        }
        apply_pending(&mut state, Some(PendingPropertyEdit::RemoveGeometry { body_id: "coupler".into() }));
        assert!(state.blueprint.as_ref().unwrap().bodies["coupler"].geometry.is_none());
        assert!(state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.is_none());
    }

    #[test]
    fn geometry_edits_are_one_undo_step_each_and_mark_the_file_changed() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.dirty = false;
        let depth = state.undo_history.undo_count();
        add_geometry(&mut state, GeometryShape::Rectangle);
        assert_eq!(state.undo_history.undo_count(), depth + 1);
        assert!(state.dirty, "a geometry edit marks the file changed");
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetGeometryShape { body_id: "coupler".into(), shape: GeometryShape::Circle }),
        );
        assert_eq!(state.undo_history.undo_count(), depth + 2);
        apply_pending(&mut state, Some(PendingPropertyEdit::Undo));
        let (span, _) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span / 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn non_positive_or_non_finite_sizes_are_ignored() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        let before = geometry_copies(&state, "coupler");
        let bad = [0.0, -0.01, f64::NAN, f64::INFINITY];
        for value in bad {
            apply_pending(
                &mut state,
                Some(PendingPropertyEdit::UpdateGeometryDiameter { body_id: "coupler".into(), diameter: value }),
            );
            for (geo, old) in geometry_copies(&state, "coupler").iter().zip(&before) {
                assert_eq!((geo.width, geo.height), (old.width, old.height), "diameter {value}");
            }
        }
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::SetGeometryShape { body_id: "coupler".into(), shape: GeometryShape::Rectangle }),
        );
        let before = geometry_copies(&state, "coupler");
        for value in bad {
            apply_pending(
                &mut state,
                Some(PendingPropertyEdit::UpdateGeometryWidth { body_id: "coupler".into(), width: value }),
            );
            apply_pending(
                &mut state,
                Some(PendingPropertyEdit::UpdateGeometryHeight { body_id: "coupler".into(), height: value }),
            );
            for (geo, old) in geometry_copies(&state, "coupler").iter().zip(&before) {
                assert_eq!((geo.width, geo.height), (old.width, old.height), "size {value}");
            }
        }
    }

    #[test]
    fn default_geometry_sizes_a_link_with_fewer_than_two_points() {
        let one = serde_json::from_str::<crate::io::BodyJson>(
            r#"{"attachment_points":{"A":[0.3,0.1]},"mass":1.0,"cg_local":[0.0,0.0],"izz_cg":0.0}"#,
        )
        .unwrap();
        let none = serde_json::from_str::<crate::io::BodyJson>(
            r#"{"attachment_points":{},"mass":1.0,"cg_local":[0.0,0.0],"izz_cg":0.0}"#,
        )
        .unwrap();
        for (body, centre) in [(&one, [0.3, 0.1]), (&none, [0.0, 0.0])] {
            let circle = default_geometry(body, GeometryShape::Circle);
            assert!((circle.width - 0.05).abs() < 1e-12 && (circle.height - 0.05).abs() < 1e-12);
            assert!((circle.offset.x - centre[0]).abs() < 1e-12 && (circle.offset.y - centre[1]).abs() < 1e-12);
            let rect = default_geometry(body, GeometryShape::Rectangle);
            assert!((rect.width - 0.1).abs() < 1e-12 && (rect.height - 0.025).abs() < 1e-12);
        }
    }
}
