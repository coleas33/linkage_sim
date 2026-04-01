//! Property panel for the selected entity.
//!
//! Mass and inertia properties are editable for non-ground bodies when a
//! blueprint is available. Edits are applied via `AppState::set_body_mass`
//! and `AppState::set_body_izz`, which mutate the blueprint and rebuild.

mod pending_edits;
mod diagnostics;
pub(super) mod force_editor;

use eframe::egui;
use crate::core::state::GROUND_ID;
use crate::gui::state::AppState;

use pending_edits::{PendingPropertyEdit, apply_pending, draw_force_elements_inner};
use diagnostics::draw_diagnostics_section;

/// Draw the property panel showing info about the selected entity.
///
/// Mass and inertia fields are editable via `DragValue` widgets when a
/// blueprint is present and the selected body is not ground.
pub fn draw_property_panel(ui: &mut egui::Ui, state: &mut AppState) {
    let mut pending: Option<PendingPropertyEdit> = None;

    let Some(mech) = &state.mechanism else {
        ui.label("No mechanism loaded.");
        return;
    };

    // ── Compact mechanism summary ─────────────────────────────────────
    {
        let n_bodies = mech.bodies().len().saturating_sub(1);
        let n_joints = mech.joints().len();
        let total_mass: f64 = mech.bodies().values()
            .filter(|b| b.id != GROUND_ID)
            .map(|b| b.mass)
            .sum();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.small(format!("{} bodies", n_bodies));
            ui.small("\u{2022}");
            ui.small(format!("{} joints", n_joints));
            ui.small("\u{2022}");
            ui.small(format!("{:.2} kg", total_mass));
        });
    }

    // ── Link Editor (always visible, dropdown to pick body) ───────────
    let body_ids: Vec<String> = mech.body_order().to_vec();

    // Auto-select first body if none selected
    if state.link_editor_body.is_none() && !body_ids.is_empty() {
        // Can't mutate here (mech borrows state), use pending
        pending = Some(PendingPropertyEdit::SetEditorBody(body_ids[0].clone()));
    }

    let link_color = state.nc(egui::Color32::from_rgb(70, 150, 240));
    egui::CollapsingHeader::new(
        egui::RichText::new("Link Editor").color(link_color),
    )
        .id_salt("link_editor")
        .default_open(true)
        .show(ui, |ui| {
            // Body selector dropdown
            let current_label = state.link_editor_body.as_deref().unwrap_or("(none)");
            egui::ComboBox::from_label("Body")
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for bid in &body_ids {
                        let is_selected = state.link_editor_body.as_deref() == Some(bid.as_str());
                        if ui.selectable_label(is_selected, bid).clicked() {
                            pending = Some(PendingPropertyEdit::SetEditorBody(bid.clone()));
                        }
                    }
                });

            // Show editor for the selected body
            if let Some(body_id) = &state.link_editor_body {
                let body_id = body_id.clone();
                if let Some(body) = mech.bodies().get(&body_id) {
                    let mech_state = mech.state();
                    let q = &state.q;
                    let units = &state.display_units;

                    // ── Label ────────────────────────────────────────────
                    if body_id != GROUND_ID {
                        let mut label = body.label.clone();
                        if ui.text_edit_singleline(&mut label).changed() {
                            pending = Some(PendingPropertyEdit::UpdateLabel {
                                body_id: body_id.clone(),
                                label,
                            });
                        }
                    }

                    if body_id != GROUND_ID {
                        let (x, y, theta) = mech_state.get_pose(&body_id, q);
                        ui.label(format!(
                            "Pos: ({:.3}, {:.3}){}  \u{2220} {:.1}{}",
                            units.length(x), units.length(y), units.length_suffix(),
                            units.angle(theta), units.angle_suffix()
                        ));
                    }

                    // ── Geometry: lengths + orientations ──────────────────
                    let mut pts: Vec<_> = body.attachment_points.iter().collect();
                    pts.sort_by_key(|(name, _)| name.as_str());

                    if pts.len() >= 2 && body_id != GROUND_ID {
                        ui.separator();

                        // Collect segments: (name_a, name_b, length, angle_rad)
                        let mut segments: Vec<(&str, &str, f64, f64)> = Vec::new();
                        for pair in pts.windows(2) {
                            let (na, pa) = &pair[0];
                            let (nb, pb) = &pair[1];
                            let dx = pb.x - pa.x;
                            let dy = pb.y - pa.y;
                            let len = (dx*dx+dy*dy).sqrt();
                            let angle = dy.atan2(dx);
                            segments.push((na.as_str(), nb.as_str(), len, angle));
                        }
                        if pts.len() >= 3 {
                            let (na, pa) = pts.last().unwrap();
                            let (nb, pb) = &pts[0];
                            let dx = pb.x - pa.x;
                            let dy = pb.y - pa.y;
                            let len = (dx*dx+dy*dy).sqrt();
                            let angle = dy.atan2(dx);
                            segments.push((na.as_str(), nb.as_str(), len, angle));
                        }

                        for (na, nb, len, angle) in &segments {
                            let seg_label = format!("{}\u{2192}{}", na, nb);

                            // Length slider
                            let mut display_len = units.length(*len);
                            let lr = ui.add(
                                egui::Slider::new(&mut display_len, units.length(0.001)..=units.length(2.0))
                                    .text(&seg_label)
                                    .suffix(units.length_suffix())
                                    .clamping(egui::SliderClamping::Never)
                                    .logarithmic(true),
                            );
                            if lr.drag_stopped() || (lr.changed() && !lr.dragged()) {
                                pending = Some(PendingPropertyEdit::LinkLength {
                                    body_id: body_id.clone(),
                                    point_a: na.to_string(), point_b: nb.to_string(),
                                    length: units.length_to_si(display_len),
                                });
                            }

                            // Angle slider
                            let mut display_angle = units.angle(*angle);
                            let angle_speed = if matches!(units.angle, crate::gui::state::AngleUnit::Degrees) { 0.5 } else { 0.01 };
                            let ar = ui.add(
                                egui::Slider::new(&mut display_angle, units.angle(-std::f64::consts::PI)..=units.angle(std::f64::consts::PI))
                                    .text(format!("{} angle", seg_label))
                                    .suffix(units.angle_suffix())
                                    .clamping(egui::SliderClamping::Never)
                                    .step_by(angle_speed),
                            );
                            if ar.drag_stopped() || (ar.changed() && !ar.dragged()) {
                                pending = Some(PendingPropertyEdit::LinkOrientation {
                                    body_id: body_id.clone(),
                                    point_a: na.to_string(), point_b: nb.to_string(),
                                    angle_rad: units.angle_to_si(display_angle),
                                });
                            }
                        }
                    }

                    // ── Mass & Inertia ────────────────────────────────────
                    if body_id != GROUND_ID {
                        ui.separator();
                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                let mut mass = bp_body.mass;
                                let mr = ui.add(
                                    egui::Slider::new(&mut mass, 0.0..=100.0)
                                        .text("mass").suffix(" kg")
                                        .clamping(egui::SliderClamping::Never)
                                        .logarithmic(true),
                                ).on_hover_text("Body mass in kg \u{2014} affects inertial loads and reaction forces");
                                if mr.drag_stopped() || (mr.changed() && !mr.dragged()) {
                                    pending = Some(PendingPropertyEdit::Mass {
                                        body_id: body_id.clone(), value: mass,
                                    });
                                }

                                let mut izz = bp_body.izz_cg;
                                let ir = ui.add(
                                    egui::Slider::new(&mut izz, 0.0..=10.0)
                                        .text("Izz").suffix(" kg\u{00b7}m\u{00b2}")
                                        .clamping(egui::SliderClamping::Never)
                                        .logarithmic(true),
                                ).on_hover_text("Moment of inertia about the CG in kg\u{00b7}m\u{00b2} \u{2014} affects angular acceleration");
                                if ir.drag_stopped() || (ir.changed() && !ir.dragged()) {
                                    pending = Some(PendingPropertyEdit::Izz {
                                        body_id: body_id.clone(), value: izz,
                                    });
                                }
                            }
                        }
                    }

                    // ── Point Masses ─────────────────────────────────────
                    if body_id != GROUND_ID {
                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                if !bp_body.point_masses.is_empty() {
                                    ui.separator();
                                    let pm_color = state.nc(egui::Color32::from_rgb(255, 200, 50));
                                    egui::CollapsingHeader::new(
                                        egui::RichText::new(format!(
                                            "Point Masses ({})", bp_body.point_masses.len()
                                        )).color(pm_color),
                                    )
                                        .id_salt(format!("point_masses_{}", body_id))
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            let units = &state.display_units;
                                            for (i, pm) in bp_body.point_masses.iter().enumerate() {
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("#{}", i + 1));

                                                    let mut mass_val = pm.mass;
                                                    let mr = ui.add(
                                                        egui::DragValue::new(&mut mass_val)
                                                            .speed(0.01)
                                                            .range(0.001..=1000.0)
                                                            .prefix("m: ")
                                                            .suffix(" kg"),
                                                    ).on_hover_text("Point mass magnitude in kg");
                                                    if mr.drag_stopped() || (mr.changed() && !mr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: mass_val,
                                                            local_pos: pm.local_pos,
                                                        });
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(20.0);
                                                    let mut x_display = units.length(pm.local_pos[0]);
                                                    let xr = ui.add(
                                                        egui::DragValue::new(&mut x_display)
                                                            .speed(units.length(0.001))
                                                            .prefix("X ")
                                                            .suffix(units.length_suffix()),
                                                    ).on_hover_text("Body-local X position");
                                                    if xr.drag_stopped() || (xr.changed() && !xr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: pm.mass,
                                                            local_pos: [units.length_to_si(x_display), pm.local_pos[1]],
                                                        });
                                                    }

                                                    let mut y_display = units.length(pm.local_pos[1]);
                                                    let yr = ui.add(
                                                        egui::DragValue::new(&mut y_display)
                                                            .speed(units.length(0.001))
                                                            .prefix("Y ")
                                                            .suffix(units.length_suffix()),
                                                    ).on_hover_text("Body-local Y position");
                                                    if yr.drag_stopped() || (yr.changed() && !yr.dragged()) {
                                                        pending = Some(PendingPropertyEdit::UpdatePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                            mass: pm.mass,
                                                            local_pos: [pm.local_pos[0], units.length_to_si(y_display)],
                                                        });
                                                    }

                                                    if ui.small_button("x")
                                                        .on_hover_text("Remove this point mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::RemovePointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
                                                    }
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.add_space(20.0);
                                                    if ui.small_button("Move to Link")
                                                        .on_hover_text("Click a different link to reassign this mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::ReassignPointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
                                                    }
                                                    if ui.small_button("Reposition")
                                                        .on_hover_text("Click on the canvas to move this mass")
                                                        .clicked()
                                                    {
                                                        pending = Some(PendingPropertyEdit::RepositionPointMass {
                                                            body_id: body_id.clone(),
                                                            index: i,
                                                        });
                                                    }
                                                });
                                            }
                                        });
                                }
                            }
                        }
                    }

                    // ── Body Geometry ─────────────────────────────────────
                    if body_id != GROUND_ID {
                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                egui::CollapsingHeader::new("Geometry")
                                    .id_salt(format!("body_geometry_{}", body_id))
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        if let Some(ref geo) = bp_body.geometry {
                                            // Width slider (mm display, m internal)
                                            let mut width_mm = geo.width * 1e3;
                                            let wr = ui.add(
                                                egui::Slider::new(&mut width_mm, 1.0..=500.0)
                                                    .text("Width (mm)")
                                                    .clamping(egui::SliderClamping::Never)
                                                    .logarithmic(true),
                                            ).on_hover_text("Body rectangle width in mm (visual geometry for force zones)");
                                            if wr.drag_stopped() || (wr.changed() && !wr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryWidth {
                                                    body_id: body_id.clone(),
                                                    width: width_mm * 1e-3,
                                                });
                                            }

                                            // Height slider (mm display, m internal)
                                            let mut height_mm = geo.height * 1e3;
                                            let hr = ui.add(
                                                egui::Slider::new(&mut height_mm, 1.0..=500.0)
                                                    .text("Height (mm)")
                                                    .clamping(egui::SliderClamping::Never)
                                                    .logarithmic(true),
                                            ).on_hover_text("Body rectangle height in mm (visual geometry for force zones)");
                                            if hr.drag_stopped() || (hr.changed() && !hr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryHeight {
                                                    body_id: body_id.clone(),
                                                    height: height_mm * 1e-3,
                                                });
                                            }

                                            // Offset X slider (mm display, m internal)
                                            let mut ox_mm = geo.offset.x * 1e3;
                                            let oxr = ui.add(
                                                egui::Slider::new(&mut ox_mm, -250.0..=250.0)
                                                    .text("Offset X (mm)"),
                                            ).on_hover_text("Horizontal offset of geometry rectangle from body origin in mm");
                                            if oxr.drag_stopped() || (oxr.changed() && !oxr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryOffsetX {
                                                    body_id: body_id.clone(),
                                                    offset_x: ox_mm * 1e-3,
                                                });
                                            }

                                            // Offset Y slider (mm display, m internal)
                                            let mut oy_mm = geo.offset.y * 1e3;
                                            let oyr = ui.add(
                                                egui::Slider::new(&mut oy_mm, -250.0..=250.0)
                                                    .text("Offset Y (mm)"),
                                            ).on_hover_text("Vertical offset of geometry rectangle from body origin in mm");
                                            if oyr.drag_stopped() || (oyr.changed() && !oyr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryOffsetY {
                                                    body_id: body_id.clone(),
                                                    offset_y: oy_mm * 1e-3,
                                                });
                                            }

                                            if ui.button("Remove Geometry").on_hover_text("Remove the visual geometry rectangle from this body").clicked() {
                                                pending = Some(PendingPropertyEdit::RemoveGeometry {
                                                    body_id: body_id.clone(),
                                                });
                                            }
                                        } else {
                                            if ui.button("Add Geometry").on_hover_text("Attach a visual geometry rectangle to this body (required for force zones)").clicked() {
                                                pending = Some(PendingPropertyEdit::AddGeometry {
                                                    body_id: body_id.clone(),
                                                    width: 0.03,
                                                    height: 0.01,
                                                });
                                            }
                                        }
                                    });
                            }
                        }
                    }

                    // ── Mount Points ──────────────────────────────────────
                    if let Some(bp) = &state.blueprint {
                        if let Some(body_json) = bp.bodies.get(&body_id) {
                            ui.separator();
                            egui::CollapsingHeader::new("Mount Points")
                                .id_salt(format!("mount_points_{}", body_id))
                                .default_open(true)
                                .show(ui, |ui| {
                                    let mut sorted_names: Vec<&String> =
                                        body_json.mount_points.keys().collect();
                                    sorted_names.sort();
                                    for name in &sorted_names {
                                        let pos = body_json.mount_points[*name];
                                        ui.horizontal(|ui| {
                                            ui.label(name.as_str());
                                            let mut x = pos[0];
                                            let mut y = pos[1];
                                            if ui
                                                .add(
                                                    egui::DragValue::new(&mut x)
                                                        .speed(0.001)
                                                        .prefix("x: ")
                                                        .suffix(" m"),
                                                )
                                                .changed()
                                            {
                                                pending =
                                                    Some(PendingPropertyEdit::UpdateMountPointPosition {
                                                        body_id: body_id.clone(),
                                                        name: (*name).clone(),
                                                        position: [x, pos[1]],
                                                    });
                                            }
                                            if ui
                                                .add(
                                                    egui::DragValue::new(&mut y)
                                                        .speed(0.001)
                                                        .prefix("y: ")
                                                        .suffix(" m"),
                                                )
                                                .changed()
                                            {
                                                pending =
                                                    Some(PendingPropertyEdit::UpdateMountPointPosition {
                                                        body_id: body_id.clone(),
                                                        name: (*name).clone(),
                                                        position: [pos[0], y],
                                                    });
                                            }
                                            if ui.small_button("x").on_hover_text("Delete this mount point").clicked() {
                                                pending = Some(PendingPropertyEdit::DeleteMountPoint {
                                                    body_id: body_id.clone(),
                                                    name: (*name).clone(),
                                                });
                                            }
                                        });
                                    }
                                    if ui.button("+ Add Mount Point").on_hover_text("Add a new force attachment mount point on this body").clicked() {
                                        // Find next unused M<N> name, skipping collisions
                                        // with both mount_points and attachment_points
                                        let mut next_num = 1u32;
                                        while body_json
                                            .mount_points
                                            .contains_key(&format!("M{}", next_num))
                                            || body_json
                                                .attachment_points
                                                .contains_key(&format!("M{}", next_num))
                                        {
                                            next_num += 1;
                                        }
                                        let new_name = format!("M{}", next_num);
                                        pending = Some(PendingPropertyEdit::AddMountPoint {
                                            body_id: body_id.clone(),
                                            name: new_name,
                                            position: [body_json.cg_local[0], body_json.cg_local[1]],
                                        });
                                    }
                                });
                        }
                    }
                }
            }
        });

    // ── Ground Pivots ─────────────────────────────────────────────────
    draw_ground_pivots_section(ui, state, &mut pending);

    // ── Scale Mechanism ──────────────────────────────────────────────
    draw_scale_mechanism_section(ui, state, &mut pending);

    // ── Diagnostics (collapsed) ───────────────────────────────────────
    draw_diagnostics_section(ui, state);

    // ── Joint Reactions (live, always visible) ──────────────────────
    let react_color = state.nc(egui::Color32::from_rgb(255, 100, 100));
    egui::CollapsingHeader::new(
        egui::RichText::new("Joint Reactions").color(react_color),
    )
        .id_salt("joint_reactions_section")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(torque) = state.force_results.driver_torque {
                ui.label(format!("Driver Torque: {:.4} N\u{00b7}m", torque));
            } else {
                ui.label("Driver Torque: \u{2014}");
            }
            if let Some(ma) = state.force_results.mechanical_advantage {
                ui.label(format!("Mech. Advantage: {:.3}", ma));
            }

            if state.force_results.joint_reactions.is_empty() {
                ui.label("No reaction data (need driver)");
            } else {
                let mut ids: Vec<&String> = state.force_results.joint_reactions.keys().collect();
                ids.sort();
                for jid in ids {
                    let (fx, fy) = state.force_results.joint_reactions[jid];
                    let mag = (fx * fx + fy * fy).sqrt();
                    ui.label(format!("{}: {:.2} N  ({:.2}, {:.2})", jid, mag, fx, fy));
                }
            }
        });

    // ── Force Elements section ─────────────────────────────────────────
    draw_force_elements_inner(ui, state, &mut pending);

    // --- Apply any pending edits (mutable borrow now safe) ---
    apply_pending(state, pending);
}

/// Draw the "Ground Pivots" collapsing section with editable X/Y fields
/// for each ground attachment point, plus distance & angle for the ground link.
fn draw_ground_pivots_section(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let Some(bp) = &state.blueprint else { return };
    let Some(ground) = bp.bodies.get(GROUND_ID) else { return };
    if ground.attachment_points.is_empty() {
        return;
    }

    let units = &state.display_units;

    let pivot_color = state.nc(egui::Color32::from_rgb(180, 130, 220));
    egui::CollapsingHeader::new(
        egui::RichText::new("Ground Pivots").color(pivot_color),
    )
        .id_salt("ground_pivots_section")
        .default_open(true)
        .show(ui, |ui| {
            // Sorted pivot names for stable ordering
            let mut pivot_names: Vec<&String> = ground.attachment_points.keys().collect();
            pivot_names.sort();

            // ── Per-pivot X / Y editing ────────────────────────────────
            for name in &pivot_names {
                let pos = ground.attachment_points[*name];
                ui.horizontal(|ui| {
                    ui.label(format!("{}:", name));
                    let mut x_display = units.length(pos[0]);
                    let mut y_display = units.length(pos[1]);

                    let xr = ui.add(
                        egui::DragValue::new(&mut x_display)
                            .speed(units.length(0.001))
                            .prefix("X ")
                            .suffix(units.length_suffix()),
                    );
                    if xr.drag_stopped() || (xr.changed() && !xr.dragged()) {
                        *pending = Some(PendingPropertyEdit::UpdateGroundPivot {
                            name: (*name).clone(),
                            x: units.length_to_si(x_display),
                            y: pos[1],
                        });
                    }

                    let yr = ui.add(
                        egui::DragValue::new(&mut y_display)
                            .speed(units.length(0.001))
                            .prefix("Y ")
                            .suffix(units.length_suffix()),
                    );
                    if yr.drag_stopped() || (yr.changed() && !yr.dragged()) {
                        *pending = Some(PendingPropertyEdit::UpdateGroundPivot {
                            name: (*name).clone(),
                            x: pos[0],
                            y: units.length_to_si(y_display),
                        });
                    }
                });
            }

            // ── Ground link distance & angle (when exactly 2 pivots) ──
            if pivot_names.len() == 2 {
                let pos_a = ground.attachment_points[pivot_names[0]];
                let pos_b = ground.attachment_points[pivot_names[1]];

                let dx = pos_b[0] - pos_a[0];
                let dy = pos_b[1] - pos_a[1];
                let distance = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);

                ui.separator();
                ui.label(format!(
                    "Ground Link: {} \u{2192} {}",
                    pivot_names[0], pivot_names[1]
                ));

                let mut dist_display = units.length(distance);
                let dr = ui.add(
                    egui::DragValue::new(&mut dist_display)
                        .speed(units.length(0.001))
                        .prefix("Dist ")
                        .suffix(units.length_suffix())
                        .range(units.length(0.001)..=units.length(10.0)),
                );
                if dr.drag_stopped() || (dr.changed() && !dr.dragged()) {
                    let new_dist = units.length_to_si(dist_display);
                    let new_x = pos_a[0] + new_dist * angle.cos();
                    let new_y = pos_a[1] + new_dist * angle.sin();
                    *pending = Some(PendingPropertyEdit::UpdateGroundPivot {
                        name: pivot_names[1].clone(),
                        x: new_x,
                        y: new_y,
                    });
                }

                let mut angle_display = units.angle(angle);
                let ar = ui.add(
                    egui::DragValue::new(&mut angle_display)
                        .speed(if matches!(units.angle, crate::gui::state::AngleUnit::Degrees) { 0.5 } else { 0.01 })
                        .prefix("Angle ")
                        .suffix(units.angle_suffix()),
                );
                if ar.drag_stopped() || (ar.changed() && !ar.dragged()) {
                    let new_angle = units.angle_to_si(angle_display);
                    let new_x = pos_a[0] + distance * new_angle.cos();
                    let new_y = pos_a[1] + distance * new_angle.sin();
                    *pending = Some(PendingPropertyEdit::UpdateGroundPivot {
                        name: pivot_names[1].clone(),
                        x: new_x,
                        y: new_y,
                    });
                }
            }
        });
}

/// Draw the "Scale Mechanism" collapsing section with preset buttons
/// and a custom percentage input.
fn draw_scale_mechanism_section(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    if state.blueprint.is_none() {
        return;
    }

    let scale_color = state.nc(egui::Color32::from_rgb(120, 200, 140));
    egui::CollapsingHeader::new(
        egui::RichText::new("Scale Mechanism").color(scale_color),
    )
        .id_salt("scale_mechanism_section")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("50%").on_hover_text("Shrink to half size").clicked() {
                    *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: 0.5 });
                }
                if ui.button("75%").on_hover_text("Shrink to 75%").clicked() {
                    *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: 0.75 });
                }
                if ui.button("150%").on_hover_text("Enlarge to 150%").clicked() {
                    *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: 1.5 });
                }
                if ui.button("200%").on_hover_text("Enlarge to double size").clicked() {
                    *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: 2.0 });
                }
            });

            // Custom percentage via DragValue (applied on lost_focus / Enter)
            let id = ui.id().with("scale_pct_custom");
            let mut scale_pct: f64 = ui.data_mut(|d| *d.get_temp_mut_or(id, 100.0));
            let response = ui.add(
                egui::DragValue::new(&mut scale_pct)
                    .suffix("%")
                    .speed(1.0)
                    .range(1.0..=1000.0)
            ).on_hover_text("Type a percentage and press Enter to scale the mechanism");
            ui.data_mut(|d| *d.get_temp_mut_or(id, 100.0) = scale_pct);
            if response.lost_focus() && (scale_pct - 100.0).abs() > 0.01 {
                *pending = Some(PendingPropertyEdit::ScaleMechanism { factor: scale_pct / 100.0 });
                // Reset to 100% after applying
                ui.data_mut(|d| *d.get_temp_mut_or(id, 100.0) = 100.0);
            }
        });
}
