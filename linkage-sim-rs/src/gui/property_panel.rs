//! Property panel for the selected entity.
//!
//! Mass and inertia properties are editable for non-ground bodies when a
//! blueprint is available. Edits are applied via `AppState::set_body_mass`
//! and `AppState::set_body_izz`, which mutate the blueprint and rebuild.

use eframe::egui;
use meval;
use crate::analysis::envelopes::compute_envelope;
use crate::analysis::grashof::GrashofType;
use crate::analysis::motor_sizing::check_motor_sizing;
use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use crate::forces::elements::*;
use crate::solver::statics::reaction_to_local;
use crate::io::serialization::JointJson;
use super::state::{AppState, SelectedEntity}; // DisplayUnits used via state.display_units

/// Pending edit collected during UI rendering, applied after all reads
/// are done to avoid borrow conflicts.
enum PendingPropertyEdit {
    Mass { body_id: String, value: f64 },
    Izz { body_id: String, value: f64 },
    RemoveForce(usize),
    UpdateForce { index: usize, force: ForceElement },
    SetDriver(String),
    AddPointMass { body_id: String, mass: f64, local_pos: [f64; 2] },
    RemovePointMass { body_id: String, index: usize },
    UpdatePrismaticAxis { joint_id: String, axis: [f64; 2] },
    LinkLength { body_id: String, point_a: String, point_b: String, length: f64 },
    LinkOrientation { body_id: String, point_a: String, point_b: String, angle_rad: f64 },
    SelectBody(String),
    Deselect,
}

/// Draw the property panel showing info about the selected entity.
///
/// Mass and inertia fields are editable via `DragValue` widgets when a
/// blueprint is present and the selected body is not ground.
pub fn draw_property_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Properties");

    // ── Compact mechanism summary (always visible) ─────────────────────
    if let Some(mech) = &state.mechanism {
        let n_bodies = mech.bodies().len().saturating_sub(1); // exclude ground
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
        ui.separator();
    }

    // ── Diagnostics section (always shown when a mechanism is loaded) ───
    draw_diagnostics_section(ui, state);

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
            ui.label("Click a link on the canvas to edit it.");
            ui.separator();
            ui.strong("Quick select:");
            let body_ids: Vec<String> = mech.body_order().to_vec();
            for bid in &body_ids {
                if ui.button(bid).clicked() {
                    // Can't mutate state inside immutable borrow block, so
                    // use pending to defer selection.
                    pending = Some(PendingPropertyEdit::SelectBody(bid.clone()));
                }
            }
            // Still need to draw force elements below, so don't return.
            // Fall through with no selected entity content.
            ui.separator();

            // Skip to force elements section
            let _ = mech;
            draw_force_elements_inner(ui, state, &mut pending);
            apply_pending(state, pending);
            return;
        };

        match selected {
            SelectedEntity::Body(body_id) => {
                let body_id = body_id.clone();
                if let Some(body) = mech.bodies().get(&body_id) {
                    let mech_state = mech.state();
                    let q = &state.q;
                    let units = &state.display_units;

                    // ── Link Editor header ─────────────────────────────────
                    ui.horizontal(|ui| {
                        ui.heading(format!("Link Editor: {}", body_id));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("\u{2715} Close").clicked() {
                                pending = Some(PendingPropertyEdit::Deselect);
                            }
                        });
                    });
                    ui.separator();

                    if body_id != GROUND_ID {
                        let (x, y, theta) = mech_state.get_pose(&body_id, q);
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "Pos: ({:.3}, {:.3}){}  Angle: {:.1}{}",
                                units.length(x), units.length(y), units.length_suffix(),
                                units.angle(theta), units.angle_suffix()
                            ));
                        });
                    }

                    // ── Geometry: lengths + orientations ──────────────────
                    let mut pts: Vec<_> = body.attachment_points.iter().collect();
                    pts.sort_by_key(|(name, _)| name.as_str());

                    if pts.len() >= 2 && body_id != GROUND_ID {
                        ui.separator();
                        ui.strong("Geometry");

                        // Build segments with length and angle
                        let mut segments: Vec<(&str, &str, f64, f64)> = Vec::new();
                        for pair in pts.windows(2) {
                            let (name_a, pt_a) = &pair[0];
                            let (name_b, pt_b) = &pair[1];
                            let dx = pt_b.x - pt_a.x;
                            let dy = pt_b.y - pt_a.y;
                            let len = (dx * dx + dy * dy).sqrt();
                            let angle = dy.atan2(dx);
                            segments.push((name_a.as_str(), name_b.as_str(), len, angle));
                        }
                        if pts.len() >= 3 {
                            let (name_a, pt_a) = pts.last().unwrap();
                            let (name_b, pt_b) = &pts[0];
                            let dx = pt_b.x - pt_a.x;
                            let dy = pt_b.y - pt_a.y;
                            let len = (dx * dx + dy * dy).sqrt();
                            let angle = dy.atan2(dx);
                            segments.push((name_a.as_str(), name_b.as_str(), len, angle));
                        }

                        for (name_a, name_b, len, angle) in &segments {
                            ui.label(format!("  {}\u{2192}{}", name_a, name_b));

                            // Length slider — only rebuild on release
                            let mut display_len = units.length(*len);
                            let len_resp = ui.add(
                                egui::Slider::new(&mut display_len, units.length(0.001)..=units.length(2.0))
                                    .text("length")
                                    .suffix(units.length_suffix())
                                    .clamping(egui::SliderClamping::Never)
                                    .logarithmic(true),
                            );
                            if len_resp.drag_stopped() || (len_resp.changed() && !len_resp.dragged()) {
                                let new_si = units.length_to_si(display_len);
                                pending = Some(PendingPropertyEdit::LinkLength {
                                    body_id: body_id.clone(),
                                    point_a: name_a.to_string(),
                                    point_b: name_b.to_string(),
                                    length: new_si,
                                });
                            }

                            // Orientation slider — only rebuild on release
                            let mut angle_deg = angle.to_degrees();
                            let ang_resp = ui.add(
                                egui::Slider::new(&mut angle_deg, -180.0..=180.0)
                                    .text("angle")
                                    .suffix("\u{00B0}")
                                    .step_by(0.5),
                            );
                            if ang_resp.drag_stopped() || (ang_resp.changed() && !ang_resp.dragged()) {
                                let new_angle_rad = angle_deg.to_radians();
                                pending = Some(PendingPropertyEdit::LinkOrientation {
                                    body_id: body_id.clone(),
                                    point_a: name_a.to_string(),
                                    point_b: name_b.to_string(),
                                    angle_rad: new_angle_rad,
                                });
                            }
                        }
                    }

                    // ── Mass & Inertia (editable sliders) ─────────────────────
                    if body_id != GROUND_ID {
                        ui.separator();
                        ui.strong("Mass & Inertia");

                        if let Some(bp) = &state.blueprint {
                            if let Some(bp_body) = bp.bodies.get(&body_id) {
                                let mut mass = bp_body.mass;
                                let mut izz = bp_body.izz_cg;

                                ui.label("Mass");
                                let mass_resp = ui.add(
                                    egui::Slider::new(&mut mass, 0.0..=100.0)
                                        .suffix(" kg")
                                        .clamping(egui::SliderClamping::Never)
                                        .logarithmic(true),
                                );
                                if mass_resp.drag_stopped() || (mass_resp.changed() && !mass_resp.dragged()) {
                                    pending = Some(PendingPropertyEdit::Mass {
                                        body_id: body_id.clone(),
                                        value: mass,
                                    });
                                }

                                ui.label("Inertia (Izz)");
                                let izz_resp = ui.add(
                                    egui::Slider::new(&mut izz, 0.0..=10.0)
                                        .suffix(" kg\u{00b7}m\u{00b2}")
                                        .clamping(egui::SliderClamping::Never)
                                        .logarithmic(true),
                                );
                                if izz_resp.drag_stopped() || (izz_resp.changed() && !izz_resp.dragged()) {
                                    pending = Some(PendingPropertyEdit::Izz {
                                        body_id: body_id.clone(),
                                        value: izz,
                                    });
                                }
                            }
                        }

                        ui.label(format!(
                            "CG local: ({:.3}, {:.3}){}",
                            units.length(body.cg_local.x),
                            units.length(body.cg_local.y),
                            units.length_suffix()
                        ));

                        // Point masses
                        egui::CollapsingHeader::new("Point Masses")
                            .id_salt("point_masses")
                            .show(ui, |ui| {
                                if let Some(bp) = &state.blueprint {
                                    if let Some(bp_body) = bp.bodies.get(&body_id) {
                                        if bp_body.point_masses.is_empty() {
                                            ui.label("None");
                                        }
                                        let mut remove_idx = None;
                                        for (idx, pm) in bp_body.point_masses.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                ui.label(format!(
                                                    "{:.3} kg at ({:.1}, {:.1}){}",
                                                    pm.mass,
                                                    units.length(pm.local_pos[0]),
                                                    units.length(pm.local_pos[1]),
                                                    units.length_suffix()
                                                ));
                                                if ui.small_button("\u{2715}").on_hover_text("Remove").clicked() {
                                                    remove_idx = Some(idx);
                                                }
                                            });
                                        }
                                        if let Some(idx) = remove_idx {
                                            pending = Some(PendingPropertyEdit::RemovePointMass {
                                                body_id: body_id.clone(),
                                                index: idx,
                                            });
                                        }
                                        if ui.small_button("+ Point Mass").clicked() {
                                            pending = Some(PendingPropertyEdit::AddPointMass {
                                                body_id: body_id.clone(),
                                                mass: 1.0,
                                                local_pos: [0.0, 0.0],
                                            });
                                        }
                                    }
                                }
                            });
                    }

                    // ── Attachment points (read-only, collapsed) ──────────────
                    if body_id != GROUND_ID {
                        egui::CollapsingHeader::new("Attachment Points")
                            .id_salt("attach_pts")
                            .show(ui, |ui| {
                                for (name, local) in &pts {
                                    let global = mech_state.body_point_global(&body_id, local, q);
                                    ui.label(format!(
                                        "{}: ({:.3}, {:.3}){}",
                                        name,
                                        units.length(global.x),
                                        units.length(global.y),
                                        units.length_suffix()
                                    ));
                                }
                            });
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
                    let units = &state.display_units;
                    ui.label(format!(
                        "Position: ({:.3}, {:.3}){}",
                        units.length(global_pos.x),
                        units.length(global_pos.y),
                        units.length_suffix()
                    ));

                    // Prismatic axis editing (only for prismatic joints with a blueprint).
                    if joint.is_prismatic() {
                        if let Some(bp) = &state.blueprint {
                            if let Some(bpj) = bp.joints.get(joint_id) {
                                if let JointJson::Prismatic { axis_local_i, .. } = bpj {
                                    ui.separator();
                                    ui.strong("Slide Axis (body i local):");
                                    let mut ax = axis_local_i[0];
                                    let mut ay = axis_local_i[1];
                                    let mut changed = false;
                                    ui.horizontal(|ui| {
                                        ui.label("x:");
                                        if ui
                                            .add(egui::DragValue::new(&mut ax).speed(0.01).range(-1.0..=1.0))
                                            .changed()
                                        {
                                            changed = true;
                                        }
                                        ui.label("y:");
                                        if ui
                                            .add(egui::DragValue::new(&mut ay).speed(0.01).range(-1.0..=1.0))
                                            .changed()
                                        {
                                            changed = true;
                                        }
                                    });
                                    if changed {
                                        // Normalize the axis
                                        let len = (ax * ax + ay * ay).sqrt();
                                        if len > 1e-12 {
                                            ax /= len;
                                            ay /= len;
                                        }
                                        pending = Some(PendingPropertyEdit::UpdatePrismaticAxis {
                                            joint_id: joint_id.clone(),
                                            axis: [ax, ay],
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Show reaction forces for this joint if available.
                    if let Some(&(fx, fy)) = state.force_results.joint_reactions.get(joint_id) {
                        ui.separator();
                        ui.strong("Reaction Forces:");
                        ui.label(format!("Fx: {:.4} N", fx));
                        ui.label(format!("Fy: {:.4} N", fy));
                        let resultant = (fx * fx + fy * fy).sqrt();
                        ui.label(format!("Resultant: {:.4} N", resultant));

                        // Local-frame reactions (body i frame)
                        let theta_i = if mech_state.is_ground(joint.body_i_id()) {
                            0.0
                        } else {
                            mech_state.get_angle(joint.body_i_id(), q)
                        };
                        let local = reaction_to_local([fx, fy], theta_i);
                        ui.separator();
                        ui.strong("Local Frame (body i):");
                        ui.label(format!("Fx_local: {:.4} N", local[0]));
                        ui.label(format!("Fy_local: {:.4} N", local[1]));
                    }

                    // "Set as Driver" button for grounded revolute joints.
                    let grounded_ids = mech.grounded_revolute_joint_ids();
                    if grounded_ids.contains(&joint_id.to_string()) {
                        ui.separator();
                        let is_current =
                            state.driver_joint_id.as_deref() == Some(joint_id.as_str());
                        if is_current {
                            ui.label("(Current driver)");
                        } else if ui.button("Set as Driver").clicked() {
                            pending = Some(PendingPropertyEdit::SetDriver(joint_id.to_string()));
                        }
                    }
                }
            }
            SelectedEntity::Driver(_driver_id) => {
                ui.label("Driver properties not yet available.");
            }
        }

        // Show driver torque regardless of what is selected.
        if let Some(torque) = state.force_results.driver_torque {
            ui.separator();
            ui.strong("Driver Torque:");
            ui.label(format!("{:.4} N\u{00b7}m", torque));
        }

        // Show mechanical advantage regardless of what is selected.
        if let Some(ma) = state.force_results.mechanical_advantage {
            if state.force_results.driver_torque.is_none() {
                ui.separator();
            }
            ui.strong("Mechanical Advantage:");
            ui.label(format!("{:.3}", ma));
        }
    } // end immutable borrow block

    // ── Force Elements section ─────────────────────────────────────────
    draw_force_elements_inner(ui, state, &mut pending);

    // --- Apply any pending edits (mutable borrow now safe) ---
    apply_pending(state, pending);
}

/// Draw the force elements collapsible section.
fn draw_force_elements_inner(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    ui.separator();
    egui::CollapsingHeader::new("Force Elements")
        .id_salt("force_elements_section")
        .default_open(true)
        .show(ui, |ui| {
            draw_force_elements_panel(ui, state, pending);
        });
}

/// Apply a pending property edit.
fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
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
            PendingPropertyEdit::SetDriver(joint_id) => {
                state.pending_driver_reassignment = Some(joint_id);
            }
            PendingPropertyEdit::AddPointMass { body_id, mass, local_pos } => {
                state.add_point_mass(&body_id, mass, local_pos);
            }
            PendingPropertyEdit::RemovePointMass { body_id, index } => {
                state.remove_point_mass(&body_id, index);
            }
            PendingPropertyEdit::UpdatePrismaticAxis { joint_id, axis } => {
                state.update_prismatic_axis(&joint_id, axis);
            }
            PendingPropertyEdit::LinkLength { body_id, point_a, point_b, length } => {
                state.set_link_length(&body_id, &point_a, &point_b, length);
            }
            PendingPropertyEdit::LinkOrientation { body_id, point_a, point_b, angle_rad } => {
                state.set_link_orientation(&body_id, &point_a, &point_b, angle_rad);
            }
            PendingPropertyEdit::SelectBody(body_id) => {
                state.selected = Some(SelectedEntity::Body(body_id));
            }
            PendingPropertyEdit::Deselect => {
                state.selected = None;
            }
        }
    }
}

// ── Diagnostics section ──────────────────────────────────────────────────

/// Draw Grashof classification, crank recommendation, Jacobian conditioning,
/// and sweep envelope diagnostics.
///
/// Shows a collapsible "Diagnostics" header containing:
/// - Grashof classification and link lengths (4-bar mechanisms only)
/// - Crank recommendation (which link to drive for maximum rotation)
/// - Constraint Jacobian condition number (when forces have been solved)
/// - Sweep envelope statistics (torque min/max/RMS when sweep data available)
fn draw_diagnostics_section(ui: &mut egui::Ui, state: &AppState) {
    let mech = match &state.mechanism {
        Some(m) => m,
        None => return,
    };

    let has_grashof = state.grashof_result.is_some();
    let has_crank_rec = state.crank_recommendation.is_some();
    let has_condition = state.force_results.condition_number.is_some();
    // Always show diagnostics when a mechanism is loaded (mass summary is
    // always available).
    egui::CollapsingHeader::new("Diagnostics")
        .default_open(false)
        .show(ui, |ui| {
            // ── Mechanism mass summary ─────────────────────────────
            let total_mass: f64 = mech.bodies().values()
                .filter(|b| b.id != GROUND_ID)
                .map(|b| b.mass)
                .sum();
            let total_izz: f64 = mech.bodies().values()
                .filter(|b| b.id != GROUND_ID)
                .map(|b| b.izz_cg)
                .sum();
            ui.label(format!("Total mass: {:.4} kg", total_mass));
            ui.label(format!(
                "Total Izz (body CGs): {:.6} kg\u{00b7}m\u{00b2}",
                total_izz
            ));

            // ── Grashof classification ──────────────────────────────
            if let Some(ref gr) = state.grashof_result {
                ui.separator();
                let (label, is_ok) = match gr.classification {
                    GrashofType::CrankRocker => ("Crank-Rocker", true),
                    GrashofType::DoubleCrank => ("Double-Crank", true),
                    GrashofType::DoubleRocker => ("Double-Rocker", true),
                    GrashofType::ChangePoint => ("Change-Point", false),
                    GrashofType::NonGrashof => ("Non-Grashof", false),
                };

                let units = &state.display_units;

                ui.horizontal(|ui| {
                    ui.label("Grashof:");
                    if is_ok {
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 100),
                            label,
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 180, 60),
                            label,
                        );
                    }
                });

                let [ground, crank, coupler, rocker] = gr.link_lengths;
                ui.label(format!(
                    "  Ground: {:.2}{}  Crank: {:.2}{}",
                    units.length(ground),
                    units.length_suffix(),
                    units.length(crank),
                    units.length_suffix(),
                ));
                ui.label(format!(
                    "  Coupler: {:.2}{}  Rocker: {:.2}{}",
                    units.length(coupler),
                    units.length_suffix(),
                    units.length(rocker),
                    units.length_suffix(),
                ));
            }

            // ── Crank recommendation ─────────────────────────────
            if let Some(ref rec) = state.crank_recommendation {
                if !has_grashof {
                    // Grashof section already added a separator; only add one
                    // when it was skipped.
                    ui.separator();
                }

                let drive_color = if rec.full_rotation {
                    egui::Color32::from_rgb(100, 200, 100) // full rotation
                } else {
                    egui::Color32::from_rgb(220, 180, 60)  // limited range
                };

                ui.horizontal(|ui| {
                    ui.label("Drive:");
                    ui.colored_label(
                        drive_color,
                        format!("'{}'", rec.recommended_link),
                    );
                    if rec.full_rotation {
                        ui.label("(360\u{00b0})");
                    }
                });

                // Show reason for the best candidate.
                if let Some(best) = rec.candidates.first() {
                    ui.label(format!("  {}", best.reason));
                }

                // If other candidates exist with different capabilities, note them.
                for candidate in rec.candidates.iter().skip(1) {
                    if candidate.can_fully_rotate != rec.full_rotation {
                        ui.label(format!(
                            "  Alt: '{}' ~{:.0}\u{00b0}",
                            candidate.link_name, candidate.estimated_range_deg
                        ));
                    }
                }
            }

            // ── Jacobian conditioning ──────────────────────────────
            if let Some(kappa) = state.force_results.condition_number {
                if !has_grashof && !has_crank_rec {
                    // Previous sections already added separators; only add one
                    // when they were all skipped.
                    ui.separator();
                }

                let color = if kappa < 1e4 {
                    egui::Color32::from_rgb(100, 200, 100) // well-conditioned
                } else if kappa < 1e8 {
                    egui::Color32::from_rgb(220, 180, 60) // moderate
                } else {
                    egui::Color32::from_rgb(220, 80, 80) // ill-conditioned
                };

                ui.horizontal(|ui| {
                    ui.label("Conditioning:");
                    ui.colored_label(color, format!("\u{03ba} = {:.2e}", kappa));
                });

                if state.force_results.is_overconstrained {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 180, 60),
                        "  Overconstrained (pseudo-inverse used)",
                    );
                }
            }

            // ── Sweep envelope statistics ────────────────────────────
            if let Some(ref sweep) = state.sweep_data {
                if let Some(ref torques) = sweep.driver_torques {
                    if let Some(env) = compute_envelope(torques) {
                        if !has_grashof && !has_crank_rec && !has_condition {
                            ui.separator();
                        }

                        ui.label(format!(
                            "Torque: {:.3} to {:.3} N\u{00b7}m",
                            env.min_value, env.max_value
                        ));
                        ui.label(format!("RMS: {:.3} N\u{00b7}m", env.rms));
                    }
                }
            }

            // ── Force contributions ────────────────────────────────
            if !state.force_results.force_contributions.is_empty() {
                ui.separator();
                ui.strong("Force Contributions:");
                let max_norm = state
                    .force_results
                    .force_contributions
                    .iter()
                    .map(|(_, n)| *n)
                    .fold(0.0_f64, f64::max);
                for (name, norm) in &state.force_results.force_contributions {
                    let bar_frac = if max_norm > 0.0 {
                        (*norm / max_norm).min(1.0)
                    } else {
                        0.0
                    };
                    ui.horizontal(|ui| {
                        ui.label(format!("{}: {:.4}", name, norm));
                        let bar = egui::ProgressBar::new(bar_frac as f32)
                            .desired_width(60.0);
                        ui.add(bar);
                    });
                }
            }

            // ── Virtual work cross-check ─────────────────────────────
            if let Some((vw_torque, lm_torque, agrees)) = state.force_results.virtual_work_check {
                ui.separator();
                ui.strong("Virtual Work Check:");
                let color = if agrees {
                    egui::Color32::from_rgb(100, 200, 100)
                } else {
                    egui::Color32::from_rgb(220, 80, 80)
                };
                ui.colored_label(
                    color,
                    if agrees { "\u{2713} Agrees" } else { "\u{2717} Disagrees" },
                );
                ui.label(format!("VW torque: {:.4} N\u{00b7}m", vw_torque));
                ui.label(format!("\u{03bb} torque:  {:.4} N\u{00b7}m", lm_torque));
            }

            // ── Motor sizing feasibility ────────────────────────────
            draw_motor_sizing_diagnostic(ui, state);
        });

    ui.separator();
}

/// Draw motor sizing feasibility when sweep data and a MotorElement are both present.
///
/// Extracts the motor's stall_torque and no_load_speed from the force elements,
/// gets the sweep's driver angular velocities and inverse dynamics torques,
/// runs `check_motor_sizing`, and displays the result.
fn draw_motor_sizing_diagnostic(ui: &mut egui::Ui, state: &AppState) {
    // Need sweep data with inverse dynamics torques.
    let sweep = match state.sweep_data.as_ref() {
        Some(s) if !s.inverse_dynamics_torques.is_empty() => s,
        _ => return,
    };

    // Find the first MotorElement in the blueprint forces.
    let bp = match state.blueprint.as_ref() {
        Some(bp) => bp,
        None => return,
    };

    let motor = bp.forces.iter().find_map(|f| match f {
        ForceElement::Motor(m) => Some(m),
        _ => None,
    });

    let motor = match motor {
        Some(m) if m.no_load_speed > 0.0 => m,
        _ => return,
    };

    // Build the speed array for each sweep step.
    // The driver angular velocity is constant (driver_omega) for a constant-speed driver.
    let omega = state.driver_omega;
    let n = sweep.inverse_dynamics_torques.len();
    let speeds: Vec<f64> = vec![omega; n];

    // Filter out NaN torques (failed solves) -- use 0.0 as fallback.
    let torques: Vec<f64> = sweep
        .inverse_dynamics_torques
        .iter()
        .map(|&t| if t.is_finite() { t } else { 0.0 })
        .collect();

    let result = check_motor_sizing(&speeds, &torques, motor.stall_torque, motor.no_load_speed);

    ui.separator();
    if result.all_feasible {
        ui.horizontal(|ui| {
            ui.label("Motor:");
            ui.colored_label(
                egui::Color32::from_rgb(100, 200, 100),
                "\u{2713} all feasible",
            );
        });
        ui.label(format!(
            "  Worst margin: {:.0}% at {:.0}\u{00b0}",
            result.worst_margin * 100.0,
            sweep.angles_deg.get(result.worst_index).unwrap_or(&0.0),
        ));
    } else {
        ui.horizontal(|ui| {
            ui.label("Motor:");
            ui.colored_label(
                egui::Color32::from_rgb(220, 80, 80),
                format!(
                    "\u{2717} worst margin = {:.0}% at {:.0}\u{00b0}",
                    result.worst_margin * 100.0,
                    sweep.angles_deg.get(result.worst_index).unwrap_or(&0.0),
                ),
            );
        });
    }
}

// ── Force Elements panel ─────────────────────────────────────────────────

/// Draw the force elements section of the property panel.
///
/// Lists all non-gravity force elements with editable parameters in
/// collapsing headers, plus "Add ..." buttons for creating new elements.
/// Reads from the blueprint to avoid borrow conflicts with the mechanism.
fn draw_force_elements_panel(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let Some(bp) = &state.blueprint else {
        ui.label("No blueprint loaded.");
        return;
    };

    // List existing force elements (skip Gravity -- toggled via View menu).
    let mut visible_count = 0u32;
    for (bp_idx, force) in bp.forces.iter().enumerate() {
        if matches!(force, ForceElement::Gravity(_)) {
            continue;
        }
        visible_count += 1;

        let header_label = format!("{} #{}", force.type_name(), visible_count);
        egui::CollapsingHeader::new(&header_label)
            .id_salt(format!("force_{}", bp_idx))
            .show(ui, |ui| {
                draw_force_element_details(ui, bp_idx, force, pending);

                if ui.small_button("Remove").clicked() {
                    *pending = Some(PendingPropertyEdit::RemoveForce(bp_idx));
                }
            });
    }

    if visible_count == 0 {
        ui.label("No force elements.");
    }

    // Force elements are added via the toolbar ribbon (force_toolbar.rs).
}

/// Draw editable parameter fields for a single force element.
///
/// When a `DragValue` changes, the current element is cloned with the
/// modified parameter and set as an `UpdateForce` pending edit.
fn draw_force_element_details(
    ui: &mut egui::Ui,
    index: usize,
    force: &ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    match force {
        ForceElement::Gravity(_) => {} // skipped in caller

        ForceElement::LinearSpring(s) => {
            ui.label(format!("Body A: {}  Body B: {}", s.body_a, s.body_b));

            let mut stiffness = s.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if ui
                    .add(
                        egui::DragValue::new(&mut stiffness)
                            .speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N/m"),
                    )
                    .changed()
                {
                    let mut updated = s.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearSpring(updated),
                    });
                }
            });

            let mut free_len = s.free_length;
            ui.horizontal(|ui| {
                ui.label("L\u{2080}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut free_len)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"),
                    )
                    .changed()
                {
                    let mut updated = s.clone();
                    updated.free_length = free_len;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearSpring(updated),
                    });
                }
            });

            draw_point_fields(ui, "Pt A", &s.point_a, index, |pt| {
                let mut updated = s.clone();
                updated.point_a = pt;
                ForceElement::LinearSpring(updated)
            }, pending);

            draw_point_fields(ui, "Pt B", &s.point_b, index, |pt| {
                let mut updated = s.clone();
                updated.point_b = pt;
                ForceElement::LinearSpring(updated)
            }, pending);
        }

        ForceElement::LinearDamper(d) => {
            ui.label(format!("Body A: {}  Body B: {}", d.body_a, d.body_b));

            let mut damping = d.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if ui
                    .add(
                        egui::DragValue::new(&mut damping)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}s/m"),
                    )
                    .changed()
                {
                    let mut updated = d.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearDamper(updated),
                    });
                }
            });

            draw_point_fields(ui, "Pt A", &d.point_a, index, |pt| {
                let mut updated = d.clone();
                updated.point_a = pt;
                ForceElement::LinearDamper(updated)
            }, pending);

            draw_point_fields(ui, "Pt B", &d.point_b, index, |pt| {
                let mut updated = d.clone();
                updated.point_b = pt;
                ForceElement::LinearDamper(updated)
            }, pending);
        }

        ForceElement::ExternalForce(f) => {
            ui.label(format!("Body: {}", f.body_id));

            let mut fx = f.force[0];
            let mut fy = f.force[1];
            ui.horizontal(|ui| {
                ui.label("Fx:");
                if ui
                    .add(egui::DragValue::new(&mut fx).speed(0.1).suffix(" N"))
                    .changed()
                {
                    let mut updated = f.clone();
                    updated.force[0] = fx;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalForce(updated),
                    });
                }
            });
            ui.horizontal(|ui| {
                ui.label("Fy:");
                if ui
                    .add(egui::DragValue::new(&mut fy).speed(0.1).suffix(" N"))
                    .changed()
                {
                    let mut updated = f.clone();
                    updated.force[1] = fy;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalForce(updated),
                    });
                }
            });

            draw_point_fields(ui, "Local pt", &f.local_point, index, |pt| {
                let mut updated = f.clone();
                updated.local_point = pt;
                ForceElement::ExternalForce(updated)
            }, pending);

            draw_modulation_fields(ui, index, &f.modulation, |m| {
                let mut updated = f.clone();
                updated.modulation = m;
                ForceElement::ExternalForce(updated)
            }, pending);
        }

        ForceElement::ExternalTorque(t) => {
            ui.label(format!("Body: {}", t.body_id));

            let mut torque = t.torque;
            ui.horizontal(|ui| {
                ui.label("\u{03c4}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut torque)
                            .speed(0.1)
                            .suffix(" N\u{00b7}m"),
                    )
                    .changed()
                {
                    let mut updated = t.clone();
                    updated.torque = torque;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalTorque(updated),
                    });
                }
            });

            draw_modulation_fields(ui, index, &t.modulation, |m| {
                let mut updated = t.clone();
                updated.modulation = m;
                ForceElement::ExternalTorque(updated)
            }, pending);
        }

        ForceElement::TorsionSpring(s) => {
            ui.label(format!("Body I: {}  Body J: {}", s.body_i, s.body_j));

            let mut stiffness = s.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if ui
                    .add(
                        egui::DragValue::new(&mut stiffness)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m/rad"),
                    )
                    .changed()
                {
                    let mut updated = s.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::TorsionSpring(updated),
                    });
                }
            });

            let mut free_angle = s.free_angle;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}\u{2080}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut free_angle)
                            .speed(0.01)
                            .suffix(" rad"),
                    )
                    .changed()
                {
                    let mut updated = s.clone();
                    updated.free_angle = free_angle;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::TorsionSpring(updated),
                    });
                }
            });
        }

        ForceElement::RotaryDamper(d) => {
            ui.label(format!("Body I: {}  Body J: {}", d.body_i, d.body_j));

            let mut damping = d.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if ui
                    .add(
                        egui::DragValue::new(&mut damping)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"),
                    )
                    .changed()
                {
                    let mut updated = d.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::RotaryDamper(updated),
                    });
                }
            });
        }

        ForceElement::GasSpring(gs) => {
            ui.label(format!("Body A: {}  Body B: {}", gs.body_a, gs.body_b));

            let mut initial_force = gs.initial_force;
            ui.horizontal(|ui| {
                ui.label("F\u{2080}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut initial_force)
                            .speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N"),
                    )
                    .changed()
                {
                    let mut updated = gs.clone();
                    updated.initial_force = initial_force;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut extended_length = gs.extended_length;
            ui.horizontal(|ui| {
                ui.label("L\u{2091}\u{2093}\u{209c}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut extended_length)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"),
                    )
                    .changed()
                {
                    let mut updated = gs.clone();
                    updated.extended_length = extended_length;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut stroke = gs.stroke;
            ui.horizontal(|ui| {
                ui.label("Stroke:");
                if ui
                    .add(
                        egui::DragValue::new(&mut stroke)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"),
                    )
                    .changed()
                {
                    let mut updated = gs.clone();
                    updated.stroke = stroke;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut damping = gs.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if ui
                    .add(
                        egui::DragValue::new(&mut damping)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}s/m"),
                    )
                    .changed()
                {
                    let mut updated = gs.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut polytropic_exp = gs.polytropic_exp;
            ui.horizontal(|ui| {
                ui.label("n:");
                if ui
                    .add(
                        egui::DragValue::new(&mut polytropic_exp)
                            .speed(0.01)
                            .range(0.0..=f64::MAX),
                    )
                    .changed()
                {
                    let mut updated = gs.clone();
                    updated.polytropic_exp = polytropic_exp;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            draw_point_fields(ui, "Pt A", &gs.point_a, index, |pt| {
                let mut updated = gs.clone();
                updated.point_a = pt;
                ForceElement::GasSpring(updated)
            }, pending);

            draw_point_fields(ui, "Pt B", &gs.point_b, index, |pt| {
                let mut updated = gs.clone();
                updated.point_b = pt;
                ForceElement::GasSpring(updated)
            }, pending);
        }

        ForceElement::BearingFriction(bf) => {
            ui.label(format!("Body I: {}  Body J: {}", bf.body_i, bf.body_j));

            let mut constant_drag = bf.constant_drag;
            ui.horizontal(|ui| {
                ui.label("Drag:");
                if ui
                    .add(
                        egui::DragValue::new(&mut constant_drag)
                            .speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m"),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.constant_drag = constant_drag;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut viscous_coeff = bf.viscous_coeff;
            ui.horizontal(|ui| {
                ui.label("Viscous:");
                if ui
                    .add(
                        egui::DragValue::new(&mut viscous_coeff)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.viscous_coeff = viscous_coeff;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut coulomb_coeff = bf.coulomb_coeff;
            ui.horizontal(|ui| {
                ui.label("\u{03bc}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut coulomb_coeff)
                            .speed(0.001)
                            .range(0.0..=f64::MAX),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.coulomb_coeff = coulomb_coeff;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut pin_radius = bf.pin_radius;
            ui.horizontal(|ui| {
                ui.label("Pin r:");
                if ui
                    .add(
                        egui::DragValue::new(&mut pin_radius)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.pin_radius = pin_radius;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut radial_load = bf.radial_load;
            ui.horizontal(|ui| {
                ui.label("Radial:");
                if ui
                    .add(
                        egui::DragValue::new(&mut radial_load)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N"),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.radial_load = radial_load;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut v_threshold = bf.v_threshold;
            ui.horizontal(|ui| {
                ui.label("v\u{209c}\u{2095}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut v_threshold)
                            .speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" rad/s"),
                    )
                    .changed()
                {
                    let mut updated = bf.clone();
                    updated.v_threshold = v_threshold;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });
        }

        ForceElement::JointLimit(jl) => {
            ui.label(format!("Body I: {}  Body J: {}", jl.body_i, jl.body_j));

            let mut angle_min = jl.angle_min;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}\u{2098}\u{1d62}\u{2099}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut angle_min)
                            .speed(0.01)
                            .suffix(" rad"),
                    )
                    .changed()
                {
                    let mut updated = jl.clone();
                    updated.angle_min = angle_min;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut angle_max = jl.angle_max;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}\u{2098}\u{2090}\u{2093}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut angle_max)
                            .speed(0.01)
                            .suffix(" rad"),
                    )
                    .changed()
                {
                    let mut updated = jl.clone();
                    updated.angle_max = angle_max;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut stiffness = jl.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if ui
                    .add(
                        egui::DragValue::new(&mut stiffness)
                            .speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m/rad"),
                    )
                    .changed()
                {
                    let mut updated = jl.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut damping = jl.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if ui
                    .add(
                        egui::DragValue::new(&mut damping)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"),
                    )
                    .changed()
                {
                    let mut updated = jl.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut restitution = jl.restitution;
            ui.horizontal(|ui| {
                ui.label("e:");
                if ui
                    .add(
                        egui::DragValue::new(&mut restitution)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    )
                    .changed()
                {
                    let mut updated = jl.clone();
                    updated.restitution = restitution;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });
        }

        ForceElement::Motor(m) => {
            ui.label(format!("Body I: {}  Body J: {}", m.body_i, m.body_j));

            let mut stall_torque = m.stall_torque;
            ui.horizontal(|ui| {
                ui.label("\u{03c4}\u{209b}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut stall_torque)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m"),
                    )
                    .changed()
                {
                    let mut updated = m.clone();
                    updated.stall_torque = stall_torque;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });

            let mut no_load_speed = m.no_load_speed;
            ui.horizontal(|ui| {
                ui.label("\u{03c9}\u{2080}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut no_load_speed)
                            .speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" rad/s"),
                    )
                    .changed()
                {
                    let mut updated = m.clone();
                    updated.no_load_speed = no_load_speed;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });

            let mut direction = m.direction;
            ui.horizontal(|ui| {
                ui.label("Dir:");
                if ui
                    .add(
                        egui::DragValue::new(&mut direction)
                            .speed(0.1),
                    )
                    .changed()
                {
                    let mut updated = m.clone();
                    updated.direction = direction;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });
        }

        ForceElement::LinearActuator(la) => {
            ui.label(format!("Body A: {}  Body B: {}", la.body_a, la.body_b));

            let mut force = la.force;
            ui.horizontal(|ui| {
                ui.label("F:");
                if ui
                    .add(
                        egui::DragValue::new(&mut force)
                            .speed(1.0)
                            .suffix(" N"),
                    )
                    .changed()
                {
                    let mut updated = la.clone();
                    updated.force = force;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearActuator(updated),
                    });
                }
            });

            let mut speed_limit = la.speed_limit;
            ui.horizontal(|ui| {
                ui.label("v\u{2098}\u{2090}\u{2093}:");
                if ui
                    .add(
                        egui::DragValue::new(&mut speed_limit)
                            .speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" m/s"),
                    )
                    .changed()
                {
                    let mut updated = la.clone();
                    updated.speed_limit = speed_limit;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearActuator(updated),
                    });
                }
            });

            draw_point_fields(ui, "Pt A", &la.point_a, index, |pt| {
                let mut updated = la.clone();
                updated.point_a = pt;
                ForceElement::LinearActuator(updated)
            }, pending);

            draw_point_fields(ui, "Pt B", &la.point_b, index, |pt| {
                let mut updated = la.clone();
                updated.point_b = pt;
                ForceElement::LinearActuator(updated)
            }, pending);
        }
    }
}

/// Modulation type index for the ComboBox selector.
///
/// Maps TimeModulation variants to integer indices for the UI selector.
fn modulation_type_index(m: &TimeModulation) -> usize {
    match m {
        TimeModulation::Constant => 0,
        TimeModulation::Sinusoidal { .. } => 1,
        TimeModulation::Step { .. } => 2,
        TimeModulation::Ramp { .. } => 3,
        TimeModulation::Expression { .. } => 4,
    }
}

/// Human-readable label for a modulation type index.
fn modulation_type_label(idx: usize) -> &'static str {
    match idx {
        0 => "Constant",
        1 => "Sinusoidal",
        2 => "Step",
        3 => "Ramp",
        4 => "Expression",
        _ => "Unknown",
    }
}

/// Draw time modulation type selector and parameter fields.
///
/// When the modulation type or a parameter changes, the current element
/// is reconstructed via `make_element` and emitted as an UpdateForce edit.
fn draw_modulation_fields(
    ui: &mut egui::Ui,
    index: usize,
    modulation: &TimeModulation,
    make_element: impl Fn(TimeModulation) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    ui.separator();
    ui.label("Modulation:");

    let mut type_idx = modulation_type_index(modulation);
    let prev_idx = type_idx;

    egui::ComboBox::from_id_salt(format!("mod_type_{}", index))
        .selected_text(modulation_type_label(type_idx))
        .show_ui(ui, |ui| {
            for i in 0..5 {
                ui.selectable_value(&mut type_idx, i, modulation_type_label(i));
            }
        });

    // If the type changed, switch to new variant with default parameters.
    if type_idx != prev_idx {
        let new_mod = match type_idx {
            0 => TimeModulation::Constant,
            1 => TimeModulation::Sinusoidal { omega: 1.0, phase: 0.0 },
            2 => TimeModulation::Step { t_on: 0.0 },
            3 => TimeModulation::Ramp { t_start: 0.0, t_end: 1.0 },
            4 => TimeModulation::Expression { expr: "sin(2*pi*t)".into() },
            _ => TimeModulation::Constant,
        };
        *pending = Some(PendingPropertyEdit::UpdateForce {
            index,
            force: make_element(new_mod),
        });
        return;
    }

    // Draw parameter fields for the current modulation type.
    match modulation {
        TimeModulation::Constant => {} // no parameters

        TimeModulation::Sinusoidal { omega, phase } => {
            let mut w = *omega;
            ui.horizontal(|ui| {
                ui.label("\u{03c9}:");
                if ui
                    .add(egui::DragValue::new(&mut w).speed(0.1).suffix(" rad/s"))
                    .changed()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Sinusoidal {
                            omega: w,
                            phase: *phase,
                        }),
                    });
                }
            });

            let mut p = *phase;
            ui.horizontal(|ui| {
                ui.label("Phase:");
                if ui
                    .add(egui::DragValue::new(&mut p).speed(0.01).suffix(" rad"))
                    .changed()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Sinusoidal {
                            omega: *omega,
                            phase: p,
                        }),
                    });
                }
            });
        }

        TimeModulation::Step { t_on } => {
            let mut t = *t_on;
            ui.horizontal(|ui| {
                ui.label("t_on:");
                if ui
                    .add(
                        egui::DragValue::new(&mut t)
                            .speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"),
                    )
                    .changed()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Step { t_on: t }),
                    });
                }
            });
        }

        TimeModulation::Ramp { t_start, t_end } => {
            let mut ts = *t_start;
            ui.horizontal(|ui| {
                ui.label("t_start:");
                if ui
                    .add(
                        egui::DragValue::new(&mut ts)
                            .speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"),
                    )
                    .changed()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Ramp {
                            t_start: ts,
                            t_end: *t_end,
                        }),
                    });
                }
            });

            let mut te = *t_end;
            ui.horizontal(|ui| {
                ui.label("t_end:");
                if ui
                    .add(
                        egui::DragValue::new(&mut te)
                            .speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"),
                    )
                    .changed()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Ramp {
                            t_start: *t_start,
                            t_end: te,
                        }),
                    });
                }
            });
        }

        TimeModulation::Expression { expr } => {
            let mut text = expr.clone();
            ui.horizontal(|ui| {
                ui.label("f(t):");
                let response = ui.text_edit_singleline(&mut text);
                if response.lost_focus() && text != *expr {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Expression { expr: text }),
                    });
                }
            });

            // Validation hint: try parsing the expression
            if let Err(e) = expr.parse::<meval::Expr>() {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Parse error: {}", e),
                );
            } else if expr.parse::<meval::Expr>().ok().and_then(|e| e.bind("t").ok()).is_none() {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    "Error: cannot bind variable 't'",
                );
            }
        }
    }
}

/// Draw x/y DragValue fields for a 2D point, emitting an UpdateForce edit
/// when either component changes.
///
/// `make_element` takes the updated `[f64; 2]` and returns the full
/// `ForceElement` with that point replaced.
fn draw_point_fields(
    ui: &mut egui::Ui,
    label: &str,
    point: &[f64; 2],
    index: usize,
    make_element: impl Fn([f64; 2]) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let mut x = point[0];
    let mut y = point[1];
    ui.horizontal(|ui| {
        ui.label(format!("{} x:", label));
        if ui
            .add(egui::DragValue::new(&mut x).speed(0.001).suffix(" m"))
            .changed()
        {
            *pending = Some(PendingPropertyEdit::UpdateForce {
                index,
                force: make_element([x, point[1]]),
            });
        }
    });
    ui.horizontal(|ui| {
        ui.label(format!("{} y:", label));
        if ui
            .add(egui::DragValue::new(&mut y).speed(0.001).suffix(" m"))
            .changed()
        {
            *pending = Some(PendingPropertyEdit::UpdateForce {
                index,
                force: make_element([point[0], y]),
            });
        }
    });
}
