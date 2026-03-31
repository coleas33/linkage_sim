//! Diagnostics section of the property panel.
//!
//! Grashof classification, crank recommendation, Jacobian conditioning,
//! sweep envelope statistics, force contributions, virtual work cross-check,
//! and motor sizing feasibility.

use eframe::egui;
use crate::analysis::envelopes::compute_envelope;
use crate::analysis::grashof::GrashofType;
use crate::analysis::motor_sizing::check_motor_sizing;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::gui::state::AppState;

/// Draw Grashof classification, crank recommendation, Jacobian conditioning,
/// and sweep envelope diagnostics.
///
/// Shows a collapsible "Diagnostics" header containing:
/// - Grashof classification and link lengths (4-bar mechanisms only)
/// - Crank recommendation (which link to drive for maximum rotation)
/// - Constraint Jacobian condition number (when forces have been solved)
/// - Sweep envelope statistics (torque min/max/RMS when sweep data available)
pub(super) fn draw_diagnostics_section(ui: &mut egui::Ui, state: &AppState) {
    let mech = match &state.mechanism {
        Some(m) => m,
        None => return,
    };

    let has_grashof = state.grashof_result.is_some();
    let has_crank_rec = state.crank_recommendation.is_some();
    let has_condition = state.force_results.condition_number.is_some();
    // Always show diagnostics when a mechanism is loaded (mass summary is
    // always available).
    egui::CollapsingHeader::new(
        egui::RichText::new("Diagnostics").color(state.nc(egui::Color32::from_rgb(150, 160, 180))),
    )
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
