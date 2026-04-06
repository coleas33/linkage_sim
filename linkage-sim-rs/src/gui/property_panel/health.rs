//! Mechanism Health Report section of the property panel.
//!
//! A collapsible pre-flight checklist showing status indicators
//! (green/yellow/red) for Grashof classification, toggle points,
//! transmission angle, peak torque, peak joint reactions, Jacobian
//! conditioning, and constraint violations.

use eframe::egui;
use crate::analysis::envelopes::compute_envelope;
use crate::analysis::grashof::GrashofType;
use crate::gui::state::AppState;
use crate::gui::sweep::SweepData;

/// Color for OK / green status indicators.
const COLOR_OK: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);
/// Color for warning / yellow status indicators.
const COLOR_WARN: egui::Color32 = egui::Color32::from_rgb(255, 200, 60);
/// Color for problem / red status indicators.
const COLOR_RED: egui::Color32 = egui::Color32::from_rgb(255, 80, 80);

/// Draw the "Mechanism Health" collapsible section.
///
/// Shows a compact pre-flight checklist of mechanism quality indicators.
/// Items without data (e.g., transmission angle for non-4-bar mechanisms)
/// are silently omitted.
pub(super) fn draw_health_section(ui: &mut egui::Ui, state: &AppState) {
    if state.mechanism.is_none() {
        return;
    }

    let health_color = state.nc(egui::Color32::from_rgb(80, 200, 120));
    egui::CollapsingHeader::new(
        egui::RichText::new("Mechanism Health").color(health_color),
    )
    .id_salt("health_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut any_shown = false;

        // 1. Grashof classification
        if let Some(ref gr) = state.grashof_result {
            any_shown = true;
            draw_grashof_indicator(ui, state, gr);
        }

        // 2. Toggle/dead points
        if let Some(ref sweep) = state.sweep_data {
            if any_shown { ui.separator(); }
            any_shown = true;
            draw_toggle_indicator(ui, state, &sweep.toggle_angles);
        }

        // 3. Transmission angle (4-bar only)
        if let Some(ref sweep) = state.sweep_data {
            if let Some(ref trans_angles) = sweep.transmission_angles {
                if any_shown { ui.separator(); }
                any_shown = true;
                draw_transmission_angle_indicator(ui, state, trans_angles);
            }
        }

        // 4. Peak driver torque
        if let Some(ref sweep) = state.sweep_data {
            if let Some(ref torques) = sweep.driver_torques {
                if let Some(env) = compute_envelope(torques) {
                    if any_shown { ui.separator(); }
                    any_shown = true;
                    draw_torque_indicator(ui, state, &env);
                }
            }
        }

        // 4b. Peak profile torque (when a non-constant motion profile is active)
        if let Some(ref sweep) = state.sweep_data {
            if let Some(ref prof_torques) = sweep.profile_torques {
                if let Some(env) = compute_envelope(prof_torques) {
                    if any_shown { ui.separator(); }
                    any_shown = true;
                    draw_profile_torque_indicator(ui, state, &env);
                }
            }
        }

        // 5. Peak joint reactions
        if let Some(ref sweep) = state.sweep_data {
            if !sweep.joint_reaction_magnitudes.is_empty() {
                if any_shown { ui.separator(); }
                any_shown = true;
                draw_peak_reactions_indicator(ui, state, &sweep.joint_reaction_magnitudes);
            }
        }

        // 6. Actuator stroke and peak force
        if let Some(ref sweep) = state.sweep_data {
            if sweep.actuator_forces.is_some() || sweep.actuator_lengths.is_some() {
                if any_shown { ui.separator(); }
                any_shown = true;
                draw_actuator_stroke_indicator(ui, state, sweep);
            }
        }

        // 6b. RMS actuator force, power, and peak speed
        if let Some(ref sweep) = state.sweep_data {
            if sweep.actuator_forces.is_some() {
                if any_shown { ui.separator(); }
                any_shown = true;
                draw_actuator_rms_indicators(ui, state, sweep);
            }
        }

        // 6c. Actuator utilization (when rated force is set)
        if let Some(ref sweep) = state.sweep_data {
            if state.actuator_rated_force > 0.0 {
                if let Some(ref forces) = sweep.actuator_forces {
                    if any_shown { ui.separator(); }
                    any_shown = true;
                    draw_actuator_utilization_indicator(
                        ui, state, forces, &sweep.angles_deg, state.actuator_rated_force,
                    );
                }
            }
        }

        // 7. Jacobian conditioning
        if let Some(kappa) = state.force_results.condition_number {
            if any_shown { ui.separator(); }
            any_shown = true;
            draw_conditioning_indicator(ui, state, kappa);
        }

        // 8. Constraint violations
        {
            if any_shown { ui.separator(); }
            #[allow(unused_assignments)]
            { any_shown = true; }
            draw_residual_indicator(ui, state);
        }
    });
}

/// Grashof classification indicator.
fn draw_grashof_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    gr: &crate::analysis::grashof::GrashofResult,
) {
    let (label, color) = match gr.classification {
        GrashofType::CrankRocker => ("Crank-Rocker (Grashof)", state.nc(COLOR_OK)),
        GrashofType::DoubleCrank => ("Double-Crank (Grashof)", state.nc(COLOR_OK)),
        GrashofType::DoubleRocker => ("Double-Rocker (Grashof)", state.nc(COLOR_OK)),
        GrashofType::ChangePoint => ("Change-Point", state.nc(COLOR_WARN)),
        GrashofType::NonGrashof => ("Non-Grashof", state.nc(COLOR_WARN)),
    };

    ui.horizontal(|ui| {
        ui.label("Grashof:");
        ui.colored_label(color, label);
    });
}

/// Toggle/dead-point indicator.
fn draw_toggle_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    toggle_angles: &[f64],
) {
    ui.horizontal(|ui| {
        ui.label("Toggle points:");
        if toggle_angles.is_empty() {
            ui.colored_label(state.nc(COLOR_OK), "None");
        } else {
            // Check if any toggle angles fall within the active sweep range
            let in_range = has_toggles_in_range(state, toggle_angles);
            let color = if in_range {
                state.nc(COLOR_RED)
            } else {
                state.nc(COLOR_WARN)
            };

            let angle_strs: Vec<String> = toggle_angles
                .iter()
                .map(|a| format!("{:.0}\u{00b0}", a))
                .collect();
            ui.colored_label(color, angle_strs.join(", "));
        }
    });
}

/// Check whether any toggle angles fall within the active sweep range.
fn has_toggles_in_range(state: &AppState, toggle_angles: &[f64]) -> bool {
    if !state.sweep_range_enabled {
        // Full 360 -- all toggles are in range
        return !toggle_angles.is_empty();
    }
    let min_deg = state.sweep_angle_min_deg;
    let max_deg = state.sweep_angle_max_deg;
    toggle_angles.iter().any(|&a| a >= min_deg && a <= max_deg)
}

/// Transmission angle indicator (4-bar only).
fn draw_transmission_angle_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    trans_angles: &[f64],
) {
    if let Some(env) = compute_envelope(trans_angles) {
        let min_ta = env.min_value;
        let (label, color) = if min_ta >= 40.0 {
            (format!("Min: {:.0}\u{00b0} (OK)", min_ta), state.nc(COLOR_OK))
        } else {
            (format!("Min: {:.0}\u{00b0} (POOR)", min_ta), state.nc(COLOR_RED))
        };

        ui.horizontal(|ui| {
            ui.label("Transmission \u{2220}:");
            ui.colored_label(color, label);
        });
    }
}

/// Peak driver torque indicator.
fn draw_torque_indicator(
    ui: &mut egui::Ui,
    _state: &AppState,
    env: &crate::analysis::envelopes::SignalEnvelope,
) {
    ui.horizontal(|ui| {
        ui.label("Driver torque:");
        ui.label(format!(
            "Peak {:.3} N\u{00b7}m / RMS {:.3} N\u{00b7}m",
            env.max_value.abs().max(env.min_value.abs()),
            env.rms,
        ));
    });
}

/// Profile torque indicator (trapezoidal motion profile).
fn draw_profile_torque_indicator(
    ui: &mut egui::Ui,
    _state: &AppState,
    env: &crate::analysis::envelopes::SignalEnvelope,
) {
    ui.horizontal(|ui| {
        ui.label("Profile torque:");
        ui.label(format!(
            "Peak {:.3} N\u{00b7}m / RMS {:.3} N\u{00b7}m",
            env.max_value.abs().max(env.min_value.abs()),
            env.rms,
        ));
    });
}

/// Peak joint reaction indicator.
fn draw_peak_reactions_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    reactions: &std::collections::HashMap<String, Vec<f64>>,
) {
    // Find the joint with the highest peak reaction force
    let mut worst_joint = String::new();
    let mut worst_peak = 0.0_f64;

    for (jid, magnitudes) in reactions {
        if let Some(env) = compute_envelope(magnitudes) {
            if env.max_value > worst_peak {
                worst_peak = env.max_value;
                worst_joint = jid.clone();
            }
        }
    }

    if worst_peak > 0.0 {
        let color = if worst_peak < 100.0 {
            state.nc(COLOR_OK)
        } else if worst_peak < 1000.0 {
            state.nc(COLOR_WARN)
        } else {
            state.nc(COLOR_RED)
        };

        ui.horizontal(|ui| {
            ui.label("Peak reaction:");
            ui.colored_label(
                color,
                format!("{:.1} N at {}", worst_peak, worst_joint),
            );
        });
    }
}

/// Jacobian conditioning indicator.
fn draw_conditioning_indicator(ui: &mut egui::Ui, state: &AppState, kappa: f64) {
    let (label, color) = if kappa < 1e4 {
        ("Well-conditioned".to_string(), state.nc(COLOR_OK))
    } else if kappa < 1e8 {
        (format!("Moderate (\u{03ba}={:.1e})", kappa), state.nc(COLOR_WARN))
    } else {
        (format!("Near-singular (\u{03ba}={:.1e})", kappa), state.nc(COLOR_RED))
    };

    ui.horizontal(|ui| {
        ui.label("Jacobian:");
        ui.colored_label(color, label);
    });
}

/// Actuator stroke and peak force indicator.
///
/// Shows the actuator stroke (max - min length) in mm and peak force from both
/// statics and inverse dynamics analyses.
fn draw_actuator_stroke_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    sweep: &SweepData,
) {
    let units = &state.display_units;

    // Compute min/max actuator length from sweep data.
    if let Some(ref lengths) = sweep.actuator_lengths {
        let finite_lengths: Vec<f64> = lengths.iter().copied().filter(|v| v.is_finite()).collect();
        if let (Some(&min_len), Some(&max_len)) = (
            finite_lengths.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            finite_lengths.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            let stroke = max_len - min_len;
            ui.horizontal(|ui| {
                ui.label("Actuator stroke:");
                ui.colored_label(
                    state.nc(COLOR_OK),
                    format!(
                        "{:.1}{} (min: {:.1}{}, max: {:.1}{})",
                        units.length(stroke), units.length_suffix(),
                        units.length(min_len), units.length_suffix(),
                        units.length(max_len), units.length_suffix(),
                    ),
                );
            });
        }
    }

    // Compute peak actuator forces.
    let peak_statics = sweep.actuator_forces.as_ref().and_then(|f| {
        compute_envelope(f).map(|env| env.max_value.abs().max(env.min_value.abs()))
    });
    let peak_id = sweep.actuator_forces_id.as_ref().and_then(|f| {
        compute_envelope(f).map(|env| env.max_value.abs().max(env.min_value.abs()))
    });

    if let Some(ps) = peak_statics {
        ui.horizontal(|ui| {
            ui.label("Peak actuator force:");
            let label = match peak_id {
                Some(pi) => format!("{:.0} N (statics) / {:.0} N (inertia)", ps, pi),
                None => format!("{:.0} N", ps),
            };
            ui.colored_label(state.nc(COLOR_OK), label);
        });
    }
}

/// RMS actuator force, RMS actuator power, and peak actuator speed indicators.
///
/// Displays duty-cycle summary statistics computed from the full sweep:
/// - RMS actuator force from statics (and inverse dynamics if available)
/// - RMS actuator power from statics
/// - Peak actuator speed in mm/s
fn draw_actuator_rms_indicators(
    ui: &mut egui::Ui,
    state: &AppState,
    sweep: &SweepData,
) {
    // RMS Actuator Force
    let rms_statics = sweep.actuator_forces.as_ref().and_then(|f| {
        compute_envelope(f).map(|env| env.rms)
    });
    let rms_id = sweep.actuator_forces_id.as_ref().and_then(|f| {
        compute_envelope(f).map(|env| env.rms)
    });

    if let Some(rms_s) = rms_statics {
        ui.horizontal(|ui| {
            ui.label("RMS actuator force:");
            let label = match rms_id {
                Some(rms_i) => format!("{:.0} N (statics) / {:.0} N (with inertia)", rms_s, rms_i),
                None => format!("{:.0} N", rms_s),
            };
            ui.colored_label(state.nc(COLOR_OK), label);
        });
    }

    // RMS Actuator Power
    let rms_power = sweep.actuator_power.as_ref().and_then(|p| {
        compute_envelope(p).map(|env| env.rms)
    });

    if let Some(rms_p) = rms_power {
        ui.horizontal(|ui| {
            ui.label("RMS actuator power:");
            ui.colored_label(state.nc(COLOR_OK), format!("{:.1} W", rms_p));
        });
    }

    // Peak Actuator Speed (convert m/s to mm/s for display)
    let peak_speed = sweep.actuator_speeds.as_ref().and_then(|s| {
        compute_envelope(s).map(|env| env.max_value.abs().max(env.min_value.abs()))
    });

    if let Some(ps) = peak_speed {
        ui.horizontal(|ui| {
            ui.label("Peak actuator speed:");
            ui.colored_label(state.nc(COLOR_OK), format!("{:.1} mm/s", ps * 1000.0));
        });
    }
}

/// Constraint residual / solver status indicator.
fn draw_residual_indicator(ui: &mut egui::Ui, state: &AppState) {
    let status = &state.solver_status;
    let norm = status.residual_norm;

    let (label, color) = if !status.converged {
        (format!("NOT CONVERGED (res={:.2e})", norm), state.nc(COLOR_RED))
    } else if norm < 1e-8 {
        (format!("OK (res={:.1e})", norm), state.nc(COLOR_OK))
    } else {
        (format!("Marginal (res={:.2e})", norm), state.nc(COLOR_WARN))
    };

    ui.horizontal(|ui| {
        ui.label("Constraints:");
        ui.colored_label(color, label);
    });
}

/// Actuator utilization indicator (shown when rated force > 0).
///
/// Displays peak utilization percentage and the angle range where the
/// actuator exceeds 80% of its rated capacity.
fn draw_actuator_utilization_indicator(
    ui: &mut egui::Ui,
    state: &AppState,
    forces: &[f64],
    angles_deg: &[f64],
    rated: f64,
) {
    // Compute peak utilization.
    let peak_abs = forces
        .iter()
        .filter(|f| f.is_finite())
        .map(|f| f.abs())
        .fold(0.0_f64, f64::max);
    let peak_pct = (peak_abs / rated) * 100.0;

    let (pct_label, color) = if peak_pct < 50.0 {
        (format!("{:.0}% peak", peak_pct), state.nc(COLOR_OK))
    } else if peak_pct < 80.0 {
        (format!("{:.0}% peak", peak_pct), state.nc(COLOR_WARN))
    } else {
        (format!("{:.0}% peak", peak_pct), state.nc(COLOR_RED))
    };

    ui.horizontal(|ui| {
        ui.label("Actuator utilization:");
        ui.colored_label(color, pct_label);
    });

    // Find contiguous angle ranges exceeding 80% rated.
    let exceeding: Vec<f64> = angles_deg
        .iter()
        .zip(forces.iter())
        .filter(|&(_, &f)| f.is_finite() && f.abs() / rated >= 0.8)
        .map(|(&a, _)| a)
        .collect();

    if !exceeding.is_empty() {
        let min_angle = exceeding.first().copied().unwrap_or(0.0);
        let max_angle = exceeding.last().copied().unwrap_or(0.0);
        ui.horizontal(|ui| {
            ui.label("Exceeding 80%:");
            ui.colored_label(
                state.nc(COLOR_RED),
                format!("{:.0}\u{00b0} to {:.0}\u{00b0}", min_angle, max_angle),
            );
        });
    }
}
