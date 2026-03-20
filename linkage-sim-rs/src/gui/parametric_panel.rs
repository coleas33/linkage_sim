//! Parametric study panel: sweep a design variable, plot output sensitivity.
//!
//! The user selects a parameter (link length, mass, spring k, etc.), defines
//! a range and step count, then runs the study. Results are plotted as
//! metric vs. parameter value.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use super::state::{AppState, ParametricMetric, SweepParameter};

/// Draw the parametric study panel.
pub fn draw_parametric_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Parametric Study");

    // ── Parameter selection ─────────────────────────────────────────
    let available = state.available_parameters();
    if available.is_empty() {
        ui.label("Load a mechanism to configure a parametric study.");
        return;
    }

    ui.label("Parameter to sweep:");
    let current_label = state.parametric_config.parameter.label();
    egui::ComboBox::from_id_salt("param_select")
        .selected_text(&current_label)
        .show_ui(ui, |ui| {
            for param in &available {
                let label = param.label();
                if ui
                    .selectable_label(*param == state.parametric_config.parameter, &label)
                    .clicked()
                {
                    state.parametric_config.parameter = param.clone();
                    // Set reasonable default range based on current value
                    let current = get_current_value(state, param);
                    if current.abs() > 1e-12 {
                        state.parametric_config.min_value = current * 0.5;
                        state.parametric_config.max_value = current * 1.5;
                    }
                }
            }
        });

    // ── Range inputs ────────────────────────────────────────────────
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Min:");
        ui.add(
            egui::DragValue::new(&mut state.parametric_config.min_value)
                .speed(0.01)
                .max_decimals(4),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Max:");
        ui.add(
            egui::DragValue::new(&mut state.parametric_config.max_value)
                .speed(0.01)
                .max_decimals(4),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Steps:");
        let mut steps = state.parametric_config.num_steps as i32;
        if ui
            .add(egui::DragValue::new(&mut steps).range(2..=50))
            .changed()
        {
            state.parametric_config.num_steps = steps.max(2) as usize;
        }
    });

    // ── Metric selection ────────────────────────────────────────────
    ui.add_space(4.0);
    ui.label("Output metric:");
    egui::ComboBox::from_id_salt("metric_select")
        .selected_text(state.parametric_config.metric.label())
        .show_ui(ui, |ui| {
            for metric in ParametricMetric::all() {
                if ui
                    .selectable_label(
                        *metric == state.parametric_config.metric,
                        metric.label(),
                    )
                    .clicked()
                {
                    state.parametric_config.metric = *metric;
                    // Recompute metric values from cached sweeps if available
                }
            }
        });

    // ── Run button ──────────────────────────────────────────────────
    ui.add_space(8.0);
    if ui.button("Run Study").clicked() {
        state.run_parametric_study();
    }

    // ── Results plot ────────────────────────────────────────────────
    if let Some(ref result) = state.parametric_result {
        ui.add_space(8.0);
        ui.separator();

        let points: Vec<[f64; 2]> = result
            .parameter_values
            .iter()
            .zip(result.metric_values.iter())
            .filter(|(_, y)| y.is_finite())
            .map(|(x, y)| [*x, *y])
            .collect();

        let x_label = result.config.parameter.label();
        let y_label = result.config.metric.label();

        Plot::new("parametric_plot")
            .x_axis_label(x_label.as_str())
            .y_axis_label(y_label)
            .height(200.0)
            .allow_drag(false)
            .show(ui, |plot_ui| {
                let line = Line::new(
                    result.config.metric.label(),
                    PlotPoints::new(points.clone()),
                );
                plot_ui.line(line);
            });

        // Show numeric summary
        let valid: Vec<f64> = result
            .metric_values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if !valid.is_empty() {
            let min = valid.iter().copied().fold(f64::INFINITY, f64::min);
            let max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ui.horizontal(|ui| {
                ui.small(format!("Range: {:.4} to {:.4}", min, max));
            });
        }
    }
}

// ── Counterbalance section ────────────────────────────────────────────────

/// Draw the counterbalance assistant panel.
pub fn draw_counterbalance_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Counterbalance Assistant");
    ui.label("Find optimal spring to minimize torque ripple.");

    let Some(ref bp) = state.blueprint else {
        ui.label("Load a mechanism first.");
        return;
    };

    // Collect all attachment points for the dropdown
    let mut all_points: Vec<(String, String)> = Vec::new();
    let mut body_ids: Vec<&String> = bp.bodies.keys().collect();
    body_ids.sort();
    for body_id in &body_ids {
        let body = &bp.bodies[*body_id];
        let mut pts: Vec<&String> = body.attachment_points.keys().collect();
        pts.sort();
        for pt in pts {
            all_points.push(((*body_id).clone(), pt.clone()));
        }
    }

    if all_points.len() < 2 {
        ui.label("Need at least 2 attachment points.");
        return;
    }

    // Spring attachment point A
    ui.add_space(4.0);
    ui.label("Spring point A:");
    let label_a = format!("{}.{}", state.counterbalance_config.body_a, state.counterbalance_config.point_a);
    egui::ComboBox::from_id_salt("cb_point_a")
        .selected_text(&label_a)
        .show_ui(ui, |ui| {
            for (body, pt) in &all_points {
                let l = format!("{}.{}", body, pt);
                if ui.selectable_label(
                    *body == state.counterbalance_config.body_a && *pt == state.counterbalance_config.point_a,
                    &l,
                ).clicked() {
                    state.counterbalance_config.body_a = body.clone();
                    state.counterbalance_config.point_a = pt.clone();
                }
            }
        });

    // Spring attachment point B
    ui.label("Spring point B:");
    let label_b = format!("{}.{}", state.counterbalance_config.body_b, state.counterbalance_config.point_b);
    egui::ComboBox::from_id_salt("cb_point_b")
        .selected_text(&label_b)
        .show_ui(ui, |ui| {
            for (body, pt) in &all_points {
                let l = format!("{}.{}", body, pt);
                if ui.selectable_label(
                    *body == state.counterbalance_config.body_b && *pt == state.counterbalance_config.point_b,
                    &l,
                ).clicked() {
                    state.counterbalance_config.body_b = body.clone();
                    state.counterbalance_config.point_b = pt.clone();
                }
            }
        });

    // Stiffness range
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("k min:");
        ui.add(egui::DragValue::new(&mut state.counterbalance_config.k_min).speed(1.0).suffix(" N/m"));
    });
    ui.horizontal(|ui| {
        ui.label("k max:");
        ui.add(egui::DragValue::new(&mut state.counterbalance_config.k_max).speed(1.0).suffix(" N/m"));
    });
    ui.horizontal(|ui| {
        ui.label("k steps:");
        let mut s = state.counterbalance_config.k_steps as i32;
        if ui.add(egui::DragValue::new(&mut s).range(2..=30)).changed() {
            state.counterbalance_config.k_steps = s.max(2) as usize;
        }
    });

    // Free length range
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("L0 min:");
        ui.add(egui::DragValue::new(&mut state.counterbalance_config.free_length_min).speed(0.001).suffix(" m"));
    });
    ui.horizontal(|ui| {
        ui.label("L0 max:");
        ui.add(egui::DragValue::new(&mut state.counterbalance_config.free_length_max).speed(0.001).suffix(" m"));
    });
    ui.horizontal(|ui| {
        ui.label("L0 steps:");
        let mut s = state.counterbalance_config.free_length_steps as i32;
        if ui.add(egui::DragValue::new(&mut s).range(1..=20)).changed() {
            state.counterbalance_config.free_length_steps = s.max(1) as usize;
        }
    });

    // Run button
    ui.add_space(8.0);
    if ui.button("Optimize Counterbalance").clicked() {
        state.run_counterbalance_study();
    }

    // Results
    if let Some(ref result) = state.counterbalance_result {
        ui.add_space(8.0);
        ui.separator();
        ui.strong("Results:");
        ui.label(format!("Optimal k = {:.1} N/m", result.best_k));
        ui.label(format!("Optimal L0 = {:.4} m", result.best_free_length));
        ui.label(format!(
            "Torque P-P: {:.3} N*m \u{2192} {:.3} N*m ({:.0}% reduction)",
            result.baseline_peak_to_peak,
            result.best_peak_to_peak,
            (1.0 - result.best_peak_to_peak / result.baseline_peak_to_peak.max(1e-12)) * 100.0
        ));

        // Before/after torque overlay plot
        if !result.baseline_torques.is_empty() && !result.optimized_torques.is_empty() {
            ui.add_space(4.0);
            let baseline_points: Vec<[f64; 2]> = result.angles_deg.iter()
                .zip(result.baseline_torques.iter())
                .map(|(x, y)| [*x, *y])
                .collect();
            let optimized_points: Vec<[f64; 2]> = result.angles_deg.iter()
                .zip(result.optimized_torques.iter())
                .map(|(x, y)| [*x, *y])
                .collect();

            Plot::new("counterbalance_plot")
                .x_axis_label("Driver angle (deg)")
                .y_axis_label("Driver torque (N*m)")
                .height(180.0)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Baseline", PlotPoints::new(baseline_points)));
                    plot_ui.line(Line::new("Optimized", PlotPoints::new(optimized_points)));
                });
        }
    }
}

/// Get the current value of a parameter from the blueprint.
fn get_current_value(state: &AppState, param: &SweepParameter) -> f64 {
    let Some(ref bp) = state.blueprint else {
        return 1.0;
    };
    match param {
        SweepParameter::BodyMass(id) => bp.bodies.get(id).map(|b| b.mass).unwrap_or(1.0),
        SweepParameter::BodyIzz(id) => bp.bodies.get(id).map(|b| b.izz_cg).unwrap_or(0.01),
        SweepParameter::AttachmentX(body_id, pt) => bp
            .bodies
            .get(body_id)
            .and_then(|b| b.attachment_points.get(pt))
            .map(|p| p[0])
            .unwrap_or(0.0),
        SweepParameter::AttachmentY(body_id, pt) => bp
            .bodies
            .get(body_id)
            .and_then(|b| b.attachment_points.get(pt))
            .map(|p| p[1])
            .unwrap_or(0.0),
        SweepParameter::ForceParam(idx, field) => bp
            .forces
            .get(*idx)
            .and_then(|f| get_force_field_value(f, field))
            .unwrap_or(1.0),
        SweepParameter::DriverOmega => state.driver_omega,
    }
}

/// Read a scalar field value from a force element.
fn get_force_field_value(force: &crate::forces::elements::ForceElement, field: &str) -> Option<f64> {
    use crate::forces::elements::ForceElement;
    match force {
        ForceElement::LinearSpring(e) => match field {
            "stiffness" => Some(e.stiffness),
            "free_length" => Some(e.free_length),
            _ => None,
        },
        ForceElement::TorsionSpring(e) => match field {
            "stiffness" => Some(e.stiffness),
            "free_angle" => Some(e.free_angle),
            _ => None,
        },
        ForceElement::LinearDamper(e) => match field {
            "damping" => Some(e.damping),
            _ => None,
        },
        ForceElement::RotaryDamper(e) => match field {
            "damping" => Some(e.damping),
            _ => None,
        },
        ForceElement::GasSpring(e) => match field {
            "initial_force" => Some(e.initial_force),
            "extended_length" => Some(e.extended_length),
            "stroke" => Some(e.stroke),
            _ => None,
        },
        ForceElement::Motor(e) => match field {
            "stall_torque" => Some(e.stall_torque),
            "no_load_speed" => Some(e.no_load_speed),
            _ => None,
        },
        ForceElement::ExternalForce(e) => match field {
            "force_x" => Some(e.force[0]),
            "force_y" => Some(e.force[1]),
            _ => None,
        },
        ForceElement::ExternalTorque(e) => match field {
            "torque" => Some(e.torque),
            _ => None,
        },
        ForceElement::BearingFriction(e) => match field {
            "constant_drag" => Some(e.constant_drag),
            "viscous_coeff" => Some(e.viscous_coeff),
            "coulomb_coeff" => Some(e.coulomb_coeff),
            _ => None,
        },
        ForceElement::JointLimit(e) => match field {
            "stiffness" => Some(e.stiffness),
            _ => None,
        },
        ForceElement::LinearActuator(e) => match field {
            "force" => Some(e.force),
            "speed_limit" => Some(e.speed_limit),
            _ => None,
        },
        _ => None,
    }
}
