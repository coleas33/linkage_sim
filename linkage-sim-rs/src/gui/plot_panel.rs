//! Plot panel: sweep data visualization using egui_plot.
//!
//! Shows tabbed plots for coupler trace, body angles, and transmission angle.
//!
//! Clicking on any plot whose X-axis is driver angle scrubs the mechanism to
//! that angle.

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text as PlotText, VLine};

use super::state::{AngleUnit, AppState, DisplayUnits};
use super::sweep::SweepData;

/// Selected plot tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlotTab {
    CouplerTrace,
    BodyAngles,
    TransmissionAngle,
    DriverTorque,
    InverseDynamics,
    Energy,
    MechanicalAdvantage,
    JointReactions,
    CouplerVelocity,
    CouplerAcceleration,
    ActuatorForce,
    ActuatorSpeed,
    ActuatorPower,
    OutputForce,
}

/// Draw the plot panel with tabbed plots.
///
/// When the user clicks on a plot whose X-axis is driver angle, the mechanism
/// is scrubbed to that angle.
pub fn draw_plot_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        ui.label("No mechanism loaded.");
        return;
    }

    let Some(sweep) = &state.sweep_data else {
        ui.label("No sweep data available.");
        return;
    };

    if sweep.angles_deg.is_empty() {
        ui.label("Sweep produced no data (solver failed at all angles).");
        return;
    }

    // Persistent tab selection via egui's memory.
    let tab_id = ui.id().with("plot_tab");
    let mut selected_tab = ui
        .memory(|mem| mem.data.get_temp::<PlotTab>(tab_id))
        .unwrap_or(PlotTab::CouplerTrace);

    ui.horizontal(|ui| {
        ui.selectable_value(&mut selected_tab, PlotTab::CouplerTrace, "Coupler Trace");
        ui.selectable_value(&mut selected_tab, PlotTab::BodyAngles, "Body Angles");

        // Only show transmission angle tab if data exists.
        let has_ta = sweep.transmission_angles.is_some();
        ui.add_enabled_ui(has_ta, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::TransmissionAngle,
                "Transmission Angle",
            );
        });

        // Only show driver torque tab if data exists.
        let has_dt = sweep.driver_torques.is_some();
        let torque_tab_label = if sweep.sweep_mode.is_stroke() {
            "Actuator Force"
        } else {
            "Driver Torque"
        };
        ui.add_enabled_ui(has_dt, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::DriverTorque,
                torque_tab_label,
            );
        });

        // Only show inverse dynamics tab if data exists.
        let has_id = !sweep.inverse_dynamics_torques.is_empty();
        ui.add_enabled_ui(has_id, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::InverseDynamics,
                "Inv. Dynamics",
            );
        });

        // Only show energy tab if data exists.
        let has_energy = !sweep.kinetic_energy.is_empty();
        ui.add_enabled_ui(has_energy, |ui| {
            ui.selectable_value(&mut selected_tab, PlotTab::Energy, "Energy");
        });

        // Only show mechanical advantage tab if data exists.
        let has_ma = !sweep.mechanical_advantage.is_empty()
            && sweep.mechanical_advantage.iter().any(|v| v.is_finite());
        ui.add_enabled_ui(has_ma, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::MechanicalAdvantage,
                "Mech. Advantage",
            );
        });

        // Only show joint reactions tab if data exists.
        let has_jr = !sweep.joint_reaction_magnitudes.is_empty();
        ui.add_enabled_ui(has_jr, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::JointReactions,
                "Joint Reactions",
            );
        });

        // Only show coupler velocity tab if data exists.
        let has_cv = !sweep.coupler_velocities.is_empty();
        ui.add_enabled_ui(has_cv, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::CouplerVelocity,
                "Coupler Vel.",
            );
        });

        // Only show coupler acceleration tab if data exists.
        let has_ca = !sweep.coupler_accelerations.is_empty();
        ui.add_enabled_ui(has_ca, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::CouplerAcceleration,
                "Coupler Accel.",
            );
        });

        // Only show actuator force tab when a LinearActuator is present.
        let has_af = sweep.actuator_forces.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_af, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorForce,
                "Actuator Force",
            );
        });

        // Only show actuator speed tab when actuator speed data exists.
        let has_as = sweep.actuator_speeds.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_as, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorSpeed,
                "Actuator Speed",
            );
        });

        // Only show actuator power tab when actuator power data exists.
        let has_ap = sweep.actuator_power.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_ap, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorPower,
                "Actuator Power",
            );
        });

        // Only show output force tab when force zone data exists.
        let has_of = sweep.output_forces.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_of, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::OutputForce,
                "Output Force",
            );
        });
    });

    ui.memory_mut(|mem| mem.data.insert_temp(tab_id, selected_tab));

    ui.horizontal(|ui| {
        ui.separator();
        ui.label(egui::RichText::new("Scroll to zoom, double-click to reset").small().weak());
    });

    let current_driver_display = state.display_units.angle(state.driver_angle);
    let nm = state.nathan_mode;

    // Each driver-angle-on-X-axis plot returns Some(x) when clicked, where x
    // is the X coordinate in display angle units. CouplerTrace (X vs Y) does
    // not participate in scrubbing.
    let clicked_display_angle: Option<f64> = match selected_tab {
        PlotTab::CouplerTrace => {
            draw_coupler_trace(ui, sweep, &state.display_units, nm);
            None
        }
        PlotTab::BodyAngles => {
            draw_body_angles(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::TransmissionAngle => {
            draw_transmission_angle(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::DriverTorque => {
            draw_driver_torque(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::InverseDynamics => {
            draw_inverse_dynamics(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::Energy => {
            draw_energy(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::MechanicalAdvantage => {
            draw_mechanical_advantage(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::JointReactions => {
            draw_joint_reactions(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::CouplerVelocity => {
            draw_coupler_velocity(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::CouplerAcceleration => {
            draw_coupler_acceleration(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::ActuatorForce => {
            // Rated force input above the plot.
            ui.horizontal(|ui| {
                ui.label("Rated Force:");
                ui.add(egui::DragValue::new(&mut state.actuator_rated_force)
                    .speed(10.0)
                    .range(0.0..=1e6)
                    .suffix(" N"));
                if state.actuator_rated_force > 0.0 {
                    if ui.small_button("Clear").clicked() {
                        state.actuator_rated_force = 0.0;
                    }
                }
            });
            draw_actuator_force(ui, sweep, current_driver_display, &state.display_units, nm, state.actuator_rated_force)
        }
        PlotTab::ActuatorSpeed => {
            draw_actuator_speed(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::ActuatorPower => {
            draw_actuator_power(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::OutputForce => {
            draw_output_force(ui, sweep, current_driver_display, &state.display_units, nm)
        }
    };

    // Scrub the mechanism to the clicked angle.
    if let Some(display_angle) = clicked_display_angle {
        let angle_rad = display_to_radians(display_angle, &state.display_units);
        state.solve_at_angle(angle_rad);
    }
}

/// Convert a display-unit angle back to radians.
///
/// This is the inverse of `DisplayUnits::angle()`.
fn display_to_radians(display_angle: f64, units: &DisplayUnits) -> f64 {
    match units.angle {
        AngleUnit::Radians => display_angle,
        AngleUnit::Degrees => display_angle.to_radians(),
    }
}

/// Return the x-axis label string for plots that sweep over the driver variable.
///
/// In angle mode this is `"Driver Angle (deg)"` or `"Driver Angle (rad)"`.
/// In stroke mode this is `"Actuator Stroke (mm)"`.
fn x_axis_label_for_sweep(sweep: &SweepData, units: &DisplayUnits) -> String {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Stroke (mm)".to_string()
    } else {
        let angle_label = match units.angle {
            AngleUnit::Degrees => "deg",
            AngleUnit::Radians => "rad",
        };
        format!("Driver Angle ({})", angle_label)
    }
}

/// Return the y-axis label for the driver effort plot.
///
/// In angle mode (revolute driver) this is `"Driver Torque (N*m)"`.
/// In stroke mode (linear driver) this is `"Actuator Force (N)"`.
fn driver_effort_y_label(sweep: &SweepData) -> &'static str {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Force (N)"
    } else {
        "Driver Torque (N\u{00b7}m)"
    }
}

/// Return the series name for the driver effort in the plot legend.
///
/// In angle mode: `"Driver Torque"`. In stroke mode: `"Actuator Force"`.
fn driver_effort_series_name(sweep: &SweepData) -> &'static str {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Force"
    } else {
        "Driver Torque"
    }
}

/// Detect a click on the plot and return the X coordinate in plot space.
///
/// Call this inside a `plot_ui` closure. Returns `Some(x)` when the user
/// clicks (not drags) on the plot area.
fn detect_plot_click(plot_ui: &egui_plot::PlotUi) -> Option<f64> {
    if plot_ui.response().clicked() {
        if let Some(coord) = plot_ui.pointer_coordinate() {
            return Some(coord.x);
        }
    }
    None
}

/// Plot coupler point traces: x vs y, converted to display length units.
///
/// When `sweep.active_range` is set, the full curve is drawn faded/dashed for
/// context and the active sub-range is overdrawn solid.
fn draw_coupler_trace(ui: &mut egui::Ui, sweep: &SweepData, units: &DisplayUnits, nathan_mode: bool) {
    let axis_label = units.length_axis_label();
    let plot = Plot::new("coupler_trace_plot")
        .data_aspect(1.0) // equal axis scaling
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(format!("X ({})", axis_label))
        .y_axis_label(format!("Y ({})", axis_label))
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    plot.show(ui, |plot_ui| {
        let colors = series_colors(nathan_mode);
        let mut color_idx = 0;

        let mut keys: Vec<&String> = sweep.coupler_traces.keys().collect();
        keys.sort();

        for key in keys {
            let trace = &sweep.coupler_traces[key];
            if trace.is_empty() {
                continue;
            }

            let color = colors[color_idx % colors.len()];

            if let Some((start, end)) = sweep.active_range {
                // Full curve -- faded/dashed for context.
                let full_points: PlotPoints = trace
                    .iter()
                    .map(|[x, y]| [units.length(*x), units.length(*y)])
                    .collect();
                let faded = egui::Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(), 60,
                );
                plot_ui.line(
                    Line::new(format!("{} (full)", key), full_points)
                        .color(faded)
                        .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                        .width(0.75),
                );

                // Active range -- solid, full color.
                let end_clamped = end.min(trace.len().saturating_sub(1));
                if start <= end_clamped {
                    let active_points: PlotPoints = trace[start..=end_clamped]
                        .iter()
                        .map(|[x, y]| [units.length(*x), units.length(*y)])
                        .collect();
                    plot_ui.line(
                        Line::new(key.as_str(), active_points)
                            .color(color)
                            .width(2.0),
                    );
                }
            } else {
                // No range limit -- draw normally.
                let points: PlotPoints = trace
                    .iter()
                    .map(|[x, y]| [units.length(*x), units.length(*y)])
                    .collect();
                plot_ui.line(
                    Line::new(key.as_str(), points)
                        .color(color)
                        .width(1.5),
                );
            }
            color_idx += 1;
        }
    });
}

/// Plot body angles vs driver angle, using the current display angle unit.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_body_angles(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    // Sweep stores angles in degrees (or stroke in meters); convert on the fly.
    let angle_label = match units.angle {
        AngleUnit::Degrees => "deg",
        AngleUnit::Radians => "rad",
    };
    let plot = Plot::new("body_angles_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label(format!("Body Angle ({})", angle_label))
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let colors = series_colors(nathan_mode);
        let mut color_idx = 0;

        let mut body_ids: Vec<&String> = sweep.body_angles.keys().collect();
        body_ids.sort();

        for body_id in body_ids {
            let angles = &sweep.body_angles[body_id];
            // Body angles use y in display angle units, so we pass the
            // converted y through the helper's (deg, y) pairs.
            let pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(angles.iter())
                .map(|(&x_deg, &y_deg)| (x_deg, units.angle(y_deg.to_radians())))
                .collect();

            let color = colors[color_idx % colors.len()];
            draw_angle_series_with_range(
                plot_ui,
                body_id.as_str(),
                color,
                1.5,
                &pairs,
                sweep,
                units,
                nathan_mode,
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle (already in display units).
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot transmission angle (degrees) vs driver angle.
///
/// Transmission angle data is always in degrees; the x-axis driver angle is
/// shown in the current display unit.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_transmission_angle(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    let Some(ta) = &sweep.transmission_angles else {
        ui.label("Transmission angle not available for this mechanism.");
        return None;
    };

    // The x-axis span in display units (0 to 2pi).
    let x_max = units.angle(2.0 * std::f64::consts::PI);

    let plot = Plot::new("transmission_angle_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Transmission Angle (deg)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(ta.iter())
            .map(|(&x_deg, &y)| (x_deg, y))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Transmission Angle",
            egui::Color32::from_rgb(100, 200, 255),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Ideal zone: 40-140 degrees (y-axis stays in degrees always).
        let ideal_low: PlotPoints = [[0.0, 40.0], [x_max, 40.0]].into_iter().collect();
        let ideal_high: PlotPoints = [[0.0, 140.0], [x_max, 140.0]].into_iter().collect();
        plot_ui.line(
            Line::new("Poor threshold (40 deg)", ideal_low)
                .color(egui::Color32::from_rgba_premultiplied(200, 60, 60, 120))
                .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                .width(1.0),
        );
        plot_ui.line(
            Line::new("Poor threshold (140 deg)", ideal_high)
                .color(egui::Color32::from_rgba_premultiplied(200, 60, 60, 120))
                .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                .width(1.0),
        );

        // Text annotations inside the plot labelling the poor output zones.
        let text_x = x_max * 0.5;
        plot_ui.text(
            PlotText::new(
                "poor_zone_low",
                PlotPoint::new(text_x, 20.0),
                "Poor output zone (< 40\u{00b0})",
            )
            .color(egui::Color32::from_rgb(255, 80, 80))
            .anchor(egui::Align2::CENTER_CENTER),
        );
        plot_ui.text(
            PlotText::new(
                "poor_zone_high",
                PlotPoint::new(text_x, 160.0),
                "Poor output zone (> 140\u{00b0})",
            )
            .color(egui::Color32::from_rgb(255, 80, 80))
            .anchor(egui::Align2::CENTER_CENTER),
        );

        // Vertical marker at current driver angle (already in display units).
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Plot driver torque (N*m) vs driver angle.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_driver_torque(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    let Some(torques) = &sweep.driver_torques else {
        ui.label("Driver torque data not available.");
        return None;
    };

    let plot = Plot::new("driver_torque_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label(driver_effort_y_label(sweep))
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(torques.iter())
            .map(|(&x_deg, &y)| (x_deg, y))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            driver_effort_series_name(sweep),
            egui::Color32::from_rgb(255, 150, 80),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Plot inverse dynamics driver torque vs driver angle,
/// with optional statics torque overlay for comparison.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_inverse_dynamics(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.inverse_dynamics_torques.is_empty() {
        ui.label("Inverse dynamics data not available.");
        return None;
    }

    let plot = Plot::new("inverse_dynamics_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label(driver_effort_y_label(sweep))
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Inverse dynamics torque (cyan).
        let id_pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(sweep.inverse_dynamics_torques.iter())
            .filter(|&(_, &t)| t.is_finite())
            .map(|(&x_deg, &t)| (x_deg, t))
            .collect();
        draw_angle_series_with_range(
            plot_ui,
            "Inverse Dynamics Torque",
            egui::Color32::from_rgb(100, 200, 255),
            2.0,
            &id_pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Overlay statics torque/force if available (orange, dashed).
        // This is always drawn dashed as a reference, so no faded/solid split.
        if let Some(statics_torques) = &sweep.driver_torques {
            let is_stroke = sweep.sweep_mode.is_stroke();
            let st_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(statics_torques.iter())
                .filter(|&(_, &t)| t.is_finite())
                .map(|(&x, &t)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, t]
                })
                .collect();
            let statics_label = if is_stroke { "Statics Force" } else { "Statics Torque" };
            let statics_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(255, 150, 80))
            } else {
                egui::Color32::from_rgb(255, 150, 80)
            };
            plot_ui.line(
                Line::new(statics_label, st_points)
                    .color(statics_color)
                    .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                    .width(1.5),
            );
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Plot energy (KE, PE, total) vs driver angle.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_energy(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.kinetic_energy.is_empty() {
        ui.label("Energy data not available (velocity solve needed).");
        return None;
    }

    let plot = Plot::new("energy_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Energy (J)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Kinetic energy
        let ke_pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(sweep.kinetic_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| (x_deg, e))
            .collect();
        draw_angle_series_with_range(
            plot_ui,
            "Kinetic Energy",
            egui::Color32::from_rgb(255, 150, 80),
            2.0,
            &ke_pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Potential energy
        let pe_pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(sweep.potential_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| (x_deg, e))
            .collect();
        draw_angle_series_with_range(
            plot_ui,
            "Potential Energy",
            egui::Color32::from_rgb(100, 200, 255),
            2.0,
            &pe_pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Total energy
        let te_pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(sweep.total_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| (x_deg, e))
            .collect();
        draw_angle_series_with_range(
            plot_ui,
            "Total Energy",
            egui::Color32::from_rgb(120, 220, 120),
            2.0,
            &te_pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot mechanical advantage (dimensionless) vs driver angle.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_mechanical_advantage(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.mechanical_advantage.is_empty() {
        ui.label("Mechanical advantage data not available.");
        return None;
    }

    let plot = Plot::new("mechanical_advantage_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Mechanical Advantage")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(sweep.mechanical_advantage.iter())
            .filter(|&(_, &ma)| ma.is_finite())
            .map(|(&x_deg, &ma)| (x_deg, ma))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Mechanical Advantage",
            egui::Color32::from_rgb(200, 150, 255),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Unity reference line (MA = 1).
        let x_max = units.angle(2.0 * std::f64::consts::PI);
        let unity: PlotPoints = [[0.0, 1.0], [x_max, 1.0]].into_iter().collect();
        plot_ui.line(
            Line::new("MA = 1", unity)
                .color(egui::Color32::from_rgba_premultiplied(200, 200, 200, 80))
                .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                .width(1.0),
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot joint reaction force magnitudes (N) vs driver angle.
///
/// Each joint gets its own series with a distinct color from the standard
/// palette. Only finite values are plotted (NaN from failed statics solves
/// is silently skipped).
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_joint_reactions(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.joint_reaction_magnitudes.is_empty() {
        ui.label("Joint reaction data not available.");
        return None;
    }

    let plot = Plot::new("joint_reactions_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Reaction Force (N)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let colors = series_colors(nathan_mode);
        let mut color_idx = 0;

        let mut joint_ids: Vec<&String> = sweep.joint_reaction_magnitudes.keys().collect();
        joint_ids.sort();

        for joint_id in joint_ids {
            let magnitudes = &sweep.joint_reaction_magnitudes[joint_id];
            let pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(magnitudes.iter())
                .filter(|&(_, &m)| m.is_finite())
                .map(|(&x_deg, &m)| (x_deg, m))
                .collect();

            let color = colors[color_idx % colors.len()];
            draw_angle_series_with_range(
                plot_ui,
                joint_id.as_str(),
                color,
                1.5,
                &pairs,
                sweep,
                units,
                nathan_mode,
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot coupler point velocity magnitudes (m/s) vs driver angle.
///
/// Each coupler point gets its own series with a distinct color from the
/// standard palette. Only finite values are plotted.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_coupler_velocity(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.coupler_velocities.is_empty() {
        ui.label("Coupler velocity data not available.");
        return None;
    }

    let plot = Plot::new("coupler_velocity_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Velocity (m/s)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let colors = series_colors(nathan_mode);
        let mut color_idx = 0;

        let mut keys: Vec<&String> = sweep.coupler_velocities.keys().collect();
        keys.sort();

        for key in keys {
            let velocities = &sweep.coupler_velocities[key];
            let pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(velocities.iter())
                .filter(|&(_, &v)| v.is_finite())
                .map(|(&x_deg, &v)| (x_deg, v))
                .collect();

            let color = colors[color_idx % colors.len()];
            draw_angle_series_with_range(
                plot_ui,
                key.as_str(),
                color,
                1.5,
                &pairs,
                sweep,
                units,
                nathan_mode,
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot coupler point acceleration magnitudes (m/s^2) vs driver angle.
///
/// Each coupler point gets its own series with a distinct color from the
/// standard palette. Only finite values are plotted.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_coupler_acceleration(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    if sweep.coupler_accelerations.is_empty() {
        ui.label("Coupler acceleration data not available.");
        return None;
    }

    let plot = Plot::new("coupler_acceleration_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Acceleration (m/s\u{00b2})")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let colors = series_colors(nathan_mode);
        let mut color_idx = 0;

        let mut keys: Vec<&String> = sweep.coupler_accelerations.keys().collect();
        keys.sort();

        for key in keys {
            let accelerations = &sweep.coupler_accelerations[key];
            let pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(accelerations.iter())
                .filter(|&(_, &a)| a.is_finite())
                .map(|(&x_deg, &a)| (x_deg, a))
                .collect();

            let color = colors[color_idx % colors.len()];
            draw_angle_series_with_range(
                plot_ui,
                key.as_str(),
                color,
                1.5,
                &pairs,
                sweep,
                units,
                nathan_mode,
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });
    clicked_x
}

/// Plot required actuator force (N) vs driver angle.
///
/// Only available when a LinearActuator force element is present.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_actuator_force(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
    actuator_rated_force: f64,
) -> Option<f64> {
    let Some(forces) = &sweep.actuator_forces else {
        ui.label("Actuator force data not available (no LinearActuator in mechanism).");
        return None;
    };

    // Rated force input above the plot.
    // Note: actuator_rated_force is passed by value; the DragValue writes
    // through the mutable reference on the *caller's* copy inside
    // draw_plot_panel which owns `state`.  We re-read the value each frame.
    // Because this function doesn't own state mutably, we render the input
    // in the caller instead (see draw_plot_panel).

    let plot = Plot::new("actuator_force_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Actuator Force (N)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Statics-based actuator force (solid line).
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(forces.iter())
            .filter(|&(_, &f)| f.is_finite())
            .map(|(&x_deg, &f)| (x_deg, f))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Statics",
            egui::Color32::from_rgb(255, 100, 100),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Inverse dynamics actuator force (dashed overlay, includes inertia).
        if let Some(ref id_forces) = sweep.actuator_forces_id {
            let is_stroke = sweep.sweep_mode.is_stroke();
            let id_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(id_forces.iter())
                .filter(|&(_, &f)| f.is_finite())
                .map(|(&x, &f)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, f]
                })
                .collect();
            let id_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(100, 200, 255))
            } else {
                egui::Color32::from_rgb(100, 200, 255)
            };
            plot_ui.line(
                Line::new("With Inertia", id_points)
                    .color(id_color)
                    .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                    .width(1.5),
            );
        }

        // ── Force Margin Visualization (Feature 4) ──────────────────────
        // When the user has entered a rated force, draw horizontal reference
        // lines and safety-factor color-coded scatter overlay.
        if actuator_rated_force > 0.0 {
            let rated = actuator_rated_force;

            // Horizontal rated force line (positive / tension).
            let rated_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(80, 200, 80))
            } else {
                egui::Color32::from_rgb(80, 200, 80)
            };
            plot_ui.hline(
                HLine::new("Rated Force", rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );

            // Horizontal rated force line (negative / compression).
            plot_ui.hline(
                HLine::new("Rated (compression)", -rated)
                    .color(rated_color)
                    .style(egui_plot::LineStyle::Dashed { length: 6.0 })
                    .width(2.0),
            );

            // ── Safety Factor Overlay (Feature 5) ───────────────────────
            // Color-code the force data by utilization ratio |F| / rated.
            let is_stroke = sweep.sweep_mode.is_stroke();
            let to_display = |x_deg: f64| -> f64 {
                if is_stroke { x_deg * 1000.0 } else { units.angle(x_deg.to_radians()) }
            };

            let mut green_pts: Vec<[f64; 2]> = Vec::new();
            let mut yellow_pts: Vec<[f64; 2]> = Vec::new();
            let mut red_pts: Vec<[f64; 2]> = Vec::new();

            for &(x_deg, f) in &pairs {
                let ratio = f.abs() / rated;
                let pt = [to_display(x_deg), f];
                if ratio >= 0.8 {
                    red_pts.push(pt);
                } else if ratio >= 0.5 {
                    yellow_pts.push(pt);
                } else {
                    green_pts.push(pt);
                }
            }

            let apply_nm = |c: egui::Color32| -> egui::Color32 {
                if nathan_mode { crate::gui::canvas::to_grayscale(c) } else { c }
            };

            if !green_pts.is_empty() {
                plot_ui.points(
                    Points::new("< 50% rated", green_pts)
                        .radius(3.0)
                        .color(apply_nm(egui::Color32::GREEN)),
                );
            }
            if !yellow_pts.is_empty() {
                plot_ui.points(
                    Points::new("50-80% rated", yellow_pts)
                        .radius(3.0)
                        .color(apply_nm(egui::Color32::YELLOW)),
                );
            }
            if !red_pts.is_empty() {
                plot_ui.points(
                    Points::new("> 80% rated", red_pts)
                        .radius(3.0)
                        .color(apply_nm(egui::Color32::from_rgb(255, 60, 60))),
                );
            }
        }

        // Stroke annotation: show min/max actuator length and peak forces.
        draw_actuator_stroke_annotation(plot_ui, sweep, units);

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Draw a text annotation on the actuator force plot with peak/RMS force info.
///
/// Shows peak and RMS actuator forces from both statics and inverse dynamics.
/// Placed in the top-left corner of the plot.
fn draw_actuator_stroke_annotation(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    _units: &DisplayUnits,
) {
    use crate::analysis::envelopes::compute_envelope;

    let env_statics = sweep.actuator_forces.as_ref().and_then(|f| compute_envelope(f));
    let env_id = sweep.actuator_forces_id.as_ref().and_then(|f| compute_envelope(f));

    let mut lines: Vec<String> = Vec::new();
    if let Some(ref es) = env_statics {
        let peak = es.max_value.abs().max(es.min_value.abs());
        let peak_label = match &env_id {
            Some(ei) => {
                let peak_id = ei.max_value.abs().max(ei.min_value.abs());
                format!("Peak: {:.0} N (statics) / {:.0} N (inertia)", peak, peak_id)
            }
            None => format!("Peak: {:.0} N", peak),
        };
        lines.push(peak_label);

        let rms_label = match &env_id {
            Some(ei) => format!("RMS: {:.0} N (statics) / {:.0} N (inertia)", es.rms, ei.rms),
            None => format!("RMS: {:.0} N", es.rms),
        };
        lines.push(rms_label);
    }

    if lines.is_empty() {
        return;
    }

    let text = lines.join("\n");
    // Place at top-left of plot using the first data point as anchor.
    let bounds = plot_ui.plot_bounds();
    let x_pos = bounds.min()[0] + (bounds.max()[0] - bounds.min()[0]) * 0.02;
    let y_pos = bounds.max()[1] - (bounds.max()[1] - bounds.min()[1]) * 0.02;
    plot_ui.text(
        PlotText::new("actuator_annotation", PlotPoint::new(x_pos, y_pos), text)
            .anchor(egui::Align2::LEFT_TOP)
            .color(egui::Color32::from_rgba_premultiplied(200, 200, 200, 180)),
    );
}

/// Plot actuator extension rate (mm/s) vs driver angle.
///
/// Only available when a LinearActuator force element is present.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_actuator_speed(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    let Some(speeds) = &sweep.actuator_speeds else {
        ui.label("Actuator speed data not available (no LinearActuator in mechanism).");
        return None;
    };

    let plot = Plot::new("actuator_speed_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Actuator Speed (mm/s)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Convert m/s to mm/s for display.
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(speeds.iter())
            .filter(|&(_, &s)| s.is_finite())
            .map(|(&x_deg, &s)| (x_deg, s * 1000.0))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Actuator Speed",
            egui::Color32::from_rgb(120, 220, 120),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Plot required actuator power (W) vs driver angle.
///
/// Shows both statics-based power (solid) and inverse-dynamics power (dashed).
/// Only available when a LinearActuator force element is present.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_actuator_power(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    let Some(power) = &sweep.actuator_power else {
        ui.label("Actuator power data not available (no LinearActuator in mechanism).");
        return None;
    };

    let plot = Plot::new("actuator_power_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Power (W)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        // Statics-based power (solid line).
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(power.iter())
            .filter(|&(_, &p)| p.is_finite())
            .map(|(&x_deg, &p)| (x_deg, p))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Statics",
            egui::Color32::from_rgb(255, 100, 100),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Inverse dynamics power (dashed overlay, includes inertia).
        if let Some(ref id_power) = sweep.actuator_power_id {
            let is_stroke = sweep.sweep_mode.is_stroke();
            let id_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(id_power.iter())
                .filter(|&(_, &p)| p.is_finite())
                .map(|(&x, &p)| {
                    let x_display = if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) };
                    [x_display, p]
                })
                .collect();
            let id_color = if nathan_mode {
                crate::gui::canvas::to_grayscale(egui::Color32::from_rgb(100, 200, 255))
            } else {
                egui::Color32::from_rgb(100, 200, 255)
            };
            plot_ui.line(
                Line::new("With Inertia", id_points)
                    .color(id_color)
                    .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                    .width(1.5),
            );
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Plot output force (N) vs driver angle.
///
/// Shows the net force zone applied force at each crank angle. The output
/// force is the force the mechanism exerts at the output link, computed from
/// force zone overlap at each position.
///
/// Returns the clicked X coordinate (display angle units) if the user clicked.
fn draw_output_force(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
) -> Option<f64> {
    let Some(forces) = &sweep.output_forces else {
        ui.label("Output force data not available (no ForceZone in mechanism).");
        return None;
    };

    let plot = Plot::new("output_force_plot")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label("Output Force (N)")
        .legend(egui_plot::Legend::default())
        .height(ui.available_height().max(50.0));

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        let pairs: Vec<(f64, f64)> = sweep
            .angles_deg
            .iter()
            .zip(forces.iter())
            .filter(|&(_, &f)| f.is_finite())
            .map(|(&x_deg, &f)| (x_deg, f))
            .collect();

        draw_angle_series_with_range(
            plot_ui,
            "Output Force",
            egui::Color32::from_rgb(255, 180, 60),
            2.0,
            &pairs,
            sweep,
            units,
            nathan_mode,
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

/// Draw faint red dashed vertical lines at toggle/dead-point angles.
///
/// Toggle angles are in degrees; they are converted to the current display
/// angle unit before being drawn. In stroke mode, toggle angles are stored
/// in meters and are converted to mm.
fn draw_toggle_markers(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
) {
    let is_stroke = sweep.sweep_mode.is_stroke();
    for (i, &toggle_val) in sweep.toggle_angles.iter().enumerate() {
        let toggle_display = if is_stroke { toggle_val * 1000.0 } else { units.angle(toggle_val.to_radians()) };
        plot_ui.vline(
            VLine::new(format!("toggle_{}", i), toggle_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 60, 60, 100))
                .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                .width(2.5),
        );
    }
}

/// Draw vertical boundary markers at the sweep range limits.
///
/// When `active_range` is set, draws faint white dashed VLines at the
/// min and max angles (or stroke values) of the active range.
fn draw_range_boundary_markers(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
) {
    if let Some((start, end)) = sweep.active_range {
        if let (Some(&min_val), Some(&max_val)) =
            (sweep.angles_deg.get(start), sweep.angles_deg.get(end))
        {
            let is_stroke = sweep.sweep_mode.is_stroke();
            let min_display = if is_stroke { min_val * 1000.0 } else { units.angle(min_val.to_radians()) };
            let max_display = if is_stroke { max_val * 1000.0 } else { units.angle(max_val.to_radians()) };
            let boundary_color =
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 80);
            plot_ui.vline(
                VLine::new("range_min", min_display)
                    .color(boundary_color)
                    .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                    .width(1.0),
            );
            plot_ui.vline(
                VLine::new("range_max", max_display)
                    .color(boundary_color)
                    .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                    .width(1.0),
            );
        }
    }
}

/// Create a faded version of a color for out-of-range plot data.
fn faded_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 60)
}

/// Helper for angle-based plots: draws a single data series with faded/solid
/// treatment when an active range is set.
///
/// - `name`: legend label for the series
/// - `color`: full-opacity color for the active range
/// - `width`: line width for the active (solid) portion
/// - `x_deg_and_y`: iterator of `(angle_deg, y_value)` pairs for the full sweep
/// - `convert_x`: function to convert angle in degrees to display x-value
/// - `sweep`: used to read `active_range` and `angles_deg`
///
/// When `active_range` is `None`, draws normally. When set, draws the full
/// curve faded/dashed and overdraws the active slice solid.
///
/// In angle mode, x values are in degrees and are converted to the display
/// angle unit. In stroke mode, x values are in meters and are converted to mm.
fn draw_angle_series_with_range(
    plot_ui: &mut egui_plot::PlotUi,
    name: &str,
    color: egui::Color32,
    width: f32,
    x_deg_and_y: &[(f64, f64)],
    sweep: &SweepData,
    units: &DisplayUnits,
    nathan_mode: bool,
) {
    use crate::gui::canvas::to_grayscale;
    let color = if nathan_mode { to_grayscale(color) } else { color };
    if x_deg_and_y.is_empty() {
        return;
    }

    let is_stroke = sweep.sweep_mode.is_stroke();
    let to_display = |x: f64| -> f64 {
        if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) }
    };

    if let Some((start, end)) = sweep.active_range {
        // Full curve -- faded/dashed for context.
        let full: PlotPoints = x_deg_and_y
            .iter()
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(format!("{} (full)", name), full)
                .color(faded_color(color))
                .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                .width(width * 0.5),
        );

        // Active range -- solid.
        // Use the degree boundaries to select the active subset.
        let min_deg = sweep.angles_deg.get(start).copied().unwrap_or(0.0);
        let max_deg = sweep.angles_deg.get(end).copied().unwrap_or(360.0);
        let active: PlotPoints = x_deg_and_y
            .iter()
            .filter(|&&(x_deg, _)| x_deg >= min_deg && x_deg <= max_deg)
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(name, active)
                .color(color)
                .width(width),
        );
    } else {
        // No range limit -- draw normally.
        let pts: PlotPoints = x_deg_and_y
            .iter()
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(name, pts)
                .color(color)
                .width(width),
        );
    }
}

/// A palette of distinguishable colors for plot series.
///
/// When `nathan_mode` is true, all colors are converted to grayscale.
fn series_colors(nathan_mode: bool) -> Vec<egui::Color32> {
    use crate::gui::canvas::to_grayscale;
    let raw = vec![
        egui::Color32::from_rgb(100, 200, 255), // light blue
        egui::Color32::from_rgb(255, 150, 80),  // orange
        egui::Color32::from_rgb(120, 220, 120), // green
        egui::Color32::from_rgb(255, 100, 100), // red
        egui::Color32::from_rgb(200, 150, 255), // purple
        egui::Color32::from_rgb(255, 220, 100), // yellow
        egui::Color32::from_rgb(150, 200, 200), // teal
        egui::Color32::from_rgb(255, 150, 200), // pink
    ];
    if nathan_mode {
        raw.into_iter().map(to_grayscale).collect()
    } else {
        raw
    }
}
