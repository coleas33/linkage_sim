//! Plot panel: sweep data visualization using egui_plot.
//!
//! Shows tabbed plots for coupler trace, body angles, and transmission angle.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};

use super::state::{AppState, DisplayUnits};

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
}

/// Draw the plot panel with tabbed plots.
pub fn draw_plot_panel(ui: &mut egui::Ui, state: &AppState) {
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
        ui.add_enabled_ui(has_dt, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::DriverTorque,
                "Driver Torque",
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
    });

    ui.memory_mut(|mem| mem.data.insert_temp(tab_id, selected_tab));

    ui.separator();

    let current_driver_display = state.display_units.angle(state.driver_angle);
    match selected_tab {
        PlotTab::CouplerTrace => draw_coupler_trace(ui, sweep, &state.display_units),
        PlotTab::BodyAngles => draw_body_angles(ui, sweep, current_driver_display, &state.display_units),
        PlotTab::TransmissionAngle => {
            draw_transmission_angle(ui, sweep, current_driver_display, &state.display_units);
        }
        PlotTab::DriverTorque => {
            draw_driver_torque(ui, sweep, current_driver_display, &state.display_units);
        }
        PlotTab::InverseDynamics => {
            draw_inverse_dynamics(ui, sweep, current_driver_display, &state.display_units);
        }
        PlotTab::Energy => {
            draw_energy(ui, sweep, current_driver_display, &state.display_units);
        }
        PlotTab::MechanicalAdvantage => {
            draw_mechanical_advantage(ui, sweep, current_driver_display, &state.display_units);
        }
    }
}

/// Plot coupler point traces: x vs y, converted to display length units.
fn draw_coupler_trace(ui: &mut egui::Ui, sweep: &super::state::SweepData, units: &DisplayUnits) {
    let axis_label = units.length_axis_label();
    let plot = Plot::new("coupler_trace_plot")
        .data_aspect(1.0) // equal axis scaling
        .x_axis_label(format!("X ({})", axis_label))
        .y_axis_label(format!("Y ({})", axis_label))
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let colors = series_colors();
        let mut color_idx = 0;

        let mut keys: Vec<&String> = sweep.coupler_traces.keys().collect();
        keys.sort();

        for key in keys {
            let trace = &sweep.coupler_traces[key];
            if trace.is_empty() {
                continue;
            }

            let points: PlotPoints = trace
                .iter()
                .map(|[x, y]| [units.length(*x), units.length(*y)])
                .collect();
            let color = colors[color_idx % colors.len()];
            plot_ui.line(
                Line::new(key.as_str(), points)
                    .color(color)
                    .width(1.5),
            );
            color_idx += 1;
        }
    });
}

/// Plot body angles vs driver angle, using the current display angle unit.
fn draw_body_angles(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    // Sweep stores angles in degrees; convert to display unit on the fly.
    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };
    let plot = Plot::new("body_angles_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label(format!("Body Angle ({})", angle_label))
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let colors = series_colors();
        let mut color_idx = 0;

        let mut body_ids: Vec<&String> = sweep.body_angles.keys().collect();
        body_ids.sort();

        for body_id in body_ids {
            let angles = &sweep.body_angles[body_id];
            let points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(angles.iter())
                // Sweep data is in degrees; convert both axes to display unit.
                .map(|(&x_deg, &y_deg)| {
                    let x = units.angle(x_deg.to_radians());
                    let y = units.angle(y_deg.to_radians());
                    [x, y]
                })
                .collect();

            let color = colors[color_idx % colors.len()];
            plot_ui.line(
                Line::new(body_id.as_str(), points)
                    .color(color)
                    .width(1.5),
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle (already in display units).
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );
    });
}

/// Plot transmission angle (degrees) vs driver angle.
///
/// Transmission angle data is always in degrees; the x-axis driver angle is
/// shown in the current display unit.
fn draw_transmission_angle(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    let Some(ta) = &sweep.transmission_angles else {
        ui.label("Transmission angle not available for this mechanism.");
        return;
    };

    // The x-axis driver angle is shown in the current display angle unit.
    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };
    // The x-axis span in display units (0 to 2π).
    let x_max = units.angle(2.0 * std::f64::consts::PI);

    let plot = Plot::new("transmission_angle_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label("Transmission Angle (deg)")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(ta.iter())
            // x: driver angle in display unit; y: transmission angle always in degrees.
            .map(|(&x_deg, &y)| [units.angle(x_deg.to_radians()), y])
            .collect();

        plot_ui.line(
            Line::new("Transmission Angle", points)
                .color(egui::Color32::from_rgb(100, 200, 255))
                .width(2.0),
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

        // Vertical marker at current driver angle (already in display units).
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );
    });
}

/// Plot driver torque (N*m) vs driver angle.
fn draw_driver_torque(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    let Some(torques) = &sweep.driver_torques else {
        ui.label("Driver torque data not available.");
        return;
    };

    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };

    let plot = Plot::new("driver_torque_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label("Driver Torque (N\u{00b7}m)")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(torques.iter())
            .map(|(&x_deg, &y)| [units.angle(x_deg.to_radians()), y])
            .collect();

        plot_ui.line(
            Line::new("Driver Torque", points)
                .color(egui::Color32::from_rgb(255, 150, 80))
                .width(2.0),
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );
    });
}

/// Plot inverse dynamics driver torque vs driver angle,
/// with optional statics torque overlay for comparison.
fn draw_inverse_dynamics(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    if sweep.inverse_dynamics_torques.is_empty() {
        ui.label("Inverse dynamics data not available.");
        return;
    }

    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };

    let plot = Plot::new("inverse_dynamics_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label("Driver Torque (N\u{00b7}m)")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        // Inverse dynamics torque (cyan).
        let id_points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(sweep.inverse_dynamics_torques.iter())
            .filter(|&(_, &t)| t.is_finite())
            .map(|(&x_deg, &t)| [units.angle(x_deg.to_radians()), t])
            .collect();
        plot_ui.line(
            Line::new("Inverse Dynamics Torque", id_points)
                .color(egui::Color32::from_rgb(100, 200, 255))
                .width(2.0),
        );

        // Overlay statics torque if available (orange, dashed).
        if let Some(statics_torques) = &sweep.driver_torques {
            let st_points: PlotPoints = sweep
                .angles_deg
                .iter()
                .zip(statics_torques.iter())
                .filter(|&(_, &t)| t.is_finite())
                .map(|(&x_deg, &t)| [units.angle(x_deg.to_radians()), t])
                .collect();
            plot_ui.line(
                Line::new("Statics Torque", st_points)
                    .color(egui::Color32::from_rgb(255, 150, 80))
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
    });
}

/// Plot energy (KE, PE, total) vs driver angle.
fn draw_energy(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    if sweep.kinetic_energy.is_empty() {
        ui.label("Energy data not available (velocity solve needed).");
        return;
    }

    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };

    let plot = Plot::new("energy_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label("Energy (J)")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        // Kinetic energy
        let ke_points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(sweep.kinetic_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| [units.angle(x_deg.to_radians()), e])
            .collect();
        plot_ui.line(
            Line::new("Kinetic Energy", ke_points)
                .color(egui::Color32::from_rgb(255, 150, 80))
                .width(2.0),
        );

        // Potential energy
        let pe_points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(sweep.potential_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| [units.angle(x_deg.to_radians()), e])
            .collect();
        plot_ui.line(
            Line::new("Potential Energy", pe_points)
                .color(egui::Color32::from_rgb(100, 200, 255))
                .width(2.0),
        );

        // Total energy
        let te_points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(sweep.total_energy.iter())
            .filter(|&(_, &e)| e.is_finite())
            .map(|(&x_deg, &e)| [units.angle(x_deg.to_radians()), e])
            .collect();
        plot_ui.line(
            Line::new("Total Energy", te_points)
                .color(egui::Color32::from_rgb(120, 220, 120))
                .width(2.0),
        );

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );
    });
}

/// Plot mechanical advantage (dimensionless) vs driver angle.
fn draw_mechanical_advantage(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
) {
    if sweep.mechanical_advantage.is_empty() {
        ui.label("Mechanical advantage data not available.");
        return;
    }

    let angle_label = match units.angle {
        super::state::AngleUnit::Degrees => "deg",
        super::state::AngleUnit::Radians => "rad",
    };

    let plot = Plot::new("mechanical_advantage_plot")
        .x_axis_label(format!("Driver Angle ({})", angle_label))
        .y_axis_label("Mechanical Advantage")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(sweep.mechanical_advantage.iter())
            .filter(|&(_, &ma)| ma.is_finite())
            .map(|(&x_deg, &ma)| [units.angle(x_deg.to_radians()), ma])
            .collect();

        plot_ui.line(
            Line::new("Mechanical Advantage", points)
                .color(egui::Color32::from_rgb(200, 150, 255))
                .width(2.0),
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
    });
}

/// A palette of distinguishable colors for plot series.
fn series_colors() -> Vec<egui::Color32> {
    vec![
        egui::Color32::from_rgb(100, 200, 255), // light blue
        egui::Color32::from_rgb(255, 150, 80),  // orange
        egui::Color32::from_rgb(120, 220, 120), // green
        egui::Color32::from_rgb(255, 100, 100), // red
        egui::Color32::from_rgb(200, 150, 255), // purple
        egui::Color32::from_rgb(255, 220, 100), // yellow
        egui::Color32::from_rgb(150, 200, 200), // teal
        egui::Color32::from_rgb(255, 150, 200), // pink
    ]
}
