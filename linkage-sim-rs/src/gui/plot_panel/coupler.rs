//! Coupler and output force plots: coupler trace, velocity, acceleration, output force.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};

use crate::gui::state::{AngleUnit, DisplayUnits};
use crate::gui::sweep::SweepData;

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, series_colors, x_axis_label_for_sweep,
};

pub(super) fn draw_coupler_trace(ui: &mut egui::Ui, sweep: &SweepData, units: &DisplayUnits, nathan_mode: bool) {
    let axis_label = units.length_axis_label();
    let plot = Plot::new("coupler_trace_plot")
        .data_aspect(1.0) // equal axis scaling
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(format!("X ({})", axis_label))
        .y_axis_label(format!("Y ({})", axis_label))
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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

pub(super) fn draw_coupler_velocity(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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

pub(super) fn draw_coupler_acceleration(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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

pub(super) fn draw_output_force(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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
