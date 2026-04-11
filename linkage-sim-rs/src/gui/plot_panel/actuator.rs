//! Actuator plots: force, speed, power.

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text as PlotText, VLine};

use crate::gui::state::{AngleUnit, DisplayUnits};
use crate::gui::sweep::SweepData;

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, filter_actuator_outliers, series_colors, x_axis_label_for_sweep,
};

pub(super) fn draw_actuator_force(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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
        let pairs = filter_actuator_outliers(pairs);

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
            let id_pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(id_forces.iter())
                .filter(|&(_, &f)| f.is_finite())
                .map(|(&x_deg, &f)| (x_deg, f))
                .collect();
            let id_pairs = filter_actuator_outliers(id_pairs);

            let is_stroke = sweep.sweep_mode.is_stroke();
            let id_points: PlotPoints = id_pairs
                .iter()
                .map(|&(x, f)| {
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

pub(super) fn draw_actuator_stroke_annotation(
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

pub(super) fn draw_actuator_speed(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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

pub(super) fn draw_actuator_power(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
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
