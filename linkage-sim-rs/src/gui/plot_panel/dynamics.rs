//! Dynamics plots: driver torque, inverse dynamics, energy.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints, Text as PlotText, VLine};

use crate::gui::state::{AngleUnit, DisplayUnits};
use crate::gui::sweep::SweepData;

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, driver_effort_series_name, driver_effort_y_label, faded_color,
    series_colors, with_default_x_bounds, x_axis_label_for_sweep,
};

pub(super) fn draw_driver_torque(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "driver_torque_plot", sweep, units);

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

pub(super) fn draw_inverse_dynamics(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "inverse_dynamics_plot", sweep, units);

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

        // Overlay profile torque if a non-constant motion profile is active (green).
        if let Some(ref profile_torques) = sweep.profile_torques {
            let prof_pairs: Vec<(f64, f64)> = sweep
                .angles_deg
                .iter()
                .zip(profile_torques.iter())
                .filter(|&(_, &t)| t.is_finite())
                .map(|(&x_deg, &t)| (x_deg, t))
                .collect();
            draw_angle_series_with_range(
                plot_ui,
                "Profile Torque",
                egui::Color32::from_rgb(120, 220, 120),
                2.0,
                &prof_pairs,
                sweep,
                units,
                nathan_mode,
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

pub(super) fn draw_energy(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "energy_plot", sweep, units);

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
