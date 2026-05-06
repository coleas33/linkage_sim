//! Mechanics plots: body angles, transmission angle, mechanical advantage, joint reactions.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints, Text as PlotText, VLine};

use crate::gui::state::{AngleUnit, DisplayUnits};
use crate::gui::sweep::SweepData;

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, series_colors, with_default_x_bounds, x_axis_label_for_sweep,
};

pub(super) fn draw_body_angles(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "body_angles_plot", sweep, units);

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

pub(super) fn draw_transmission_angle(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "transmission_angle_plot", sweep, units);

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

pub(super) fn draw_mechanical_advantage(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "mechanical_advantage_plot", sweep, units);

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

pub(super) fn draw_joint_reactions(
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
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, "joint_reactions_plot", sweep, units);

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
