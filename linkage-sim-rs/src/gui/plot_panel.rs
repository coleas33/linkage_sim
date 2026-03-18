//! Plot panel: sweep data visualization using egui_plot.
//!
//! Shows tabbed plots for coupler trace, body angles, and transmission angle.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};

use super::state::AppState;

/// Selected plot tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlotTab {
    CouplerTrace,
    BodyAngles,
    TransmissionAngle,
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
    });

    ui.memory_mut(|mem| mem.data.insert_temp(tab_id, selected_tab));

    ui.separator();

    match selected_tab {
        PlotTab::CouplerTrace => draw_coupler_trace(ui, sweep),
        PlotTab::BodyAngles => draw_body_angles(ui, sweep, state.driver_angle.to_degrees()),
        PlotTab::TransmissionAngle => {
            draw_transmission_angle(ui, sweep, state.driver_angle.to_degrees());
        }
    }
}

/// Plot coupler point traces: x vs y in world coordinates.
fn draw_coupler_trace(ui: &mut egui::Ui, sweep: &super::state::SweepData) {
    let plot = Plot::new("coupler_trace_plot")
        .data_aspect(1.0) // equal axis scaling
        .x_axis_label("X (m)")
        .y_axis_label("Y (m)")
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

            let points: PlotPoints = trace.iter().map(|[x, y]| [*x, *y]).collect();
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

/// Plot body angles (degrees) vs driver angle (degrees).
fn draw_body_angles(ui: &mut egui::Ui, sweep: &super::state::SweepData, current_angle_deg: f64) {
    let plot = Plot::new("body_angles_plot")
        .x_axis_label("Driver Angle (deg)")
        .y_axis_label("Body Angle (deg)")
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
                .map(|(&x, &y)| [x, y])
                .collect();

            let color = colors[color_idx % colors.len()];
            plot_ui.line(
                Line::new(body_id.as_str(), points)
                    .color(color)
                    .width(1.5),
            );
            color_idx += 1;
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_angle_deg)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );
    });
}

/// Plot transmission angle (degrees) vs driver angle (degrees).
fn draw_transmission_angle(
    ui: &mut egui::Ui,
    sweep: &super::state::SweepData,
    current_angle_deg: f64,
) {
    let Some(ta) = &sweep.transmission_angles else {
        ui.label("Transmission angle not available for this mechanism.");
        return;
    };

    let plot = Plot::new("transmission_angle_plot")
        .x_axis_label("Driver Angle (deg)")
        .y_axis_label("Transmission Angle (deg)")
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        let points: PlotPoints = sweep
            .angles_deg
            .iter()
            .zip(ta.iter())
            .map(|(&x, &y)| [x, y])
            .collect();

        plot_ui.line(
            Line::new("Transmission Angle", points)
                .color(egui::Color32::from_rgb(100, 200, 255))
                .width(2.0),
        );

        // Ideal zone: 40-140 degrees.
        let ideal_low: PlotPoints = [[0.0, 40.0], [360.0, 40.0]].into_iter().collect();
        let ideal_high: PlotPoints = [[0.0, 140.0], [360.0, 140.0]].into_iter().collect();
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

        // Vertical marker at current angle.
        plot_ui.vline(
            VLine::new("cursor", current_angle_deg)
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
