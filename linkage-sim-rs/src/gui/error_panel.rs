//! Collapsible error/log panel displayed at the bottom of the window.

use eframe::egui;
use super::state::AppState;

/// Draw the error panel. Shows recent error messages with a clear button.
pub fn draw_error_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.strong("Errors");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Clear").clicked() {
                state.error_log.clear();
            }
            if ui.small_button("Hide").clicked() {
                state.show_error_panel = false;
            }
        });
    });

    ui.separator();

    egui::ScrollArea::vertical()
        .max_height(150.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for msg in &state.error_log {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), msg);
            }
            if state.error_log.is_empty() {
                ui.label("No errors.");
            }
        });
}
