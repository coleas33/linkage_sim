//! Angle slider and solver status display.

use eframe::egui;
use super::state::AppState;

/// Draw the input panel (in the left side panel).
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    let mut angle_deg = state.driver_angle.to_degrees();
    let prev_angle = angle_deg;

    ui.horizontal(|ui| {
        ui.label("Crank angle:");
        ui.add(
            egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                .suffix("°")
                .step_by(0.5),
        );
    });

    // Re-solve if angle changed
    if (angle_deg - prev_angle).abs() > 1e-6 {
        state.solve_at_angle(angle_deg.to_radians());
    }

    // Solver status
    ui.separator();
    let status = &state.solver_status;
    ui.horizontal(|ui| {
        let (icon, color) = if status.converged {
            ("●", egui::Color32::from_rgb(80, 200, 80))
        } else {
            ("●", egui::Color32::from_rgb(200, 60, 60))
        };
        ui.colored_label(color, icon);
        if status.converged {
            ui.label(format!(
                "Converged in {} iters (‖Φ‖ = {:.2e})",
                status.iterations, status.residual_norm
            ));
        } else {
            ui.label(format!(
                "FAILED — ‖Φ‖ = {:.2e} (showing last good pose)",
                status.residual_norm
            ));
        }
    });
}
