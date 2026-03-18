//! Angle slider, playback controls, and solver status display.

use eframe::egui;
use super::state::AppState;

/// Draw the input panel with animation controls.
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    // ── Playback controls ────────────────────────────────────────────
    ui.horizontal(|ui| {
        let play_label = if state.playing { "Pause" } else { "Play" };
        if ui.button(play_label).clicked() {
            state.playing = !state.playing;
            if state.playing && !state.loop_mode {
                state.animation_direction = 1.0;
            }
        }

        ui.label("Speed:");
        ui.add(
            egui::Slider::new(&mut state.animation_speed_deg_per_sec, 10.0..=720.0)
                .suffix(" \u{00B0}/s")
                .logarithmic(true)
                .clamping(egui::SliderClamping::Always),
        );
    });

    ui.horizontal(|ui| {
        let mode_label = if state.loop_mode { "Loop" } else { "Once" };
        if ui.button(mode_label).clicked() {
            state.loop_mode = !state.loop_mode;
            state.animation_direction = 1.0;
        }
        if state.loop_mode {
            ui.label("(continuous, ping-pong at limits)");
        } else {
            ui.label("(sweep forward, stop at 360\u{00B0})");
        }
    });

    // ── Angle slider ─────────────────────────────────────────────────
    let mut angle_deg = state.driver_angle.to_degrees();
    let prev_angle = angle_deg;

    ui.horizontal(|ui| {
        ui.label("Crank angle:");
        let response = ui.add(
            egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                .suffix("\u{00B0}")
                .step_by(0.5),
        );
        if response.dragged() && state.playing {
            state.playing = false;
            state.animation_direction = 1.0;
        }
    });

    if (angle_deg - prev_angle).abs() > 1e-6 {
        state.solve_at_angle(angle_deg.to_radians());
    }

    // ── Solver status ────────────────────────────────────────────────
    ui.separator();
    let status = &state.solver_status;
    ui.horizontal(|ui| {
        let color = if status.converged {
            egui::Color32::from_rgb(80, 200, 80)
        } else {
            egui::Color32::from_rgb(200, 60, 60)
        };
        ui.colored_label(color, "\u{25CF}");
        if status.converged {
            ui.label(format!(
                "Converged in {} iters (r = {:.2e})",
                status.iterations, status.residual_norm
            ));
        } else {
            ui.label(format!(
                "FAILED (r = {:.2e}) \u{2014} last good pose",
                status.residual_norm
            ));
        }
    });

    // ── Driver info ──────────────────────────────────────────────────
    if let Some(joint_id) = &state.driver_joint_id {
        ui.label(format!("Driver: {} (right-click joint to change)", joint_id));
    }
}
