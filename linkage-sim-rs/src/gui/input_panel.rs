//! Angle slider, playback controls, load case selector, and solver status display.

use eframe::egui;
use super::state::AppState;

/// Draw the input panel with animation controls and load case management.
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    // ── Load case selector ──────────────────────────────────────────
    draw_load_case_selector(ui, state);

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

/// Draw the load case selector: ComboBox for switching, +/- buttons, and
/// an editable name field for the active case.
fn draw_load_case_selector(ui: &mut egui::Ui, state: &mut AppState) {
    if state.load_cases.cases.is_empty() {
        return;
    }

    ui.separator();
    ui.strong("Load Case");

    // Collect values needed for the combo box to avoid borrow issues
    let active_index = state.load_cases.active_index;
    let active_name = state.load_cases.cases[active_index].name.clone();
    let case_count = state.load_cases.cases.len();

    // Combo box + add/remove buttons on the same row
    let mut new_active: Option<usize> = None;
    let mut add_case = false;
    let mut remove_case = false;

    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("load_case_selector")
            .selected_text(&active_name)
            .show_ui(ui, |ui| {
                for i in 0..case_count {
                    let case_name = state.load_cases.cases[i].name.clone();
                    if ui
                        .selectable_label(i == active_index, &case_name)
                        .clicked()
                    {
                        new_active = Some(i);
                    }
                }
            });

        if ui.button("+").on_hover_text("Add load case (copy current)").clicked() {
            add_case = true;
        }

        let remove_enabled = case_count > 1;
        if ui
            .add_enabled(remove_enabled, egui::Button::new("-"))
            .on_hover_text("Remove current load case")
            .clicked()
        {
            remove_case = true;
        }
    });

    // Editable name for the active case
    ui.horizontal(|ui| {
        ui.label("Name:");
        let name = &mut state.load_cases.cases[active_index].name;
        ui.text_edit_singleline(name);
    });

    // Apply deferred actions after UI rendering to avoid borrow conflicts
    if let Some(idx) = new_active {
        state.apply_load_case(idx);
    }
    if add_case {
        state.add_load_case();
    }
    if remove_case {
        state.remove_active_load_case();
    }
}
