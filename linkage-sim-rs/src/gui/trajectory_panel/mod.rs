//! Top-level UI for `SweepMode::Trajectory`.
//!
//! Three sections (top to bottom):
//!   1. Target picker — `ControlTarget` variant + body / point / axis fields.
//!   2. Profile editor — `TrajectoryProfile` + inline h(t) preview.
//!   3. Severity & advanced options.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §8

mod profile_input;
mod target_picker;

use eframe::egui;

use crate::gui::state::AppState;

/// Render the trajectory input panel. Called from `gui/input_panel.rs` when
/// `SweepMode::Trajectory` is active.
pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Target observable")
        .default_open(true)
        .show(ui, |ui| {
            target_picker::draw(state, ui);
        });

    egui::CollapsingHeader::new("Profile")
        .default_open(true)
        .show(ui, |ui| {
            profile_input::draw(state, ui);
        });

    egui::CollapsingHeader::new("Solve options")
        .default_open(false)
        .show(ui, |ui| {
            draw_severity_toggle(state, ui);
        });
}

fn draw_severity_toggle(state: &mut AppState, ui: &mut egui::Ui) {
    use crate::solver::inverse_kinematics::Severity;
    let current = state.trajectory_severity;
    let mut new = current;
    ui.horizontal(|ui| {
        ui.label("Severity:");
        ui.radio_value(&mut new, Severity::Analysis, "Analysis (annotate failures)");
        ui.radio_value(&mut new, Severity::Strict, "Strict (abort on failure)");
    });
    if new != current {
        state.trajectory_severity = new;
    }
}
