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
use crate::gui::sweep::SweepMode;

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

    egui::CollapsingHeader::new("Visualization")
        .default_open(false)
        .show(ui, |ui| {
            draw_visualization_section(state, ui);
        });

    egui::CollapsingHeader::new("Failure handling")
        .default_open(false)
        .show(ui, |ui| {
            draw_severity_toggle(state, ui);
        });

    // Explicit recompute trigger. Auto-debounced recompute already fires after
    // a short delay when fields change, but new users don't know it exists —
    // a visible button makes the action discoverable and gives a way to force
    // an immediate recompute.
    ui.separator();
    ui.horizontal(|ui| {
        let compute_btn = egui::Button::new(
            egui::RichText::new("\u{23F5} Compute trajectory")
                .color(egui::Color32::WHITE)
                .strong(),
        )
        .fill(egui::Color32::from_rgb(40, 100, 200));
        if ui
            .add(compute_btn)
            .on_hover_text(
                "Run the inverse-kinematics solve over the active target and profile. \
                 Auto-recomputes after a short delay when fields change; this button \
                 forces an immediate recompute.",
            )
            .clicked()
        {
            state.mark_sweep_dirty();
        }
    });
}

fn draw_severity_toggle(state: &mut AppState, ui: &mut egui::Ui) {
    use crate::solver::inverse_kinematics::Severity;
    let current = state.trajectory_severity;
    let mut new = current;
    ui.horizontal(|ui| {
        ui.label("On failure:");
        ui.radio_value(&mut new, Severity::Analysis, "Continue")
            .on_hover_text(
                "Annotate failed samples in the trajectory; continue solving the rest.",
            );
        ui.radio_value(&mut new, Severity::Strict, "Abort")
            .on_hover_text("Abort the entire trajectory on the first failed sample.");
    });
    if new != current {
        state.trajectory_severity = new;
        // Sync into the active SweepMode::Trajectory payload so the next
        // compute_sweep picks up the change. Without this, the radio appears
        // effective but doesn't reach compute_trajectory until a separate
        // dirty-trigger fires.
        if let SweepMode::Trajectory { severity, .. } = &mut state.sweep_mode {
            *severity = new;
        }
        state.mark_sweep_dirty();
    }
}

/// Draw the "Visualization" section: motion-ribbon toggle + density slider,
/// plus the trajectory-time playback controls (Play / Pause / Stop, speed,
/// loop). Both features are trajectory-mode-only and require completed
/// sweep data to do anything visible.
fn draw_visualization_section(state: &mut AppState, ui: &mut egui::Ui) {
    // ── Motion ribbon ────────────────────────────────────────────────
    ui.checkbox(
        &mut state.show_motion_ribbon,
        "Show motion ribbon (ghost poses)",
    )
    .on_hover_text(
        "Render N evenly-spaced ghost poses of the mechanism along the \
         back-solved trajectory, faded behind the live pose. Useful for \
         visualising the swept path without animating.",
    );
    ui.add_enabled(
        state.show_motion_ribbon,
        egui::Slider::new(&mut state.motion_ribbon_n_ghosts, 2..=20).text("Ghosts"),
    )
    .on_hover_text("Number of ghost poses sampled across the trajectory.");

    ui.separator();

    // ── Trajectory playback ──────────────────────────────────────────
    let duration = if let SweepMode::Trajectory { trajectory, .. } = &state.sweep_mode {
        trajectory.duration()
    } else {
        0.0
    };

    ui.horizontal(|ui| {
        let play_label = if state.trajectory_playback_active {
            "\u{23F8} Pause"
        } else {
            "\u{25B6} Play trajectory"
        };
        if ui
            .button(play_label)
            .on_hover_text(
                "Animate the canvas through the back-solved q(t) at the \
                 trajectory's actual time scale (not the constant-omega \
                 driver-animation speed).",
            )
            .clicked()
        {
            state.trajectory_playback_active = !state.trajectory_playback_active;
            if state.trajectory_playback_active {
                // Reset t if at end, otherwise resume.
                if duration > 0.0 && state.trajectory_playback_t >= duration {
                    state.trajectory_playback_t = 0.0;
                }
                // Suspend the constant-omega kinematic animation —
                // both can't drive the canvas simultaneously.
                state.playing = false;
            }
        }
        if ui
            .button("\u{23F9} Stop")
            .on_hover_text("Stop playback and reset trajectory time to 0.")
            .clicked()
        {
            state.trajectory_playback_active = false;
            state.trajectory_playback_t = 0.0;
            state.last_trajectory_scrub_t = Some(0.0);
        }
        ui.add(
            egui::DragValue::new(&mut state.trajectory_playback_speed)
                .speed(0.05)
                .range(0.05..=4.0)
                .suffix("x"),
        )
        .on_hover_text("Playback speed multiplier (1.0 = real time).");
        ui.checkbox(&mut state.trajectory_playback_loop, "Loop")
            .on_hover_text("Loop back to t=0 at the end of the trajectory.");
    });

    if duration > 0.0 {
        ui.label(format!(
            "t = {:.3} s / {:.3} s",
            state.trajectory_playback_t.min(duration),
            duration,
        ));
    }
}
