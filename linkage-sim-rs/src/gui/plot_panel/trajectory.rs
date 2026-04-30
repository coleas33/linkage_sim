//! Trajectory-mode plot rendering. X-axis is time (s).
//!
//! Traces:
//!   - target(t)    (orange)
//!   - achieved(t)  (cyan)
//!   - residual(t)  (faint red, separate stacked plot)
//!   - u(t)         (separate stacked plot)
//!   - Failure bands at samples with status != Converged
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §8.3

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};

use crate::gui::state::AppState;
use crate::gui::sweep::{SweepData, SweepMode};
use crate::solver::inverse_kinematics::InverseSolveStatus;

/// Render trajectory-mode plots: target/achieved/residual on the main axis
/// and the back-solved input parameter `u(t)` stacked below.
///
/// Rendered when `state.sweep_mode == SweepMode::Trajectory { .. }`.
pub(super) fn render(state: &AppState, data: &SweepData, ui: &mut egui::Ui) {
    // Required trajectory series. If any are missing the data was generated
    // by a different sweep mode (or compute_trajectory bailed early); show
    // a hint and bail rather than panicking on the unwraps below.
    let (Some(target), Some(achieved), Some(residual), Some(u)) = (
        data.target_values.as_ref(),
        data.achieved_values.as_ref(),
        data.tracking_residual.as_ref(),
        data.u_values.as_ref(),
    ) else {
        ui.label(
            "No trajectory data yet — switch to Trajectory mode and click Compute.",
        );
        return;
    };

    let n = target.len();
    if n < 2 {
        ui.label("Trajectory has fewer than 2 samples; nothing to plot.");
        return;
    }

    // Time axis: uniform across [0, duration]. Duration is sourced from the
    // active SweepMode::Trajectory variant; if state.sweep_mode is somehow
    // out of sync (shouldn't happen — caller dispatches on it) fall back to
    // a unit-duration axis so we still render something useful.
    let duration = match &state.sweep_mode {
        SweepMode::Trajectory { profile, .. } => profile.duration,
        _ => 1.0,
    };
    let times: Vec<f64> = (0..n)
        .map(|i| (i as f64) * duration / ((n - 1) as f64))
        .collect();

    let statuses = data.inverse_solve_statuses.as_ref();

    // Available height is split between the two stacked plots: ~65% for
    // the main target/achieved/residual plot, the rest for u(t).
    let total_h = ui.available_height().max(180.0);
    let main_h = (total_h * 0.65).max(120.0);
    let u_h = (total_h - main_h - 8.0).max(80.0);

    // ── Main plot: target / achieved / residual ──────────────────────────
    Plot::new("trajectory_main")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label("Time (s)")
        .y_axis_label("Target / Achieved")
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(main_h)
        .show(ui, |plot_ui| {
            // Failure bands first, so series lines draw on top of them.
            if let Some(statuses) = statuses {
                draw_failure_bands(plot_ui, &times, statuses);
            }

            let pts_target: PlotPoints = times
                .iter()
                .zip(target.iter())
                .map(|(&t, &v)| [t, v])
                .collect();
            let pts_achieved: PlotPoints = times
                .iter()
                .zip(achieved.iter())
                .map(|(&t, &v)| [t, v])
                .collect();
            let pts_residual: PlotPoints = times
                .iter()
                .zip(residual.iter())
                .map(|(&t, &v)| [t, v])
                .collect();

            plot_ui.line(
                Line::new("target", pts_target)
                    .color(egui::Color32::from_rgb(255, 150, 80))
                    .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                    .width(1.5),
            );
            plot_ui.line(
                Line::new("achieved", pts_achieved)
                    .color(egui::Color32::from_rgb(100, 200, 255))
                    .width(2.0),
            );
            plot_ui.line(
                Line::new("residual (achieved - target)", pts_residual)
                    .color(egui::Color32::from_rgba_unmultiplied(255, 100, 100, 160))
                    .width(1.0),
            );
        });

    // ── Stacked u(t) plot ────────────────────────────────────────────────
    Plot::new("trajectory_u")
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label("Time (s)")
        .y_axis_label("u(t)")
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(u_h)
        .show(ui, |plot_ui| {
            if let Some(statuses) = statuses {
                draw_failure_bands(plot_ui, &times, statuses);
            }
            let pts: PlotPoints = times
                .iter()
                .zip(u.iter())
                .map(|(&t, &v)| [t, v])
                .collect();
            plot_ui.line(
                Line::new("u(t)", pts)
                    .color(egui::Color32::from_rgb(120, 220, 120))
                    .width(2.0),
            );
        });
}

/// Draw faint red vertical lines at sample times whose `InverseSolveStatus`
/// is anything other than `Converged`.
fn draw_failure_bands(
    plot_ui: &mut egui_plot::PlotUi,
    times: &[f64],
    statuses: &[InverseSolveStatus],
) {
    let band_color = egui::Color32::from_rgba_unmultiplied(220, 50, 50, 80);
    for (i, status) in statuses.iter().enumerate() {
        if matches!(status, InverseSolveStatus::Converged) {
            continue;
        }
        let Some(&t) = times.get(i) else { continue };
        plot_ui.vline(
            VLine::new(format!("failure_{}", i), t)
                .color(band_color)
                .width(1.0),
        );
    }
}
