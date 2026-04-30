//! Trajectory-mode plot rendering. X-axis is time (s).
//!
//! Traces:
//!   - target(t)    (orange)
//!   - achieved(t)  (cyan)
//!   - residual(t)  (faint red, separate stacked plot)
//!   - u(t)         (separate stacked plot)
//!   - Failure bands at samples with status != Converged
//!
//! Clicking the main plot scrubs the canvas-displayed mechanism to the
//! configuration corresponding to the clicked time `t_k` by calling
//! `state.solve_for_trajectory_target(target, h)` with `h = profile.evaluate(t_k).0`.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §8.3

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, VLine};

use crate::gui::state::AppState;
use crate::gui::sweep::SweepMode;
use crate::solver::inverse_kinematics::InverseSolveStatus;

/// Render trajectory-mode plots: target/achieved/residual on the main axis
/// and the back-solved input parameter `u(t)` stacked below.
///
/// Rendered when `state.sweep_mode == SweepMode::Trajectory { .. }`.
/// Takes `&mut state` so that clicks on the main plot can drive the canvas
/// mechanism to the clicked sample's configuration via
/// `state.solve_for_trajectory_target`.
pub(super) fn render(state: &mut AppState, ui: &mut egui::Ui) {
    // Required trajectory series. If any are missing the data was generated
    // by a different sweep mode (or compute_trajectory bailed early); show
    // a hint and bail rather than panicking on the unwraps below.
    let Some(data) = state.sweep_data.as_ref() else {
        ui.label("No sweep data available.");
        return;
    };
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
    // Capture click coordinate inside the closure; act on it after the
    // immutable borrow of `state.sweep_data` ends (we need `&mut state` to
    // call solve_for_trajectory_target).
    let mut clicked_t: Option<f64> = None;
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

            clicked_t = super::detect_plot_click(plot_ui);
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

    // ── Failure summary ──────────────────────────────────────────────────
    // egui_plot 0.33's VLine has no native per-shape hover tooltip. Rather
    // than fiddle with a synthetic Text marker, list the failure samples in
    // a collapsing section under the plots — the band positions on the plot
    // make the t-mapping obvious, and this gives full-text payloads.
    if let Some(statuses) = statuses {
        let failures: Vec<(usize, &InverseSolveStatus)> = statuses
            .iter()
            .enumerate()
            .filter(|(_, s)| !matches!(s, InverseSolveStatus::Converged))
            .collect();
        if !failures.is_empty() {
            ui.collapsing(
                format!("\u{26A0} {} sample(s) failed to converge", failures.len()),
                |ui| {
                    for (i, status) in &failures {
                        ui.label(format!(
                            "t={:.3}s: {}",
                            times[*i],
                            format_failure_tooltip(status)
                        ));
                    }
                },
            );
        }
    }

    // ── Click-to-scrub ───────────────────────────────────────────────────
    // The plot closures above hold an immutable borrow on `state.sweep_data`;
    // by this point those closures have returned and the borrow is dropped.
    // Now resolve the click into (target, h) and drive the canvas pose.
    if let Some(t_raw) = clicked_t {
        let t_clicked = t_raw.clamp(0.0, duration);
        // Pull target+h out of the trajectory sweep mode and drop the borrow
        // before calling the &mut self method.
        let payload = if let SweepMode::Trajectory { target, profile, .. } = &state.sweep_mode {
            let (h, _, _) = profile.evaluate(t_clicked);
            Some((target.clone(), h))
        } else {
            None
        };
        if let Some((target, h)) = payload {
            state.solve_for_trajectory_target(&target, h);
        }
    }
}

/// Render an `InverseSolveStatus` failure variant as a single-line summary
/// string suitable for the failure-band tooltip / collapsing list.
///
/// Returns the literal `"Converged"` for the no-failure variant, but in
/// practice this is only called from the failure-summary path where
/// `Converged` is filtered out.
fn format_failure_tooltip(status: &InverseSolveStatus) -> String {
    match status {
        InverseSolveStatus::Converged => "Converged".to_string(),
        InverseSolveStatus::Reachability {
            target,
            achieved_clamp,
            workspace_min,
            workspace_max,
        } => {
            let min = workspace_min
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-inf".to_string());
            let max = workspace_max
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "+inf".to_string());
            format!(
                "Reachability: target={:.4} clamped to {:.4} (workspace [{}, {}])",
                target, achieved_clamp, min, max
            )
        }
        InverseSolveStatus::Singularity { dg_du } => {
            format!("Singularity: |dg/du| = {:.2e}", dg_du)
        }
        InverseSolveStatus::BranchJump { delta_q_norm } => {
            format!("Branch jump: \u{2016}\u{0394}q\u{2016} = {:.4}", delta_q_norm)
        }
        InverseSolveStatus::NonConvergent {
            iterations,
            residual,
        } => {
            format!(
                "Non-convergent after {} iter (residual = {:.2e})",
                iterations, residual
            )
        }
    }
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
