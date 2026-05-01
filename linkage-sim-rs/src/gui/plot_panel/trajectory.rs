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
use egui_plot::{Line, Plot, PlotPoint, PlotPoints, Text as PlotText, VLine};

use crate::gui::state::AppState;
use crate::gui::sweep::{SweepData, SweepMode};
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
            "No trajectory data yet \u{2014} switch to Trajectory mode and click Compute trajectory.",
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
        SweepMode::Trajectory { trajectory, .. } => trajectory.duration(),
        _ => 1.0,
    };
    let times: Vec<f64> = (0..n)
        .map(|i| (i as f64) * duration / ((n - 1) as f64))
        .collect();

    let statuses = data.inverse_solve_statuses.as_ref();

    // Optional comparison snapshot — a frozen SweepData captured by the
    // user via "Save as comparison" in the trajectory panel. Rendered
    // beneath the live traces in dotted style so the user can A/B two
    // profiles or targets on the same axes. Time axis comes from the
    // snapshot's own SweepMode::Trajectory duration (which may differ
    // from the live duration).
    let comparison = state.trajectory_comparison.as_ref();
    let comparison_traces = comparison.and_then(comparison_traces_from);

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
    let scrub_t_for_cursor = state.last_trajectory_scrub_t;
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

            // Comparison overlay: dotted, faded, and labelled "(prev)" so
            // the live traces stay visually dominant.
            if let Some(cmp) = &comparison_traces {
                let pts_t: PlotPoints =
                    cmp.times.iter().zip(cmp.target.iter()).map(|(&t, &v)| [t, v]).collect();
                let pts_a: PlotPoints =
                    cmp.times.iter().zip(cmp.achieved.iter()).map(|(&t, &v)| [t, v]).collect();
                plot_ui.line(
                    Line::new("target (prev)", pts_t)
                        .color(egui::Color32::from_rgba_unmultiplied(255, 150, 80, 130))
                        .style(egui_plot::LineStyle::Dotted { spacing: 6.0 })
                        .width(1.2),
                );
                plot_ui.line(
                    Line::new("achieved (prev)", pts_a)
                        .color(egui::Color32::from_rgba_unmultiplied(100, 200, 255, 130))
                        .style(egui_plot::LineStyle::Dotted { spacing: 6.0 })
                        .width(1.5),
                );
            }

            // Persistent click-to-scrub cursor at the most recently clicked
            // time. Gold/yellow distinguishes from the red failure bands.
            if let Some(t_cur) = scrub_t_for_cursor {
                plot_ui.vline(
                    VLine::new("scrub_cursor", t_cur)
                        .color(egui::Color32::from_rgba_premultiplied(255, 215, 0, 200))
                        .style(egui_plot::LineStyle::Solid),
                );
            }

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
            if let Some(cmp) = &comparison_traces {
                let pts_u: PlotPoints =
                    cmp.times.iter().zip(cmp.u.iter()).map(|(&t, &v)| [t, v]).collect();
                plot_ui.line(
                    Line::new("u(t) (prev)", pts_u)
                        .color(egui::Color32::from_rgba_unmultiplied(120, 220, 120, 130))
                        .style(egui_plot::LineStyle::Dotted { spacing: 6.0 })
                        .width(1.5),
                );
            }
        });

    // ── Failure summary ──────────────────────────────────────────────────
    // The failure bands now carry a glyph (R/S/B/N) at the top of the plot
    // so severity is readable at a glance, but egui_plot 0.33's Text item
    // doesn't surface per-shape hover tooltips (it has PlotGeometry::None,
    // which the hit-test routines skip). The collapsing list below the
    // plot remains the place for full-text per-failure detail.
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
        // Persist the cursor position so the VLine renders next frame.
        state.last_trajectory_scrub_t = Some(t_clicked);
        // Pull target+h out of the trajectory sweep mode and drop the borrow
        // before calling the &mut self method.
        let payload = if let SweepMode::Trajectory { target, trajectory, .. } = &state.sweep_mode {
            let (h, _, _) = trajectory.evaluate(t_clicked);
            Some((target.clone(), h))
        } else {
            None
        };
        if let Some((target, h)) = payload {
            state.solve_for_trajectory_target(&target, h);
        }
    }
}

/// Comparison-overlay traces extracted from a frozen `SweepData` snapshot.
/// Carries its own time axis because the snapshot's trajectory duration may
/// differ from the live trajectory's (e.g. the user is comparing 1.0s vs
/// 2.0s profiles).
struct ComparisonTraces<'a> {
    times: Vec<f64>,
    target: &'a [f64],
    achieved: &'a [f64],
    u: &'a [f64],
}

/// Build a `ComparisonTraces` view if the snapshot is in trajectory mode and
/// has the required series populated. Returns `None` for non-trajectory or
/// partially-populated snapshots so the plot just skips overlay rendering.
fn comparison_traces_from(snapshot: &SweepData) -> Option<ComparisonTraces<'_>> {
    let target = snapshot.target_values.as_deref()?;
    let achieved = snapshot.achieved_values.as_deref()?;
    let u = snapshot.u_values.as_deref()?;
    let n = target.len();
    if n < 2 || achieved.len() != n || u.len() != n {
        return None;
    }
    let duration = match &snapshot.sweep_mode {
        SweepMode::Trajectory { trajectory, .. } => trajectory.duration(),
        _ => return None,
    };
    let times: Vec<f64> = (0..n)
        .map(|i| (i as f64) * duration / ((n - 1) as f64))
        .collect();
    Some(ComparisonTraces {
        times,
        target,
        achieved,
        u,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::{MotionProfile, Trajectory, TrajectoryProfile};
    use crate::gui::sweep::{empty_trajectory_sweep_data, SweepMode};
    use crate::solver::inverse_kinematics::{ControlTarget, Severity};

    fn fixture(n: usize, duration: f64) -> SweepData {
        let mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: MotionProfile::ConstantSpeed,
                start_value: 0.0,
                end_value: 1.0,
                duration,
            }),
            severity: Severity::Analysis,
            n_samples: n,
        };
        let mut data = empty_trajectory_sweep_data(mode);
        let v: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        data.target_values = Some(v.clone());
        data.achieved_values = Some(v.clone());
        data.u_values = Some(v);
        data
    }

    #[test]
    fn comparison_traces_uses_snapshot_duration_not_live_duration() {
        // Snapshot at 2.0s; live duration would be different. The overlay
        // must render along the snapshot's own time axis so the user
        // sees an honest A/B of two different durations.
        let snap = fixture(5, 2.0);
        let cmp = comparison_traces_from(&snap).expect("trajectory data should yield traces");
        assert_eq!(cmp.times.len(), 5);
        assert!((cmp.times[0]).abs() < 1e-9);
        assert!((cmp.times[4] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn comparison_traces_returns_none_for_non_trajectory_mode() {
        // A SweepData carrying SweepMode::Angle is not eligible for overlay,
        // even if the trajectory series happen to be populated.
        let mut snap = fixture(5, 1.0);
        snap.sweep_mode = SweepMode::Angle;
        assert!(comparison_traces_from(&snap).is_none());
    }

    #[test]
    fn comparison_traces_returns_none_for_partial_snapshot() {
        // Drop u_values: the overlay needs the full target/achieved/u set
        // to render meaningfully.
        let mut snap = fixture(5, 1.0);
        snap.u_values = None;
        assert!(comparison_traces_from(&snap).is_none());
    }

    #[test]
    fn comparison_traces_returns_none_for_too_few_samples() {
        let snap = fixture(1, 1.0);
        assert!(comparison_traces_from(&snap).is_none());
    }
}

/// Draw faint red vertical lines at sample times whose `InverseSolveStatus`
/// is anything other than `Converged`, plus a single-character severity
/// glyph (R/S/B/N) at the top of each band so the failure type is visible
/// at a glance from the plot.
///
/// The collapsing failure-summary list below the plot remains the place
/// for full-text payloads — egui_plot 0.33's `Text` item has
/// `PlotGeometry::None` and so does not surface per-shape hover tooltips,
/// which is why the glyph + collapsing-list pairing is the workaround.
fn draw_failure_bands(
    plot_ui: &mut egui_plot::PlotUi,
    times: &[f64],
    statuses: &[InverseSolveStatus],
) {
    let band_color = egui::Color32::from_rgba_unmultiplied(220, 50, 50, 80);
    let glyph_color = egui::Color32::from_rgb(220, 60, 60);
    // Glyphs anchor at a fixed offset below the plot's current upper
    // bound. Reading bounds inside the closure is the same pattern used in
    // gui::plot_panel::actuator for its annotation overlay.
    let bounds = plot_ui.plot_bounds();
    let y_top = bounds.max()[1];
    let y_range = (bounds.max()[1] - bounds.min()[1]).max(1e-9);
    let glyph_y = y_top - 0.02 * y_range;

    for (i, status) in statuses.iter().enumerate() {
        let glyph = match status {
            InverseSolveStatus::Converged => continue,
            InverseSolveStatus::Reachability { .. } => "R",
            InverseSolveStatus::Singularity { .. } => "S",
            InverseSolveStatus::BranchJump { .. } => "B",
            InverseSolveStatus::NonConvergent { .. } => "N",
        };
        let Some(&t) = times.get(i) else { continue };
        plot_ui.vline(
            VLine::new(format!("failure_{}", i), t)
                .color(band_color)
                .width(1.0),
        );
        plot_ui.text(
            PlotText::new(
                format!("failure_glyph_{}", i),
                PlotPoint::new(t, glyph_y),
                glyph,
            )
            .anchor(egui::Align2::CENTER_BOTTOM)
            .color(glyph_color),
        );
    }
}
