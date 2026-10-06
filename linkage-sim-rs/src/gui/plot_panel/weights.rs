//! Weight Breakdown plot and braking bands (payload weights, spec Track 2
//! section 3, "Plots").
//!
//! - [`draw_weight_breakdown`]: one line per weight (link self-weights
//!   included) plus the non-gravity remainder and the required total, as
//!   force shares or power shares. A weight's line is coloured per sample
//!   by whether that weight helps, hurts or is neutral there, with the same
//!   `canvas::classification_color` the canvas weight arrows use.
//! - [`draw_braking_bands`]: shades the driver ranges where the load drives
//!   the actuator (`WeightBreakdown::braking`) behind the Actuator Force and
//!   Actuator Power curves.
//!
//! The data assembly ([`braking_bands`], [`band_y_extent`],
//! [`breakdown_lines`]) is pure and unit-tested; the draw functions only map
//! it onto egui_plot items.

use eframe::egui;
use egui_plot::{Plot, Polygon, VLine};

use crate::analysis::gravity_breakdown::{name_with_id, Classification, WeightSource};
use crate::gui::canvas::{classification_color, to_grayscale};
use crate::gui::state::DisplayUnits;
use crate::gui::sweep::{ShareBasis, SweepData, WeightBreakdown};

use super::{
    detect_plot_click, draw_angle_series_with_range, draw_range_boundary_markers,
    draw_toggle_markers, sweep_x_to_display, with_default_x_bounds, x_axis_label_for_sweep,
};

/// Braking band fill: (90, 140, 255) at alpha 50, premultiplied.
const BRAKING_BAND_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(18, 27, 50, 50);
/// Legend entry shared by every braking band (one checkbox hides them all).
const BRAKING_LEGEND_NAME: &str = "Braking";
/// Colour of the non-gravity remainder line.
const OTHER_LOADS_COLOR: egui::Color32 = egui::Color32::from_rgb(100, 200, 255);
/// Colour of the required-total line.
const TOTAL_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 235, 235);
const OTHER_LINE_NAME: &str = "Other loads";
const TOTAL_LINE_NAME: &str = "Total";

/// What a plotted run is coloured by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunColor {
    /// A weight's line: helping / hurting / neutral at these samples.
    Weight(Classification),
    /// The non-gravity remainder.
    Other,
    /// The required total.
    Total,
}

/// A contiguous single-colour piece of a plotted line, as
/// `(sweep x, y)` pairs (x in degrees, or metres in stroke mode).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Run {
    pub(super) color: RunColor,
    pub(super) points: Vec<(f64, f64)>,
}

/// One line of the Weight Breakdown plot: its legend name and its runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BreakdownLine {
    pub(super) name: String,
    pub(super) runs: Vec<Run>,
}

/// Sweep-x ranges `(lo, hi)`, `lo < hi`, where the actuator brakes.
///
/// A run of consecutive braking samples `k0..=k1` covers from halfway to
/// its left neighbour to halfway to its right neighbour (its own sample at
/// the sweep ends), so a lone braking sample shows as one sample step wide
/// and bands of neighbouring runs never touch. A sample with a non-finite
/// x never brakes, and a band next to it ends at its own sample. Only the
/// samples both slices cover are read; zero-width bands (a one-sample
/// sweep) are dropped.
pub(super) fn braking_bands(xs: &[f64], braking: &[bool]) -> Vec<(f64, f64)> {
    let n = xs.len().min(braking.len());
    let in_band = |k: usize| braking[k] && xs[k].is_finite();
    // Band edge at sample `k`, halfway towards `neighbour` when it has an x.
    let edge = |k: usize, neighbour: Option<usize>| match neighbour {
        Some(j) if xs[j].is_finite() => 0.5 * (xs[k] + xs[j]),
        _ => xs[k],
    };
    let mut bands = Vec::new();
    let mut k = 0;
    while k < n {
        if !in_band(k) {
            k += 1;
            continue;
        }
        let first = k;
        while k + 1 < n && in_band(k + 1) {
            k += 1;
        }
        let a = edge(first, first.checked_sub(1));
        let b = edge(k, (k + 1 < n).then_some(k + 1));
        let (lo, hi) = (a.min(b), a.max(b));
        if hi > lo {
            bands.push((lo, hi));
        }
        k += 1;
    }
    bands
}

/// Finite y range `(lo, hi)` over every series: the height of the braking
/// bands, so they cover the curves without stretching the auto-fitted
/// axes. A flat range is widened by 5 % of its magnitude (at least 0.05)
/// so the band stays visible; `None` when no value is finite.
pub(super) fn band_y_extent(series: &[&[f64]]) -> Option<(f64, f64)> {
    let (lo, hi) = series
        .iter()
        .flat_map(|s| s.iter())
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if !lo.is_finite() {
        return None;
    }
    if hi > lo {
        return Some((lo, hi));
    }
    let pad = 0.05 * lo.abs().max(1.0);
    Some((lo - pad, hi + pad))
}

/// Shade the ranges where the actuator brakes (`WeightBreakdown::braking`)
/// behind the curves of an actuator plot. The bands span the finite y
/// range of `extent_series` (every curve and reference line the plot
/// draws) and share one legend entry, "Braking". Nothing is drawn when the
/// sweep has no breakdown or nothing brakes. Call before drawing the
/// curves so they stay on top.
pub(super) fn draw_braking_bands(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
    extent_series: &[&[f64]],
    nathan_mode: bool,
) {
    let Some(breakdown) = &sweep.weight_breakdown else {
        return;
    };
    let bands = braking_bands(&sweep.angles_deg, &breakdown.braking);
    if bands.is_empty() {
        return;
    }
    let Some((y_lo, y_hi)) = band_y_extent(extent_series) else {
        return;
    };
    let fill = if nathan_mode { to_grayscale(BRAKING_BAND_FILL) } else { BRAKING_BAND_FILL };
    for (lo, hi) in bands {
        let (x_lo, x_hi) = (sweep_x_to_display(lo, sweep, units), sweep_x_to_display(hi, sweep, units));
        plot_ui.polygon(
            Polygon::new(BRAKING_LEGEND_NAME, vec![[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])
                .fill_color(fill)
                .stroke(egui::Stroke::new(0.0, fill))
                .allow_hover(false),
        );
    }
}

/// Legend name of a weight's line, unique per source because ids are:
/// `"coupler (link)"` or `"Arm (link b2)"` for a link self-weight (body
/// label, plus the body id when the label differs), `"W1"` or
/// `"Robot (W1)"` for a point mass.
pub(crate) fn source_line_name(source: &WeightSource) -> String {
    if source.is_link_self_weight {
        if source.name == source.body_id {
            format!("{} (link)", source.name)
        } else {
            format!("{} (link {})", source.name, source.body_id)
        }
    } else {
        name_with_id(&source.name, &source.id)
    }
}

/// Split `(xs[k], ys[k])` into runs of consecutive finite samples with one
/// colour. A non-finite sample ends the run (a gap in the plot). Where the
/// colour changes, the new run starts at the previous sample, so the line
/// stays connected and the joining segment takes the new colour.
fn split_runs(xs: &[f64], ys: &[f64], color_at: impl Fn(usize) -> RunColor) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    let mut previous: Option<(f64, f64)> = None;
    for (k, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if !(x.is_finite() && y.is_finite()) {
            previous = None;
            continue;
        }
        let color = color_at(k);
        match (runs.last_mut(), previous) {
            (Some(run), Some(_)) if run.color == color => run.points.push((x, y)),
            (_, Some(p)) => runs.push(Run { color, points: vec![p, (x, y)] }),
            (_, None) => runs.push(Run { color, points: vec![(x, y)] }),
        }
        previous = Some((x, y));
    }
    runs
}

/// The Weight Breakdown lines over sweep x values `xs`: one per source (in
/// `b.sources` order, coloured by its classification at each sample, the
/// same in both views), then "Other loads" and "Total". `show_power` picks
/// the power shares (W) instead of the force shares (N, or driver-torque
/// shares); force shares are NaN near stroke reversal, which leaves a gap.
pub(super) fn breakdown_lines(xs: &[f64], b: &WeightBreakdown, show_power: bool) -> Vec<BreakdownLine> {
    let per_source = if show_power { &b.power_share } else { &b.force_share };
    let mut lines: Vec<BreakdownLine> = b
        .sources
        .iter()
        .zip(per_source)
        .enumerate()
        .map(|(i, (source, ys))| {
            let classes = b.classifications(i);
            let color_at = |k: usize| RunColor::Weight(classes.get(k).copied().unwrap_or(Classification::Neutral));
            BreakdownLine { name: source_line_name(source), runs: split_runs(xs, ys, color_at) }
        })
        .collect();
    let (other, total) = if show_power { (&b.other_power, &b.total_power) } else { (&b.other_force, &b.total_force) };
    lines.push(BreakdownLine { name: OTHER_LINE_NAME.to_string(), runs: split_runs(xs, other, |_| RunColor::Other) });
    lines.push(BreakdownLine { name: TOTAL_LINE_NAME.to_string(), runs: split_runs(xs, total, |_| RunColor::Total) });
    lines
}

/// Y-axis label of the Weight Breakdown plot: actuator or driver shares,
/// force (N), torque (N*m; N for a linear driver in stroke mode) or power
/// (W).
pub(super) fn breakdown_y_label(basis: ShareBasis, is_stroke: bool, show_power: bool) -> &'static str {
    match (basis, show_power) {
        (ShareBasis::ActuatorForce, false) => "Actuator Force Share (N)",
        (ShareBasis::ActuatorForce, true) => "Actuator Power Share (W)",
        (ShareBasis::DriverTorque, false) if is_stroke => "Driver Force Share (N)",
        (ShareBasis::DriverTorque, false) => "Driver Torque Share (N\u{00b7}m)",
        (ShareBasis::DriverTorque, true) => "Driver Power Share (W)",
    }
}

/// Plot colour and width of a run.
fn run_style(color: RunColor) -> (egui::Color32, f32) {
    match color {
        RunColor::Weight(class) => (classification_color(class), 1.5),
        RunColor::Other => (OTHER_LOADS_COLOR, 1.5),
        RunColor::Total => (TOTAL_COLOR, 2.5),
    }
}

/// Plot each weight's force share (or power share when `show_power`) vs
/// the driver, plus the non-gravity remainder and the required total.
///
/// Returns the clicked X coordinate (display units) if the user clicked.
pub(super) fn draw_weight_breakdown(
    ui: &mut egui::Ui,
    sweep: &SweepData,
    current_driver_display: f64,
    units: &DisplayUnits,
    nathan_mode: bool,
    show_power: bool,
) -> Option<f64> {
    let Some(breakdown) = &sweep.weight_breakdown else {
        ui.label("Weight breakdown not available (no link mass or weight in the mechanism).");
        return None;
    };
    let lines = breakdown_lines(&sweep.angles_deg, breakdown, show_power);

    // Separate plot ids per view, so a zoom set on newtons is not reused
    // on watts.
    let plot_id = if show_power { "weight_breakdown_power_plot" } else { "weight_breakdown_force_plot" };
    let plot = Plot::new(plot_id)
        .allow_zoom(true)
        .allow_drag(true)
        .x_axis_label(x_axis_label_for_sweep(sweep, units))
        .y_axis_label(breakdown_y_label(breakdown.basis, sweep.sweep_mode.is_stroke(), show_power))
        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
        .height(ui.available_height().max(50.0));
    let plot = with_default_x_bounds(plot, plot_id, sweep, units);

    let mut clicked_x: Option<f64> = None;
    plot.show(ui, |plot_ui| {
        for line in &lines {
            for run in &line.runs {
                let (color, width) = run_style(run.color);
                draw_angle_series_with_range(plot_ui, &line.name, color, width, &run.points, sweep, units, nathan_mode);
            }
        }

        // Vertical marker at current driver angle.
        plot_ui.vline(
            VLine::new("cursor", current_driver_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 255, 255, 100))
                .width(1.0),
        );

        draw_toggle_markers(plot_ui, sweep, units);
        draw_range_boundary_markers(plot_ui, sweep, units);
        clicked_x = detect_plot_click(plot_ui);
    });

    clicked_x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{max_abs_finite, BRAKE_TOL_REL};
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::AppState;
    use crate::gui::test_support::set_actuator_stored_force;
    use Classification::{Helping, Hurting, Neutral};

    const NAN: f64 = f64::NAN;

    #[test]
    fn braking_bands_run_from_midpoint_to_midpoint() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let braking = [false, true, true, false, false, true, false, false];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.5, 2.5), (4.5, 5.5)]);
    }

    #[test]
    fn braking_bands_stop_at_the_sweep_ends() {
        let xs = [0.0, 10.0, 20.0, 30.0, 40.0];
        let braking = [true, true, false, false, true];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.0, 15.0), (35.0, 40.0)]);
    }

    #[test]
    fn braking_bands_none_when_nothing_brakes_one_when_everything_does() {
        let xs = [0.0, 1.0, 2.0];
        assert!(braking_bands(&xs, &[false; 3]).is_empty());
        assert_eq!(braking_bands(&xs, &[true; 3]), vec![(0.0, 2.0)]);
        assert!(braking_bands(&[], &[]).is_empty());
    }

    #[test]
    fn braking_bands_skip_zero_width_and_non_finite_samples() {
        // A one-sample sweep has no width to shade.
        assert!(braking_bands(&[5.0], &[true]).is_empty());
        // Sample 2 has no x: it never brakes, and the bands next to it end
        // at their own sample instead of a NaN midpoint.
        let xs = [0.0, 1.0, NAN, 3.0, 4.0];
        let braking = [false, true, true, true, false];
        assert_eq!(braking_bands(&xs, &braking), vec![(0.5, 1.0), (3.0, 3.5)]);
    }

    #[test]
    fn braking_bands_are_ordered_on_a_descending_sweep() {
        let xs = [3.0, 2.0, 1.0, 0.0];
        assert_eq!(braking_bands(&xs, &[false, true, true, false]), vec![(0.5, 2.5)]);
    }

    #[test]
    fn braking_bands_read_only_the_common_length() {
        // Mismatched lengths break the sweep invariant; never panic, shade
        // only the samples both series cover.
        let xs = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(braking_bands(&xs, &[false, true]), vec![(0.5, 1.0)]);
        assert_eq!(braking_bands(&xs[..2], &[false, true, true, true]), vec![(0.5, 1.0)]);
    }

    #[test]
    fn band_y_extent_spans_the_finite_values_of_every_series() {
        let a = [1.0, NAN, -3.0];
        let b = [f64::INFINITY, 7.0];
        assert_eq!(band_y_extent(&[&a, &b]), Some((-3.0, 7.0)));
        assert_eq!(band_y_extent(&[&[NAN, f64::NEG_INFINITY]]), None);
        assert_eq!(band_y_extent(&[]), None);
        // A flat series is widened so the band stays visible.
        assert_eq!(band_y_extent(&[&[-200.0, -200.0]]), Some((-210.0, -190.0)));
        assert_eq!(band_y_extent(&[&[0.0]]), Some((-0.05, 0.05)));
    }

    fn source(id: &str, name: &str, body_id: &str, is_link_self_weight: bool) -> WeightSource {
        WeightSource {
            id: id.to_string(),
            name: name.to_string(),
            body_id: body_id.to_string(),
            local_pos: [0.0, 0.0],
            mass: 1.0,
            is_link_self_weight,
        }
    }

    #[test]
    fn source_line_names_are_unique_per_source() {
        assert_eq!(source_line_name(&source("link:coupler", "coupler", "coupler", true)), "coupler (link)");
        assert_eq!(source_line_name(&source("link:b2", "Arm", "b2", true)), "Arm (link b2)");
        assert_eq!(source_line_name(&source("W2", "W2", "b2", false)), "W2");
        assert_eq!(source_line_name(&source("W1", "Robot", "b2", false)), "Robot (W1)");
        // Two weights with the same label still get separate legend entries.
        assert_ne!(
            source_line_name(&source("W3", "Robot", "b2", false)),
            source_line_name(&source("W1", "Robot", "b2", false))
        );
    }

    const HAND_XS: [f64; 6] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0];

    /// Six samples, two sources. Source 0 (the coupler's own weight) comes
    /// down, drifts into its neutral band (0.01 W < 1 % of 4 W) and goes up;
    /// source 1 (W1 "Robot") always goes up. Sample 40 failed (NaN
    /// everywhere); at 20 the actuator reverses (force shares NaN); at 50
    /// it retracts, so the hurting weights' force shares are negative.
    fn hand_breakdown() -> WeightBreakdown {
        WeightBreakdown {
            sources: vec![
                source("link:coupler", "coupler", "coupler", true),
                source("W1", "Robot", "coupler", false),
            ],
            basis: ShareBasis::ActuatorForce,
            gravity_power: vec![vec![4.0, 3.0, 0.01, -2.0, NAN, -4.0], vec![-1.0, -1.0, -1.0, -1.0, NAN, -1.0]],
            force_share: vec![vec![-2.0, -1.5, NAN, 1.0, NAN, -2.0], vec![0.5, 0.5, NAN, 0.5, NAN, -0.5]],
            power_share: vec![vec![-4.0, -3.0, -0.01, 2.0, NAN, 4.0], vec![1.0, 1.0, 1.0, 1.0, NAN, 1.0]],
            other_force: vec![5.0, 5.0, NAN, 5.0, NAN, 5.0],
            other_power: vec![6.0, 6.0, 6.0, 6.0, NAN, 6.0],
            total_force: vec![7.0, 7.0, NAN, 7.0, NAN, 7.0],
            total_power: vec![8.0, 8.0, 8.0, 8.0, NAN, 8.0],
            braking: vec![false; 6],
        }
    }

    fn run(color: RunColor, points: &[(f64, f64)]) -> Run {
        Run { color, points: points.to_vec() }
    }

    #[test]
    fn power_share_lines_change_colour_with_the_classification_and_break_at_failures() {
        let lines = breakdown_lines(&HAND_XS, &hand_breakdown(), true);
        let names: Vec<&str> = lines.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["coupler (link)", "Robot (W1)", "Other loads", "Total"]);
        // Each colour change restarts from the previous sample, so the line
        // stays connected; the failed sample at 40 leaves a gap.
        assert_eq!(
            lines[0].runs,
            vec![
                run(RunColor::Weight(Helping), &[(0.0, -4.0), (10.0, -3.0)]),
                run(RunColor::Weight(Neutral), &[(10.0, -3.0), (20.0, -0.01)]),
                run(RunColor::Weight(Hurting), &[(20.0, -0.01), (30.0, 2.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, 4.0)]),
            ]
        );
        assert_eq!(
            lines[1].runs,
            vec![
                run(RunColor::Weight(Hurting), &[(0.0, 1.0), (10.0, 1.0), (20.0, 1.0), (30.0, 1.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, 1.0)]),
            ]
        );
        assert_eq!(
            lines[2].runs,
            vec![
                run(RunColor::Other, &[(0.0, 6.0), (10.0, 6.0), (20.0, 6.0), (30.0, 6.0)]),
                run(RunColor::Other, &[(50.0, 6.0)]),
            ]
        );
        assert_eq!(
            lines[3].runs,
            vec![
                run(RunColor::Total, &[(0.0, 8.0), (10.0, 8.0), (20.0, 8.0), (30.0, 8.0)]),
                run(RunColor::Total, &[(50.0, 8.0)]),
            ]
        );
    }

    #[test]
    fn force_share_lines_keep_the_gravity_power_colours_and_gap_near_reversal() {
        let lines = breakdown_lines(&HAND_XS, &hand_breakdown(), false);
        // The colour follows the gravity power, not the sign of the force
        // share: at 50 both weights are hurting although their shares are
        // negative (the actuator retracts). The reversal at 20 and the
        // failure at 40 are gaps.
        assert_eq!(
            lines[0].runs,
            vec![
                run(RunColor::Weight(Helping), &[(0.0, -2.0), (10.0, -1.5)]),
                run(RunColor::Weight(Hurting), &[(30.0, 1.0)]),
                run(RunColor::Weight(Hurting), &[(50.0, -2.0)]),
            ]
        );
        assert_eq!(
            lines[1].runs,
            vec![
                run(RunColor::Weight(Hurting), &[(0.0, 0.5), (10.0, 0.5)]),
                run(RunColor::Weight(Hurting), &[(30.0, 0.5)]),
                run(RunColor::Weight(Hurting), &[(50.0, -0.5)]),
            ]
        );
        assert_eq!(
            lines[2].runs,
            vec![
                run(RunColor::Other, &[(0.0, 5.0), (10.0, 5.0)]),
                run(RunColor::Other, &[(30.0, 5.0)]),
                run(RunColor::Other, &[(50.0, 5.0)]),
            ]
        );
        assert_eq!(
            lines[3].runs,
            vec![
                run(RunColor::Total, &[(0.0, 7.0), (10.0, 7.0)]),
                run(RunColor::Total, &[(30.0, 7.0)]),
                run(RunColor::Total, &[(50.0, 7.0)]),
            ]
        );
    }

    #[test]
    fn breakdown_lines_without_sources_or_with_short_series_never_panic() {
        let mut b = hand_breakdown();
        b.sources.clear();
        b.gravity_power.clear();
        b.force_share.clear();
        b.power_share.clear();
        let names: Vec<String> = breakdown_lines(&HAND_XS, &b, true).into_iter().map(|l| l.name).collect();
        assert_eq!(names, ["Other loads", "Total"]);

        // Series shorter than the sweep (a broken invariant): only the
        // common samples are drawn, uncoloured samples count as neutral.
        let mut b = hand_breakdown();
        b.gravity_power[0].truncate(1);
        let lines = breakdown_lines(&HAND_XS, &b, true);
        assert_eq!(
            lines[0].runs[..2],
            [
                run(RunColor::Weight(Helping), &[(0.0, -4.0)]),
                run(RunColor::Weight(Neutral), &[(0.0, -4.0), (10.0, -3.0), (20.0, -0.01), (30.0, 2.0)]),
            ]
        );
        assert!(breakdown_lines(&HAND_XS[..2], &hand_breakdown(), true).iter().all(|l| l.runs.len() == 1));
    }

    #[test]
    fn y_label_names_the_quantity_and_unit_for_each_basis() {
        use ShareBasis::{ActuatorForce, DriverTorque};
        assert_eq!(breakdown_y_label(ActuatorForce, false, false), "Actuator Force Share (N)");
        assert_eq!(breakdown_y_label(ActuatorForce, true, false), "Actuator Force Share (N)");
        assert_eq!(breakdown_y_label(ActuatorForce, false, true), "Actuator Power Share (W)");
        assert_eq!(breakdown_y_label(DriverTorque, false, false), "Driver Torque Share (N\u{00b7}m)");
        assert_eq!(breakdown_y_label(DriverTorque, true, false), "Driver Force Share (N)");
        assert_eq!(breakdown_y_label(DriverTorque, false, true), "Driver Power Share (W)");
        assert_eq!(breakdown_y_label(DriverTorque, true, true), "Driver Power Share (W)");
    }

    /// ParallelogramActuator with a 50 kg weight on the rocker, with the
    /// sample's shipped stored force and in sizing mode: the shaded bands
    /// cover exactly the samples where the plotted actuator power is
    /// negative beyond the braking tolerance, and the cycle both brakes and
    /// motors. Since BL-026 the plotted power is the required power in both
    /// modes, so the bands must follow the curve in stored-force mode too.
    #[test]
    fn braking_bands_cover_exactly_the_negative_actuator_power_samples() {
        for sizing in [false, true] {
            let mode = if sizing { "sizing" } else { "stored force" };
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramActuator);
            if sizing {
                set_actuator_stored_force(&mut state, 0.0);
            }
            state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
            state.compute_sweep();
            let data = state.sweep_data.as_ref().expect("sweep computed");
            let power = data.actuator_power.as_ref().expect("actuator power");
            let braking = &data.weight_breakdown.as_ref().expect("breakdown computed").braking;
            let bands = braking_bands(&data.angles_deg, braking);
            let tol = BRAKE_TOL_REL * max_abs_finite(power);
            let (mut n_braking, mut n_motoring) = (0, 0);
            for (k, &x) in data.angles_deg.iter().enumerate() {
                let shaded = bands.iter().any(|&(lo, hi)| lo <= x && x <= hi);
                let brakes = power[k] < -tol;
                assert_eq!(shaded, brakes, "{mode}, at {x} deg: P = {} W", power[k]);
                if brakes {
                    n_braking += 1;
                } else {
                    n_motoring += 1;
                }
            }
            assert!(n_braking > 0 && n_motoring > 0, "{mode}: {n_braking} braking, {n_motoring} motoring samples");
        }
    }

    /// With the ParallelogramActuator's shipped stored force, the Total line
    /// is the plotted actuator force in the force view and the plotted
    /// actuator power in the power view (both the required values since
    /// BL-026) at every sample where the actuator plot has a value.
    #[test]
    fn total_line_is_the_plotted_required_actuator_force_and_power() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
        state.compute_sweep();
        let data = state.sweep_data.as_ref().expect("sweep computed");
        let b = data.weight_breakdown.as_ref().expect("breakdown computed");
        let views = [
            (false, data.actuator_forces.as_ref().expect("actuator force")),
            (true, data.actuator_power.as_ref().expect("actuator power")),
        ];
        for (show_power, plotted) in views {
            let lines = breakdown_lines(&data.angles_deg, b, show_power);
            let total = lines.iter().find(|l| l.name == TOTAL_LINE_NAME).expect("Total line");
            // One colour, so the runs only break at gaps: no repeated points.
            let points: Vec<(f64, f64)> = total.runs.iter().flat_map(|r| r.points.iter().copied()).collect();
            let tol = 1e-9 * max_abs_finite(plotted);
            let mut compared = 0;
            for (&x, &want) in data.angles_deg.iter().zip(plotted) {
                if !want.is_finite() {
                    continue;
                }
                let (_, got) = points
                    .iter()
                    .find(|p| p.0 == x)
                    .unwrap_or_else(|| panic!("power view {show_power}: no Total point at {x} deg"));
                assert!((got - want).abs() <= tol, "power view {show_power}, at {x} deg: Total {got}, plotted {want}");
                compared += 1;
            }
            assert!(compared > 300, "power view {show_power}: only {compared} samples compared");
        }
    }
}
