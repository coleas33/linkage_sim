//! The plot tabs (spec M4 "Layout": "torque vs temperature (hot/cold allowance band,
//! requirement line); gap sweep; pole sweep; slip heating over time vs limit; torque vs rotation
//! angle (pull-out point)"), drawn with egui_plot (the 0.33 line linkage-sim-rs uses, locked to
//! its version: gate 11).
//!
//! Every series comes from the results computed this frame, plus three inputs of the design
//! shown (the minimum temperature, the variation allowance and the required minimum torque,
//! which the results do not repeat): a few hundred evaluations of closed forms, and only for the
//! tab shown. The closed forms are the engine's own (decision M42-5):
//!
//! - **Torque against temperature**: the pull-out at 20 °C times each ring's remanence factor,
//!   `model::ring_pair_factor` (Br_i(T) Br_o(T) / (Br_i Br_o), decision A2-7), the expression
//!   the Metal design and Temperature design torques use, so the curve passes through the
//!   pull-out at the operating temperature and its band edges through the hot-low and cold-high
//!   torques; from the minimum temperature to 10 °C past the governing limit.
//! - **Gap and pole sweeps**: the sweep tables' pull-out at the operating temperature against
//!   the swept variable, each row coloured by its status (the dashboard's colours: nominal
//!   green, below the hot minimum amber, a fit failure red) and greyed when its own f_end <= 0
//!   (`model::end_effect_in_range`), with the design shown as a marker. The pole sweep's rows
//!   take the smallest apothem that fits each count (the engine's `sweeps::pole_sweep`), so its
//!   line's legend says so: the design, at its own apothem, can sit off that line.
//! - **Slip heating**: the thermal network's first-order rise, start + rise (1 - exp(-t / tau)),
//!   for the estimate and the high case, against the governing limit, over five time constants
//!   (longer when the high case reaches the limit later), with the time to the limit marked.
//! - **Torque against rotation**: area-lever product × f_end × calibration factor ×
//!   Σ amp_n sin(n x) over the harmonics summed (`model.amp*_Pa`, the E7 torque-angle
//!   amplitudes), against the relative rotation in mechanical degrees over one pole pair
//!   (x = N/2 × the mechanical angle), with the pull-out point at the E7 angle.
//!
//! When f_end <= 0 (audit M9) every torque computed from the pull-out (the temperature and the
//! rotation plots) is drawn grey, as the dashboard greys those rows (decision M42-6).

use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

use crate::engine::meta::NumOrText;
use crate::engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor};
use crate::engine::sweeps::SweepRow;
use crate::gui::dashboard::{Level, end_effect_out_of_range};
use crate::{DesignInputs, DesignResults};

/// Samples of the torque-temperature curve.
pub const TEMPERATURE_SAMPLES: usize = 121;

/// How far past the governing limit the temperature axis runs [°C].
pub const PAST_THE_LIMIT_C: f64 = 10.0;

/// Samples of each slip-heating curve.
pub const HEATING_SAMPLES: usize = 121;

/// The slip-heating time axis, in thermal time constants (at least).
pub const HEATING_SPAN_TAUS: f64 = 5.0;

/// Samples of the torque-rotation curve.
pub const ANGLE_SAMPLES: usize = 361;

/// The radius of a sweep row's marker [points].
pub const POINT_RADIUS: f32 = 4.0;

/// The plots, one tab each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlotKind {
    TorqueTemperature,
    GapSweep,
    PoleSweep,
    SlipHeating,
    TorqueAngle,
}

impl PlotKind {
    /// Every plot, in tab order.
    pub const ALL: [PlotKind; 5] = [
        PlotKind::TorqueTemperature,
        PlotKind::GapSweep,
        PlotKind::PoleSweep,
        PlotKind::SlipHeating,
        PlotKind::TorqueAngle,
    ];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            PlotKind::TorqueTemperature => "Torque vs temperature",
            PlotKind::GapSweep => "Gap sweep",
            PlotKind::PoleSweep => "Pole sweep",
            PlotKind::SlipHeating => "Slip heating",
            PlotKind::TorqueAngle => "Torque vs rotation",
        }
    }

    /// The plot's id: egui_plot keeps its bounds and transform under it
    /// (`egui_plot::PlotMemory::load`).
    pub fn id(self) -> egui::Id {
        egui::Id::new(match self {
            PlotKind::TorqueTemperature => TORQUE_TEMPERATURE_ID,
            PlotKind::GapSweep => GAP_SWEEP_ID,
            PlotKind::PoleSweep => POLE_SWEEP_ID,
            PlotKind::SlipHeating => SLIP_HEATING_ID,
            PlotKind::TorqueAngle => TORQUE_ANGLE_ID,
        })
    }
}

/// The plots' id names ([`PlotKind::id`]).
pub const TORQUE_TEMPERATURE_ID: &str = "magcoupling_plot_torque_temperature";
pub const GAP_SWEEP_ID: &str = "magcoupling_plot_gap_sweep";
pub const POLE_SWEEP_ID: &str = "magcoupling_plot_pole_sweep";
pub const SLIP_HEATING_ID: &str = "magcoupling_plot_slip_heating";
pub const TORQUE_ANGLE_ID: &str = "magcoupling_plot_torque_angle";

/// The legend names.
pub const PULL_OUT: &str = "Pull-out torque";
/// The pole sweep's line: its rows are not at the design's apothem.
pub const PULL_OUT_SMALLEST_APOTHEM: &str =
    "Pull-out torque (rows at their smallest fitting apothem)";
pub const LOW_BAND: &str = "Low: minus the variation allowance";
pub const HIGH_BAND: &str = "High: plus the variation allowance";
pub const REQUIRED: &str = "Required minimum";
pub const OPERATING: &str = "Operating temperature";
pub const LIMIT: &str = "Governing limit";
pub const THIS_DESIGN: &str = "This design";
pub const NOMINAL: &str = "Nominal: test needed";
pub const BELOW_MINIMUM: &str = "Below hot minimum";
pub const NO_FIT: &str = "Does not fit";
pub const OUT_OF_RANGE: &str = "End-effect model out of range";
pub const REQUIRED_FLOOR: &str = "Required floor";
pub const ESTIMATE: &str = "Estimate";
pub const HIGH_CASE: &str = "High case";
pub const TIME_TO_LIMIT: &str = "Time to the limit (high case)";
pub const PULL_OUT_POINT: &str = "Pull-out point";

/// The text shown instead of a plot with nothing finite to draw.
pub const NOTHING_TO_PLOT: &str = "Nothing to plot: the values are not numbers";

/// `n` evenly spaced values from `lo` to `hi`, both included.
fn samples(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
}

/// `points` without those that are not finite (egui_plot's bounds need finite values).
fn finite(points: impl IntoIterator<Item = [f64; 2]>) -> Vec<[f64; 2]> {
    points
        .into_iter()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect()
}

/// The pull-out torque at magnet temperature `temp_C` [N·m]: the pull-out at 20 °C times each
/// ring's remanence factor (`model::ring_pair_factor`).
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub fn torque_at_temperature(results: &DesignResults, temp_C: f64) -> f64 {
    let m = &results.model;
    m.pullout_20C_Nm * ring_pair_factor(m.inner_alpha_br_per_C, m.outer_alpha_br_per_C, temp_C)
}

/// The torque-temperature plot's series.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix
pub struct TorqueTemperature {
    pub nominal: Vec<[f64; 2]>,
    pub low: Vec<[f64; 2]>,
    pub high: Vec<[f64; 2]>,
    pub required_Nm: f64,
    pub operating_C: f64,
    /// The governing limit, when finite.
    pub limit_C: Option<f64>,
    /// The pull-out at the operating temperature.
    pub operating: [f64; 2],
    /// f_end <= 0: the torques are greyed.
    pub greyed: bool,
}

/// The torque-temperature series of the design shown.
pub fn torque_temperature(inputs: &DesignInputs, results: &DesignResults) -> TorqueTemperature {
    let md = &inputs.metal;
    let op = inputs.coupling.op_temp_C;
    let limit = results.temperature.summary.governing_limit_C;
    let finite_limit = limit.is_finite().then_some(limit);
    let top = op.max(finite_limit.unwrap_or(op)) + PAST_THE_LIMIT_C;
    let temps: Vec<f64> = if top > md.min_temp_C {
        samples(md.min_temp_C, top, TEMPERATURE_SAMPLES).collect()
    } else {
        Vec::new()
    };
    let curve = |factor: f64| {
        finite(
            temps
                .iter()
                .map(|&t| [t, torque_at_temperature(results, t) * factor]),
        )
    };
    TorqueTemperature {
        nominal: curve(1.0),
        low: curve(1.0 - md.variation),
        high: curve(1.0 + md.variation),
        required_Nm: md.required_min_Nm,
        operating_C: op,
        limit_C: finite_limit,
        operating: [op, results.model.pullout_Nm],
        greyed: end_effect_out_of_range(results).is_some(),
    }
}

/// A sweep's rows as markers, by status, and the line through the rows in the end-effect
/// model's range.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SweepPoints {
    /// "nominal: test needed".
    pub nominal: Vec<[f64; 2]>,
    /// "below hot minimum".
    pub below_minimum: Vec<[f64; 2]>,
    /// A flat too narrow or outside the OD envelope.
    pub no_fit: Vec<[f64; 2]>,
    /// f_end <= 0, whatever the status.
    pub out_of_range: Vec<[f64; 2]>,
    pub line: Vec<[f64; 2]>,
}

/// The markers of a sweep's rows: (swept variable, pull-out at the operating temperature).
pub fn sweep_points(rows: &[SweepRow]) -> SweepPoints {
    let mut points = SweepPoints::default();
    for row in rows {
        let p = [row.variable, row.pullout_op_Nm];
        if !(p[0].is_finite() && p[1].is_finite()) {
            continue;
        }
        if !end_effect_in_range(row.f_end) {
            points.out_of_range.push(p);
            continue;
        }
        points.line.push(p);
        match row.status.as_str() {
            "nominal: test needed" => points.nominal.push(p),
            "below hot minimum" => points.below_minimum.push(p),
            _ => points.no_fit.push(p),
        }
    }
    points
}

/// The slip-heating plot's series.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix
pub struct SlipHeating {
    pub estimate: Vec<[f64; 2]>,
    pub high: Vec<[f64; 2]>,
    pub limit_C: f64,
    /// (time to the limit, the limit) in the high case, when it reaches it.
    pub time_to_limit: Option<[f64; 2]>,
}

/// The slip-heating series: continuous slip from the starting temperature.
pub fn slip_heating(results: &DesignResults) -> SlipHeating {
    let th = &results.temperature.thermal;
    let limit = results.temperature.summary.governing_limit_C;
    let tau = th.time_constant_s;
    let reached = match th.time_to_limit_high {
        NumOrText::Num(t) if t.is_finite() => Some(t),
        _ => None,
    };
    let span = (HEATING_SPAN_TAUS * tau).max(reached.map_or(0.0, |t| 1.1 * t));
    let times: Vec<f64> = if tau.is_finite() && tau > 0.0 && span.is_finite() {
        samples(0.0, span, HEATING_SAMPLES).collect()
    } else {
        Vec::new()
    };
    let curve = |rise: f64| {
        finite(
            times
                .iter()
                .map(|&t| [t, th.start_C + rise * (1.0 - (-t / tau).exp())]),
        )
    };
    SlipHeating {
        estimate: curve(th.steady_rise_est_C),
        high: curve(th.steady_rise_high_C),
        limit_C: limit,
        time_to_limit: reached.map(|t| [t, limit]),
    }
}

/// The torque-rotation plot's series.
#[derive(Clone, Debug, PartialEq)]
pub struct TorqueAngle {
    /// (relative rotation [° mechanical], torque [N·m]) over one pole pair.
    pub curve: Vec<[f64; 2]>,
    /// The pull-out point: the E7 angle and the pull-out torque.
    pub pull_out: [f64; 2],
    /// f_end <= 0: the torques are greyed.
    pub greyed: bool,
}

/// The torque at electrical angle `x` [rad] of the design shown [N·m]: area-lever product ×
/// f_end × calibration factor × Σ amp_n sin(n x) over the harmonics summed; NaN for a harmonic
/// set outside its choices.
pub fn torque_at_angle(inputs: &DesignInputs, results: &DesignResults, x: f64) -> f64 {
    let m = &results.model;
    let amps = [
        m.amp1_Pa, m.amp3_Pa, m.amp5_Pa, m.amp7_Pa, m.amp9_Pa, m.amp11_Pa,
    ];
    let Some(count) = harmonic_count(inputs.coupling.max_harmonic) else {
        return f64::NAN;
    };
    let tau = amps[..count]
        .iter()
        .zip([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        .fold(0.0, |acc, (amp, n)| acc + amp * (n * x).sin());
    tau * m.area_lever_m3 * m.f_end * m.f_cal
}

/// The torque-rotation series of the design shown.
pub fn torque_angle(inputs: &DesignInputs, results: &DesignResults) -> TorqueAngle {
    let pairs = inputs.coupling.npole as f64 / 2.0;
    let pole_pair_deg = 360.0 / pairs;
    let curve = if pole_pair_deg.is_finite() && pole_pair_deg > 0.0 {
        finite(samples(0.0, pole_pair_deg, ANGLE_SAMPLES).map(|deg| {
            let x = (deg * pairs).to_radians();
            [deg, torque_at_angle(inputs, results, x)]
        }))
    } else {
        Vec::new()
    };
    TorqueAngle {
        curve,
        pull_out: [
            (results.model.pullout_angle_rad / pairs).to_degrees(),
            results.model.pullout_Nm,
        ],
        greyed: end_effect_out_of_range(results).is_some(),
    }
}

/// The colour of a torque series: `color`, or grey when the torques are greyed.
fn tint(color: Color32, greyed: bool, visuals: &egui::Visuals) -> Color32 {
    if greyed {
        visuals.weak_text_color()
    } else {
        color
    }
}

const BLUE: Color32 = Color32::from_rgb(100, 180, 255);
const AMBER: Color32 = Color32::from_rgb(230, 170, 70);
const VIOLET: Color32 = Color32::from_rgb(180, 140, 230);

/// A plot with its id ([`PlotKind::id`]), the panel's legend and axis labels, filling the
/// space left.
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
    Plot::new(kind.id())
        .id(kind.id())
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label(x.to_owned())
        .y_axis_label(y.to_owned())
        .height(ui.available_height().max(150.0))
}

/// Draws the plot `kind` of the design shown (`inputs`, and the `results` computed from them).
pub fn plot_ui(ui: &mut egui::Ui, kind: PlotKind, inputs: &DesignInputs, results: &DesignResults) {
    match kind {
        PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
        PlotKind::GapSweep => sweep_ui(
            ui,
            kind,
            "Corner gap [mm]",
            PULL_OUT,
            &results.gap_sweep,
            [results.model.corner_gap_mm, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::PoleSweep => sweep_ui(
            ui,
            kind,
            "Poles per ring",
            PULL_OUT_SMALLEST_APOTHEM,
            &results.pole_sweep,
            [inputs.coupling.npole as f64, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::SlipHeating => slip_heating_ui(ui, results),
        PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
    }
}

/// Adds a dashed horizontal reference line `name` at `y`, only when `y` is finite (egui_plot's
/// bounds need finite values; a value that is not a number has no place on the axis).
fn reference_hline(p: &mut PlotUi<'_>, name: &str, y: f64, color: Color32) {
    if y.is_finite() {
        p.hline(
            HLine::new(name, y)
                .color(color)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
    }
}

/// Adds a dashed vertical reference line `name` at `x`, only when `x` is finite.
fn reference_vline(p: &mut PlotUi<'_>, name: &str, x: f64, color: Color32) {
    if x.is_finite() {
        p.vline(
            VLine::new(name, x)
                .color(color)
                .style(LineStyle::Dashed { length: 4.0 }),
        );
    }
}

fn torque_temperature_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    let s = torque_temperature(inputs, results);
    if s.nominal.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    let band = tint(AMBER, s.greyed, &visuals);
    plot(
        ui,
        PlotKind::TorqueTemperature,
        "Magnet temperature [°C]",
        "Torque [N·m]",
    )
    .show(ui, |p| {
        p.line(
            Line::new(PULL_OUT, s.nominal)
                .color(tint(BLUE, s.greyed, &visuals))
                .width(2.0),
        );
        p.line(Line::new(LOW_BAND, s.low).color(band).width(1.0));
        p.line(Line::new(HIGH_BAND, s.high).color(band).width(1.0));
        reference_hline(p, REQUIRED, s.required_Nm, Level::Bad.color(&visuals));
        reference_vline(p, OPERATING, s.operating_C, visuals.weak_text_color());
        if let Some(limit) = s.limit_C {
            reference_vline(p, LIMIT, limit, VIOLET);
        }
        let operating = finite([s.operating]);
        if !operating.is_empty() {
            p.points(
                Points::new(THIS_DESIGN, PlotPoints::from(operating))
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(BLUE, s.greyed, &visuals)),
            );
        }
    });
}

/// A sweep's plot: `line` names the line through the rows in the end-effect model's range.
#[allow(non_snake_case)] // unit suffix
fn sweep_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    x: &str,
    line: &str,
    rows: &[SweepRow],
    design: [f64; 2],
    floor_Nm: f64,
) {
    let s = sweep_points(rows);
    let visuals = ui.visuals().clone();
    plot(ui, kind, x, "Pull-out at the operating temperature [N·m]").show(ui, |p| {
        p.line(Line::new(line, s.line).color(BLUE).width(1.5));
        for (name, points, color) in [
            (NOMINAL, s.nominal, Level::Good.color(&visuals)),
            (
                BELOW_MINIMUM,
                s.below_minimum,
                Level::Caution.color(&visuals),
            ),
            (NO_FIT, s.no_fit, Level::Bad.color(&visuals)),
            (OUT_OF_RANGE, s.out_of_range, visuals.weak_text_color()),
        ] {
            if !points.is_empty() {
                p.points(Points::new(name, points).radius(POINT_RADIUS).color(color));
            }
        }
        reference_hline(p, REQUIRED_FLOOR, floor_Nm, Level::Bad.color(&visuals));
        let design = finite([design]);
        if !design.is_empty() {
            p.points(
                Points::new(THIS_DESIGN, design)
                    .radius(POINT_RADIUS + 2.0)
                    .filled(false)
                    .color(visuals.strong_text_color()),
            );
        }
    });
}

fn slip_heating_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let s = slip_heating(results);
    if s.high.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    plot(
        ui,
        PlotKind::SlipHeating,
        "Time slipping [s]",
        "Magnet temperature [°C]",
    )
    .show(ui, |p| {
        p.line(Line::new(ESTIMATE, s.estimate).color(BLUE).width(1.5));
        p.line(Line::new(HIGH_CASE, s.high).color(AMBER).width(2.0));
        reference_hline(p, LIMIT, s.limit_C, Level::Bad.color(&visuals));
        if let Some(point) = s.time_to_limit {
            p.points(
                Points::new(TIME_TO_LIMIT, finite([point]))
                    .radius(POINT_RADIUS + 1.0)
                    .color(Level::Bad.color(&visuals)),
            );
        }
    });
}

fn torque_angle_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    let s = torque_angle(inputs, results);
    if s.curve.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    let color = tint(BLUE, s.greyed, &visuals);
    plot(
        ui,
        PlotKind::TorqueAngle,
        "Relative rotation [° mechanical]",
        "Torque [N·m]",
    )
    .show(ui, |p| {
        p.line(Line::new(PULL_OUT, s.curve).color(color).width(2.0));
        let point = finite([s.pull_out]);
        if !point.is_empty() {
            p.points(
                Points::new(PULL_OUT_POINT, point)
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(Level::Bad.color(&visuals), s.greyed, &visuals)),
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::test_support::{drawn_texts, flat_shapes, short_magnets, sized_frame};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    #[test]
    fn the_torque_temperature_curve_meets_the_engine_s_torques() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let md = &r.metal;
        assert!(close(torque_at_temperature(&r, 50.0), r.model.pullout_Nm));
        assert!(close(
            torque_at_temperature(&r, 20.0),
            r.model.pullout_20C_Nm
        ));
        // The band edges: the hot-low torque at the operating temperature and the cold-high
        // torque at the minimum temperature (bit for bit: the engine's own expression).
        assert!(close(
            torque_at_temperature(&r, 50.0) * (1.0 - inputs.metal.variation),
            md.torque_hot_low_Nm
        ));
        assert_eq!(
            torque_at_temperature(&r, -40.0) * (1.0 + inputs.metal.variation),
            md.torque_cold_high_Nm
        );
        let s = torque_temperature(&inputs, &r);
        assert_eq!(s.nominal.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.low.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.high.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.nominal[0][0], -40.0);
        let limit = r.temperature.summary.governing_limit_C;
        assert_eq!(s.limit_C, Some(limit));
        assert!(close(
            s.nominal[TEMPERATURE_SAMPLES - 1][0],
            limit + PAST_THE_LIMIT_C
        ));
        assert_eq!(s.required_Nm, 2.5);
        assert_eq!(s.operating, [50.0, r.model.pullout_Nm]);
        assert!(!s.greyed);
        // Torque falls as the magnets warm (NdFeB's negative alpha).
        assert!(s.nominal.windows(2).all(|w| w[1][1] < w[0][1]));
    }

    #[test]
    fn a_grade_ring_s_own_alpha_shapes_the_curve() {
        // Decision A2-7: a grade-mode ring takes its grade's alpha; the curve still meets the
        // engine at the operating temperature and the band at the cold-high torque.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.grade_inner = "Y30".to_owned();
        let r = compute_all(&inputs);
        assert_ne!(r.model.inner_alpha_br_per_C, r.model.outer_alpha_br_per_C);
        assert!(close(torque_at_temperature(&r, 50.0), r.model.pullout_Nm));
        assert_eq!(
            torque_at_temperature(&r, inputs.metal.min_temp_C) * (1.0 + inputs.metal.variation),
            r.metal.torque_cold_high_Nm
        );
    }

    #[test]
    fn sweep_rows_are_sorted_by_status_and_end_effect_range() {
        let r = compute_all(&DesignInputs::default());
        let gap = sweep_points(&r.gap_sweep);
        assert_eq!(
            (gap.nominal.len(), gap.below_minimum.len(), gap.no_fit.len()),
            (4, 2, 7)
        );
        assert!(gap.out_of_range.is_empty());
        assert_eq!(gap.line.len(), 13);
        assert_eq!(gap.line[0], [0.5, r.gap_sweep[0].pullout_op_Nm]);
        let pole = sweep_points(&r.pole_sweep);
        assert_eq!(
            (
                pole.nominal.len(),
                pole.below_minimum.len(),
                pole.no_fit.len()
            ),
            (1, 1, 4)
        );
        // Short magnets put every row outside the end-effect model: all greyed, no line.
        let short = compute_all(&short_magnets());
        let greyed = sweep_points(&short.gap_sweep);
        assert_eq!(greyed.out_of_range.len(), 13);
        assert!(greyed.line.is_empty() && greyed.nominal.is_empty());
    }

    #[test]
    fn slip_heating_rises_to_the_steady_temperatures_and_marks_the_limit() {
        let r = compute_all(&DesignInputs::default());
        let th = &r.temperature.thermal;
        let s = slip_heating(&r);
        assert_eq!(s.estimate.len(), HEATING_SAMPLES);
        assert_eq!(s.high.len(), HEATING_SAMPLES);
        assert_eq!(s.high[0], [0.0, th.start_C]);
        let end = s.high[HEATING_SAMPLES - 1];
        assert!(close(end[0], HEATING_SPAN_TAUS * th.time_constant_s));
        let rise = th.steady_rise_high_C * (1.0 - (-HEATING_SPAN_TAUS).exp());
        assert!((end[1] - (th.start_C + rise)).abs() < 1e-9);
        // The default high case stays below the limit ("never").
        assert_eq!(s.time_to_limit, None);
        // A hotter start reaches it: the marker sits on the limit, on the curve.
        let mut inputs = DesignInputs::default();
        inputs.temperature.duty.hot_ambient_C += 5.0;
        let hot = compute_all(&inputs);
        let s = slip_heating(&hot);
        let [t, limit] = s.time_to_limit.expect("the high case reaches the limit");
        assert_eq!(limit, hot.temperature.summary.governing_limit_C);
        let th = &hot.temperature.thermal;
        let at = th.start_C + th.steady_rise_high_C * (1.0 - (-t / th.time_constant_s).exp());
        assert!((at - limit).abs() < 1e-9, "{at} vs {limit}");
        assert!(
            s.high.last().unwrap()[0] >= t,
            "the axis reaches the marker"
        );
    }

    #[test]
    fn the_torque_rotation_curve_peaks_at_the_pull_out() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let x = r.model.pullout_angle_rad;
        assert!(close(torque_at_angle(&inputs, &r, x), r.model.pullout_Nm));
        let s = torque_angle(&inputs, &r);
        assert_eq!(s.curve.len(), ANGLE_SAMPLES);
        // One pole pair of 10 poles: 72 mechanical degrees; the pull-out at a quarter of it.
        assert_eq!(s.curve[ANGLE_SAMPLES - 1][0], 72.0);
        assert!(close(s.pull_out[0], 18.0));
        assert_eq!(s.pull_out[1], r.model.pullout_Nm);
        let max = s.curve.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        assert!(max <= r.model.pullout_Nm * (1.0 + 1e-12));
        // A peak off half a pitch (E7): every harmonic up to 11 at the default design.
        let mut eleven = DesignInputs::default();
        eleven.coupling.max_harmonic = 11;
        let r11 = compute_all(&eleven);
        assert!(close(
            torque_at_angle(&eleven, &r11, r11.model.pullout_angle_rad),
            r11.model.pullout_Nm
        ));
        // A harmonic set outside its choices: no curve, never another set.
        let mut bad = DesignInputs::default();
        bad.coupling.max_harmonic = 4;
        assert!(torque_angle(&bad, &compute_all(&bad)).curve.is_empty());
    }

    #[test]
    fn out_of_range_end_effect_greys_the_torque_plots() {
        let inputs = short_magnets();
        let r = compute_all(&inputs);
        assert!(torque_temperature(&inputs, &r).greyed);
        assert!(torque_angle(&inputs, &r).greyed);
        assert!(
            !torque_temperature(
                &DesignInputs::default(),
                &compute_all(&DesignInputs::default())
            )
            .greyed
        );
    }

    /// The point count of every path egui painted, and the radius of every circle.
    fn paths_and_circles(output: &egui::FullOutput) -> (Vec<usize>, Vec<f32>) {
        let shapes = flat_shapes(output);
        let paths = shapes
            .iter()
            .filter_map(|shape| match shape {
                egui::Shape::Path(path) => Some(path.points.len()),
                _ => None,
            })
            .collect();
        let circles = shapes
            .iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(circle) => Some(circle.radius),
                _ => None,
            })
            .collect();
        (paths, circles)
    }

    /// Two frames of `kind` for `inputs` (a plot settles its bounds on the first).
    fn draw(kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput {
        let ctx = egui::Context::default();
        let results = compute_all(inputs);
        let mut output = None;
        for _ in 0..2 {
            output = Some(sized_frame(
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results),
            ));
        }
        output.unwrap()
    }

    fn count(values: &[usize], want: usize) -> usize {
        values.iter().filter(|v| **v == want).count()
    }

    #[test]
    fn each_plot_draws_its_series_with_their_point_counts() {
        let inputs = DesignInputs::default();
        let at = |radius: f32, circles: &[f32]| circles.iter().filter(|r| **r == radius).count();
        let output = draw(PlotKind::TorqueTemperature, &inputs);
        let (paths, _) = paths_and_circles(&output);
        assert_eq!(
            count(&paths, TEMPERATURE_SAMPLES),
            3,
            "nominal and both band edges"
        );
        let texts = drawn_texts(&output);
        for name in [
            PULL_OUT,
            LOW_BAND,
            HIGH_BAND,
            REQUIRED,
            OPERATING,
            LIMIT,
            THIS_DESIGN,
        ] {
            assert!(
                texts.iter().any(|t| t == name),
                "legend {name:?} in {texts:?}"
            );
        }
        let (paths, circles) = paths_and_circles(&draw(PlotKind::GapSweep, &inputs));
        assert_eq!(count(&paths, 13), 1, "the line through the 13 rows");
        assert_eq!(at(POINT_RADIUS, &circles), 13, "a marker per row");
        // Short magnets put every row outside the end-effect model: a grey marker per row and
        // no line.
        let (paths, circles) = paths_and_circles(&draw(PlotKind::GapSweep, &short_magnets()));
        assert_eq!(at(POINT_RADIUS, &circles), 13);
        assert_eq!(count(&paths, 13), 0);
        let output = draw(PlotKind::PoleSweep, &inputs);
        let (paths, circles) = paths_and_circles(&output);
        assert_eq!(count(&paths, 6), 1);
        assert_eq!(at(POINT_RADIUS, &circles), 6);
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == PULL_OUT_SMALLEST_APOTHEM)
        );
        let output = draw(PlotKind::SlipHeating, &inputs);
        let (paths, _) = paths_and_circles(&output);
        assert_eq!(count(&paths, HEATING_SAMPLES), 2, "estimate and high case");
        let output = draw(PlotKind::TorqueAngle, &inputs);
        let (paths, circles) = paths_and_circles(&output);
        assert_eq!(count(&paths, ANGLE_SAMPLES), 1);
        assert_eq!(at(POINT_RADIUS + 1.0, &circles), 1, "the pull-out point");
        assert!(drawn_texts(&output).iter().any(|t| t == PULL_OUT_POINT));
    }

    #[test]
    fn plots_of_values_that_are_not_numbers_draw_without_panicking() {
        // validate() refuses these at every boundary; a struct literal can still hold them.
        let mut inputs = DesignInputs::default();
        inputs.coupling.op_temp_C = f64::NAN;
        inputs.temperature.thermal.conductance_W_K = f64::NAN;
        inputs.coupling.max_harmonic = 4;
        for kind in PlotKind::ALL {
            draw(kind, &inputs);
        }
        let texts = drawn_texts(&draw(PlotKind::TorqueAngle, &inputs));
        assert!(texts.iter().any(|t| t == NOTHING_TO_PLOT));
    }
}
