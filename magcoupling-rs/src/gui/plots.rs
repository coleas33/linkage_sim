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
//!
//! Points and reference lines that are not finite are left out; a plot with nothing left shows
//! [`NOTHING_TO_PLOT`] (the torque-temperature plot [`EMPTY_TEMPERATURE_AXIS`] when its axis is
//! empty although every value is a number), and a sweep that left rows out counts them in a note
//! over the plot ([`rows_not_plotted`]).
//!
//! Over each plot a line reads out the design's values the plot shows ([`PlotKind::readouts`],
//! decision M43-11), each a readout: hover it for its equation, click it to open it (spec
//! Addendum A2 "Hover"). The curves' points have no result path, so the plot itself keeps
//! egui_plot's coordinate readout.

use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

use crate::engine::meta::{NumOrText, ResultSet, Value};
use crate::engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor};
use crate::engine::sweeps::SweepRow;
use crate::gui::dashboard::{Level, end_effect_out_of_range, hover_text, result_info};
use crate::gui::format::{format_value, with_unit};
use crate::gui::readouts::Readouts;
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

    /// The results read out over the plot (decision M43-11): the design's values it draws.
    pub const fn readouts(self) -> &'static [&'static str] {
        match self {
            PlotKind::TorqueTemperature => &[
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "temperature.summary.governing_limit_C",
            ],
            PlotKind::GapSweep => &[
                "model.corner_gap_mm",
                "model.pullout_Nm",
                "model.required_floor_Nm",
            ],
            PlotKind::PoleSweep => &["model.pullout_Nm", "model.required_floor_Nm"],
            PlotKind::SlipHeating => &[
                "temperature.thermal.time_constant_s",
                "temperature.thermal.time_to_limit_high",
                "temperature.summary.governing_limit_C",
            ],
            PlotKind::TorqueAngle => &["model.pullout_Nm", "model.pullout_angle_rad"],
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

/// The text shown instead of the torque-temperature plot when its axis is empty: a design file
/// can hold a minimum temperature at or past the axis end (`InputSet::set` checks no range).
pub const EMPTY_TEMPERATURE_AXIS: &str = "Nothing to plot: the minimum temperature is not below \
     the axis end (10 °C past the higher of the operating temperature and the governing limit)";

/// The start of the note over a sweep plot that left rows out ([`rows_not_plotted`]).
pub const NOT_PLOTTED: &str = "Not plotted";

/// The note over a sweep plot that left out `n` rows holding a value that is not a number.
pub fn rows_not_plotted(n: usize) -> String {
    if n == 1 {
        format!("{NOT_PLOTTED}: 1 row holds a value that is not a number")
    } else {
        format!("{NOT_PLOTTED}: {n} rows hold a value that is not a number")
    }
}

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
    /// The temperature axis' ends: the minimum temperature and 10 °C past the higher of the
    /// operating temperature and the governing limit.
    pub axis_C: [f64; 2],
}

impl TorqueTemperature {
    /// Whether the axis is empty although both its ends are numbers (the minimum temperature at
    /// or past the axis end), as opposed to an end that is not a number.
    pub fn axis_is_empty(&self) -> bool {
        let [lo, hi] = self.axis_C;
        lo.is_finite() && hi.is_finite() && hi <= lo
    }
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
        axis_C: [md.min_temp_C, top],
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
    /// The rows left out: a swept variable or pull-out that is not finite.
    pub dropped: usize,
}

impl SweepPoints {
    /// Whether no row is drawn (every row left out, or none at all).
    pub fn is_empty(&self) -> bool {
        [
            &self.nominal,
            &self.below_minimum,
            &self.no_fit,
            &self.out_of_range,
            &self.line,
        ]
        .iter()
        .all(|points| points.is_empty())
    }
}

/// The markers of a sweep's rows: (swept variable, pull-out at the operating temperature).
pub fn sweep_points(rows: &[SweepRow]) -> SweepPoints {
    let mut points = SweepPoints::default();
    for row in rows {
        let p = [row.variable, row.pullout_op_Nm];
        if !(p[0].is_finite() && p[1].is_finite()) {
            points.dropped += 1;
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
/// height left and no more (`height` is egui_plot's smallest height too; it keeps at least 1
/// point).
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
    Plot::new(kind.id())
        .id(kind.id())
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label(x.to_owned())
        .y_axis_label(y.to_owned())
        .height(ui.available_height().max(0.0))
}

/// Draws the plot `kind` of the design shown (`inputs`, and the `results` computed from them),
/// under its readouts (`readouts`).
pub fn plot_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    readouts_ui(ui, kind, results, readouts);
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

/// The line over a plot: each of [`PlotKind::readouts`] as "label: value", a readout.
fn readouts_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    ui.horizontal_wrapped(|ui| {
        // Whole readouts move to the next row (a wrapped text would start mid-row, and its
        // hover area with it).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        for &path in kind.readouts() {
            let Some(info) = result_info(path) else {
                continue;
            };
            let value = results.get(path).unwrap_or(Value::None);
            let text = format!(
                "{}: {}",
                info.meta.label,
                with_unit(format_value(&value), info.meta.unit)
            );
            let rect = ui.small(text).rect;
            readouts.show_over(ui, rect, path, || hover_text(path).unwrap_or_default());
        }
    });
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
        ui.weak(if s.axis_is_empty() {
            EMPTY_TEMPERATURE_AXIS
        } else {
            NOTHING_TO_PLOT
        });
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
    if s.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    if s.dropped > 0 {
        ui.weak(rows_not_plotted(s.dropped));
    }
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
    fn the_torque_temperature_series_hold_the_engine_s_curve_and_its_band() {
        // The default design and one with another allowance and minimum temperature: the band
        // takes the design's own variation, the curve its own temperatures.
        let mut variant = DesignInputs::default();
        variant.metal.variation = 0.1;
        variant.metal.min_temp_C = -20.0;
        for inputs in [DesignInputs::default(), variant] {
            let r = compute_all(&inputs);
            let (min_temp, variation) = (inputs.metal.min_temp_C, inputs.metal.variation);
            let s = torque_temperature(&inputs, &r);
            let top = s.limit_C.expect("a finite governing limit") + PAST_THE_LIMIT_C;
            let span = top - min_temp;
            assert_eq!(s.nominal.len(), TEMPERATURE_SAMPLES);
            assert_eq!(s.low.len(), TEMPERATURE_SAMPLES);
            assert_eq!(s.high.len(), TEMPERATURE_SAMPLES);
            for (i, nominal) in s.nominal.iter().enumerate() {
                // An even grid from the minimum temperature to 10 degrees past the limit.
                let t = min_temp + span * i as f64 / (TEMPERATURE_SAMPLES - 1) as f64;
                assert!(
                    (nominal[0] - t).abs() <= 1e-9 * span,
                    "{i}: {nominal:?} vs {t}"
                );
                // The curve is the engine's pull-out at that magnet temperature ...
                assert_eq!(
                    nominal[1],
                    torque_at_temperature(&r, nominal[0]),
                    "nominal {i}"
                );
                // ... and the band is that curve at each end of the allowance: the low edge
                // minus the variation, the high edge plus it, never the other way round.
                let (low, high) = (s.low[i], s.high[i]);
                assert_eq!((low[0], high[0]), (nominal[0], nominal[0]), "{i}");
                assert!(
                    close(low[1], nominal[1] * (1.0 - variation)),
                    "low {i}: {low:?}"
                );
                assert!(
                    close(high[1], nominal[1] * (1.0 + variation)),
                    "high {i}: {high:?}"
                );
                assert!(low[1] < nominal[1] && nominal[1] < high[1], "{i}");
            }
            // The curve's first point is the minimum temperature: its high edge is the engine's
            // cold-high torque there.
            assert_eq!(s.nominal[0][0], min_temp);
            assert!(
                close(s.high[0][1], r.metal.torque_cold_high_Nm),
                "{:?}",
                s.high[0]
            );
        }
    }

    #[test]
    fn the_slip_heating_curves_are_each_case_s_first_order_rise() {
        let r = compute_all(&DesignInputs::default());
        let th = &r.temperature.thermal;
        assert!(
            th.steady_rise_est_C < th.steady_rise_high_C,
            "the two cases differ, so a curve drawn with the other's rise shows"
        );
        let s = slip_heating(&r);
        let span = HEATING_SPAN_TAUS * th.time_constant_s;
        for (name, series, steady) in [
            ("estimate", &s.estimate, th.steady_rise_est_C),
            ("high case", &s.high, th.steady_rise_high_C),
        ] {
            assert_eq!(series.len(), HEATING_SAMPLES, "{name}");
            assert_eq!(
                series[0],
                [0.0, th.start_C],
                "{name} starts at the starting temperature"
            );
            for (i, p) in series.iter().enumerate() {
                let t = span * i as f64 / (HEATING_SAMPLES - 1) as f64;
                assert!((p[0] - t).abs() <= 1e-9 * span, "{name} {i}: {p:?} vs {t}");
                let want = th.start_C + steady * (1.0 - (-p[0] / th.time_constant_s).exp());
                assert!(close(p[1], want), "{name} {i}: {p:?} vs {want}");
            }
            // Five time constants: within 1% of the way to the steady temperature.
            let end = series[HEATING_SAMPLES - 1];
            assert!(close(
                end[1],
                th.start_C + steady * (1.0 - (-HEATING_SPAN_TAUS).exp())
            ));
        }
        // The estimate stays below the high case once it has started to rise.
        assert!(
            s.estimate
                .iter()
                .zip(&s.high)
                .skip(1)
                .all(|(e, h)| e[1] < h[1])
        );
    }

    #[test]
    fn the_torque_rotation_curve_is_the_engine_s_torque_at_each_angle() {
        // 10 poles and another count: the mechanical angle runs over 720 / N degrees and the
        // electrical angle is N / 2 times it.
        for npole in [10, 12] {
            let mut inputs = DesignInputs::default();
            inputs.coupling.npole = npole;
            let r = compute_all(&inputs);
            let pairs = npole as f64 / 2.0;
            let pair_deg = 360.0 / pairs;
            let s = torque_angle(&inputs, &r);
            assert_eq!(s.curve.len(), ANGLE_SAMPLES, "{npole} poles");
            assert!(
                close(s.curve[ANGLE_SAMPLES - 1][0], pair_deg),
                "{npole} poles"
            );
            // The E7 pull-out sits a quarter of the way round a pole pair (90 electrical
            // degrees); the curve is the engine's pull-out torque there ...
            let quarter = (ANGLE_SAMPLES - 1) / 4;
            assert!(
                close(s.pull_out[0], pair_deg / 4.0),
                "{npole} poles: {:?}",
                s.pull_out
            );
            assert!(
                close(s.curve[quarter][0], pair_deg / 4.0),
                "{npole} poles: {:?}",
                s.curve[quarter]
            );
            assert!(
                close(s.curve[quarter][1], r.model.pullout_Nm),
                "{npole} poles: {:?} vs {}",
                s.curve[quarter],
                r.model.pullout_Nm
            );
            // ... and, the harmonics being odd, zero at the start and half way, and the
            // negative of the pull-out three quarters of the way.
            let tolerance = 1e-12 * r.model.pullout_Nm;
            for (index, want) in [
                (0, 0.0),
                (2 * quarter, 0.0),
                (4 * quarter, 0.0),
                (3 * quarter, -r.model.pullout_Nm),
            ] {
                let [deg, torque] = s.curve[index];
                assert!(
                    (torque - want).abs() <= tolerance,
                    "{npole} poles, {deg} degrees: {torque} vs {want}"
                );
            }
            // Every point is the torque at its own electrical angle.
            for (i, [deg, torque]) in s.curve.iter().enumerate() {
                assert!(
                    (deg - pair_deg * i as f64 / (ANGLE_SAMPLES - 1) as f64).abs() <= 1e-9,
                    "{npole} poles {i}"
                );
                let x = (deg * pairs).to_radians();
                assert_eq!(
                    *torque,
                    torque_at_angle(&inputs, &r, x),
                    "{npole} poles {i}"
                );
            }
        }
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
                |ui| plot_ui(ui, kind, inputs, &results, &mut Readouts::default()),
            ));
        }
        output.unwrap()
    }

    fn count(values: &[usize], want: usize) -> usize {
        values.iter().filter(|v| **v == want).count()
    }

    #[test]
    fn each_plot_reads_out_the_design_values_it_draws() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for kind in PlotKind::ALL {
            let texts = drawn_texts(&draw(kind, &inputs));
            for path in kind.readouts() {
                let info = result_info(path).unwrap_or_else(|| panic!("{path} is a result"));
                let value = results.get(path).unwrap();
                let want = format!(
                    "{}: {}",
                    info.meta.label,
                    with_unit(format_value(&value), info.meta.unit)
                );
                assert!(texts.contains(&want), "{kind:?}: missing {want:?}");
            }
            // The first readout of each plot has an equation to show.
            let first = kind.readouts()[0];
            assert!(
                crate::gui::readouts::registry()
                    .equation_for(first)
                    .is_some(),
                "{first}"
            );
        }
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

    /// Two frames of a sweep plot of `rows` (a plot settles its bounds on the first).
    fn draw_sweep(rows: &[SweepRow], design: [f64; 2]) -> egui::FullOutput {
        let ctx = egui::Context::default();
        let mut output = None;
        for _ in 0..2 {
            output = Some(sized_frame(
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| sweep_ui(ui, PlotKind::GapSweep, "x", PULL_OUT, rows, design, 1.0),
            ));
        }
        output.unwrap()
    }

    #[test]
    fn a_sweep_without_a_finite_row_says_there_is_nothing_to_plot() {
        // A harmonic set outside its choices (a struct written by hand) leaves every row's
        // pull-out NaN: the sweep tabs say so, as the other plots do, not an empty frame.
        let mut inputs = DesignInputs::default();
        inputs.coupling.max_harmonic = 4;
        let r = compute_all(&inputs);
        for (kind, rows) in [
            (PlotKind::GapSweep, &r.gap_sweep),
            (PlotKind::PoleSweep, &r.pole_sweep),
        ] {
            let s = sweep_points(rows);
            assert!(s.is_empty(), "{kind:?}: {s:?}");
            assert_eq!(s.dropped, rows.len(), "{kind:?}");
            let texts = drawn_texts(&draw(kind, &inputs));
            assert!(
                texts.iter().any(|t| t == NOTHING_TO_PLOT),
                "{kind:?}: {texts:?}"
            );
        }
        // The default design's rows are all finite: nothing dropped, no note.
        let r = compute_all(&DesignInputs::default());
        let s = sweep_points(&r.gap_sweep);
        assert!(!s.is_empty() && s.dropped == 0);
        let texts = drawn_texts(&draw(PlotKind::GapSweep, &DesignInputs::default()));
        assert!(!texts.iter().any(|t| t == NOTHING_TO_PLOT));
        assert!(
            !texts.iter().any(|t| t.starts_with(NOT_PLOTTED)),
            "{texts:?}"
        );
    }

    #[test]
    fn sweep_rows_that_are_not_numbers_are_counted_in_a_note() {
        // Some rows not numbers: the others are drawn, and a note counts those left out.
        let r = compute_all(&DesignInputs::default());
        let design = [r.model.corner_gap_mm, r.model.pullout_Nm];
        let mut rows = r.gap_sweep.clone();
        rows[0].pullout_op_Nm = f64::NAN;
        let s = sweep_points(&rows);
        assert_eq!(s.dropped, 1);
        assert_eq!(s.line.len(), rows.len() - 1);
        let texts = drawn_texts(&draw_sweep(&rows, design));
        assert!(texts.iter().any(|t| t == &rows_not_plotted(1)), "{texts:?}");
        assert!(!texts.iter().any(|t| t == NOTHING_TO_PLOT));
        rows[1].variable = f64::INFINITY;
        assert_eq!(sweep_points(&rows).dropped, 2);
        let texts = drawn_texts(&draw_sweep(&rows, design));
        assert!(texts.iter().any(|t| t == &rows_not_plotted(2)), "{texts:?}");
        assert_eq!(
            rows_not_plotted(1),
            "Not plotted: 1 row holds a value that is not a number"
        );
        assert_eq!(
            rows_not_plotted(2),
            "Not plotted: 2 rows hold a value that is not a number"
        );
    }

    #[test]
    fn an_empty_temperature_axis_is_named_as_such() {
        // A design file can hold a minimum temperature at or past the axis end (InputSet::set
        // checks no range): every value is finite, so the message names the empty axis.
        let defaults = DesignInputs::default();
        let r = compute_all(&defaults);
        let top = r.temperature.summary.governing_limit_C.max(50.0) + PAST_THE_LIMIT_C;
        for min_temp in [top, top + 1.0, 500.0] {
            let mut inputs = DesignInputs::default();
            inputs.metal.min_temp_C = min_temp;
            let r = compute_all(&inputs);
            let s = torque_temperature(&inputs, &r);
            assert!(s.nominal.is_empty(), "{min_temp}");
            assert_eq!(s.axis_C, [min_temp, top], "{min_temp}");
            let texts = drawn_texts(&draw(PlotKind::TorqueTemperature, &inputs));
            assert!(
                texts.iter().any(|t| t == EMPTY_TEMPERATURE_AXIS),
                "{min_temp}: {texts:?}"
            );
            assert!(!texts.iter().any(|t| t == NOTHING_TO_PLOT), "{min_temp}");
        }
        // Values that are not numbers still say so: the minimum temperature itself, and a
        // harmonic set outside its choices (the axis is fine, every torque NaN).
        let mut nan_min = DesignInputs::default();
        nan_min.metal.min_temp_C = f64::NAN;
        let mut bad_set = DesignInputs::default();
        bad_set.coupling.max_harmonic = 4;
        for inputs in [nan_min, bad_set] {
            let texts = drawn_texts(&draw(PlotKind::TorqueTemperature, &inputs));
            assert!(texts.iter().any(|t| t == NOTHING_TO_PLOT), "{texts:?}");
            assert!(!texts.iter().any(|t| t == EMPTY_TEMPERATURE_AXIS));
        }
    }

    #[test]
    fn each_plot_stays_in_a_short_region() {
        // The space left above the Equation panel: each plot's height is capped by what its
        // readouts line leaves, so the plot ends at the region's foot (no 150-point floor). A
        // region a few points tall or none draws without panicking.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let size = egui::vec2(1000.0, 700.0);
        for kind in PlotKind::ALL {
            for height in [300.0, 120.0, 80.0, 5.0, 0.0] {
                let region =
                    egui::Rect::from_min_size(egui::pos2(20.0, 30.0), egui::vec2(700.0, height));
                for _ in 0..2 {
                    let (_, used) =
                        crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                            plot_ui(ui, kind, &inputs, &results, &mut Readouts::default());
                        });
                    // The readouts line is text and does not shrink: only a region that holds
                    // it holds the plot too.
                    if height >= 80.0 {
                        assert!(
                            used.bottom() <= region.bottom() + 0.01,
                            "{kind:?} at {height}: the plot runs {} points past the region",
                            used.bottom() - region.bottom()
                        );
                    }
                }
            }
        }
    }
}
