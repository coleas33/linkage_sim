//! Addendum A3: the model assumptions, apart from the design inputs.
//!
//! Every assumption of the spec's v1 set is an input flagged `.assumption()` where it is
//! declared, and its value, unit, label, slider and workbook cell live there, once. This
//! module adds what the assumptions panel shows beside them, a rationale and a source, and
//! the engine support the spec names: which assumptions differ from their workbook default
//! (the "assumptions modified" banner, [`any_modified`], [`modified`]) and the reset
//! ([`reset_to_workbook_defaults`]). The end-effect coefficient is two inputs, the
//! Calculator's and the Calibration's, which the differential data vary independently
//! (report 6.5), so a row lists its input paths.
//!
//! "Workbook default" is [`DesignInputs::default`]: no approved correction changes an
//! assumption's default (E1, E3 and E5 correct other inputs; `tests/assumptions.rs`
//! checks it), and the Rust-only harmonic set defaults to the workbook's 1, 3, 5.
//!
//! Two assumptions have documented overrides (A-1 plan decisions A2 and A9): with
//! correction E20 and the coercivity source at 1, the Hcj temperature coefficient acts only
//! for a magnet without a grade; a library back iron with its own design flux density
//! (1018) replaces the back-iron design flux density in the wall check.
//!
//! The spec's traceability test (each assumption changes every dependent result and no
//! independent one, by the equation registry's dependency graph) needs the A2 equation
//! registry and belongs to plan A-3.

use super::api::DesignInputs;
use super::meta::{InputSet, Value, input_rows};

/// One assumption of the Addendum A3 panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assumption {
    /// Stable id (the GUI's key).
    pub id: &'static str,
    /// The panel's label.
    pub label: &'static str,
    /// The input paths it sets, each flagged `.assumption()`, all in one unit.
    pub paths: &'static [&'static str],
    /// Why the value is what it is.
    pub rationale: &'static str,
    /// Where it comes from: the workbook cell of each input and the report that discusses it.
    pub source: &'static str,
}

/// The spec's v1 set, in the spec's order.
pub const ASSUMPTIONS: [Assumption; 14] = [
    Assumption {
        id: "harmonics",
        label: "Harmonics included",
        paths: &["coupling.max_harmonic"],
        rationale: "Each ring's alternating magnetization is a square wave whose odd space harmonics 1, 3, 5, ... add to the torque; the workbook sums 1, 3 and 5. A harmonic's wave number grows with its order, so its field decays faster across the gap: higher harmonics add little at the default gap and more at small gaps or low fill.",
        source: "Spec Addendum A3 (workbook 1, 3, 5; selectable up to 11); workbook Calculator rows 71 to 88; M2 decision D7 and Addendum A decision 29 (the peak search for any set).",
    },
    Assumption {
        id: "end_effect",
        label: "End-effect coefficient",
        paths: &["coupling.c_end", "calibration.c_end"],
        rationale: "Empirical factor for the finite axial length, f_end = 1 − c_end · pole pitch / L, on the Calculator and on the Calibration prototype (each keeps its own input). Within 1 % of 3D for magnets longer than about 3 mm; at or below L = c_end · pole pitch the factor is 0 or negative and the end-effect model is out of range (end_effect_check).",
        source: "Workbook Calculator!C41 and Calibration!C23 (0.15); M1 audit M8 (error band against 3D) and M9 (negative for short magnets).",
    },
    Assumption {
        id: "calibration_factor",
        label: "Calibration factor",
        paths: &["calibration.f_cal_original"],
        rationale: "The model's calibration coefficient for every design but the measured prototype: the bench correction (Calibration C9) applies only to the prototype's own rings, pole count and circuit (Calculator C42). The audit found the 2D model within a few per cent of exact 2D sections.",
        source: "Workbook Calibration!C24 (0.95); M1 audit M4 to M6.",
    },
    Assumption {
        id: "production_variation",
        label: "Production variation",
        paths: &["metal.variation"],
        rationale: "Symmetric allowance on the pull-out for magnet, gap and assembly scatter: the hot low torque is the pull-out at the operating temperature times (1 − variation), the cold high torque the cold pull-out times (1 + variation). An engineering allowance, not measured.",
        source: "Workbook Metal design!C18 (±15 %).",
    },
    Assumption {
        id: "br_temperature_coefficient",
        label: "Br temperature coefficient",
        paths: &["calibration.alpha_br_per_C"],
        rationale: "Reversible remanence coefficient: Br(T) = Br(20 °C) · (1 + α (T − 20 °C)), and torque scales with Br², on every sheet. −0.12 %/°C is the sintered NdFeB value of every library part's grade.",
        source: "Workbook Calibration!C22 (−0.0012 /°C); Addendum A grade table (K&J, sintered NdFeB).",
    },
    Assumption {
        id: "hcj_temperature_coefficient",
        label: "Hcj temperature coefficient",
        paths: &["temperature.demag.beta_hcj_per_C"],
        rationale: "Effective coefficient of the intrinsic coercivity over 20 to 150 °C, for the knee Hk(T) = knee · Hcj(20 °C) · (1 + β (T − 20 °C)). With correction E20 a magnet with a grade (every library part) uses its grade's β unless the coercivity source is set to the inputs; this value acts for a magnet without a grade or with that source.",
        source: "Workbook Temperature design!C45 (−0.50 %/°C, N42SH); Addendum A decisions 18 (Arnold's −0.55 %/°C is the reference, not the default) and 19 (E20).",
    },
    Assumption {
        id: "knee_fraction",
        label: "Demagnetization knee fraction",
        paths: &["temperature.demag.knee_fraction"],
        rationale: "The knee of the demagnetization curve as a fraction of Hcj: the reverse field at which irreversible loss starts. The onsets are calibrated to the magnet's rating; the audit found that calibration uses 9.83 °C of the 10 °C design margin, so the margin left for model-form uncertainty is thin.",
        source: "Workbook Temperature design!C46 (0.9); M1 audit rulings (onset calibration) and M13.",
    },
    Assumption {
        id: "demag_margin",
        label: "Demagnetization margin",
        paths: &["temperature.demag.design_margin_C"],
        rationale: "Margin kept from the skipping onset (like poles facing, the largest reverse field): the magnet design limit is that onset minus this margin, and with E20 a positive-beta magnet's cold limit is its cold onset plus it.",
        source: "Workbook Temperature design!C51 (10 °C); M1 audit rulings (onset calibration).",
    },
    Assumption {
        id: "backiron_design_flux_density",
        label: "Back-iron design flux density",
        paths: &["materials.steel.bsat_T"],
        rationale: "The flux density the back-iron wall is sized to, t = B_gap · pole pitch / (π · B): a design limit below saturation for annealed 4140 (1018 about 1.7 T, pre-hardened stock about 1.4 T). A library back iron with its own design value (1018) replaces it in the wall check.",
        source: "Workbook Materials!C13 (1.5 T); Addendum A decision 20 and the A-1 plan's decision A9; M1 audit M1 (the wall formula reads 12 to 16 % thin).",
    },
    Assumption {
        id: "thermal_conductance",
        label: "Thermal conductance",
        paths: &["temperature.thermal.conductance_W_K"],
        rationale: "One conductance from the rotating coupling to the housing and both shafts; it sets the steady slip rise and the thermal time constant. A placeholder, not measured.",
        source: "Workbook Temperature design!C142 (0.3 W/K); M1 audit placeholder inputs (−2.48 °C of steady high-case magnet temperature per +10 %).",
    },
    Assumption {
        id: "driving_rise",
        label: "Driving temperature rise",
        paths: &["temperature.duty.driving_rise_C"],
        rationale: "The coupling's rise above ambient while driving without slip (housing air, sun, gearbox heat). It sets the hot-day starting temperature, the most influential thermal input. A placeholder, not measured.",
        source: "Workbook Temperature design!C36 (10 °C); M1 audit placeholder inputs.",
    },
    Assumption {
        id: "slip_event_duration",
        label: "Slip-event duration",
        paths: &["metal.slip_event_s"],
        rationale: "How long one slip event lasts; it sets the life slip rotations, the heat per event and the slip duty. Illustrative: replace it with the recorded value.",
        source: "Workbook Metal design!C87 (0.1 s); M1 audit placeholder inputs.",
    },
    Assumption {
        id: "clamp_friction",
        label: "Clamp friction coefficient",
        paths: &["clamps.friction"],
        rationale: "Friction between the shaft and the clamp bore in the clamp capacity µ · preload · d · factor: degreased; 0.10 if the bore could be oily. The adapter joint's friction (Shaft clamps C66) is a design input, not this assumption.",
        source: "Workbook Shaft clamps!C18 (0.15); Addendum A report section 6.5 (which input the assumption names).",
    },
    Assumption {
        id: "preload_fraction",
        label: "Preload fraction of proof load",
        paths: &["clamps.preload_fraction"],
        rationale: "Screw preload as a share of the proof load (bolted-joint practice), capped by thread stripping in the aluminium.",
        source: "Workbook Shaft clamps!C29 (75 %); spec M1 (clamps).",
    },
];

/// One assumption as the panel shows it for a set of inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct AssumptionState {
    pub assumption: &'static Assumption,
    /// The current value of each path, in `paths` order.
    pub values: Vec<Value>,
    /// The workbook default of each path, in `paths` order.
    pub defaults: Vec<Value>,
    /// The unit of every path.
    pub unit: &'static str,
    /// Whether any path differs from its workbook default (the panel's changed dot).
    pub modified: bool,
}

/// The value at an assumption path (every path is an input: `tests/assumptions.rs`).
fn value_at(inputs: &DesignInputs, path: &str) -> Value {
    inputs
        .get(path)
        .expect("an assumption path is an input (tests/assumptions.rs)")
}

/// The panel for `inputs`: every assumption with its values, workbook defaults and unit, in
/// [`ASSUMPTIONS`] order.
pub fn states(inputs: &DesignInputs) -> Vec<AssumptionState> {
    let defaults = DesignInputs::default();
    let rows = input_rows(inputs);
    ASSUMPTIONS
        .iter()
        .map(|a| {
            let values: Vec<Value> = a.paths.iter().map(|p| value_at(inputs, p)).collect();
            let workbook: Vec<Value> = a.paths.iter().map(|p| value_at(&defaults, p)).collect();
            let unit = rows
                .iter()
                .find(|r| r.path == a.paths[0])
                .map_or("", |r| r.meta.unit);
            AssumptionState {
                assumption: a,
                modified: values != workbook,
                values,
                defaults: workbook,
                unit,
            }
        })
        .collect()
}

/// The assumptions that differ from their workbook default, in [`ASSUMPTIONS`] order.
pub fn modified(inputs: &DesignInputs) -> Vec<&'static Assumption> {
    let defaults = DesignInputs::default();
    ASSUMPTIONS
        .iter()
        .filter(|a| {
            a.paths
                .iter()
                .any(|p| value_at(inputs, p) != value_at(&defaults, p))
        })
        .collect()
}

/// Whether any assumption differs from its workbook default (the spec's "assumptions
/// modified" banner).
pub fn any_modified(inputs: &DesignInputs) -> bool {
    !modified(inputs).is_empty()
}

/// Puts every assumption back to its workbook default and leaves every design input as it
/// is (the spec's "reset to workbook defaults" button).
pub fn reset_to_workbook_defaults(inputs: &mut DesignInputs) {
    let defaults = DesignInputs::default();
    for a in &ASSUMPTIONS {
        for path in a.paths {
            inputs
                .set(path, value_at(&defaults, path))
                .expect("a default is a valid value of its own input");
        }
    }
}
