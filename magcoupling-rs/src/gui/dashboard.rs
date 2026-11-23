//! The dashboard's rows and their badges (spec M4 "Layout": "headline numbers with
//! green/amber/red badges derived from check verdicts; corrected values carry a 'corrected vs
//! workbook' marker with the deviation in its tooltip"). Since the UX overhaul (spec 4.5) no
//! one list draws them all: the right column is the check list ([`crate::gui::checks`]) over
//! the Explain panel ([`crate::gui::explain`]), which draws the rows a check's verdict badges
//! as that check's numbers ([`line_ui`]); the Results tab's Headline group lists every row, and
//! the spreadsheet's Summary writes them.
//!
//! [`DASHBOARD`] lists the rows: the 15 headline numbers in Python's order, then the shaft
//! joints check (BL-064), the gearbox check (BL-059), the A1 space claim badge and the audit M9
//! end-effect flag. A row's badge comes from a check verdict ([`verdict_level`]); its marker
//! from the deviation registry ([`crate::gui::corrections`]). Two notes from the user's
//! decisions of 2026-09-30:
//!
//! - **End effect (audit M9).** When f_end ≤ 0 the pull-out and the numbers computed from it
//!   ([`END_EFFECT_ROWS`]) are greyed, without a badge, under the banner "End-effect model out
//!   of range", which carries a "Why: <note>" link to the end-effect check's note (slice-2 spec
//!   3.7; `end_effect_banner_ui` draws it over the centre region).
//! - **Stored 3D values.** The temperature rows and the shaft joints row that read the stored
//!   3D fields ([`STORED_3D_ROWS`]; the shaft joints row through the inner ring's single-ring
//!   onset, which caps the bonded inner hub's fitting temperature, owner R-15) carry the label
//!   [`STORED_3D_LABEL`]: the fields ([`STORED_3D_INPUTS`]) are 3D results for the example
//!   design or for the starting design ([`STARTING_3D_FIELDS`]) and stay fixed when the design
//!   changes; no live 3D model runs. The check list closes with the label, and the entries of
//!   the two checks that read the fields repeat it.
//!
//! Under the check list's rows, the material warnings that fire (spec Addendum A5: "plain
//! language, colour-coded, linked to their teaching note"): each rule's text in its severity's
//! colour (a warning red, a caution amber) with a link that opens its reviewed note in the
//! Equation panel ([`warning_lines`], decision M43-8).
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). A row's text is built only while it is hovered
//! ([`DashboardLine::tooltip`]).
//!
//! Check guides (guidance spec 3.1 and 3.4, [`crate::gui::guidance`]): the design status line
//! tops the check list ([`status_ui`]), and the Magnets fit check closes its rows (a GUI check
//! of `model::blocks_fit`, so it is apart from [`DASHBOARD`], whose paths are results). A
//! check's problem, its guide's sentence, its fixes and their notes are its Explain entry. An
//! entry's lesson link ([`learn_ui`]) is a link to its reviewed note (opened in the Equation
//! panel), a link to its assumption (its row shown, the rationale in the hover text) or "no
//! note yet".

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::assumptions::ASSUMPTIONS;
use crate::engine::clamps;
use crate::engine::explain::notes;
use crate::engine::materials::CUP_WALL_TOO_THIN;
use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::engine::shaft_joint;
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::checks::state_mark;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, value_with_unit};
use crate::gui::guidance::{BANNER_NOTE, Learn, design_status, status_line};
use crate::gui::look;
use crate::gui::plain_text::{display_label, help_for, label_for};
use crate::gui::readouts::Readouts;
use crate::gui::typeset::glyph_safe;
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict. Ordered by severity (green, amber, red): the worst of
/// several levels is their maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Green: the check passes.
    Good,
    /// Amber: the check asks for a look (a temperature "CHECK", an unknown space claim).
    Caution,
    /// Red: the check fails.
    Bad,
}

impl Level {
    /// The word of the badge's hover text.
    pub const fn word(self) -> &'static str {
        match self {
            Level::Good => "OK",
            Level::Caution => "Check",
            Level::Bad => "Fails",
        }
    }

    /// The badge colour: the panel's own green ([`crate::gui::look::OK`] in a dark theme), the
    /// theme's warning colour, the theme's error colour.
    pub fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        match self {
            Level::Good if visuals.dark_mode => crate::gui::look::OK,
            Level::Good => egui::Color32::from_rgb(20, 130, 60),
            Level::Caution => visuals.warn_fg_color,
            Level::Bad => visuals.error_fg_color,
        }
    }
}

/// The heading of the material warnings.
pub const WARNINGS_HEADING: &str = "Material warnings";

/// The start of a warning's link to its teaching note.
pub const WHY: &str = "Why";

/// What a lesson link says when no reviewed note or assumption explains the row.
pub const NO_NOTE_YET: &str = "no note yet";

/// The hover text of a lesson link whose note is not reviewed (sent back to Draft): it never
/// shows.
pub const NOTE_NOT_REVIEWED: &str = "Its teaching note has not passed review.";

/// The Magnets fit check's value in its Explain entry: the blocks fit around the ring, or not.
pub const FITS: &str = "Fit around the ring";
pub const DOES_NOT_FIT: &str = "Do not fit around the ring";

/// A warning's badge colour: a warning (the coupling does not work as designed) red, a caution
/// amber.
pub const fn severity_level(severity: Severity) -> Level {
    match severity {
        Severity::Warning => Level::Bad,
        Severity::Caution => Level::Caution,
    }
}

/// The material warnings that fire for `results`, in the rules' order: each rule and its text.
pub fn warning_lines(results: &DesignResults) -> Vec<(&'static WarningRule, String)> {
    WARNING_RULES
        .iter()
        .filter_map(|rule| match results.get(&format!("warnings.{}", rule.id)) {
            Some(Value::Text(text)) if !text.is_empty() => Some((rule, text)),
            _ => None,
        })
        .collect()
}

/// The reviewed teaching note of a warning rule, if its note has passed the accuracy gate
/// ([`reviewed_note`]).
pub fn warning_note(rule: &WarningRule) -> Option<&'static notes::Note> {
    reviewed_note(rule.note_id)
}

/// The temperature summary's verdict, the badge of the three temperature rows.
const TEMPERATURE_VERDICT: &str = "temperature.summary.verdict";

/// The dashboard rows: (result path, the check whose verdict gives its badge). The first 15
/// are `HEADLINE`, in order (a test checks).
pub const DASHBOARD: [(&str, Option<&str>); 19] = [
    ("model.pullout_Nm", None),
    ("model.pullout_20C_Nm", None),
    ("metal.torque_hot_low_Nm", Some("metal.hot_min_check")),
    ("metal.hot_min_check", Some("metal.hot_min_check")),
    ("metal.torque_cold_high_Nm", None),
    ("model.gearbox_input_ripple_Nm", None),
    ("model.cup_od_mm", None),
    ("mass.total_g", None),
    (
        "metal.min_running_clearance_mm",
        Some("metal.clearance_check"),
    ),
    ("metal.clearance_check", Some("metal.clearance_check")),
    ("materials.cup_wall_check", Some("materials.cup_wall_check")),
    (
        "temperature.summary.governing_limit_C",
        Some(TEMPERATURE_VERDICT),
    ),
    (
        "temperature.summary.margin_hot_day_C",
        Some(TEMPERATURE_VERDICT),
    ),
    (TEMPERATURE_VERDICT, Some(TEMPERATURE_VERDICT)),
    ("clamps.recommended", Some("clamps.recommended")),
    ("shaft_joint.check", Some("shaft_joint.check")),
    ("gearbox.cold_high_check", Some("gearbox.cold_high_check")),
    (
        "housing.space_claim_check",
        Some("housing.space_claim_check"),
    ),
    ("model.end_effect_check", Some("model.end_effect_check")),
];

/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the rows
/// that move with the end-effect coefficient (a test probes them at back iron 1 and 0), the
/// headline's and the shaft joints and gearbox checks, both computed from cold-high. The whole
/// shaft joints row greys, its stress and fitting terms too, which do not read cold-high: one
/// row cannot grey in part, and the end-effect check fails there under its banner (BL-064).
/// Per-row greying of the results table waits for plan A-3's dependency graph; the table shows
/// the banner instead (decision M41-11), and its four checks among these rows (the hot minimum,
/// the recommended screw, the shaft joints, the gearbox check) give no badge there either
/// ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 9] = [
    "model.pullout_Nm",
    "model.pullout_20C_Nm",
    "metal.torque_hot_low_Nm",
    "metal.hot_min_check",
    "metal.torque_cold_high_Nm",
    "model.gearbox_input_ripple_Nm",
    "clamps.recommended",
    "shaft_joint.check",
    "gearbox.cold_high_check",
];

/// The dashboard rows that read the workbook's stored 3D fields (the reverse fields of the
/// demagnetization check and the slip-loss fields): the dashboard rows that move with them (a
/// test probes each at its slider ends). The three temperature rows, and the shaft joints row:
/// the bonded inner hub's allowed fitting temperature is capped at the inner ring's single-ring
/// onset less the cure margin (owner R-15), and the onset follows the stored single-ring
/// reverse field.
pub const STORED_3D_ROWS: [&str; 4] = [
    "temperature.summary.governing_limit_C",
    "temperature.summary.margin_hot_day_C",
    TEMPERATURE_VERDICT,
    "shaft_joint.check",
];

/// The label of the stored-3D rows. It names both designs that come with stored fields (the
/// example design's are the inputs' defaults, the starting design's [`STARTING_3D_FIELDS`]),
/// so it is true at each of them and at a design edited away from either: an edit never moves
/// the fields.
pub const STORED_3D_LABEL: &str = "Fixed 3D fields: starting or example design";

/// Its hover text.
pub const STORED_3D_NOTE: &str = "The temperature checks use the demagnetization reverse fields and the slip-heating fields, computed in 3D for the starting design and for the example design, each for its own geometry. The bonded inner hub's allowed fitting temperature, on the shaft joints row, uses one of the reverse fields too. Those fields stay fixed when you change the geometry, magnet count or grade; for another design, enter its own 3D fields to check the limits.";

/// The stored 3D inputs: the four reverse fields of the demagnetization check and the
/// slip-loss fields (the steel circuit's seven, and the three free-space ones). They are 3D
/// results for a named design, the example design or the starting design, and stay fixed when
/// the design changes ([`STORED_3D_SENTENCE`]); [`STORED_3D_ROWS`] are the dashboard rows that
/// move with them: the temperature rows and the shaft joints row.
pub const STORED_3D_INPUTS: [&str; 14] = [
    "temperature.demag.h_rev_aligned_kA_m",
    "temperature.demag.h_rev_pullout_kA_m",
    "temperature.demag.h_rev_likepole_kA_m",
    "temperature.demag.h_rev_single_ring_kA_m",
    "temperature.slip_loss.b_hub_T",
    "temperature.slip_loss.b_cup_T",
    "temperature.slip_loss.b_sleeve_T",
    "temperature.slip_loss.b_liner_T",
    "temperature.slip_loss.cap_integral_T2m4",
    "temperature.slip_loss.web_integral_T2m2",
    "temperature.slip_loss.b_magnet_T",
    "temperature.slip_loss.b_hub_free_T",
    "temperature.slip_loss.b_cup_free_T",
    "temperature.slip_loss.web_integral_free_T2m2",
];

/// The 3D fields the standalone app's starting design stores (`app::starting_design`; backlog
/// BL-080), in [`STORED_3D_INPUTS`]' order: the reference routine's results for its geometry
/// (8 magnets, ring radius 9.00 mm, face gap 1.80 mm), where the inputs' defaults are the
/// example design's. A test of the app holds them to a recorded run of the routine
/// (`the_starting_design_stores_the_3d_fields_of_its_own_geometry`, which says how to run it
/// again). They are here, not beside the design, because the temperature Problem names the
/// design whose fields a design holds (`guidance`).
pub const STARTING_3D_FIELDS: [f64; 14] = [
    386.0, 748.0, 850.0, 585.0, 0.202, 0.207, 0.335, 0.349, 4.04e-10, 4.405e-5, 0.17, 0.07416,
    0.08099, 6.039e-6,
];

/// The sentence the on-screen help of every stored 3D input ends with, whether the overlay
/// (`gui::plain_text`) or the engine words it (spec 3.8 and section 8, amendment A-7: the
/// stored fields never follow the design). Typed once, in the engine's
/// `engine::temperature::stored_3d_sentence!`.
pub const STORED_3D_SENTENCE: &str = crate::engine::temperature::stored_3d_sentence!();

/// The start of the banner shown when f_end ≤ 0.
pub const END_EFFECT_BANNER: &str = END_EFFECT_OUT_OF_RANGE;

/// What a greyed row's hover text says after [`END_EFFECT_OUT_OF_RANGE`].
pub const END_EFFECT_GREYED: &str = "the pull-out and every number computed from it are not valid while the end-effect factor is zero or below.";

/// The line over the dashboard's rows while the spreadsheet details show (spec S2-5): what the
/// ids beside the values are.
pub const MARKER_LEGEND: &str = "E-numbers beside a value are corrections to the original spreadsheet that change it; hover the value to see each one.";

/// Every design-level check of the results, in schema order (decision O-3): the dashboard's
/// verdicts, the coupling model's, the temperature design's and the clamp's checks and
/// screens, the material warnings, the space claim, the gearbox check and the shaft joints
/// check. Each gives a badge level ([`verdict_level`]): the results table shows it, and its
/// failing filter lists the checks that fail ([`failing_checks`]). The clamp table's per-size
/// checks and the sweeps' status texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 33] = [
    "calibration.end_effect_check",
    "model.inner_flat_check",
    "model.outer_flat_check",
    "model.end_effect_check",
    "model.verdict",
    "model.cup_ring_check",
    "model.hub_check",
    "model.inner_temp_check",
    "model.outer_temp_check",
    "metal.hot_min_check",
    "metal.clearance_check",
    "materials.cup_wall_check",
    "temperature.summary.torque_hot_day_note",
    TEMPERATURE_VERDICT,
    "temperature.demag.cold_check",
    "temperature.adhesive.fatigue_screen",
    "temperature.mismatch.reading",
    "temperature.magnet_life.torque_hot_day_check",
    "temperature.adhesive_life.hot_fatigue_screen",
    "temperature.adhesive_life.daily_screen",
    "temperature.adhesive_life.daily_entered_screen",
    "clamps.recommended",
    "clamps.head_check",
    "clamps.vent_port",
    "warnings.non_ferromagnetic_back_iron",
    "warnings.ferromagnetic_sleeve_or_liner",
    "warnings.high_conductivity_sleeve_or_liner",
    "warnings.low_saturation",
    "warnings.uncoated_low_alloy_steel",
    "warnings.cte_mismatch_with_magnets",
    "housing.space_claim_check",
    "gearbox.cold_high_check",
    "shaft_joint.check",
];

/// The badge level of a check's verdict text; `None` (no badge) for a path that is no check, a
/// verdict that says the check does not apply (the cup wall's "No back iron"), or a text the
/// check does not produce. A warning (`warnings.<rule>`) that fires has its severity's level.
pub fn verdict_level(path: &str, text: &str) -> Option<Level> {
    use Level::{Bad, Caution, Good};
    const FLATS: [&str; 2] = ["model.inner_flat_check", "model.outer_flat_check"];
    const HOT_DAY: [&str; 2] = [
        "temperature.summary.torque_hot_day_note",
        "temperature.magnet_life.torque_hot_day_check",
    ];
    // BL-063: the daily-cycle screen at the recommended bondline and at the entered one.
    const DAILY: [&str; 2] = [
        "temperature.adhesive_life.daily_screen",
        "temperature.adhesive_life.daily_entered_screen",
    ];
    // BL-064: each failing shaft joint text starts with its cause's constant.
    const SHAFT_JOINT_FAILS: [&str; 5] = [
        shaft_joint::FIT_SLIPS,
        shaft_joint::BORE_OVER_ITS_LIMIT,
        shaft_joint::CANNOT_BE_FITTED,
        shaft_joint::KEY_OVER_ITS_LIMIT,
        shaft_joint::KEY_HAS_NO_BEARING_AREA,
    ];
    match (path, text) {
        ("calibration.end_effect_check", "OK") => Some(Good),
        ("calibration.end_effect_check", END_EFFECT_OUT_OF_RANGE) => Some(Bad),
        (p, t) if FLATS.contains(&p) && t.starts_with("OK, ") => Some(Good),
        (p, t) if FLATS.contains(&p) && t.starts_with("TOO NARROW: ") => Some(Bad),
        // Arcs have no flats: the check does not apply.
        (p, "n/a (arcs)") if FLATS.contains(&p) => None,
        // The nominal pull-out covers the floor; the workbook asks for a hot test to confirm
        // it, as the temperature verdict asks for its tests (both green).
        ("model.verdict", "Nominal only: hot test") => Some(Good),
        ("model.verdict", "Below hot minimum") => Some(Bad),
        ("model.cup_ring_check" | "model.hub_check", "Thickness OK") => Some(Good),
        ("model.cup_ring_check" | "model.hub_check", "Too thin") => Some(Bad),
        // E9: without back iron no magnetic rule sizes the steel.
        ("model.cup_ring_check" | "model.hub_check", "No back iron") => None,
        ("model.inner_temp_check" | "model.outer_temp_check", "OK") => Some(Good),
        ("model.inner_temp_check" | "model.outer_temp_check", "OVER the magnet rating") => {
            Some(Bad)
        }
        // A manual magnet without a grade has no rating to check against: a look, as an
        // unknown space claim.
        ("model.inner_temp_check" | "model.outer_temp_check", "unknown") => Some(Caution),
        (p, "Meets it nominally (no variation allowance)") if HOT_DAY.contains(&p) => Some(Good),
        (p, "Below it") if HOT_DAY.contains(&p) => Some(Bad),
        ("temperature.demag.cold_check", "OK") => Some(Good),
        ("temperature.demag.cold_check", "Below the cold demagnetization limit") => Some(Bad),
        // NdFeB's coercivity rises as it cools: the cold check does not apply.
        ("temperature.demag.cold_check", t) if t.starts_with("n/a") => None,
        ("temperature.adhesive.fatigue_screen", t) if t.starts_with("OK: ") => Some(Good),
        ("temperature.adhesive.fatigue_screen", "CHECK") => Some(Caution),
        ("temperature.mismatch.reading", "Below the lap-shear strength") => Some(Good),
        ("temperature.mismatch.reading", "Above the lap-shear strength at the block ends") => {
            Some(Bad)
        }
        ("temperature.adhesive_life.hot_fatigue_screen", "OK") => Some(Good),
        ("temperature.adhesive_life.hot_fatigue_screen", "CHECK: get hot fatigue data") => {
            Some(Caution)
        }
        (p, "Below the fatigue endurance") if DAILY.contains(&p) => Some(Good),
        (p, "Above the fatigue endurance: qualify by thermal cycling") if DAILY.contains(&p) => {
            Some(Caution)
        }
        // The clamp's head and key checks read empty when no screw fits (the recommended
        // screw's "None:" is the red one).
        ("clamps.head_check", "OK") => Some(Good),
        ("clamps.head_check", "Use a hardened washer") => Some(Caution),
        ("clamps.vent_port", t) if t.starts_with("Yes: ") => Some(Good),
        ("clamps.vent_port", t) if t.starts_with("No: ") => Some(Caution),
        (p, t) if !t.is_empty() && p.starts_with("warnings.") => WARNING_RULES
            .iter()
            .find(|rule| p.strip_prefix("warnings.") == Some(rule.id))
            .map(|rule| severity_level(rule.severity)),
        ("metal.hot_min_check", "Estimate covers hot min") => Some(Good),
        ("metal.hot_min_check", "Below hot minimum") => Some(Bad),
        ("metal.clearance_check", "Meets assumed target") => Some(Good),
        ("metal.clearance_check", "Below target") => Some(Bad),
        ("materials.cup_wall_check", "OK") => Some(Good),
        // E9: no magnetic rule sizes a cup without back iron (the suggested wall reads "n/a").
        ("materials.cup_wall_check", "No back iron") => None,
        ("materials.cup_wall_check", t) if t.starts_with("Too thin: ") => Some(Bad),
        (TEMPERATURE_VERDICT, t) if t.starts_with("OK on temperature.") => Some(Good),
        (TEMPERATURE_VERDICT, t) if t.starts_with("CHECK:") => Some(Caution),
        ("clamps.recommended", t) if t.starts_with("ISO 4762 ") => Some(Good),
        ("clamps.recommended", t) if t.starts_with("None:") => Some(Bad),
        ("housing.space_claim_check", crate::engine::housing::INSIDE_THE_SPACE_CLAIM) => Some(Good),
        ("housing.space_claim_check", crate::engine::housing::SPACE_CLAIM_UNKNOWN) => Some(Caution),
        ("housing.space_claim_check", t) if t.starts_with("Exceeds the space claim:") => Some(Bad),
        ("gearbox.cold_high_check", crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING) => {
            Some(Good)
        }
        ("gearbox.cold_high_check", crate::engine::gearbox::GEARBOX_CHECK_UNKNOWN) => Some(Caution),
        ("gearbox.cold_high_check", t)
            if t.starts_with(crate::engine::gearbox::ABOVE_THE_GEARBOX_RATING) =>
        {
            Some(Bad)
        }
        ("shaft_joint.check", shaft_joint::SHAFT_JOINTS_PASS) => Some(Good),
        // Every unknown form: not a number, a safety factor not above zero (ruling C-4), a value
        // outside what it can physically be (its extension), a gearbox shaft below its slider's
        // bottom (build ruling B-5), no wall around a bore, a reversed band.
        ("shaft_joint.check", t) if t.starts_with(shaft_joint::SHAFT_JOINT_UNKNOWN) => {
            Some(Caution)
        }
        ("shaft_joint.check", t) if SHAFT_JOINT_FAILS.iter().any(|&f| t.starts_with(f)) => {
            Some(Bad)
        }
        ("model.end_effect_check", "OK") => Some(Good),
        ("model.end_effect_check", END_EFFECT_OUT_OF_RANGE) => Some(Bad),
        _ => None,
    }
}

/// What the screen draws for a check's verdict in place of the engine's text (ruling R9): the
/// cup wall's "Too thin" verdict names the wall by its input's on-screen label instead of the
/// original spreadsheet's cell. [`verdict_level`] has already read the engine text, and the
/// number is the engine's own, kept as written. `None` for any other path or text, and for a
/// number that is not a finite number: the engine text is then drawn.
pub fn verdict_display(path: &str, text: &str) -> Option<String> {
    if path != "materials.cup_wall_check" || verdict_level(path, text) != Some(Level::Bad) {
        return None;
    }
    let number = text.strip_prefix(CUP_WALL_TOO_THIN)?.strip_suffix(" mm")?;
    number.parse::<f64>().ok().filter(|x| x.is_finite())?;
    Some(format!(
        "{}{number} mm",
        crate::gui::plain_text::shown_literal(CUP_WALL_TOO_THIN)
    ))
}

/// A result's value as the screen draws it, before its unit: [`format_value`]'s text, with
/// [`verdict_display`]'s words in place of the cup wall's failing verdict. The results table and
/// the Equation panel draw it (the dashboard, [`DashboardLine::shown_value`]); the exports keep
/// the engine's text.
pub fn shown_text(path: &str, value: &Value) -> String {
    match value {
        Value::Text(text) => verdict_display(path, text).unwrap_or_else(|| text.clone()),
        other => format_value(other),
    }
}

/// The badge level of the check at `path` for `results`: [`verdict_level`] of its text; `None`
/// for a path that is no check or holds no text, a verdict that gives no badge, or a check
/// computed from an invalid pull-out ([`greyed_by_end_effect`]: the dashboard greys its row).
pub fn check_level(results: &DesignResults, path: &str) -> Option<Level> {
    if greyed_by_end_effect(results, path) {
        return None;
    }
    match results.get(path)? {
        Value::Text(text) => verdict_level(path, &text),
        _ => None,
    }
}

/// The checks of `results` that fail (red) or ask for a look (amber), the red first, each in
/// [`CHECKS`] order: what the results table's failing filter shows. While the shaft-clamp
/// section is paused (`clamps_paused`) its checks are left out ([`clamps::hides`]).
pub fn failing_checks(results: &DesignResults, clamps_paused: bool) -> Vec<(&'static str, Level)> {
    let mut failing: Vec<(&'static str, Level)> = CHECKS
        .iter()
        .filter(|&&path| !clamps::hides(path, clamps_paused))
        .filter_map(|&path| match check_level(results, path) {
            Some(level @ (Level::Bad | Level::Caution)) => Some((path, level)),
            _ => None,
        })
        .collect();
    // A stable sort, the most severe first: the red, then the amber, each in CHECKS order.
    failing.sort_by_key(|&(_, level)| std::cmp::Reverse(level));
    failing
}

/// A result's metadata and workbook cell.
#[derive(Clone, Debug)]
pub struct ResultInfo {
    pub meta: &'static ResultMeta,
    pub cell: Option<String>,
}

/// Every result's metadata and cell by path, built once (the result layout does not depend
/// on the inputs: every table has a fixed number of rows).
pub fn result_info(path: &str) -> Option<&'static ResultInfo> {
    static INDEX: OnceLock<HashMap<String, ResultInfo>> = OnceLock::new();
    INDEX
        .get_or_init(|| {
            result_rows(&compute_all(&DesignInputs::default()))
                .into_iter()
                .map(|row| {
                    let info = ResultInfo {
                        meta: row.meta,
                        cell: row.cell,
                    };
                    (row.path, info)
                })
                .collect()
        })
        .get(path)
}

/// What a readout knows about its value beyond the metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResultNotes<'a> {
    /// The corrections the registry ties to its cell.
    pub marks: &'a [Mark],
    /// Greyed: computed from the pull-out while f_end ≤ 0.
    pub greyed: bool,
    /// Reads the workbook's stored 3D fields.
    pub stored_3d: bool,
}

/// The hover text of a displayed result: its on-screen name and help (`gui::plain_text`: the
/// overlay's, else the metadata's), path, workbook cell, then the notes. The text every readout
/// shows (`Readouts::show` adds the equation under it).
///
/// The path, the workbook cell and the corrections' lines only with `show_details` (the
/// header's Show spreadsheet details, spec S2-5).
pub fn result_tooltip(
    path: &str,
    info: &ResultInfo,
    notes: ResultNotes<'_>,
    show_details: bool,
) -> String {
    let mut lines = vec![display_label(path, info.meta.label).to_owned()];
    let help = help_for(path);
    if !help.is_empty() {
        lines.push(help.to_owned());
    }
    if show_details {
        if let Some(original) = crate::gui::plain_text::original_help(path) {
            lines.push(format!("{}: {original}", crate::gui::inputs::ORIGINAL_NOTE));
        }
        lines.push(path.to_owned());
        lines.push(
            info.cell
                .clone()
                .unwrap_or_else(|| crate::gui::plain_text::NO_SPREADSHEET_CELL.to_owned()),
        );
    }
    if notes.greyed {
        lines.push(format!("{END_EFFECT_OUT_OF_RANGE}: {END_EFFECT_GREYED}"));
    }
    if notes.stored_3d {
        lines.push(format!("{STORED_3D_LABEL}: {STORED_3D_NOTE}"));
    }
    if show_details && !notes.marks.is_empty() {
        lines.push(marker_tooltip(notes.marks));
    }
    lines.join("\n")
}

/// The hover text of the result at `path` with its corrections' marks, with the spreadsheet
/// details when `show_details`: the hook's text for a readout outside the dashboard (a
/// results-table row, a geometry callout); `None` for a path that is no result.
pub fn hover_text(path: &str, show_details: bool) -> Option<String> {
    let info = result_info(path)?;
    let marks = CorrectionIndex::get().marks(info.cell.as_deref());
    Some(result_tooltip(
        path,
        info,
        ResultNotes {
            marks,
            ..ResultNotes::default()
        },
        show_details,
    ))
}

/// One dashboard row, ready to draw.
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardLine {
    pub path: &'static str,
    /// The result's catalogue label: the spreadsheet's Summary writes it. The dashboard draws
    /// the path's on-screen name, [`label_for`].
    pub label: &'static str,
    /// The value with its unit.
    pub value: String,
    /// The badge; none when the row has no check or is greyed.
    pub level: Option<Level>,
    pub greyed: bool,
    pub stored_3d: bool,
    /// The corrections' marker text (`E3 E7 E8`), empty for none.
    pub marker: String,
    /// The corrections the registry ties to its cell (the hover text's notes).
    pub marks: &'static [Mark],
}

impl DashboardLine {
    /// The row's hover text (built only while the row is hovered), with the spreadsheet
    /// details when `show_details`.
    pub fn tooltip(&self, show_details: bool) -> String {
        let info = result_info(self.path).expect("a dashboard row is a result");
        result_tooltip(
            self.path,
            info,
            ResultNotes {
                marks: self.marks,
                greyed: self.greyed,
                stored_3d: self.stored_3d,
            },
            show_details,
        )
    }

    /// The value as the dashboard draws it: [`verdict_display`]'s words for the cup wall's
    /// failing verdict, else the value as the engine gives it (the exports keep that).
    pub fn shown_value(&self) -> String {
        verdict_display(self.path, &self.value).unwrap_or_else(|| self.value.clone())
    }
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
pub fn end_effect_out_of_range(results: &DesignResults) -> Option<f64> {
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
}

/// Whether the result at `path` is computed from an invalid pull-out: f_end <= 0 and `path` is
/// one of [`END_EFFECT_ROWS`] (audit M9). The dashboard greys such a row, and a check among them
/// gives no badge anywhere: the dashboard, the results table, its headings and its failing
/// filter all read [`check_level`].
pub fn greyed_by_end_effect(results: &DesignResults, path: &str) -> bool {
    END_EFFECT_ROWS.contains(&path) && end_effect_out_of_range(results).is_some()
}

/// The banner shown over the dashboard and the results table when f_end ≤ 0.
pub fn end_effect_banner(results: &DesignResults) -> Option<String> {
    end_effect_out_of_range(results).map(|f_end| {
        format!(
            "{END_EFFECT_BANNER} (end-effect factor {}): the pull-out and the numbers computed from it are greyed.",
            format_value(&Value::Num(f_end))
        )
    })
}

/// The text of a Why link drawn outside a dashboard row: "Why: " and the note's title, one
/// link (a warning's; the Size tab's and the end-effect banner's, spec 3.7). Not a "Why:" label
/// and the bare title: Start here draws its step's title exactly once, and the Size tab links
/// its first step's note.
pub fn why_note_text(note: &notes::Note) -> String {
    format!("{WHY}: {}", glyph_safe(note.title))
}

/// Draws the Why link to the note `id` ([`why_note_text`]); nothing for a note that has not
/// passed review (the release gate keeps every linked note reviewed). Returns whether it was
/// clicked.
pub(crate) fn why_note_link(ui: &mut egui::Ui, id: &str) -> bool {
    reviewed_note(id).is_some_and(|note| ui.link(why_note_text(note)).clicked())
}

/// The end-effect banner while f_end <= 0 ([`end_effect_banner`]), drawn over the dashboard and
/// over the centre region (decision M42-1), with the Why link to the end-effect check's note
/// under it (spec 3.7, CG-10), which asks for that note. Returns whether the banner shows.
pub(crate) fn end_effect_banner_ui(
    ui: &mut egui::Ui,
    results: &DesignResults,
    readouts: &mut Readouts,
) -> bool {
    let Some(banner) = end_effect_banner(results) else {
        return false;
    };
    ui.colored_label(ui.visuals().error_fg_color, banner);
    if why_note_link(ui, BANNER_NOTE) {
        readouts.open_note(BANNER_NOTE);
    }
    true
}

/// The check whose level badges the dashboard row at `path`; `None` for a row with no badge
/// or a path that is no row.
pub(crate) fn badging_check(path: &str) -> Option<&'static str> {
    DASHBOARD
        .iter()
        .find(|&&(row, _)| row == path)
        .and_then(|&(_, check)| check)
}

/// The dashboard rows of `results`.
pub fn dashboard_lines(results: &DesignResults) -> Vec<DashboardLine> {
    DASHBOARD
        .iter()
        .map(|&(path, badge)| {
            let info = result_info(path).unwrap_or_else(|| panic!("DASHBOARD: no result {path}"));
            let value = results.get(path).unwrap_or(Value::None);
            let greyed = greyed_by_end_effect(results, path);
            let stored_3d = STORED_3D_ROWS.contains(&path);
            // A greyed row has no badge; any other takes its check's level, the level the
            // results table shows on the check's own row.
            let level = match (greyed, badge) {
                (false, Some(check)) => check_level(results, check),
                _ => None,
            };
            let marks = CorrectionIndex::get().marks(info.cell.as_deref());
            DashboardLine {
                path,
                label: info.meta.label,
                value: value_with_unit(&value, format_value(&value), info.meta.unit),
                level,
                greyed,
                stored_3d,
                marker: marker_text(marks),
                marks,
            }
        })
        .collect()
}

/// The dashboard rows drawn while the shaft-clamp section is paused (`clamps_paused`) or not.
pub fn shown_lines(results: &DesignResults, clamps_paused: bool) -> Vec<DashboardLine> {
    dashboard_lines(results)
        .into_iter()
        .filter(|line| !clamps::hides(line.path, clamps_paused))
        .collect()
}

/// Draws one dashboard row: its badge and label, the label quieter than the value
/// ([`look::quiet_text`]), then its value and, with the spreadsheet details, its correction
/// ids, so a narrow side still reads. The row is a readout
/// (`readouts`): its hover text and equation while hovered, a click opens it in the Equation
/// panel. The Explain panel draws a check's rows with it (UX overhaul spec 4.5: "Numbers behind
/// this check").
pub(crate) fn line_ui(ui: &mut egui::Ui, line: &DashboardLine, readouts: &mut Readouts) {
    let show_details = readouts.show_details();
    let weak = ui.visuals().weak_text_color();
    let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
    let title = ui.horizontal(|ui| {
        state_mark(ui, line.level);
        let label = egui::RichText::new(label_for(line.path)).color(look::quiet_text(ui.visuals()));
        ui.add(egui::Label::new(tint(label)).wrap());
    });
    let value = ui.horizontal(|ui| {
        ui.add_space(16.0);
        // One wrapped flow, indented: the correction ids follow the value on its last line
        // or the next. Beside a value wrapped to the full width they would run past the row,
        // and the resizable side would widen every frame.
        ui.horizontal_wrapped(|ui| {
            let value = egui::RichText::new(line.shown_value()).strong();
            ui.add(egui::Label::new(tint(value)).wrap());
            if show_details && !line.marker.is_empty() {
                ui.add(egui::Label::new(egui::RichText::new(&line.marker).small()).extend());
            }
        });
    });
    let rect = title.response.rect.union(value.response.rect);
    readouts.show_over(ui, rect, line.path, |show_details| {
        line.tooltip(show_details)
    });
}

/// The design status line (guidance spec 3.1): whose design it is against the session's
/// `baseline` and how many guided checks fail (those that count with the shaft-clamp section's
/// pause, `clamps_paused`), in the colour of the worst level, over a separator; at the example
/// design and at the starting design its hover says what the design is (spec S2-5; UX overhaul
/// spec 7.1).
pub(crate) fn status_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    baseline: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) {
    let status = design_status(inputs, results, clamps_paused);
    let level = if status.bad > 0 {
        Level::Bad
    } else if status.caution > 0 {
        Level::Caution
    } else {
        Level::Good
    };
    let (text, hover) = status_line(&status, inputs, baseline);
    let text = egui::RichText::new(text)
        .strong()
        .color(level.color(ui.visuals()));
    let line = ui.add(egui::Label::new(text).wrap());
    if let Some(hover) = hover {
        line.on_hover_text(hover);
    }
    ui.separator();
}

/// The lesson link of an Explain entry: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
/// input's row; the rationale is its hover text), or [`NO_NOTE_YET`] with the reason as its
/// hover text.
pub(crate) fn learn_ui(ui: &mut egui::Ui, learn: Learn, readouts: &mut Readouts) {
    match learn {
        Learn::Note(id) => match reviewed_note(id) {
            Some(note) => {
                if ui.link(glyph_safe(note.title)).clicked() {
                    readouts.open_note(note.id);
                }
            }
            None => {
                ui.weak(NO_NOTE_YET).on_hover_text(NOTE_NOT_REVIEWED);
            }
        },
        Learn::Assumption(id) => {
            let assumption = ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .expect("a row's assumption is in the registry (guidance tests)");
            if ui
                .link(assumption.label)
                .on_hover_text(assumption.rationale)
                .clicked()
            {
                readouts.focus_input(assumption.paths[0]);
            }
        }
        Learn::None(reason) => {
            ui.weak(NO_NOTE_YET).on_hover_text(reason);
        }
    }
}

/// The material warnings that fire: the severity's shape ([`state_mark`]) and the text in its
/// colour, then a link
/// to the teaching note (it asks the panel to open the note).
pub(crate) fn warnings_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = warning_lines(results);
    if lines.is_empty() {
        return;
    }
    ui.strong(WARNINGS_HEADING);
    for (rule, text) in lines {
        let level = severity_level(rule.severity);
        ui.horizontal(|ui| {
            state_mark(ui, Some(level));
            ui.add(
                egui::Label::new(egui::RichText::new(text).color(level.color(ui.visuals()))).wrap(),
            );
        });
        if let Some(note) = warning_note(rule) {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                if ui.link(why_note_text(note)).clicked() {
                    readouts.open_note(rule.note_id);
                }
            });
        }
    }
    ui.separator();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::meta::InputSet;
    use crate::gui::corrections::MARKER_HEADING;
    use crate::gui::guidance::{self, CHECK_GUIDES, FIT_CHECK};
    use crate::gui::test_support::{
        SCREEN, hub_overstressed, only_the_clamp_fails, only_the_joint_fails, passing_design,
        reversed_hub_band, short_magnets, sized_frame, text_color, with_poles,
    };
    use std::collections::BTreeSet;

    #[test]
    fn the_dashboard_starts_with_the_headline_in_order() {
        let paths: Vec<&str> = DASHBOARD.iter().map(|(path, _)| *path).collect();
        let headline: Vec<&str> = HEADLINE.iter().map(|(_, path)| *path).collect();
        assert_eq!(paths[..15], headline[..]);
        assert_eq!(
            paths[15..],
            [
                "shaft_joint.check",
                "gearbox.cold_high_check",
                "housing.space_claim_check",
                "model.end_effect_check"
            ]
        );
        for path in END_EFFECT_ROWS.iter().chain(STORED_3D_ROWS.iter()) {
            assert!(paths.contains(path), "{path}");
        }
    }

    /// The defaults with `edit` applied.
    fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    #[test]
    fn every_verdict_each_check_gives_is_classified() {
        use Level::{Bad, Caution, Good};
        let cases: Vec<(&str, DesignInputs, Level)> = vec![
            ("metal.hot_min_check", DesignInputs::default(), Bad),
            (
                "metal.hot_min_check",
                design(|i| i.metal.required_min_Nm = 0.1),
                Good,
            ),
            ("metal.clearance_check", DesignInputs::default(), Bad),
            (
                "metal.clearance_check",
                design(|i| i.metal.face_gap_mm = 5.0),
                Good,
            ),
            ("materials.cup_wall_check", DesignInputs::default(), Bad),
            (
                "materials.cup_wall_check",
                design(|i| i.metal.cup_wall_corner_mm = 6.0),
                Good,
            ),
            (TEMPERATURE_VERDICT, DesignInputs::default(), Good),
            (TEMPERATURE_VERDICT, hot_day(), Caution),
            ("clamps.recommended", DesignInputs::default(), Good),
            (
                "clamps.recommended",
                design(|i| {
                    i.clamps.boss_od_mm = 12.0;
                    i.clamps.clamp_length_mm = 3.0;
                }),
                Bad,
            ),
            ("gearbox.cold_high_check", DesignInputs::default(), Bad),
            ("gearbox.cold_high_check", passing_design(), Good),
            (
                "gearbox.cold_high_check",
                design(|i| i.coupling.gearbox_input_rating_Nm = f64::NAN),
                Caution,
            ),
            ("housing.space_claim_check", DesignInputs::default(), Good),
            (
                "housing.space_claim_check",
                design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)),
                Bad,
            ),
            (
                "housing.space_claim_check",
                design(|i| i.metal.max_diameter_mm = f64::NAN),
                Caution,
            ),
            ("model.end_effect_check", DesignInputs::default(), Good),
            ("model.end_effect_check", short_magnets(), Bad),
        ];
        for (check, inputs, want) in cases {
            let Some(Value::Text(text)) = compute_all(&inputs).get(check) else {
                panic!("{check} is text")
            };
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
        }
        // The gearbox check's texts as its engine writes them: at the rating, above it, an
        // infinite torque, and a torque or a rating that is not a number.
        let gearbox = |torque: f64, rating: f64| {
            let text = crate::engine::gearbox::compute(torque, rating).cold_high_check;
            verdict_level("gearbox.cold_high_check", &text)
        };
        assert_eq!(gearbox(0.6, 0.6), Some(Good));
        assert_eq!(gearbox(0.8049, 0.6), Some(Bad));
        assert_eq!(gearbox(f64::INFINITY, 0.6), Some(Bad));
        assert_eq!(gearbox(f64::NAN, 0.6), Some(Caution));
        assert_eq!(gearbox(0.6, f64::NAN), Some(Caution));
        // BL-064 (spec 7.5): every text form the shaft joints check writes, each reached as the
        // badge by a design from the defaults, and its level: the pass; the five failing causes
        // (the key's two at a 0.3 mm keyway and at a contact height typed 0); the six unknown
        // forms (a value that is not a number, a safety factor typed 0 (ruling C-4), a friction
        // typed below zero (its extension, the whole-branch review's MF-1), a gearbox shaft
        // typed below its slider's bottom (build ruling B-5), no wall around a bore, a reversed
        // band). So no form is classified from a constant alone.
        type Edit = fn(&mut DesignInputs);
        let joint_forms: [(Edit, &str, Level); 12] = [
            (|_| {}, shaft_joint::SHAFT_JOINTS_PASS, Good),
            (
                |i| i.shaft_joint.hub_interference_min_um = 8.0,
                shaft_joint::FIT_SLIPS,
                Bad,
            ),
            (hub_overstressed, shaft_joint::BORE_OVER_ITS_LIMIT, Bad),
            (
                |i| i.shaft_joint.wheel_shaft_fit_C = 20.0,
                shaft_joint::CANNOT_BE_FITTED,
                Bad,
            ),
            (
                |i| i.coupling.keyway_depth_mm = 0.3,
                shaft_joint::KEY_OVER_ITS_LIMIT,
                Bad,
            ),
            (
                |i| i.shaft_joint.key_contact_mm = 0.0,
                shaft_joint::KEY_HAS_NO_BEARING_AREA,
                Bad,
            ),
            (
                |i| i.temperature.adhesive.selected = 0,
                shaft_joint::UNKNOWN_NOT_A_NUMBER,
                Caution,
            ),
            (
                |i| i.shaft_joint.safety_factor = 0.0,
                shaft_joint::UNKNOWN_SAFETY_FACTOR,
                Caution,
            ),
            (
                |i| i.shaft_joint.friction = -0.14,
                shaft_joint::UNKNOWN_IMPOSSIBLE_VALUE,
                Caution,
            ),
            (
                |i| i.shaft_joint.output_shaft_fit_C = -21.0,
                shaft_joint::UNKNOWN_GEARBOX_SHAFT_TOO_COLD,
                Caution,
            ),
            (
                |i| {
                    i.coupling.bore_mm = 16.0;
                    i.metal.boss_od_mm = 12.0;
                    i.shaft_joint.hub_interference_min_um = 15.0;
                },
                shaft_joint::UNKNOWN_NO_WALL,
                Caution,
            ),
            (
                reversed_hub_band,
                shaft_joint::UNKNOWN_BAND_REVERSED,
                Caution,
            ),
        ];
        for (edit, start, want) in joint_forms {
            let results = compute_all(&design(edit));
            let text = &results.shaft_joint.check;
            assert!(text.starts_with(start), "{text:?} starts with {start:?}");
            assert_eq!(
                verdict_level("shaft_joint.check", text),
                Some(want),
                "{text:?}"
            );
            assert_eq!(
                check_level(&results, "shaft_joint.check"),
                Some(want),
                "{text:?}"
            );
        }
        // No back iron (E9): the wall rule does not apply, so the check gives no badge.
        let text = compute_all(&design(|i| i.coupling.backiron = 0))
            .materials
            .cup_wall_check;
        assert_eq!(text, "No back iron");
        assert_eq!(verdict_level("materials.cup_wall_check", &text), None);
        assert_eq!(verdict_level("metal.hot_min_check", "OK"), None);
        assert_eq!(verdict_level("model.pullout_Nm", "OK"), None);
    }

    /// Manual magnets of the ferrite grade Y30 (positive beta: the cold side is checked).
    fn ferrite(inputs: &mut DesignInputs) {
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.coupling.magnets.grade_inner = "Y30".to_owned();
        inputs.coupling.magnets.grade_outer = "Y30".to_owned();
    }

    #[test]
    fn every_verdict_of_every_other_check_is_classified() {
        // Decision O-3: the checks off the dashboard, each from a design that reaches the
        // branch (the dashboard's own are in the test above); the two texts no design reaches
        // are checked as written below.
        use Level::{Bad, Caution, Good};
        let default = DesignInputs::default;
        let cases: Vec<(&str, DesignInputs, Level)> = vec![
            ("calibration.end_effect_check", default(), Good),
            (
                "calibration.end_effect_check",
                design(|i| {
                    i.calibration.c_end = 0.5;
                    i.calibration.magnet_length_mm = 2.0;
                }),
                Bad,
            ),
            ("model.inner_flat_check", default(), Good),
            (
                "model.inner_flat_check",
                design(|i| i.coupling.npole = 40),
                Bad,
            ),
            ("model.outer_flat_check", default(), Good),
            (
                "model.outer_flat_check",
                design(|i| i.coupling.npole = 40),
                Bad,
            ),
            ("model.verdict", default(), Good),
            (
                "model.verdict",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            ("model.cup_ring_check", default(), Bad),
            (
                "model.cup_ring_check",
                design(|i| i.metal.cup_wall_corner_mm = 6.0),
                Good,
            ),
            ("model.hub_check", default(), Good),
            (
                "model.hub_check",
                design(|i| i.coupling.bore_mm = 17.0),
                Bad,
            ),
            ("model.inner_temp_check", default(), Good),
            (
                "model.inner_temp_check",
                design(|i| i.coupling.op_temp_C = 200.0),
                Bad,
            ),
            ("model.outer_temp_check", default(), Good),
            (
                "model.outer_temp_check",
                design(|i| i.coupling.op_temp_C = 200.0),
                Bad,
            ),
            ("temperature.summary.torque_hot_day_note", default(), Good),
            (
                "temperature.summary.torque_hot_day_note",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            (
                "temperature.magnet_life.torque_hot_day_check",
                default(),
                Good,
            ),
            (
                "temperature.magnet_life.torque_hot_day_check",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            ("temperature.demag.cold_check", design(ferrite), Bad),
            (
                "temperature.demag.cold_check",
                design(|i| {
                    ferrite(i);
                    i.metal.min_temp_C = 20.0;
                    i.temperature.demag.h_rev_aligned_kA_m = 10.0;
                    i.temperature.demag.h_rev_pullout_kA_m = 10.0;
                    i.temperature.demag.h_rev_likepole_kA_m = 10.0;
                    i.temperature.demag.h_rev_single_ring_kA_m = 10.0;
                }),
                Good,
            ),
            ("temperature.adhesive.fatigue_screen", default(), Good),
            (
                "temperature.adhesive.fatigue_screen",
                design(|i| i.temperature.adhesive_life.fatigue_endurance = 0.05),
                Caution,
            ),
            ("temperature.mismatch.reading", default(), Good),
            (
                "temperature.mismatch.reading",
                design(|i| i.materials.steel.cte_per_C = 3e-5),
                Bad,
            ),
            (
                "temperature.adhesive_life.hot_fatigue_screen",
                default(),
                Good,
            ),
            (
                "temperature.adhesive_life.hot_fatigue_screen",
                design(|i| i.temperature.adhesive_life.hot_strength_retained = 0.05),
                Caution,
            ),
            ("temperature.adhesive_life.daily_screen", default(), Good),
            (
                "temperature.adhesive_life.daily_screen",
                design(|i| i.temperature.adhesive_life.daily_swing_C = 300.0),
                Caution,
            ),
            // BL-063: the default 0.05 mm entered bondline is above the endurance; 0.2 mm,
            // thicker than the 0.1 mm recommended one, is below it.
            (
                "temperature.adhesive_life.daily_entered_screen",
                default(),
                Caution,
            ),
            (
                "temperature.adhesive_life.daily_entered_screen",
                design(|i| i.metal.bond_inner_mm = 0.2),
                Good,
            ),
            ("clamps.head_check", default(), Good),
            ("clamps.head_check", design(|i| i.clamps.alloy = 2), Caution),
            ("clamps.vent_port", default(), Good),
            (
                "warnings.non_ferromagnetic_back_iron",
                design(|i| i.materials.parts.back_iron = 7),
                Bad,
            ),
            (
                "warnings.cte_mismatch_with_magnets",
                design(|i| i.materials.parts.back_iron = 7),
                Caution,
            ),
            (
                "warnings.low_saturation",
                design(|i| i.materials.parts.back_iron = 3),
                Caution,
            ),
            (
                "warnings.high_conductivity_sleeve_or_liner",
                design(|i| i.temperature.slip_loss.sigma_316_S_m = 5e6),
                Caution,
            ),
            (
                "warnings.uncoated_low_alloy_steel",
                design(|i| i.materials.nickel.thickness_mm = 0.0),
                Caution,
            ),
        ];
        for (check, inputs, want) in cases {
            let results = compute_all(&inputs);
            let Some(Value::Text(text)) = results.get(check) else {
                panic!("{check} is text")
            };
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
            assert_eq!(check_level(&results, check), Some(want), "{check}");
        }
        // The key of an M6 or larger screw misses the vent port: no design of the clamp
        // model's bore reaches it, so its text (clamps.rs) is checked as written.
        assert_eq!(
            verdict_level("clamps.vent_port", "No: key too large"),
            Some(Caution)
        );
        // No sleeve or liner of the material library is ferromagnetic, so no design fires that
        // warning: its text (warnings.rs) is checked as written, red as a warning.
        let ferromagnetic = WARNING_RULES
            .iter()
            .find(|rule| rule.id == "ferromagnetic_sleeve_or_liner")
            .unwrap();
        assert_eq!(
            verdict_level("warnings.ferromagnetic_sleeve_or_liner", ferromagnetic.text),
            Some(Bad)
        );
        // A check that does not apply gives no badge: arcs have no flats, no back iron needs
        // no wall, NdFeB's coercivity rises as it cools, no screw fits (no head, no key), and
        // a warning that does not fire is empty.
        let none = |inputs: DesignInputs, checks: &[&str]| {
            let results = compute_all(&inputs);
            for check in checks {
                let Some(Value::Text(text)) = results.get(check) else {
                    panic!("{check} is text")
                };
                assert_eq!(verdict_level(check, &text), None, "{check}: {text:?}");
                assert_eq!(check_level(&results, check), None, "{check}");
            }
        };
        none(
            design(|i| i.coupling.faceted = 0),
            &["model.inner_flat_check", "model.outer_flat_check"],
        );
        none(
            design(|i| i.coupling.backiron = 0),
            &["model.cup_ring_check", "model.hub_check"],
        );
        none(default(), &["temperature.demag.cold_check"]);
        none(
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            &["clamps.head_check", "clamps.vent_port"],
        );
        let warnings: Vec<&str> = CHECKS
            .iter()
            .copied()
            .filter(|c| c.starts_with("warnings."))
            .collect();
        none(default(), &warnings);
        // A manual magnet without a grade has no rating to check against: amber.
        let manual = compute_all(&design(|i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        }));
        assert_eq!(manual.model.inner_temp_check, "unknown");
        assert_eq!(
            check_level(&manual, "model.inner_temp_check"),
            Some(Caution)
        );
        // No arm reads a text of another check, or of a path that is no check.
        assert_eq!(verdict_level("model.hub_check", "OK"), None);
        assert_eq!(
            verdict_level(
                "housing.space_claim_check",
                crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING
            ),
            None
        );
        assert_eq!(
            verdict_level(
                "gearbox.cold_high_check",
                crate::engine::housing::INSIDE_THE_SPACE_CLAIM
            ),
            None
        );
        assert_eq!(
            verdict_level("gearbox.cold_high_check", shaft_joint::SHAFT_JOINTS_PASS),
            None
        );
        assert_eq!(
            verdict_level(
                "shaft_joint.check",
                crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING
            ),
            None
        );
        assert_eq!(verdict_level("shaft_joint.check", "OK"), None);
        assert_eq!(verdict_level("warnings.no_such_rule", "text"), None);
        assert_eq!(check_level(&manual, "model.pullout_Nm"), None);
    }

    #[test]
    fn every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed() {
        let results = compute_all(&DesignInputs::default());
        let rows = result_rows(&results);
        // In schema order, each a text.
        let mut last = 0;
        for check in CHECKS {
            let index = rows
                .iter()
                .position(|r| r.path == check)
                .unwrap_or_else(|| panic!("CHECKS: no result {check}"));
            assert!(index >= last, "{check} out of schema order");
            last = index;
            assert!(matches!(rows[index].value, Value::Text(_)), "{check}");
        }
        // A new check (a text named check, *_check, *_screen, *verdict or *reading, or a
        // warning) fails here until it is listed and classified. The clamp table's and the
        // sweeps' rows rate candidates, not the design.
        for row in &rows {
            let name = row.path.rsplit('.').next().unwrap();
            let named = name == "check"
                || name.ends_with("_check")
                || name.ends_with("_screen")
                || name.ends_with("verdict")
                || name.ends_with("reading")
                || row.path.starts_with("warnings.");
            let per_row = row.path.contains('[');
            if named && !per_row && matches!(row.value, Value::Text(_)) {
                assert!(
                    CHECKS.contains(&row.path.as_str()),
                    "{} is not in CHECKS",
                    row.path
                );
            }
        }
        // Every warning rule is a check.
        for rule in WARNING_RULES {
            assert!(
                CHECKS.contains(&format!("warnings.{}", rule.id).as_str()),
                "{}",
                rule.id
            );
        }
    }

    #[test]
    fn the_failing_checks_are_the_red_then_the_amber() {
        use Level::{Bad, Caution, Good};
        // The levels order by severity: the worst of several is their maximum.
        assert!(Good < Caution && Caution < Bad);
        assert_eq!([Caution, Bad, Good].into_iter().max(), Some(Bad));
        // The defaults fail six checks: five red, then the amber daily-cycle screen at the
        // entered bondline (BL-063).
        let defaults = compute_all(&DesignInputs::default());
        assert_eq!(
            failing_checks(&defaults, false),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("gearbox.cold_high_check", Bad),
                ("temperature.adhesive_life.daily_entered_screen", Caution),
            ]
        );
        // Manual magnets add two amber rating checks, after the red.
        let manual = compute_all(&design(|i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        }));
        assert_eq!(
            failing_checks(&manual, false),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("gearbox.cold_high_check", Bad),
                ("model.inner_temp_check", Caution),
                ("model.outer_temp_check", Caution),
                ("temperature.adhesive_life.daily_entered_screen", Caution),
            ]
        );
        // Exactly the checks whose level is red or amber, each once.
        for inputs in [DesignInputs::default(), short_magnets(), design(ferrite)] {
            let results = compute_all(&inputs);
            let failing: Vec<&str> = failing_checks(&results, false)
                .iter()
                .map(|(p, _)| *p)
                .collect();
            let want: Vec<&str> = CHECKS
                .iter()
                .copied()
                .filter(|c| matches!(check_level(&results, c), Some(Bad | Caution)))
                .collect();
            let mut sorted = failing.clone();
            sorted.sort_unstable();
            let mut want_sorted = want.clone();
            want_sorted.sort_unstable();
            assert_eq!(sorted, want_sorted);
        }
    }

    /// The dashboard rows whose value moves when `inputs` take any of `values`, from the
    /// defaults at back iron 1 and 0.
    fn dashboard_moved_by(
        inputs: &[String],
        values: impl Fn(&str) -> Vec<f64>,
    ) -> BTreeSet<&'static str> {
        let mut moved = BTreeSet::new();
        for backiron in [1, 0] {
            let base = design(|i| i.coupling.backiron = backiron);
            let before = compute_all(&base);
            for path in inputs {
                for x in values(path) {
                    let mut edited = base.clone();
                    edited.set(path, Value::Num(x)).unwrap();
                    let after = compute_all(&edited);
                    for &(row, _) in &DASHBOARD {
                        if before.get(row) != after.get(row) {
                            moved.insert(row);
                        }
                    }
                }
            }
        }
        moved
    }

    #[test]
    fn the_end_effect_rows_are_the_dashboard_rows_that_move_with_the_end_effect_coefficient() {
        // c_end = 0 (f_end = 1) is the one value that flips the hot-minimum verdict.
        let moved = dashboard_moved_by(&["coupling.c_end".to_owned()], |_| {
            vec![0.0, 0.05, 0.3, 0.5]
        });
        assert_eq!(moved, END_EFFECT_ROWS.into_iter().collect());
        // The headline's rows, and two rows past it: the shaft joints and the gearbox checks,
        // both computed from cold-high.
        let past_the_headline: Vec<&str> = END_EFFECT_ROWS
            .into_iter()
            .filter(|row| !HEADLINE.iter().any(|&(_, path)| path == *row))
            .collect();
        assert_eq!(
            past_the_headline,
            ["shaft_joint.check", "gearbox.cold_high_check"]
        );
    }

    #[test]
    fn the_stored_3d_rows_are_the_dashboard_rows_that_move_with_the_stored_3d_inputs() {
        // Owner R-15: the shaft joints row is one of them, through the inner ring's single-ring
        // onset that caps the bonded inner hub's fitting temperature (it moves with the
        // single-ring reverse field at its slider's top, back iron 1).
        let inputs: Vec<String> = STORED_3D_INPUTS.map(str::to_owned).to_vec();
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        let moved = dashboard_moved_by(&inputs, |path| {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            vec![range.min, range.max]
        });
        assert_eq!(moved, STORED_3D_ROWS.into_iter().collect());
    }

    #[test]
    fn the_stored_3d_label_and_note_say_the_fields_are_fixed_in_plain_words() {
        // Spec 3.8: "3D values from the workbook ... until M3" becomes a plain statement that
        // the values are a named design's and do not follow the design; no live 3D model is
        // promised (section 8, slice-1 trap 6). Amendment A-7: "example design", the name the
        // screen gives the spreadsheet's values. BL-080: the starting design stores the fields
        // of its own geometry, so each text names both designs; with the example design alone
        // it would be false on the design the app opens on.
        for text in [STORED_3D_LABEL, STORED_3D_NOTE, STORED_3D_SENTENCE] {
            assert_eq!(
                guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text}"
            );
            assert!(text.contains("example design"), "{text}");
            assert!(text.contains("starting"), "{text}");
            assert!(!text.contains("live"), "{text}");
        }
        assert_eq!(
            STORED_3D_LABEL,
            "Fixed 3D fields: starting or example design"
        );
        assert_eq!(
            STORED_3D_SENTENCE,
            "Stored 3D value, starting or example design; it does not follow yours."
        );
        assert!(
            STORED_3D_NOTE.contains(
                "computed in 3D for the starting design and for the example design, each for its own geometry."
            ),
            "{STORED_3D_NOTE}"
        );
        // The two headings over the fields' input rows say the same.
        for section in ["calibration.stored_demag", "calibration.stored_slip"] {
            let label = crate::gui::inputs::WORKFLOW
                .iter()
                .flat_map(|group| group.sections)
                .find(|s| s.id == section)
                .unwrap_or_else(|| panic!("no section {section}"))
                .label;
            assert!(
                label.ends_with("from 3D analysis (starting or example design)"),
                "{label}"
            );
        }
        assert!(STORED_3D_NOTE.contains("stay fixed"), "{STORED_3D_NOTE}");
        assert!(
            STORED_3D_SENTENCE.contains("does not follow"),
            "{STORED_3D_SENTENCE}"
        );
    }

    #[test]
    fn the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits() {
        // The reverse fields set the magnet limit, so the governing limit and the hot-day margin;
        // the slip-heating fields set only the slip fault's peak, which the verdict compares with
        // the limits. So the note's sentence that names the slip-heating fields never says they
        // set a limit.
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        let ends = |path: &str| {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            vec![range.min, range.max]
        };
        let fields = |group: &str| -> Vec<String> {
            STORED_3D_INPUTS
                .iter()
                .filter(|path| path.starts_with(group))
                .map(|path| (*path).to_owned())
                .collect()
        };
        assert_eq!(
            dashboard_moved_by(&fields("temperature.slip_loss."), ends),
            BTreeSet::from([TEMPERATURE_VERDICT])
        );
        let by_reverse = dashboard_moved_by(&fields("temperature.demag."), ends);
        for limit in [
            "temperature.summary.governing_limit_C",
            "temperature.summary.margin_hot_day_C",
        ] {
            assert!(by_reverse.contains(limit), "{limit}: {by_reverse:?}");
        }
        assert!(STORED_3D_NOTE.contains("slip-heating"), "{STORED_3D_NOTE}");
        for sentence in STORED_3D_NOTE.split_inclusive(". ") {
            if sentence.contains("slip-heating") {
                assert!(!sentence.contains("limit"), "{sentence}");
            }
        }
        // The shaft joints row's sentence calls the bonded hub's cap by the name the result row
        // and the Problem give it, never a third name (T5 text re-review SF-5), and so does the
        // help of the reverse field that sets it (G-3 re-review of the review leftovers, MF-2).
        assert!(
            STORED_3D_NOTE.contains("The bonded inner hub's allowed fitting temperature, on the shaft joints row, uses one of the reverse fields too.")
                && !STORED_3D_NOTE.contains("fitting limit"),
            "{STORED_3D_NOTE}"
        );
        let single_ring = help_for("temperature.demag.h_rev_single_ring_kA_m");
        assert!(!single_ring.contains("fitting limit"), "{single_ring}");
    }

    #[test]
    fn every_stored_3d_input_says_whose_it_is_and_stays_fixed() {
        // The help on screen of every stored 3D input ends with one sentence, whether the
        // overlay or the engine words it.
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        for path in STORED_3D_INPUTS {
            assert!(catalogue.entry(path).is_some(), "{path}: not an input");
            let help = help_for(path);
            assert!(help.ends_with(STORED_3D_SENTENCE), "{path}: {help:?}");
            assert_eq!(
                guidance::plain_text_violations(help),
                Vec::<&str>::new(),
                "{path}: {help:?}"
            );
        }
    }

    #[test]
    fn the_default_dashboard_has_badges_markers_and_the_3d_rows() {
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
        assert_eq!(lines.len(), DASHBOARD.len());
        let line = |path: &str| lines.iter().find(|l| l.path == path).unwrap();
        assert_eq!(line("model.pullout_Nm").value, "2.688 N·m");
        assert_eq!(line("model.pullout_Nm").level, None);
        assert_eq!(line("model.pullout_Nm").marker, "E3 E7 E8");
        // With the spreadsheet details the hover says what differs from the original
        // spreadsheet (corrections.rs tests the values); without them it does not.
        let pullout = line("model.pullout_Nm");
        assert!(
            pullout
                .tooltip(true)
                .contains(&format!("{MARKER_HEADING}:"))
        );
        assert!(!pullout.tooltip(false).contains(MARKER_HEADING));
        assert_eq!(line("metal.torque_hot_low_Nm").level, Some(Level::Bad));
        assert_eq!(line("materials.cup_wall_check").level, Some(Level::Bad));
        assert_eq!(line(TEMPERATURE_VERDICT).level, Some(Level::Good));
        assert_eq!(
            line("housing.space_claim_check").value,
            "Inside the space claim"
        );
        assert_eq!(line("housing.space_claim_check").level, Some(Level::Good));
        assert_eq!(
            line("housing.space_claim_check").marker,
            "",
            "Rust-only: no cell"
        );
        assert_eq!(line("model.end_effect_check").level, Some(Level::Good));
        // BL-064: the shaft joints pass at the example design, Rust-only.
        let joint = line("shaft_joint.check");
        assert_eq!(joint.value, crate::engine::shaft_joint::SHAFT_JOINTS_PASS);
        assert_eq!(joint.level, Some(Level::Good));
        assert_eq!(joint.marker, "", "Rust-only: no cell");
        assert!(lines.iter().all(|l| !l.greyed));
        for path in STORED_3D_ROWS {
            assert!(line(path).stored_3d);
            assert!(line(path).tooltip(false).contains(STORED_3D_LABEL));
        }
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 4);
        // The stored-3D label is drawn after the last stored-3D row shown: the shaft joints row,
        // whether the clamp section shows or not (owner R-15).
        for paused in [true, false] {
            let shown = shown_lines(&compute_all(&DesignInputs::default()), paused);
            let last = shown.iter().rev().find(|l| l.stored_3d).map(|l| l.path);
            assert_eq!(last, Some("shaft_joint.check"), "paused {paused}");
        }
    }

    #[test]
    fn out_of_range_end_effect_greys_the_pull_out_rows_without_badges() {
        let results = compute_all(&short_magnets());
        assert!(end_effect_out_of_range(&results).is_some_and(|f| f < 0.0));
        let lines = dashboard_lines(&results);
        for line in &lines {
            let derived = END_EFFECT_ROWS.contains(&line.path);
            assert_eq!(line.greyed, derived, "{}", line.path);
            if derived {
                assert_eq!(line.level, None, "{}", line.path);
                assert!(
                    line.tooltip(false).contains(END_EFFECT_OUT_OF_RANGE),
                    "{}",
                    line.path
                );
            }
        }
        let flag = lines
            .iter()
            .find(|l| l.path == "model.end_effect_check")
            .unwrap();
        assert_eq!(flag.level, Some(Level::Bad));
        assert_eq!(
            end_effect_out_of_range(&compute_all(&DesignInputs::default())),
            None
        );
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter() {
        // The twin of the test above for the results table's badges and its failing filter
        // (both read `check_level`): a verdict computed from an invalid pull-out is no verdict,
        // so the four checks among the greyed rows give no badge and are not listed as failing.
        use Level::{Bad, Caution};
        let results = compute_all(&short_magnets());
        assert!(end_effect_out_of_range(&results).is_some());
        let greyed: Vec<&str> = END_EFFECT_ROWS
            .into_iter()
            .filter(|path| CHECKS.contains(path))
            .collect();
        assert_eq!(
            greyed,
            [
                "metal.hot_min_check",
                "clamps.recommended",
                "shaft_joint.check",
                "gearbox.cold_high_check"
            ]
        );
        for path in &greyed {
            assert!(greyed_by_end_effect(&results, path), "{path}");
            // The verdict has a level of its own (red, then green three times here): the rule
            // drops it.
            let Some(Value::Text(text)) = results.get(path) else {
                panic!("{path} is text")
            };
            assert!(verdict_level(path, &text).is_some(), "{path}: {text:?}");
            assert_eq!(check_level(&results, path), None, "{path}");
        }
        assert_eq!(
            failing_checks(&results, false),
            [
                ("model.end_effect_check", Bad),
                ("model.verdict", Bad),
                ("model.cup_ring_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("temperature.summary.torque_hot_day_note", Bad),
                ("temperature.magnet_life.torque_hot_day_check", Bad),
                ("model.inner_temp_check", Caution),
                ("model.outer_temp_check", Caution),
                ("temperature.adhesive.fatigue_screen", Caution),
                ("temperature.adhesive_life.hot_fatigue_screen", Caution),
            ]
        );
        // On every dashboard row with a check, the check's level is the row's badge, in range
        // and out of it: what the table shows on a check's row is what the dashboard shows.
        for inputs in [DesignInputs::default(), short_magnets()] {
            let results = compute_all(&inputs);
            for (line, &(path, check)) in dashboard_lines(&results).iter().zip(&DASHBOARD) {
                assert_eq!(line.path, path);
                if let Some(check) = check {
                    assert_eq!(check_level(&results, check), line.level, "{path}");
                }
            }
        }
        // In range nothing is dropped: the defaults' hot minimum is red, the screw green, the
        // shaft joints green, the gearbox check red.
        let defaults = compute_all(&DesignInputs::default());
        assert!(!greyed_by_end_effect(&defaults, "metal.hot_min_check"));
        assert_eq!(check_level(&defaults, "metal.hot_min_check"), Some(Bad));
        assert_eq!(
            check_level(&defaults, "clamps.recommended"),
            Some(Level::Good)
        );
        assert!(!greyed_by_end_effect(&defaults, "shaft_joint.check"));
        assert_eq!(
            check_level(&defaults, "shaft_joint.check"),
            Some(Level::Good)
        );
        assert!(!greyed_by_end_effect(&defaults, "gearbox.cold_high_check"));
        assert_eq!(check_level(&defaults, "gearbox.cold_high_check"), Some(Bad));
        // A row outside END_EFFECT_ROWS is never dropped, out of range or not.
        assert!(!greyed_by_end_effect(&results, "metal.clearance_check"));
    }

    #[test]
    fn out_of_range_end_effect_greys_the_gearbox_check_whatever_its_text_reads() {
        // Cold-high comes from the pull-out, so while the end-effect model is out of range it is
        // no torque: on short magnets it is below zero and the engine's text reads within the
        // rating, a pass from a meaningless number; a rating below it (set on the struct) turns
        // the text red. Either way the row is greyed without a badge, the check has no level in
        // the dashboard, the results table or its failing filter, its guide shows no Problem,
        // and the status line counts it neither way.
        use crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING};
        use crate::gui::results_table::{entry_index, entry_level, table_entries};
        let gearbox = "gearbox.cold_high_check";
        let guide = guidance::guide(gearbox).expect("the gearbox check's guide");
        let entry = &table_entries()[entry_index(gearbox).expect("the gearbox check's row")];
        let mut below = short_magnets();
        below.coupling.gearbox_input_rating_Nm = -10.0;
        let mut statuses = Vec::new();
        for (inputs, text) in [
            (short_magnets(), WITHIN_THE_GEARBOX_RATING),
            (below, ABOVE_THE_GEARBOX_RATING),
        ] {
            let results = compute_all(&inputs);
            assert!(end_effect_out_of_range(&results).is_some());
            let torque = results.metal.cold_high_input_Nm;
            assert!(torque < 0.0, "{torque}");
            let verdict = &results.gearbox.cold_high_check;
            assert!(verdict.starts_with(text), "{verdict}");
            assert!(verdict_level(gearbox, verdict).is_some(), "{verdict}");
            assert!(greyed_by_end_effect(&results, gearbox));
            assert_eq!(check_level(&results, gearbox), None);
            let line = dashboard_lines(&results)
                .into_iter()
                .find(|l| l.path == gearbox)
                .expect("the gearbox row");
            assert!(line.greyed);
            assert_eq!(line.level, None);
            assert_eq!(entry_level(&results, entry), None);
            assert!(
                failing_checks(&results, false)
                    .iter()
                    .all(|&(path, _)| path != gearbox)
            );
            assert_eq!(guidance::level(guide, &inputs, &results), Level::Good);
            assert_eq!((guide.problem)(&inputs, &results), None);
            statuses.push(design_status(&inputs, &results, false));
        }
        assert_eq!(statuses[0], statuses[1]);
    }

    #[test]
    fn out_of_range_end_effect_greys_the_shaft_joint_check_whatever_its_text_reads() {
        // BL-064 (spec 5.1, 7.6). The check reads cold-high, which comes from the pull-out: on
        // short magnets it is below zero, so the engine's text reads a pass from a meaningless
        // number. The whole row greys, so a bore stress above its limit, which does not read
        // cold-high, is hidden there too (review 2, SF-13: one row cannot grey in part; the
        // end-effect check fails there under its banner). Either way the row is greyed without
        // a badge, keeps its stored-3D note beside the greyed one, has no level in the
        // dashboard, the results table or its failing filter, its guide shows no Problem, and
        // the status line counts it neither way.
        use crate::engine::shaft_joint::{BORE_OVER_ITS_LIMIT, SHAFT_JOINTS_PASS};
        use crate::gui::results_table::{entry_index, entry_level, table_entries};
        let joint = "shaft_joint.check";
        let guide = guidance::guide(joint).expect("the shaft joints check's guide");
        let entry = &table_entries()[entry_index(joint).expect("the shaft joints check's row")];
        let mut overstressed = short_magnets();
        hub_overstressed(&mut overstressed);
        let mut statuses = Vec::new();
        for (inputs, text) in [
            (short_magnets(), SHAFT_JOINTS_PASS),
            (overstressed, BORE_OVER_ITS_LIMIT),
        ] {
            let results = compute_all(&inputs);
            assert!(end_effect_out_of_range(&results).is_some());
            assert!(results.metal.torque_cold_high_Nm < 0.0);
            let verdict = &results.shaft_joint.check;
            assert!(verdict.starts_with(text), "{verdict}");
            assert!(verdict_level(joint, verdict).is_some(), "{verdict}");
            assert!(greyed_by_end_effect(&results, joint));
            assert_eq!(check_level(&results, joint), None);
            let line = dashboard_lines(&results)
                .into_iter()
                .find(|l| l.path == joint)
                .expect("the shaft joints row");
            assert!(line.greyed && line.stored_3d);
            assert_eq!(line.level, None);
            let hover = line.tooltip(false);
            assert!(hover.contains(END_EFFECT_OUT_OF_RANGE), "{hover}");
            assert!(hover.contains(STORED_3D_LABEL), "{hover}");
            assert_eq!(entry_level(&results, entry), None);
            assert!(
                failing_checks(&results, false)
                    .iter()
                    .all(|&(path, _)| path != joint)
            );
            assert_eq!(guidance::level(guide, &inputs, &results), Level::Good);
            assert_eq!((guide.problem)(&inputs, &results), None);
            statuses.push(design_status(&inputs, &results, false));
        }
        assert_eq!(statuses[0], statuses[1]);
    }

    #[test]
    fn a_design_past_the_space_claim_shows_the_overshoot_in_red() {
        let results = compute_all(&design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        let lines = dashboard_lines(&results);
        let claim = lines
            .iter()
            .find(|l| l.path == "housing.space_claim_check")
            .unwrap();
        assert!(
            claim
                .value
                .starts_with("Exceeds the space claim: overall length "),
            "{}",
            claim.value
        );
        assert_eq!(claim.level, Some(Level::Bad));
    }

    #[test]
    fn the_result_tooltip_names_the_path_cell_and_notes() {
        let info = result_info("model.pullout_Nm").unwrap();
        // With the spreadsheet details (spec S2-5).
        let tooltip = result_tooltip("model.pullout_Nm", info, ResultNotes::default(), true);
        // The on-screen name and the help first (gui::plain_text).
        let first = format!(
            "{}\n{}\n",
            label_for("model.pullout_Nm"),
            help_for("model.pullout_Nm")
        );
        assert!(!help_for("model.pullout_Nm").is_empty());
        assert!(tooltip.starts_with(&first), "{tooltip}");
        assert!(
            tooltip.contains("\nmodel.pullout_Nm\nCalculator!C93"),
            "{tooltip}"
        );
        let rust_only = result_info("housing.space_claim_check").unwrap();
        assert!(
            result_tooltip(
                "housing.space_claim_check",
                rust_only,
                ResultNotes::default(),
                true
            )
            .contains(crate::gui::plain_text::NO_SPREADSHEET_CELL)
        );
        assert!(result_info("no.such").is_none());
        assert_eq!(
            result_info("gap_sweep[0].f_end").unwrap().cell.as_deref(),
            Some("Gap sweep!W6")
        );
    }

    #[test]
    fn the_hover_text_carries_the_marks_and_needs_a_result() {
        let text = hover_text("model.pullout_Nm", true).unwrap();
        assert!(
            text.starts_with(&format!("{}\n", label_for("model.pullout_Nm"))),
            "{text}"
        );
        assert!(text.contains(&format!("{MARKER_HEADING}:")), "{text}");
        assert!(
            hover_text("housing.length_overshoot_mm", true)
                .unwrap()
                .contains(crate::gui::plain_text::NO_SPREADSHEET_CELL)
        );
        assert_eq!(hover_text("no.such", true), None);
        assert_eq!(hover_text("no.such", false), None);
    }

    #[test]
    fn every_result_s_hover_names_its_path_cell_and_corrections_only_with_the_details() {
        // Spec S2-5: the path, the original spreadsheet cell (or the line that says there is
        // none) and the corrections that change the value show with Show spreadsheet details;
        // without them the hover is the same text with exactly those lines left out.
        for entry in crate::gui::results_table::table_entries() {
            let path = entry.path.as_str();
            let on = hover_text(path, true).expect("every table row is a result");
            let off = hover_text(path, false).expect("every table row is a result");
            let marks = CorrectionIndex::get().marks(entry.info.cell.as_deref());
            let marker_lines = if marks.is_empty() {
                0
            } else {
                marker_tooltip(marks).lines().count()
            };
            // Where the overlay replaced the engine's help, the original note is one more line.
            let original_lines = usize::from(crate::gui::plain_text::original_help(path).is_some());
            assert_eq!(
                on.lines().count(),
                off.lines().count() + 2 + marker_lines + original_lines,
                "{path}: {on:?} against {off:?}"
            );
            assert!(on.lines().any(|line| line == path), "{path}: {on:?}");
            assert!(!off.lines().any(|line| line == path), "{path}: {off:?}");
            if let Some(cell) = entry.info.cell.as_deref() {
                assert!(on.lines().any(|line| line == cell), "{path}: {on:?}");
                assert!(!off.lines().any(|line| line == cell), "{path}: {off:?}");
            }
            assert_eq!(on.contains(MARKER_HEADING), !marks.is_empty(), "{path}");
            assert!(!off.contains(MARKER_HEADING), "{path}");
            assert!(
                off.lines().all(|line| on.lines().any(|o| o == line)),
                "{path}: {off:?} is not part of {on:?}"
            );
        }
    }

    #[test]
    fn each_row_is_exported_under_its_catalogue_label_and_four_have_a_display_label() {
        // The Explain panel draws a check's rows under their display labels
        // (explain.rs: the_numbers_behind_a_check_are_the_dashboard_rows_its_verdict_badges).
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
        for line in &lines {
            // The line keeps the catalogue label: the spreadsheet's Summary writes it.
            assert_eq!(line.label, result_info(line.path).unwrap().meta.label);
        }
        // The overlay renames four rows (spec 3.5's three and BL-059's ripple row).
        let renamed: Vec<&str> = lines
            .iter()
            .filter(|line| label_for(line.path) != line.label)
            .map(|line| line.path)
            .collect();
        assert_eq!(
            renamed,
            [
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "model.gearbox_input_ripple_Nm"
            ]
        );
    }

    #[test]
    fn every_hover_text_with_the_details_off_is_in_plain_words() {
        // Spec 3.4: every drawn tooltip with Show spreadsheet details off passes the guard:
        // every input's, every result's and every dashboard row's (in range and with the end
        // effect out of range).
        let mut texts: Vec<String> = crate::gui::inputs::InputCatalogue::get()
            .all()
            .map(|entry| crate::gui::inputs::input_tooltip(entry, &entry.default, false))
            .collect();
        for entry in crate::gui::results_table::table_entries() {
            texts.push(hover_text(&entry.path, false).expect("every table row is a result"));
        }
        for inputs in [DesignInputs::default(), short_magnets()] {
            texts.extend(
                dashboard_lines(&compute_all(&inputs))
                    .iter()
                    .map(|line| line.tooltip(false)),
            );
        }
        for text in &texts {
            assert_eq!(
                guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn the_warnings_that_fire_are_listed_with_their_reviewed_notes() {
        assert!(warning_lines(&compute_all(&DesignInputs::default())).is_empty());
        // 304 stainless back iron: an open circuit, and its expansion against the magnets'.
        let results = compute_all(&design(|i| i.materials.parts.back_iron = 7));
        let ids: Vec<&str> = warning_lines(&results)
            .iter()
            .map(|(rule, _)| rule.id)
            .collect();
        assert_eq!(
            ids,
            ["non_ferromagnetic_back_iron", "cte_mismatch_with_magnets"]
        );
        let (rule, text) = &warning_lines(&results)[0];
        assert_eq!(text, rule.text);
        assert_eq!(severity_level(rule.severity), Level::Bad);
        assert_eq!(severity_level(Severity::Caution), Level::Caution);
        // Every rule's note has passed the accuracy gate, so every warning links to it.
        for rule in &WARNING_RULES {
            assert_eq!(warning_note(rule).map(|n| n.id), Some(rule.note_id));
        }
    }

    #[test]
    fn a_passing_check_is_the_ok_token_in_a_dark_theme() {
        // UX overhaul spec 6.3: the panel's own green takes the token's value (#4cc38a). A
        // light theme's green is not a token and stays.
        assert_eq!(
            Level::Good.color(&egui::Visuals::dark()),
            crate::gui::look::OK
        );
        assert_eq!(
            Level::Good.color(&egui::Visuals::light()),
            egui::Color32::from_rgb(20, 130, 60)
        );
    }

    #[test]
    fn a_row_s_label_is_quieter_than_its_value_in_a_theme_and_in_egui_s_own_visuals() {
        // UX overhaul spec 6.1: labels and captions are quieter than values
        // (`look::quiet_text`). The label and the value of the pull-out's row, as the Explain
        // panel draws it under "Numbers behind this check", at the example design.
        let results = compute_all(&DesignInputs::default());
        let lines = shown_lines(&results, false);
        let line = lines
            .iter()
            .find(|line| line.path == "model.pullout_Nm")
            .expect("the pull-out has a row");
        let colours = |ctx: &egui::Context| {
            let output = sized_frame(ctx, SCREEN, Vec::new(), |ui| {
                line_ui(ui, line, &mut Readouts::default());
            });
            [label_for("model.pullout_Nm"), "2.688 N·m"].map(|text| {
                text_color(&output, text).unwrap_or_else(|| panic!("no {text:?} drawn"))
            })
        };
        // A theme that sets a weak colour (the standalone app's): the label takes it, the value
        // keeps the strong colour.
        let themed = egui::Context::default();
        let weak = egui::Color32::from_rgb(1, 2, 3);
        themed.style_mut(|style| style.visuals.weak_text_color = Some(weak));
        let strong = themed.style().visuals.strong_text_color();
        assert_eq!(colours(&themed), [weak, strong]);
        // egui's own visuals set none: their weak colour is too faint for a name (the text
        // colour at 60 %), and their text colour is already a step under the strong colour.
        let plain = egui::Context::default();
        let visuals = plain.style().visuals.clone();
        assert_eq!(visuals.weak_text_color, None);
        assert_ne!(visuals.text_color(), visuals.strong_text_color());
        assert_eq!(
            colours(&plain),
            [visuals.text_color(), visuals.strong_text_color()]
        );
    }

    #[test]
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
        assert_eq!(Level::Bad.color(&dark), dark.error_fg_color);
        assert_eq!(Level::Caution.color(&light), light.warn_fg_color);
        assert_ne!(Level::Good.color(&dark), Level::Good.color(&light));
        assert_eq!(Level::Good.word(), "OK");
    }

    // ------------------------------------------------------------------ the check guides

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(crate::gui::test_support::hot_day)
    }

    /// The defaults at 40 poles: the blocks no longer fit around the ring.
    fn too_many_poles() -> DesignInputs {
        with_poles(40)
    }

    #[test]
    fn while_the_clamp_section_is_paused_its_checks_leave_the_failing_list() {
        let results = compute_all(&only_the_clamp_fails());
        let shown = failing_checks(&results, false);
        assert!(
            shown.contains(&("clamps.recommended", Level::Bad)),
            "{shown:?}"
        );
        let want: Vec<(&str, Level)> = shown
            .iter()
            .copied()
            .filter(|&(path, _)| !clamps::hides(path, true))
            .collect();
        assert_eq!(failing_checks(&results, true), want);
        assert!(want.len() < shown.len());
    }

    #[test]
    fn every_guide_has_its_own_row_but_the_fit_row_which_is_drawn_apart() {
        // A guide draws under the row of its own check (the rows its verdict only badges have
        // none); the fit check is no result, so its row is drawn after the table's.
        for guide in &CHECK_GUIDES {
            let own_row = DASHBOARD.contains(&(guide.check, Some(guide.check)));
            assert_eq!(own_row, guide.check != FIT_CHECK, "{}", guide.check);
        }
        assert!(result_info(FIT_CHECK).is_none());
        let checks: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.check).collect();
        assert_eq!(checks.len(), CHECK_GUIDES.len(), "one guide per check");
    }

    #[test]
    fn a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it() {
        // A block shows exactly under a red or amber badge: a check greyed by f_end <= 0 (the
        // hot minimum, the screw) and a cup without back iron have no badge, and no block.
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            too_many_poles(),
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
            design(|i| i.coupling.backiron = 0),
            only_the_joint_fails(),
            design(reversed_hub_band),
        ] {
            let results = compute_all(&inputs);
            for line in dashboard_lines(&results) {
                if let Some(guide) = guidance::guide(line.path) {
                    assert_eq!(
                        guidance::level(guide, &inputs, &results),
                        line.level.unwrap_or(Level::Good),
                        "{}",
                        line.path
                    );
                }
            }
        }
    }

    #[test]
    fn the_production_variation_assumption_is_named_as_its_input() {
        // Map-jargon T-15: the Assumptions view's heading and the input row name the allowance
        // alike (spec 3.5).
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        assert_eq!(assumption.paths, ["metal.variation"]);
        assert_eq!(assumption.label, label_for("metal.variation"));
    }

    #[test]
    fn the_cup_wall_verdict_is_shown_with_the_wall_input_s_label() {
        use crate::engine::compat::fmt_fixed;
        use crate::engine::meta::NumOrText;
        // Ruling R9: the default design's cup wall fails; the screen names the wall by its
        // input's label, with the engine's own number.
        let results = compute_all(&DesignInputs::default());
        let engine = &results.materials.cup_wall_check;
        let NumOrText::Num(wall) = results.materials.cup_wall_suggested_mm else {
            panic!("a steel circuit has a wall rule")
        };
        let want = format!(
            "Too thin: raise {} to at least {} mm",
            label_for("metal.cup_wall_corner_mm"),
            fmt_fixed(wall, 1)
        );
        assert_eq!(
            verdict_display("materials.cup_wall_check", engine).as_deref(),
            Some(want.as_str())
        );
        assert_eq!(
            guidance::plain_text_violations(&want),
            Vec::<&str>::new(),
            "{want}"
        );
        // The engine text is unchanged: the badge reads it, the line keeps it, and the
        // exports, which read the results, write it.
        assert!(engine.starts_with(CUP_WALL_TOO_THIN), "{engine}");
        assert_eq!(
            verdict_level("materials.cup_wall_check", engine),
            Some(Level::Bad)
        );
        let line = |results: &DesignResults| {
            dashboard_lines(results)
                .into_iter()
                .find(|l| l.path == "materials.cup_wall_check")
                .expect("the cup wall row")
        };
        assert_eq!(&line(&results).value, engine);
        assert_eq!(line(&results).shown_value(), want);
        assert_eq!(
            shown_text("materials.cup_wall_check", &Value::Text(engine.clone())),
            want
        );
        // Every other row of the default design draws its value as the line holds it.
        for other in dashboard_lines(&results)
            .iter()
            .filter(|l| l.path != "materials.cup_wall_check")
        {
            assert_eq!(other.shown_value(), other.value, "{}", other.path);
        }
        // A passing wall and a design without back iron draw the engine text.
        for inputs in [passing_design(), design(|i| i.coupling.backiron = 0)] {
            let results = compute_all(&inputs);
            let row = line(&results);
            assert_eq!(row.shown_value(), row.value, "{}", row.value);
        }
        // Not that verdict, or its number not a finite number: no display text (the engine text
        // is drawn), never a panic.
        let nan = format!("{CUP_WALL_TOO_THIN}nan mm");
        let inches = format!("{CUP_WALL_TOO_THIN}2.0 inches");
        let words = format!("{CUP_WALL_TOO_THIN}two mm");
        for (path, text) in [
            ("materials.cup_wall_check", "OK"),
            ("materials.cup_wall_check", "No back iron"),
            ("materials.cup_wall_check", nan.as_str()),
            ("materials.cup_wall_check", inches.as_str()),
            ("materials.cup_wall_check", words.as_str()),
            ("metal.hot_min_check", engine.as_str()),
        ] {
            assert_eq!(verdict_display(path, text), None, "{path}: {text}");
            assert_eq!(
                shown_text(path, &Value::Text(text.to_owned())),
                text,
                "{path}"
            );
        }
        // A value that is no text reads as the readouts format it.
        for value in [Value::Num(2.5), Value::Int(3), Value::None] {
            assert_eq!(
                shown_text("materials.cup_wall_check", &value),
                format_value(&value)
            );
        }
    }

    #[test]
    fn a_why_link_outside_the_rows_is_one_link_naming_its_note() {
        // Spec 3.7 and F11: "Why: <title>" as one link, never the bare title, which Start here
        // draws exactly once while its step is open.
        for note in notes::NOTES {
            let text = why_note_text(note);
            assert_eq!(
                text,
                format!("{WHY}: {}", glyph_safe(note.title)),
                "{}",
                note.id
            );
            assert_ne!(text, glyph_safe(note.title), "{}", note.id);
        }
    }
}
