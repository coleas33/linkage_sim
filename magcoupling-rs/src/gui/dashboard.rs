//! The dashboard on the right (spec M4 "Layout": "headline numbers with green/amber/red
//! badges derived from check verdicts; corrected values carry a 'corrected vs workbook' marker
//! with the deviation in its tooltip").
//!
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the gearbox
//! check (BL-059), the A1 space claim badge and the audit M9 end-effect flag. A row's badge
//! comes from a check verdict ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
//!
//! - **End effect (audit M9).** When f_end ≤ 0 the pull-out and the numbers computed from it
//!   ([`END_EFFECT_ROWS`]) are greyed, without a badge, under the banner "End-effect model out
//!   of range", which carries a "Why: <note>" link to the end-effect check's note (slice-2 spec
//!   3.7; `end_effect_banner_ui` draws it here and over the centre region).
//! - **Stored 3D values.** The temperature rows that read the stored 3D fields
//!   ([`STORED_3D_ROWS`]) carry the label [`STORED_3D_LABEL`]: the fields
//!   ([`STORED_3D_INPUTS`]) are 3D results for the example design and stay fixed when the
//!   design changes; no live 3D model runs.
//!
//! Over the rows, the material warnings that fire (spec Addendum A5: "plain language,
//! colour-coded, linked to their teaching note"): each rule's text in its severity's colour (a
//! warning red, a caution amber) with a link that opens its reviewed note in the Equation panel
//! ([`warning_lines`], decision M43-8).
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). The dashboard builds a row's text only while it is hovered
//! ([`DashboardLine::tooltip`]).
//!
//! Check guides (guidance spec 3.1 and 3.4, [`crate::gui::guidance`]): the design status line
//! tops the dashboard, and the Magnets fit row follows the end-effect flag (a GUI check of
//! `model::blocks_fit`, so it is drawn apart from [`DASHBOARD`], whose paths are results). Under
//! a guided check's own row, a red level opens its block: the problem from this design's
//! results, the guide's sentence and its fixes (each a link that shows its input's row, the
//! Outside diameter's the Size tab), or a line saying no listed change helps, then any
//! trade-off and side-effect notes. An amber level keeps the block shut under a one-line header, so the
//! dashboard stays short. Every row, at any level, has its Why line: a link to its reviewed
//! note (opened in the Equation panel), a link to its assumption (its row shown, the rationale
//! in the hover text) or "no note yet" ([`crate::gui::guidance::row_learn`]).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::assumptions::ASSUMPTIONS;
use crate::engine::clamps;
use crate::engine::explain::notes;
use crate::engine::materials::CUP_WALL_TOO_THIN;
use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, with_unit};
use crate::gui::guidance::{
    self, BANNER_NOTE, CHECK_GUIDES, CheckGuide, Direction, FIT_CHECK, Learn, NO_FIX,
    design_status, design_status_text, fix_text, guide_notes, row_learn,
};
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

    /// The badge colour: green, the theme's warning colour, the theme's error colour.
    pub fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        match self {
            Level::Good if visuals.dark_mode => egui::Color32::from_rgb(90, 200, 120),
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

/// The head of a red or opened amber check's problem line (guidance spec 3.4).
pub const PROBLEM: &str = "Problem:";

/// The head of a red or opened amber check's fix line: each fix a link that shows its input.
pub const FIX: &str = "Fix:";

/// The head of every row's Why line: under an open block the guide's sentence, then the link
/// to read more.
pub const WHY_LINK: &str = "Why:";

/// What a Why line says when no reviewed note or assumption explains the row.
pub const NO_NOTE_YET: &str = "no note yet";

/// The hover text of a Why line whose note is not reviewed (sent back to Draft): it never shows.
pub const NOTE_NOT_REVIEWED: &str = "Its teaching note has not passed review.";

/// The one-line header an amber check's block is shut under: a click opens it.
pub const CAUTION_DETAILS: &str = "Problem and fixes";

/// The Magnets fit row's value: the blocks fit around the ring, or not.
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
pub const DASHBOARD: [(&str, Option<&str>); 18] = [
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
    ("gearbox.cold_high_check", Some("gearbox.cold_high_check")),
    (
        "housing.space_claim_check",
        Some("housing.space_claim_check"),
    ),
    ("model.end_effect_check", Some("model.end_effect_check")),
];

/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the rows
/// that move with the end-effect coefficient (a test probes them at back iron 1 and 0), the
/// headline's and the gearbox check, computed from cold-high. Per-row greying of the results
/// table waits for plan A-3's dependency graph; the table shows the banner instead (decision
/// M41-11), and its three checks among these rows (the hot minimum, the recommended screw, the
/// gearbox check) give no badge there either ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 8] = [
    "model.pullout_Nm",
    "model.pullout_20C_Nm",
    "metal.torque_hot_low_Nm",
    "metal.hot_min_check",
    "metal.torque_cold_high_Nm",
    "model.gearbox_input_ripple_Nm",
    "clamps.recommended",
    "gearbox.cold_high_check",
];

/// The dashboard rows that read the workbook's stored 3D fields (the reverse fields of the
/// demagnetization check and the slip-loss fields): the headline rows that move with them (a
/// test probes each at its slider ends).
pub const STORED_3D_ROWS: [&str; 3] = [
    "temperature.summary.governing_limit_C",
    "temperature.summary.margin_hot_day_C",
    TEMPERATURE_VERDICT,
];

/// The label of the stored-3D rows.
pub const STORED_3D_LABEL: &str = "Fixed 3D fields from the example design";

/// Its hover text.
pub const STORED_3D_NOTE: &str = "The temperature checks use the demagnetization reverse fields and the slip-heating fields, computed in 3D for the example design. Those fields stay fixed when you change the geometry, magnet count or grade; for another design, enter its own 3D fields to check the limits.";

/// The stored 3D inputs: the four reverse fields of the demagnetization check and the
/// slip-loss fields (the steel circuit's eleven, and the three free-space ones). They are 3D
/// results for the example design and stay fixed when the design changes
/// ([`STORED_3D_SENTENCE`]); [`STORED_3D_ROWS`] are the headline rows that move with them.
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
/// screens, the material warnings, the space claim and the gearbox check. Each gives a badge
/// level ([`verdict_level`]): the results table shows it, and its failing filter lists the
/// checks that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps'
/// status texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 31] = [
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
        ("temperature.adhesive_life.daily_screen", "Below the fatigue endurance") => Some(Good),
        (
            "temperature.adhesive_life.daily_screen",
            "Above the fatigue endurance: qualify by thermal cycling",
        ) => Some(Caution),
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
                value: with_unit(format_value(&value), info.meta.unit),
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

/// Draws the dashboard of `results`, computed from `inputs`: the design status line, the
/// end-effect banner when f_end <= 0, then a row per line (badge and label, then the value and,
/// with the spreadsheet details, its correction ids under a one-line legend, so a narrow side
/// still reads) with its Why line, a guided check's
/// block over its Why line, the stored-3D label after the last temperature row, and the
/// Magnets fit row last. `fixes` holds every guide's fixes in [`CHECK_GUIDES`]' order
/// (`guidance::FixCache`). Each result row is a readout (`readouts`): its hover text and
/// equation while hovered, a click opens it in the Equation panel. While the shaft-clamp
/// section is paused (`clamps_paused`) its rows, their guide and their Why links are left out
/// and the status line does not count its check.
pub fn dashboard_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    clamps_paused: bool,
    readouts: &mut Readouts,
) {
    status_ui(ui, inputs, results, clamps_paused);
    let lines = shown_lines(results, clamps_paused);
    if end_effect_banner_ui(ui, results, readouts) {
        ui.separator();
    }
    warnings_ui(ui, results, readouts);
    // The correction ids beside the values, and what they are, only with the spreadsheet
    // details (spec S2-5).
    let show_details = readouts.show_details();
    if show_details {
        ui.add(egui::Label::new(egui::RichText::new(MARKER_LEGEND).small().weak()).wrap());
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        let title = ui.horizontal(|ui| {
            badge(ui, line.level);
            ui.add(egui::Label::new(tint(egui::RichText::new(label_for(line.path)))).wrap());
        });
        let value = ui.horizontal(|ui| {
            ui.add_space(16.0);
            // One wrapped flow, indented: the correction ids follow the value on its last line
            // or the next. Beside a value wrapped to the full width they would run past the row,
            // and the resizable dashboard would widen every frame.
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
        // A check's block under its own row (the rows its verdict only badges have none), then
        // every row's Why line.
        match CHECK_GUIDES.iter().position(|g| g.check == line.path) {
            Some(guide) => indexed_guide_ui(ui, guide, inputs, results, fixes, readouts),
            None => {
                let learn = row_learn(line.path).expect("every row has a Why link (guidance)");
                why_line(ui, None, learn, readouts);
            }
        }
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
            });
        }
        ui.add_space(4.0);
    }
    fit_ui(ui, inputs, results, fixes, readouts);
}

/// The design status line (guidance spec 3.1): whose design it is and how many guided checks
/// fail (those that count with the shaft-clamp section's pause, `clamps_paused`), in the colour
/// of the worst level, over a separator; at the defaults its hover says where the example's
/// values come from (spec S2-5).
fn status_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
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
    let text = egui::RichText::new(design_status_text(&status))
        .strong()
        .color(level.color(ui.visuals()));
    let line = ui.add(egui::Label::new(text).wrap());
    if status.is_default {
        line.on_hover_text(guidance::EXAMPLE_DESIGN_HOVER);
    }
    ui.separator();
}

/// The Magnets fit row: its badge and label, whether the blocks fit (from the guide's level: red
/// when they do not), then its guide.
fn fit_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    readouts: &mut Readouts,
) {
    let index = CHECK_GUIDES
        .iter()
        .position(|g| g.check == FIT_CHECK)
        .expect("the Magnets fit row has a guide");
    let guide = &CHECK_GUIDES[index];
    let level = guidance::level(guide, inputs, results);
    ui.horizontal(|ui| {
        badge(ui, Some(level));
        ui.add(egui::Label::new(guide.label).wrap());
    });
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        let value = if level == Level::Bad {
            DOES_NOT_FIT
        } else {
            FITS
        };
        ui.add(egui::Label::new(egui::RichText::new(value).strong()).wrap());
    });
    indexed_guide_ui(ui, index, inputs, results, fixes, readouts);
}

/// What a guide's lines are drawn with: its check's level, the fixes to show (empty: the
/// no-fix line) and the notes under them ([`guide_notes`]).
struct GuideShown<'a> {
    level: Level,
    fixes: &'a [(&'static str, Direction)],
    notes: &'a [String],
}

/// The lines of the guide at `index` of [`CHECK_GUIDES`] in this design ([`guide_ui`]): its
/// level, its fixes from `fixes`, and its notes ([`guide_notes`]).
fn indexed_guide_ui(
    ui: &mut egui::Ui,
    index: usize,
    inputs: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    readouts: &mut Readouts,
) {
    let guide = &CHECK_GUIDES[index];
    let notes = guide_notes(fixes, index, inputs, results);
    let shown = GuideShown {
        level: guidance::level(guide, inputs, results),
        fixes: fixes.get(index).map_or(&[], Vec::as_slice),
        notes: &notes,
    };
    guide_ui(ui, guide, &shown, inputs, results, readouts);
}

/// A guided check's lines under its row. Its block, open under red and under amber once its
/// header is clicked: the problem (from `inputs` and `results`, in the level's colour), the
/// Why line with the guide's sentence, and the Fix line ([`fix_line`]). Otherwise the Why
/// line's link alone ([`why_line`]).
fn guide_ui(
    ui: &mut egui::Ui,
    guide: &CheckGuide,
    shown: &GuideShown,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    let open = match shown.level {
        Level::Bad => true,
        Level::Caution => caution_header_ui(ui, guide.check),
        Level::Good => false,
    };
    if open && let Some(problem) = (guide.problem)(inputs, results) {
        guide_line(ui, PROBLEM, |ui| {
            let text = egui::RichText::new(problem).color(shown.level.color(ui.visuals()));
            ui.add(egui::Label::new(text).wrap());
        });
    }
    why_line(ui, open.then_some(guide.why), guide.learn, readouts);
    if open {
        fix_line(ui, shown, readouts);
    }
}

/// The header an amber check's block is shut under ([`CAUTION_DETAILS`]), shut at first: a
/// click on it or on its arrow opens or shuts the block. Whether the block is open.
fn caution_header_ui(ui: &mut egui::Ui, check: &str) -> bool {
    let id = ui.make_persistent_id(("check_guide", check));
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
        let header = egui::Label::new(CAUTION_DETAILS).sense(egui::Sense::click());
        if ui.add(header).clicked() {
            state.toggle(ui);
        }
    });
    state.store(ui.ctx());
    state.is_open()
}

/// The Fix line of an open block: a link per fix, which asks for its input's row (a two-pick
/// magnet fix: its part's row), or [`NO_FIX`] when no listed change helps; then the notes
/// ([`guide_notes`]).
fn fix_line(ui: &mut egui::Ui, shown: &GuideShown, readouts: &mut Readouts) {
    guide_line(ui, FIX, |ui| {
        if shown.fixes.is_empty() {
            ui.add(egui::Label::new(NO_FIX).wrap());
        }
        for &(path, direction) in shown.fixes {
            if ui.link(fix_text(path, direction)).clicked() {
                readouts.focus_input(path);
            }
        }
        for note in shown.notes {
            ui.add(egui::Label::new(note).wrap());
        }
    });
}

/// A row's Why line: under an open block the guide's `sentence`, then the link ([`learn_ui`]).
fn why_line(ui: &mut egui::Ui, sentence: Option<&str>, learn: Learn, readouts: &mut Readouts) {
    guide_line(ui, WHY_LINK, |ui| {
        if let Some(sentence) = sentence {
            ui.add(egui::Label::new(sentence).wrap());
        }
        learn_ui(ui, learn, readouts);
    });
}

/// The link at the end of a Why line: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
/// input's row; the rationale is its hover text), or [`NO_NOTE_YET`] with the reason as its
/// hover text.
fn learn_ui(ui: &mut egui::Ui, learn: Learn, readouts: &mut Readouts) {
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

/// One line of a row's guide, 16 points in under its row: `head`, then `body` in a column
/// beside it that wraps within the width left.
fn guide_line(ui: &mut egui::Ui, head: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal_top(|ui| {
        ui.add_space(16.0);
        ui.label(head);
        ui.vertical(body);
    });
}

/// The material warnings that fire: a badge and the text in the severity's colour, then a link
/// to the teaching note (it asks the panel to open the note).
fn warnings_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = warning_lines(results);
    if lines.is_empty() {
        return;
    }
    ui.strong(WARNINGS_HEADING);
    for (rule, text) in lines {
        let level = severity_level(rule.severity);
        ui.horizontal(|ui| {
            badge(ui, Some(level));
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

/// A badge: a filled circle in the level's colour, or an empty cell (the results table's check
/// rows draw it too).
pub(crate) fn badge(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    if let Some(level) = level {
        ui.painter()
            .circle_filled(rect.center(), 5.0, level.color(ui.visuals()));
        response.on_hover_text(level.word());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::meta::InputSet;
    use crate::gui::corrections::MARKER_HEADING;
    use crate::gui::guidance::{CHECK_GUIDES, Direction, Fixes, ROW_LEARN, all_fixes, trade_offs};
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, hover_frames, only_the_clamp_fails, only_the_gearbox_fails,
        passing_design, primary_button, short_magnets, sized_frame, text_center, text_color,
        text_rect, tooltips_at_once, with_poles,
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
        // A new check (a text named *_check, *_screen, *verdict or *reading, or a warning)
        // fails here until it is listed and classified. The clamp table's and the sweeps' rows
        // rate candidates, not the design.
        for row in &rows {
            let name = row.path.rsplit('.').next().unwrap();
            let named = name.ends_with("_check")
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
        // The defaults fail five checks, all red.
        let defaults = compute_all(&DesignInputs::default());
        assert_eq!(
            failing_checks(&defaults, false),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("gearbox.cold_high_check", Bad),
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
        // The headline's rows, and one row past it: the gearbox check, computed from cold-high.
        let past_the_headline: Vec<&str> = END_EFFECT_ROWS
            .into_iter()
            .filter(|row| !HEADLINE.iter().any(|&(_, path)| path == *row))
            .collect();
        assert_eq!(past_the_headline, ["gearbox.cold_high_check"]);
    }

    #[test]
    fn the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs() {
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
        // the values are the example design's and do not follow the design; no live 3D model is
        // promised (section 8, slice-1 trap 6). Amendment A-7: "example design", the name the
        // screen gives the defaults.
        for text in [STORED_3D_LABEL, STORED_3D_NOTE, STORED_3D_SENTENCE] {
            assert_eq!(
                guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text}"
            );
            assert!(text.contains("example design"), "{text}");
            assert!(!text.contains("live"), "{text}");
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
    }

    #[test]
    fn every_stored_3d_input_says_it_is_the_example_design_s_and_stays_fixed() {
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
        assert!(lines.iter().all(|l| !l.greyed));
        for path in STORED_3D_ROWS {
            assert!(line(path).stored_3d);
            assert!(line(path).tooltip(false).contains(STORED_3D_LABEL));
        }
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 3);
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
        // so the three checks among the greyed rows give no badge and are not listed as failing.
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
                "gearbox.cold_high_check"
            ]
        );
        for path in &greyed {
            assert!(greyed_by_end_effect(&results, path), "{path}");
            // The verdict has a level of its own (red, green and green here): the rule drops it.
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
        // gearbox check red.
        let defaults = compute_all(&DesignInputs::default());
        assert!(!greyed_by_end_effect(&defaults, "metal.hot_min_check"));
        assert_eq!(check_level(&defaults, "metal.hot_min_check"), Some(Bad));
        assert_eq!(
            check_level(&defaults, "clamps.recommended"),
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
    fn the_correction_ids_and_their_legend_show_only_with_the_spreadsheet_details() {
        use crate::gui::typeset::TermColors;
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let markers: Vec<String> = dashboard_lines(&results)
            .into_iter()
            .map(|line| line.marker)
            .filter(|marker| !marker.is_empty())
            .collect();
        assert!(!markers.is_empty(), "the defaults carry correction ids");
        let status = design_status_text(&design_status(&inputs, &results, false));
        for show_details in [false, true] {
            let ctx = egui::Context::default();
            let output = sized_frame(&ctx, egui::vec2(1280.0, 4000.0), Vec::new(), |ui| {
                let mut readouts = Readouts::new(TermColors::default(), show_details);
                dashboard_ui(ui, &inputs, &results, &fixes, false, &mut readouts);
            });
            let texts = drawn_texts(&output);
            for marker in &markers {
                assert_eq!(texts.contains(marker), show_details, "{marker}");
            }
            let legends = texts.iter().filter(|t| *t == MARKER_LEGEND).count();
            assert_eq!(legends, usize::from(show_details));
            // The status line still tops the dashboard.
            assert_eq!(texts.first(), Some(&status));
        }
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
    fn each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label() {
        let ctx = egui::Context::default();
        let texts = drawn_texts(&dashboard_frame(&ctx, &DesignInputs::default()));
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
        for line in &lines {
            assert!(
                texts.iter().any(|t| t == label_for(line.path)),
                "{}",
                line.path
            );
            // The line keeps the catalogue label: the spreadsheet's Summary writes it.
            assert_eq!(line.label, result_info(line.path).unwrap().meta.label);
        }
        // The overlay renames four rows (spec 3.5's three and BL-059's ripple row); their
        // catalogue labels are not drawn.
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
        for line in lines.iter().filter(|line| renamed.contains(&line.path)) {
            assert!(!texts.iter().any(|t| t == line.label), "{}", line.path);
        }
    }

    #[test]
    fn every_hover_text_with_the_details_off_is_in_plain_words() {
        // Spec 3.4: every drawn tooltip with Show spreadsheet details off passes the guard:
        // every input's, every result's and every dashboard row's (in range and with the end
        // effect out of range).
        let mut texts: Vec<String> = crate::gui::inputs::InputCatalogue::get()
            .all()
            .map(|entry| crate::gui::inputs::input_tooltip(entry, false))
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

    /// The defaults under a 30 mm overall-length claim: the hot-low torque needs longer
    /// magnets and the space claim shorter ones, both failing (a trade-off note in each).
    fn length_conflict() -> DesignInputs {
        design(|i| i.metal.max_overall_axial_mm = 30.0)
    }

    /// One frame of the dashboard of `inputs` at full width on a screen tall enough for every
    /// row: what it drew.
    fn dashboard_frame(ctx: &egui::Context, inputs: &DesignInputs) -> egui::FullOutput {
        dashboard_frame_with(ctx, inputs, false)
    }

    /// [`dashboard_frame`] with the shaft-clamp section paused (`clamps_paused`) or not.
    fn dashboard_frame_with(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        clamps_paused: bool,
    ) -> egui::FullOutput {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, clamps_paused);
        sized_frame(ctx, egui::vec2(1280.0, 4000.0), Vec::new(), |ui| {
            dashboard_ui(
                ui,
                inputs,
                &results,
                &fixes,
                clamps_paused,
                &mut Readouts::default(),
            );
        })
    }

    #[test]
    fn while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard() {
        // Ruling 11: a design that fails only the clamp reads as passing while it is paused;
        // with the section shown its row, its Problem and its Why sentence are drawn.
        let ctx = egui::Context::default();
        let inputs = only_the_clamp_fails();
        let results = compute_all(&inputs);
        let clamp = guidance::guide("clamps.recommended").expect("the clamp's guide");
        let problem = (clamp.problem)(&inputs, &results).expect("the clamp fails");
        for paused in [false, true] {
            let output = dashboard_frame_with(&ctx, &inputs, paused);
            let texts = drawn_texts(&output);
            let count = |text: &str| texts.iter().filter(|t| *t == text).count();
            let status = design_status_text(&design_status(&inputs, &results, paused));
            assert_eq!(texts.first(), Some(&status), "paused {paused}");
            let shown = usize::from(!paused);
            assert_eq!(
                count(label_for("clamps.recommended")),
                shown,
                "paused {paused}"
            );
            assert_eq!(count(&problem), shown, "paused {paused}");
            assert_eq!(count(clamp.why), shown, "paused {paused}");
        }
        assert_eq!(
            design_status_text(&design_status(&inputs, &results, true)),
            "Your design: passes all 8 checks"
        );
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

    /// The text of the Why link to `learn`, as a reader would name it: a reviewed note's title,
    /// an assumption's label, else "no note yet".
    fn learn_link_text(learn: Learn) -> String {
        match learn {
            Learn::Note(id) => {
                reviewed_note(id).map_or(NO_NOTE_YET.to_owned(), |n| glyph_safe(n.title))
            }
            Learn::Assumption(id) => ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .label
                .to_owned(),
            Learn::None(_) => NO_NOTE_YET.to_owned(),
        }
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
    fn a_red_check_shows_its_block_an_amber_one_its_header_and_every_row_its_why_link() {
        let count = |texts: &[String], text: &str| texts.iter().filter(|t| *t == text).count();
        // Every fix text any guide can show: the static lists, and the fixes of the designs.
        let mut fix_texts: BTreeSet<String> = CHECK_GUIDES
            .iter()
            .filter_map(|g| match g.fixes {
                Fixes::Static(list) => Some(list),
                Fixes::PerTerm(_) => None,
            })
            .flatten()
            .map(|&(path, dir)| fix_text(path, dir))
            .collect();
        let ctx = egui::Context::default();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            too_many_poles(),
            length_conflict(),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
        ] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            fix_texts.extend(
                fixes
                    .iter()
                    .flatten()
                    .map(|&(path, dir)| fix_text(path, dir)),
            );
            let texts = drawn_texts(&dashboard_frame(&ctx, &inputs));
            let level = |g: &CheckGuide| guidance::level(g, &inputs, &results);
            // Red blocks are open; amber ones are shut under their header.
            let open: Vec<usize> = (0..CHECK_GUIDES.len())
                .filter(|&i| level(&CHECK_GUIDES[i]) == Level::Bad)
                .collect();
            let amber = CHECK_GUIDES
                .iter()
                .filter(|g| level(g) == Level::Caution)
                .count();
            assert_eq!(count(&texts, CAUTION_DETAILS), amber);
            let problems: Vec<String> = open
                .iter()
                .filter_map(|&i| (CHECK_GUIDES[i].problem)(&inputs, &results))
                .collect();
            assert_eq!(count(&texts, PROBLEM), problems.len());
            for problem in &problems {
                assert_eq!(count(&texts, problem), 1, "{problem:?}");
            }
            for (i, guide) in CHECK_GUIDES.iter().enumerate() {
                assert_eq!(
                    count(&texts, guide.why),
                    usize::from(open.contains(&i)),
                    "{}",
                    guide.check
                );
            }
            // Each open block has its Fix line: its fixes, each once, or the no-fix line; then
            // its notes (trade-offs and side effects).
            assert_eq!(count(&texts, FIX), open.len());
            let mut want: Vec<String> = open
                .iter()
                .flat_map(|&i| fixes[i].iter().map(|&(path, dir)| fix_text(path, dir)))
                .collect();
            let mut drawn: Vec<String> = texts
                .iter()
                .filter(|t| fix_texts.contains(*t))
                .cloned()
                .collect();
            want.sort();
            drawn.sort();
            assert_eq!(drawn, want);
            let no_fix = open.iter().filter(|&&i| fixes[i].is_empty()).count();
            assert_eq!(count(&texts, guidance::NO_FIX), no_fix);
            let notes: Vec<String> = open
                .iter()
                .flat_map(|&i| guide_notes(&fixes, i, &inputs, &results))
                .collect();
            for note in &notes {
                assert_eq!(
                    count(&texts, note),
                    notes.iter().filter(|n| *n == note).count()
                );
            }
            // Every row has its Why line, the fit row's included, each leading where
            // `row_learn` (the fit row: its guide) says.
            assert_eq!(count(&texts, WHY_LINK), DASHBOARD.len() + 1);
            let fit = guidance::guide(FIT_CHECK).unwrap().learn;
            let links: Vec<String> = ROW_LEARN
                .iter()
                .map(|&(_, learn)| learn)
                .chain([fit])
                .map(learn_link_text)
                .collect();
            for link in &links {
                assert_eq!(
                    count(&texts, link),
                    links.iter().filter(|l| *l == link).count(),
                    "{link:?}"
                );
            }
        }
        // The workbook example's four red checks are the ones with blocks.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let red: BTreeSet<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| guidance::level(g, &defaults, &results) == Level::Bad)
            .map(|g| g.check)
            .collect();
        let want = [
            "metal.hot_min_check",
            "metal.clearance_check",
            "materials.cup_wall_check",
            "gearbox.cold_high_check",
        ];
        assert_eq!(red, BTreeSet::from(want));
        // The designs above reach a trade-off note and an amber header.
        let conflict = length_conflict();
        let fixes = all_fixes(&conflict, &compute_all(&conflict), false);
        assert!((0..CHECK_GUIDES.len()).any(|i| !trade_offs(&fixes, i).is_empty()));
        let texts = drawn_texts(&dashboard_frame(&ctx, &hot_day()));
        assert_eq!(count(&texts, CAUTION_DETAILS), 1);
    }

    #[test]
    fn the_example_design_s_status_line_says_on_hover_where_its_values_come_from() {
        // Spec S2-5: "Example design" at the defaults, its hover the original spreadsheet;
        // "Your design" has no hover.
        for (inputs, example) in [(DesignInputs::default(), true), (passing_design(), false)] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            let text = design_status_text(&design_status(&inputs, &results, false));
            assert_eq!(
                text.starts_with(guidance::EXAMPLE_DESIGN),
                example,
                "{text}"
            );
            let (_, output) = hover_frames(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                |output| text_center(output, &text),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &results,
                        &fixes,
                        false,
                        &mut Readouts::default(),
                    )
                },
            );
            let hover = guidance::EXAMPLE_DESIGN_HOVER.to_owned();
            assert_eq!(drawn_texts(&output).contains(&hover), example, "{text}");
        }
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let mut passing_hot_day = passing_design();
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
        let cases = [
            // The workbook example: four red checks.
            (DesignInputs::default(), 4, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                3,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The passing design with the example's gearbox rating: the gearbox check alone.
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 4, 1, Level::Bad),
            // The passing design on the hot day: amber alone.
            (passing_hot_day, 0, 1, Level::Caution),
        ];
        for (inputs, bad, caution, colour) in cases {
            let results = compute_all(&inputs);
            let status = design_status(&inputs, &results, false);
            assert_eq!((status.bad, status.caution), (bad, caution));
            assert_eq!(status.total, CHECK_GUIDES.len());
            assert_eq!(status.is_default, inputs == DesignInputs::default());
            let output = dashboard_frame(&ctx, &inputs);
            let text = design_status_text(&status);
            assert_eq!(
                drawn_texts(&output).first(),
                Some(&text),
                "the status line tops the dashboard"
            );
            assert_eq!(text_color(&output, &text), Some(colour.color(&visuals)));
        }
    }

    #[test]
    fn the_fit_row_says_whether_the_blocks_fit_and_guides_a_misfit() {
        let ctx = egui::Context::default();
        let fit = guidance::guide(FIT_CHECK).unwrap();
        let count = |output: &egui::FullOutput, text: &str| {
            drawn_texts(output).iter().filter(|t| *t == text).count()
        };
        let output = dashboard_frame(&ctx, &DesignInputs::default());
        assert_eq!(count(&output, fit.label), 1);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (1, 0));
        assert_eq!(count(&output, fit.why), 0);
        // 40 poles: the blocks do not fit; a red badge, the problem and the guide's fixes.
        let inputs = too_many_poles();
        let results = compute_all(&inputs);
        assert!(!crate::engine::model::blocks_fit(
            &inputs.coupling,
            &results.model
        ));
        assert_eq!(guidance::level(fit, &inputs, &results), Level::Bad);
        let output = dashboard_frame(&ctx, &inputs);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (0, 1));
        assert_eq!(count(&output, fit.why), 1);
        let problem = (fit.problem)(&inputs, &results).expect("a misfit has a Problem");
        assert_eq!(count(&output, &problem), 1);
        // The last badge drawn is the fit row's, red.
        let visuals = ctx.style().visuals.clone();
        let last_badge = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .rev()
            .find_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == 5.0 => Some(c.fill),
                _ => None,
            });
        assert_eq!(last_badge, Some(Level::Bad.color(&visuals)));
    }

    /// The cup wall fix the test guide offers.
    const CUP_WALL_FIX: [(&str, Direction); 1] = [("metal.cup_wall_corner_mm", Direction::Raise)];

    /// A guide of the cup wall check, the test's own, linking to `learn`.
    fn guide_with(learn: Learn) -> CheckGuide {
        CheckGuide {
            check: "materials.cup_wall_check",
            label: "Cup wall",
            problem: |_, results| Some(results.materials.cup_wall_check.clone()),
            why: "A test sentence.",
            fixes: Fixes::Static(&CUP_WALL_FIX),
            learn,
        }
    }

    /// A test guide's lines at `level`, with the cup wall fix and no trade-off note.
    fn shown_at(level: Level) -> GuideShown<'static> {
        GuideShown {
            level,
            fixes: &CUP_WALL_FIX,
            notes: &[],
        }
    }

    /// Frames of `guide` drawn as `block` on the defaults in `ctx`, a first one with no input,
    /// then one per entry of `events`: what the last drew and what the readouts heard in it.
    fn guide_frames(
        ctx: &egui::Context,
        guide: &CheckGuide,
        block: &GuideShown,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, ReadoutEvents) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::default();
            let output = sized_frame(ctx, egui::vec2(600.0, 400.0), input, |ui| {
                guide_ui(ui, guide, block, &inputs, &results, &mut readouts);
            });
            last = Some((output, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Where `guide` drawn as `block` draws `text` on the defaults in `ctx`.
    fn guide_text_at(
        ctx: &egui::Context,
        guide: &CheckGuide,
        block: &GuideShown,
        text: &str,
    ) -> egui::Pos2 {
        let (output, _) = guide_frames(ctx, guide, block, Vec::new());
        text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center()
    }

    /// The frames of a click at `at` (move, press, release).
    fn click_at(at: egui::Pos2) -> Vec<Vec<egui::Event>> {
        vec![
            vec![egui::Event::PointerMoved(at)],
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ]
    }

    /// A click on `text` drawn by `guide` as `block`: what the readouts heard.
    fn click_in_guide(guide: &CheckGuide, block: &GuideShown, text: &str) -> ReadoutEvents {
        let ctx = egui::Context::default();
        let at = guide_text_at(&ctx, guide, block, text);
        guide_frames(&ctx, guide, block, click_at(at)).1
    }

    /// The texts drawn while `text` of `guide` as `block` is hovered, tooltips shown at once
    /// (no delay, the pointer need not rest): its hover text among them.
    fn hover_in_guide(guide: &CheckGuide, block: &GuideShown, text: &str) -> Vec<String> {
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let at = guide_text_at(&ctx, guide, block, text);
        let hover = vec![vec![egui::Event::PointerMoved(at)], Vec::new()];
        drawn_texts(&guide_frames(&ctx, guide, block, hover).0)
    }

    #[test]
    fn a_why_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row() {
        let green = shown_at(Level::Good);
        // A reviewed note: its title (glyph-safe) asks for the note.
        let note = reviewed_note("end_effect").expect("a reviewed note");
        let events = click_in_guide(
            &guide_with(Learn::Note(note.id)),
            &green,
            &glyph_safe(note.title),
        );
        assert_eq!((events.note, events.focus), (Some(note.id), None));
        // An assumption: its label asks for its first input's row; its hover text is the
        // rationale.
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        let guide = guide_with(Learn::Assumption(assumption.id));
        let events = click_in_guide(&guide, &green, assumption.label);
        assert_eq!(
            (events.note, events.focus),
            (None, Some(assumption.paths[0]))
        );
        let texts = hover_in_guide(&guide, &green, assumption.label);
        assert!(texts.iter().any(|t| t == assumption.rationale));
        // No note, or a note that has not passed review (no note has this id): "no note yet",
        // which asks for nothing; its hover text says why.
        for (learn, reason) in [
            (Learn::None("A test reason."), "A test reason."),
            (Learn::Note("no_such_note"), NOTE_NOT_REVIEWED),
        ] {
            let guide = guide_with(learn);
            assert_eq!(
                click_in_guide(&guide, &green, NO_NOTE_YET),
                ReadoutEvents::default()
            );
            assert!(hover_in_guide(&guide, &green, NO_NOTE_YET).contains(&reason.to_owned()));
        }
        // A red row's fix asks for its input's row; a two-pick magnet fix for its part's row.
        let guide = guide_with(Learn::None("A test reason."));
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let events = click_in_guide(&guide, &shown_at(Level::Bad), &fix);
        assert_eq!(
            (events.note, events.focus),
            (None, Some("metal.cup_wall_corner_mm"))
        );
        let part = [("coupling.magnets.part_inner", Direction::CustomHotterGrade)];
        let two_picks = GuideShown {
            level: Level::Bad,
            fixes: &part,
            notes: &[],
        };
        let events = click_in_guide(&guide, &two_picks, &fix_text(part[0].0, part[0].1));
        assert_eq!(events.focus, Some("coupling.magnets.part_inner"));
    }

    #[test]
    fn the_block_is_open_under_red_shut_under_amber_until_asked_and_absent_under_green() {
        let guide = guide_with(Learn::None("A test reason."));
        let problem = compute_all(&DesignInputs::default())
            .materials
            .cup_wall_check;
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let block_texts = [PROBLEM, problem.as_str(), guide.why, FIX, fix.as_str()];
        let y = |output: &egui::FullOutput, text: &str| text_rect(output, text).unwrap().top();
        // Red: open, in order (the problem in red, the sentence then its link, the fixes).
        let (output, _) = guide_frames(&ctx, &guide, &shown_at(Level::Bad), Vec::new());
        let texts = drawn_texts(&output);
        for want in block_texts.iter().chain(&[WHY_LINK, NO_NOTE_YET]) {
            assert!(texts.iter().any(|t| t == want), "no {want:?}");
        }
        assert!(!texts.iter().any(|t| t == CAUTION_DETAILS));
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Bad.color(&visuals))
        );
        assert!(y(&output, &problem) < y(&output, guide.why));
        assert!(y(&output, guide.why) < y(&output, NO_NOTE_YET));
        assert!(y(&output, NO_NOTE_YET) < y(&output, &fix));
        // Amber: a header and the Why link, the block shut; a click on the header opens it,
        // the problem in amber, and the Why link stays.
        let ctx = egui::Context::default();
        let amber = shown_at(Level::Caution);
        let (output, _) = guide_frames(&ctx, &guide, &amber, Vec::new());
        assert_eq!(
            drawn_texts(&output),
            [CAUTION_DETAILS, WHY_LINK, NO_NOTE_YET]
        );
        let header = guide_text_at(&ctx, &guide, &amber, CAUTION_DETAILS);
        let mut frames = click_at(header);
        frames.push(Vec::new());
        let (output, events) = guide_frames(&ctx, &guide, &amber, frames);
        assert_eq!(events, ReadoutEvents::default(), "opening asks for nothing");
        let texts = drawn_texts(&output);
        for want in block_texts
            .iter()
            .chain(&[CAUTION_DETAILS, WHY_LINK, NO_NOTE_YET])
        {
            assert!(
                texts.iter().any(|t| t == want),
                "amber, opened: no {want:?}"
            );
        }
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Caution.color(&visuals))
        );
        // Green: the Why line alone.
        let (output, _) = guide_frames(&ctx, &guide, &shown_at(Level::Good), Vec::new());
        assert_eq!(drawn_texts(&output), [WHY_LINK, NO_NOTE_YET]);
    }

    #[test]
    fn a_red_block_without_a_fix_says_so_and_its_trade_off_notes_follow_the_fixes() {
        let guide = guide_with(Learn::None("A test reason."));
        let ctx = egui::Context::default();
        let none = GuideShown {
            level: Level::Bad,
            fixes: &[],
            notes: &[],
        };
        let (output, _) = guide_frames(&ctx, &guide, &none, Vec::new());
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == FIX));
        assert!(texts.iter().any(|t| t == guidance::NO_FIX));
        let note = "A test trade-off.".to_owned();
        let notes = [note.clone()];
        let noted = GuideShown {
            level: Level::Bad,
            fixes: &CUP_WALL_FIX,
            notes: &notes,
        };
        let (output, _) = guide_frames(&ctx, &guide, &noted, Vec::new());
        assert!(!drawn_texts(&output).iter().any(|t| t == guidance::NO_FIX));
        let fix = fix_text(CUP_WALL_FIX[0].0, CUP_WALL_FIX[0].1);
        let y = |text: &str| text_rect(&output, text).unwrap().top();
        assert!(y(&fix) < y(&note));
    }

    #[test]
    fn every_guide_line_has_glyphs_in_the_default_fonts_and_plain_words() {
        // On designs where each check fails: the status line, the fit row, every row's Why
        // line, and each guide at red whatever the design (its problem, its sentence, its
        // fixes for the design and its static ones, its notes); the fixed texts, and
        // the hover texts.
        let ctx = egui::Context::default();
        let mut plain: Vec<String> = Vec::new();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            too_many_poles(),
            length_conflict(),
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)),
        ] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            let output = sized_frame(&ctx, egui::vec2(600.0, 8000.0), Vec::new(), |ui| {
                let mut readouts = Readouts::default();
                status_ui(ui, &inputs, &results, false);
                fit_ui(ui, &inputs, &results, &fixes, &mut readouts);
                for &(_, learn) in &ROW_LEARN {
                    why_line(ui, None, learn, &mut readouts);
                }
                for (i, guide) in CHECK_GUIDES.iter().enumerate() {
                    let notes = guide_notes(&fixes, i, &inputs, &results);
                    let listed = match guide.fixes {
                        Fixes::Static(list) => list,
                        Fixes::PerTerm(_) => &fixes[i][..],
                    };
                    for fixes in [&fixes[i][..], listed] {
                        let shown = GuideShown {
                            level: Level::Bad,
                            fixes,
                            notes: &notes,
                        };
                        guide_ui(ui, guide, &shown, &inputs, &results, &mut readouts);
                    }
                }
            });
            plain.extend(drawn_texts(&output));
        }
        plain.extend(
            [
                NOTE_NOT_REVIEWED,
                FITS,
                DOES_NOT_FIT,
                CAUTION_DETAILS,
                guidance::NO_FIX,
            ]
            .map(str::to_owned),
        );
        // An assumption's rationale is the hover text of its Why link: plain words too (spec
        // 3.4, ruling R10; the slice-1 exemption for cells is gone). The branch stays even while
        // no row links an assumption: it covers a row linked to one again.
        let fit = guidance::guide(FIT_CHECK).unwrap().learn;
        for learn in ROW_LEARN.iter().map(|&(_, learn)| learn).chain([fit]) {
            match learn {
                Learn::None(reason) => plain.push(reason.to_owned()),
                Learn::Assumption(id) => plain.push(
                    ASSUMPTIONS
                        .iter()
                        .find(|a| a.id == id)
                        .unwrap()
                        .rationale
                        .to_owned(),
                ),
                Learn::Note(_) => {}
            }
        }
        for text in &plain {
            assert_glyphs(&ctx, text, "the check guides");
        }
        for text in &plain {
            assert!(
                guidance::plain_text_violations(text).is_empty(),
                "{text:?}: {:?}",
                guidance::plain_text_violations(text)
            );
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
    fn the_dashboard_draws_the_cup_wall_verdict_in_plain_words() {
        let ctx = egui::Context::default();
        let results = compute_all(&DesignInputs::default());
        let shown = verdict_display(
            "materials.cup_wall_check",
            &results.materials.cup_wall_check,
        )
        .expect("the default cup wall fails");
        let texts = drawn_texts(&dashboard_frame(&ctx, &DesignInputs::default()));
        assert_eq!(texts.iter().filter(|t| **t == shown).count(), 1, "{shown}");
        let engine_words = CUP_WALL_TOO_THIN.trim_end();
        assert!(
            !texts.iter().any(|t| t.contains(engine_words)),
            "the engine's cell-named verdict is drawn"
        );
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
