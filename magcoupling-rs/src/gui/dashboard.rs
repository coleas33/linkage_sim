//! The dashboard on the right (spec M4 "Layout": "headline numbers with green/amber/red
//! badges derived from check verdicts; corrected values carry a 'corrected vs workbook' marker
//! with the deviation in its tooltip").
//!
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the A1 space
//! claim badge and the audit M9 end-effect flag. A row's badge comes from a check verdict
//! ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
//!
//! - **End effect (audit M9).** When f_end ≤ 0 the pull-out and the numbers computed from it
//!   ([`END_EFFECT_ROWS`]) are greyed, without a badge, under the banner "End-effect model out
//!   of range".
//! - **Stored 3D values.** M3 (the live 3D field model) comes after M4, so the temperature
//!   rows that read the workbook's stored 3D fields ([`STORED_3D_ROWS`]) carry the label
//!   "3D values from the workbook" instead of a "3D updating" badge.
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

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::explain::notes;
use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, with_unit};
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
pub const DASHBOARD: [(&str, Option<&str>); 17] = [
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
    (
        "housing.space_claim_check",
        Some("housing.space_claim_check"),
    ),
    ("model.end_effect_check", Some("model.end_effect_check")),
];

/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the
/// headline rows that move with the end-effect coefficient (a test probes it at back iron 1
/// and 0). Per-row greying of the results table waits for plan A-3's dependency graph; the
/// table shows the banner instead (decision M41-11).
pub const END_EFFECT_ROWS: [&str; 7] = [
    "model.pullout_Nm",
    "model.pullout_20C_Nm",
    "metal.torque_hot_low_Nm",
    "metal.hot_min_check",
    "metal.torque_cold_high_Nm",
    "model.gearbox_input_ripple_Nm",
    "clamps.recommended",
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
pub const STORED_3D_LABEL: &str = "3D values from the workbook";

/// Its hover text.
pub const STORED_3D_NOTE: &str = "The demagnetization reverse fields and the slip-loss fields are the workbook's stored 3D values: they do not follow geometry changes until M3 computes them live.";

/// The start of the banner shown when f_end ≤ 0.
pub const END_EFFECT_BANNER: &str = END_EFFECT_OUT_OF_RANGE;

/// Every design-level check of the results, in schema order (decision O-3): the dashboard's
/// verdicts, the coupling model's, the temperature design's and the clamp's checks and
/// screens, the material warnings and the space claim. Each gives a badge level
/// ([`verdict_level`]): the results table shows it, and its failing filter lists the checks
/// that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps' status
/// texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 30] = [
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
        ("model.end_effect_check", "OK") => Some(Good),
        ("model.end_effect_check", END_EFFECT_OUT_OF_RANGE) => Some(Bad),
        _ => None,
    }
}

/// The badge level of the check at `path` for `results`: [`verdict_level`] of its text; `None`
/// for a path that is no check or holds no text, or a verdict that gives no badge.
pub fn check_level(results: &DesignResults, path: &str) -> Option<Level> {
    match results.get(path)? {
        Value::Text(text) => verdict_level(path, &text),
        _ => None,
    }
}

/// The checks of `results` that fail (red) or ask for a look (amber), the red first, each in
/// [`CHECKS`] order: what the results table's failing filter shows.
pub fn failing_checks(results: &DesignResults) -> Vec<(&'static str, Level)> {
    let mut failing: Vec<(&'static str, Level)> = CHECKS
        .iter()
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

/// The hover text of a displayed result: label, help, path, workbook cell, then the notes.
/// The text every readout shows (`Readouts::show` adds the equation under it).
pub fn result_tooltip(path: &str, info: &ResultInfo, notes: ResultNotes<'_>) -> String {
    let meta = info.meta;
    let mut lines = vec![meta.label.to_owned()];
    if !meta.help.is_empty() {
        lines.push(meta.help.to_owned());
    }
    lines.push(path.to_owned());
    lines.push(
        info.cell
            .clone()
            .unwrap_or_else(|| "Rust-only result (no workbook cell)".to_owned()),
    );
    if notes.greyed {
        lines.push(format!(
            "{END_EFFECT_OUT_OF_RANGE}: computed from the pull-out, which is not valid while f_end <= 0."
        ));
    }
    if notes.stored_3d {
        lines.push(format!("{STORED_3D_LABEL}: {STORED_3D_NOTE}"));
    }
    if !notes.marks.is_empty() {
        lines.push(marker_tooltip(notes.marks));
    }
    lines.join("\n")
}

/// The hover text of the result at `path` with its corrections' marks: the hook's text for a
/// readout outside the dashboard (a results-table row, a geometry callout); `None` for a path
/// that is no result.
pub fn hover_text(path: &str) -> Option<String> {
    let info = result_info(path)?;
    let marks = CorrectionIndex::get().marks(info.cell.as_deref());
    Some(result_tooltip(
        path,
        info,
        ResultNotes {
            marks,
            ..ResultNotes::default()
        },
    ))
}

/// One dashboard row, ready to draw.
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardLine {
    pub path: &'static str,
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
    /// The row's hover text (built only while the row is hovered).
    pub fn tooltip(&self) -> String {
        let info = result_info(self.path).expect("a dashboard row is a result");
        result_tooltip(
            self.path,
            info,
            ResultNotes {
                marks: self.marks,
                greyed: self.greyed,
                stored_3d: self.stored_3d,
            },
        )
    }
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
pub fn end_effect_out_of_range(results: &DesignResults) -> Option<f64> {
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
}

/// The banner shown over the dashboard and the results table when f_end ≤ 0.
pub fn end_effect_banner(results: &DesignResults) -> Option<String> {
    end_effect_out_of_range(results).map(|f_end| {
        format!(
            "{END_EFFECT_BANNER} (f_end = {}): the pull-out and the numbers computed from it are greyed.",
            format_value(&Value::Num(f_end))
        )
    })
}

/// The dashboard rows of `results`.
pub fn dashboard_lines(results: &DesignResults) -> Vec<DashboardLine> {
    let out_of_range = end_effect_out_of_range(results).is_some();
    DASHBOARD
        .iter()
        .map(|&(path, badge)| {
            let info = result_info(path).unwrap_or_else(|| panic!("DASHBOARD: no result {path}"));
            let value = results.get(path).unwrap_or(Value::None);
            let greyed = out_of_range && END_EFFECT_ROWS.contains(&path);
            let stored_3d = STORED_3D_ROWS.contains(&path);
            let level = match (
                greyed,
                badge.and_then(|check| results.get(check).map(|v| (check, v))),
            ) {
                (false, Some((check, Value::Text(text)))) => verdict_level(check, &text),
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

/// Draws the dashboard: the end-effect banner when f_end <= 0, then a row per line (badge
/// and label, then the value and the marker under them, so a narrow side still reads), the
/// stored-3D label after the last temperature row. Each row is a readout (`readouts`): its
/// hover text and equation while hovered, a click opens it in the Equation panel.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = dashboard_lines(results);
    if let Some(banner) = end_effect_banner(results) {
        ui.colored_label(ui.visuals().error_fg_color, banner);
        ui.separator();
    }
    warnings_ui(ui, results, readouts);
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        let title = ui.horizontal(|ui| {
            badge(ui, line.level);
            ui.add(egui::Label::new(tint(egui::RichText::new(line.label))).wrap());
        });
        let value = ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.add(egui::Label::new(tint(egui::RichText::new(&line.value).strong())).wrap());
            if !line.marker.is_empty() {
                ui.small(&line.marker);
            }
        });
        let rect = title.response.rect.union(value.response.rect);
        readouts.show_over(ui, rect, line.path, || line.tooltip());
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
            });
        }
        ui.add_space(4.0);
    }
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
                if ui
                    .link(format!("{WHY}: {}", glyph_safe(note.title)))
                    .clicked()
                {
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
    use crate::gui::test_support::short_magnets;
    use std::collections::BTreeSet;

    #[test]
    fn the_dashboard_starts_with_the_headline_in_order() {
        let paths: Vec<&str> = DASHBOARD.iter().map(|(path, _)| *path).collect();
        let headline: Vec<&str> = HEADLINE.iter().map(|(_, path)| *path).collect();
        assert_eq!(paths[..15], headline[..]);
        assert_eq!(
            paths[15..],
            ["housing.space_claim_check", "model.end_effect_check"]
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
            (
                TEMPERATURE_VERDICT,
                design(|i| {
                    i.temperature.duty.hot_ambient_C = 80.0;
                    i.temperature.duty.driving_rise_C = 60.0;
                }),
                Caution,
            ),
            ("clamps.recommended", DesignInputs::default(), Good),
            (
                "clamps.recommended",
                design(|i| {
                    i.clamps.boss_od_mm = 12.0;
                    i.clamps.clamp_length_mm = 3.0;
                }),
                Bad,
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
        // The defaults fail four checks, all red.
        let defaults = compute_all(&DesignInputs::default());
        assert_eq!(
            failing_checks(&defaults),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
            ]
        );
        // Manual magnets add two amber rating checks, after the red.
        let manual = compute_all(&design(|i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        }));
        assert_eq!(
            failing_checks(&manual),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("model.inner_temp_check", Caution),
                ("model.outer_temp_check", Caution),
            ]
        );
        // Exactly the checks whose level is red or amber, each once.
        for inputs in [DesignInputs::default(), short_magnets(), design(ferrite)] {
            let results = compute_all(&inputs);
            let failing: Vec<&str> = failing_checks(&results).iter().map(|(p, _)| *p).collect();
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

    /// The headline paths whose value moves when `inputs` take any of `values`, from the
    /// defaults at back iron 1 and 0.
    fn headline_moved_by(
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
                    for (_, result) in HEADLINE {
                        if before.get(result) != after.get(result) {
                            moved.insert(result);
                        }
                    }
                }
            }
        }
        moved
    }

    #[test]
    fn the_end_effect_rows_are_the_headline_rows_that_move_with_the_end_effect_coefficient() {
        // c_end = 0 (f_end = 1) is the one value that flips the hot-minimum verdict.
        let moved = headline_moved_by(&["coupling.c_end".to_owned()], |_| {
            vec![0.0, 0.05, 0.3, 0.5]
        });
        assert_eq!(moved, END_EFFECT_ROWS.into_iter().collect());
    }

    #[test]
    fn the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs() {
        // The stored 3D inputs: the four reverse fields of the demagnetization check and the
        // slip-loss fields (steel circuit, and E17's free-space ones), which M3 computes live.
        let inputs: Vec<String> = [
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
        ]
        .map(str::to_owned)
        .to_vec();
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        let moved = headline_moved_by(&inputs, |path| {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            vec![range.min, range.max]
        });
        assert_eq!(moved, STORED_3D_ROWS.into_iter().collect());
    }

    #[test]
    fn the_default_dashboard_has_badges_markers_and_the_3d_rows() {
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
        assert_eq!(lines.len(), DASHBOARD.len());
        let line = |path: &str| lines.iter().find(|l| l.path == path).unwrap();
        assert_eq!(line("model.pullout_Nm").value, "2.688 N·m");
        assert_eq!(line("model.pullout_Nm").level, None);
        assert_eq!(line("model.pullout_Nm").marker, "E3 E7 E8");
        assert!(
            line("model.pullout_Nm")
                .tooltip()
                .contains("with E3 alone: workbook 2.647, corrected 2.688")
        );
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
            assert!(line(path).tooltip().contains(STORED_3D_LABEL));
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
                    line.tooltip().contains(END_EFFECT_OUT_OF_RANGE),
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
        let tooltip = result_tooltip("model.pullout_Nm", info, ResultNotes::default());
        assert!(
            tooltip.starts_with("Pull-out torque at operating temperature\n"),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("\nmodel.pullout_Nm\nCalculator!C93"),
            "{tooltip}"
        );
        let rust_only = result_info("housing.space_claim_check").unwrap();
        assert!(
            result_tooltip(
                "housing.space_claim_check",
                rust_only,
                ResultNotes::default()
            )
            .contains("Rust-only result (no workbook cell)")
        );
        assert!(result_info("no.such").is_none());
        assert_eq!(
            result_info("gap_sweep[0].f_end").unwrap().cell.as_deref(),
            Some("Gap sweep!W6")
        );
    }

    #[test]
    fn the_hover_text_carries_the_marks_and_needs_a_result() {
        let text = hover_text("model.pullout_Nm").unwrap();
        assert!(
            text.starts_with("Pull-out torque at operating temperature\n"),
            "{text}"
        );
        assert!(text.contains("Corrected vs workbook:"), "{text}");
        assert!(
            hover_text("housing.length_overshoot_mm")
                .unwrap()
                .contains("Rust-only result (no workbook cell)")
        );
        assert_eq!(hover_text("no.such"), None);
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
}
