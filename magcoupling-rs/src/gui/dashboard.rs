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
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! one hook the dashboard and the results table (and later the geometry callouts and the
//! equation explorer) go through.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// The badge level of a check's verdict text; `None` (no badge) for a path that is no check, a
/// verdict that says the check does not apply (the cup wall's "No back iron"), or a text the
/// check does not produce.
pub fn verdict_level(path: &str, text: &str) -> Option<Level> {
    use Level::{Bad, Caution, Good};
    match (path, text) {
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
/// The hook every readout goes through (plan M4-3 adds the equation here).
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
    pub tooltip: String,
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
pub fn end_effect_out_of_range(results: &DesignResults) -> Option<f64> {
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
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
                tooltip: result_tooltip(
                    path,
                    info,
                    ResultNotes {
                        marks,
                        greyed,
                        stored_3d,
                    },
                ),
            }
        })
        .collect()
}

/// Draws the dashboard: the end-effect banner when f_end <= 0, then a row per line (badge
/// and label, then the value and the marker under them, so a narrow side still reads), the
/// stored-3D label after the last temperature row.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let lines = dashboard_lines(results);
    if let Some(f_end) = end_effect_out_of_range(results) {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!(
                "{END_EFFECT_BANNER} (f_end = {}): the pull-out and the numbers computed from it are greyed.",
                format_value(&Value::Num(f_end))
            ),
        );
        ui.separator();
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        ui.horizontal(|ui| {
            badge(ui, line.level);
            ui.add(egui::Label::new(tint(egui::RichText::new(line.label))).wrap())
                .on_hover_text(&line.tooltip);
        });
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.add(egui::Label::new(tint(egui::RichText::new(&line.value).strong())).wrap())
                .on_hover_text(&line.tooltip);
            if !line.marker.is_empty() {
                ui.small(&line.marker).on_hover_text(&line.tooltip);
            }
        });
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
            });
        }
        ui.add_space(4.0);
    }
}

/// A badge: a filled circle in the level's colour, or an empty cell.
fn badge(ui: &mut egui::Ui, level: Option<Level>) {
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

    /// A design with f_end below 0 (the engine's short-magnet test): 2 mm manual blocks, c_end 0.5.
    pub(crate) fn short_magnets() -> DesignInputs {
        design(|i| {
            i.coupling.c_end = 0.5;
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
            i.coupling.magnets.manual_inner_length_mm = 2.0;
            i.coupling.magnets.manual_outer_length_mm = 2.0;
        })
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
                .tooltip
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
            assert!(line(path).tooltip.contains(STORED_3D_LABEL));
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
                    line.tooltip.contains(END_EFFECT_OUT_OF_RANGE),
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
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
        assert_eq!(Level::Bad.color(&dark), dark.error_fg_color);
        assert_eq!(Level::Caution.color(&light), light.warn_fg_color);
        assert_ne!(Level::Good.color(&dark), Level::Good.color(&light));
        assert_eq!(Level::Good.word(), "OK");
    }
}
