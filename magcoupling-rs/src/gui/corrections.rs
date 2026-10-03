//! The "corrected vs workbook" markers (spec M4 "Layout": "corrected values carry a
//! 'corrected vs workbook' marker with the deviation in its tooltip").
//!
//! The panel always computes with every approved correction on, and the test-only switch that
//! turns them off never ships, so a marker cannot come from comparing the two. It comes from
//! the deviation registry instead (decision M41-10): a workbook cell carries a correction's
//! marker when the registry ties the cell to it, in any of three ways:
//!
//! - the correction changes the cell at the default design (`changes_at_defaults`, or for the
//!   broad corrections E3, E4 and E5 their reviewed golden files, which hold most of what E3
//!   moves, the headline pull-out included);
//! - the report names the cell for the correction (`cells`);
//! - one of its probes shows the correction changing the cell off the default design.
//!
//! Only applied engine corrections mark (E14 rewords help only). [`CorrectionIndex`] is built
//! once, keyed by workbook cell; a value with no cell (a Rust-only result) has no marker. Plan
//! A-3's equation registry may later supply each record's upstream corrections instead.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value as Json;

use crate::engine::deviations::{
    Deviation, DeviationClass, DeviationId, DeviationStatus, REGISTRY,
};
use crate::engine::meta::Value;
use crate::gui::format::format_value;

/// The golden files of the broad corrections (decision D4), compiled in: the registry names
/// each in its `changes_file` (a test checks the two lists agree).
pub const GOLDEN_FILES: [(DeviationId, &str, &str); 3] = [
    (
        DeviationId::E3,
        "tests/data/deviations/E3.json",
        include_str!("../../tests/data/deviations/E3.json"),
    ),
    (
        DeviationId::E4,
        "tests/data/deviations/E4.json",
        include_str!("../../tests/data/deviations/E4.json"),
    ),
    (
        DeviationId::E5,
        "tests/data/deviations/E5.json",
        include_str!("../../tests/data/deviations/E5.json"),
    ),
];

/// How the registry ties a cell to a correction.
#[derive(Clone, Debug, PartialEq)]
pub enum Tie {
    /// The correction changes the cell at the default design: the workbook's value and the
    /// value with this correction alone applied.
    AtDefaults { workbook: Value, corrected: Value },
    /// The report names the cell for the correction, or a probe shows it changing off the
    /// default design.
    Named,
}

/// One correction's marker on a cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub id: DeviationId,
    pub tie: Tie,
}

/// The markers of every workbook cell the registry ties to an applied engine correction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectionIndex {
    by_cell: HashMap<String, Vec<Mark>>,
}

/// A golden file's value as an engine value (the other way from `session::json_value`): a
/// number or a string; anything else as none.
fn value_from_json(json: &Json) -> Value {
    match json {
        Json::Number(n) => n.as_f64().map_or(Value::None, Value::Num),
        Json::String(text) => Value::Text(text.clone()),
        _ => Value::None,
    }
}

impl CorrectionIndex {
    /// The index of [`REGISTRY`] and [`GOLDEN_FILES`].
    pub fn build() -> Self {
        let mut index = Self::default();
        for deviation in REGISTRY.iter().filter(|d| marks_values(d)) {
            let id = deviation.id;
            for change in deviation.changes_at_defaults {
                index.add(
                    change.cell,
                    id,
                    Tie::AtDefaults {
                        workbook: change.workbook.to_value(),
                        corrected: change.corrected.to_value(),
                    },
                );
            }
            if let Some((_, _, text)) = GOLDEN_FILES.iter().find(|(g, _, _)| *g == id) {
                let json: Json = serde_json::from_str(text).expect("a reviewed golden file");
                let changes = json["changes"].as_object().expect("a changes map");
                for (cell, pair) in changes {
                    index.add(
                        cell,
                        id,
                        Tie::AtDefaults {
                            workbook: value_from_json(&pair[0]),
                            corrected: value_from_json(&pair[1]),
                        },
                    );
                }
            }
            for cell in deviation.cells {
                index.add(cell, id, Tie::Named);
            }
            for probe in deviation.probes {
                for change in probe.expect {
                    index.add(change.cell, id, Tie::Named);
                }
            }
        }
        for marks in index.by_cell.values_mut() {
            marks.sort_by_key(|mark| mark.id.index());
        }
        index
    }

    /// The index, built once.
    pub fn get() -> &'static CorrectionIndex {
        static INDEX: OnceLock<CorrectionIndex> = OnceLock::new();
        INDEX.get_or_init(CorrectionIndex::build)
    }

    /// The markers of a workbook cell, by correction in report order; empty for a cell no
    /// correction touches or for no cell.
    pub fn marks(&self, cell: Option<&str>) -> &[Mark] {
        cell.and_then(|cell| self.by_cell.get(cell))
            .map_or(&[], Vec::as_slice)
    }

    /// Records a tie, once per correction and cell: the at-defaults values win over a name.
    fn add(&mut self, cell: &str, id: DeviationId, tie: Tie) {
        let marks = self.by_cell.entry(cell.to_owned()).or_default();
        match marks.iter_mut().find(|mark| mark.id == id) {
            Some(mark) => {
                if mark.tie == Tie::Named {
                    mark.tie = tie;
                }
            }
            None => marks.push(Mark { id, tie }),
        }
    }
}

/// Whether a correction marks values: applied and changing what the engine computes (the
/// spreadsheet's Summary lists these as the corrections applied).
pub(crate) fn marks_values(deviation: &Deviation) -> bool {
    deviation.status == DeviationStatus::Applied && deviation.class == DeviationClass::Engine
}

/// The marker text beside a value: the corrections' ids (`E3 E7`); empty for none.
pub fn marker_text(marks: &[Mark]) -> String {
    marks
        .iter()
        .map(|mark| mark.id.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The tooltip lines of the markers: per correction its id and title, the workbook and
/// corrected values where it changes the cell at the default design, and its evidence. The
/// corrected value is the one with that correction alone (as the registry and the golden
/// files record it), so on a cell two corrections change neither is the value shown.
pub fn marker_tooltip(marks: &[Mark]) -> String {
    let mut lines = vec!["Corrected vs workbook:".to_owned()];
    for mark in marks {
        let deviation = &REGISTRY[mark.id.index()];
        lines.push(format!("{}: {}", mark.id, deviation.title));
        if let Tie::AtDefaults {
            workbook,
            corrected,
        } = &mark.tie
        {
            lines.push(format!(
                "  at the default design, with {} alone: workbook {}, corrected {}",
                mark.id,
                format_value(workbook),
                format_value(corrected)
            ));
        }
        lines.push(format!("  {}", deviation.evidence()));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::{ResultSet, result_rows};
    use crate::{DesignInputs, compute_all};

    #[test]
    fn the_golden_files_are_the_registry_s() {
        let named: Vec<(DeviationId, &str)> = REGISTRY
            .iter()
            .filter_map(|d| d.changes_file.map(|file| (d.id, file)))
            .collect();
        let compiled: Vec<(DeviationId, &str)> = GOLDEN_FILES
            .iter()
            .map(|(id, file, _)| (*id, *file))
            .collect();
        assert_eq!(compiled, named);
        for (id, _, text) in GOLDEN_FILES {
            let json: Json = serde_json::from_str(text).unwrap();
            assert_eq!(json["id"], id.to_string());
        }
    }

    #[test]
    fn the_headline_pull_out_carries_e3_with_its_workbook_value() {
        let index = CorrectionIndex::get();
        let marks = index.marks(Some("Calculator!C93"));
        let e3 = marks.iter().find(|m| m.id == DeviationId::E3).expect("E3");
        let Tie::AtDefaults {
            workbook,
            corrected,
        } = &e3.tie
        else {
            panic!("E3 changes C93 at the defaults")
        };
        assert_eq!(format_value(workbook), "2.647");
        let pullout = compute_all(&DesignInputs::default()).model.pullout_Nm;
        assert_eq!(corrected, &Value::Num(pullout));
        // E7 and E8 have probes on it; the ids are in report order.
        assert_eq!(marker_text(marks), "E3 E7 E8");
        let tooltip = marker_tooltip(marks);
        assert!(
            tooltip.starts_with("Corrected vs workbook:\nE3: "),
            "{tooltip}"
        );
        assert!(
            tooltip
                .contains("at the default design, with E3 alone: workbook 2.647, corrected 2.688"),
            "{tooltip}"
        );
        assert!(tooltip.contains("docs/analyses/2026-09-29-magcoupling-math-audit.md, entry E3"));
    }

    #[test]
    fn a_cell_two_corrections_change_gives_each_value_with_that_correction_alone() {
        // E1 and E3 both change the peak shear at the defaults; the registry and the golden
        // files record each correction alone, so neither value is the one shown.
        let path = "temperature.mismatch.peak_shear_current_MPa";
        let cell = crate::gui::dashboard::result_info(path)
            .unwrap()
            .cell
            .as_deref();
        let marks = CorrectionIndex::get().marks(cell);
        let shown = compute_all(&DesignInputs::default()).get(path).unwrap();
        for id in [DeviationId::E1, DeviationId::E3] {
            let mark = marks.iter().find(|m| m.id == id).expect("E1 and E3");
            let Tie::AtDefaults { corrected, .. } = &mark.tie else {
                panic!("{id} changes {cell:?} at the defaults")
            };
            assert_ne!(corrected, &shown, "{id}");
        }
        let tooltip = marker_tooltip(marks);
        assert!(
            tooltip.contains("\n  at the default design, with E1 alone: workbook "),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("\n  at the default design, with E3 alone: workbook "),
            "{tooltip}"
        );
    }

    #[test]
    fn a_hand_listed_change_and_a_named_cell_mark_too() {
        let index = CorrectionIndex::get();
        // E2 lists the clamp screw at the defaults.
        let screw = index.marks(Some("Shaft clamps!C48"));
        assert!(screw.iter().any(|m| m.id == DeviationId::E2
            && m.tie
                == Tie::AtDefaults {
                    workbook: Value::Text("ISO 4762 M4 x 12, class 12.9".to_owned()),
                    corrected: Value::Text("ISO 4762 M4 x 14, class 12.9".to_owned()),
                }));
        // E6 names Calculator!C63 (and changes it).
        assert!(
            index
                .marks(Some("Calculator!C63"))
                .iter()
                .any(|m| m.id == DeviationId::E6)
        );
    }

    #[test]
    fn unmarked_cells_rust_only_results_and_documentation_corrections_have_no_marker() {
        let index = CorrectionIndex::get();
        assert!(index.marks(None).is_empty());
        assert!(index.marks(Some("No sheet!Z99")).is_empty());
        // E14 rewords Shaft clamps!C35's help only.
        assert!(
            !index
                .marks(Some("Shaft clamps!C35"))
                .iter()
                .any(|m| m.id == DeviationId::E14)
        );
        assert_eq!(marker_text(&[]), "");
    }

    #[test]
    fn every_marked_cell_belongs_to_a_registry_entry_and_most_results_are_unmarked() {
        let index = CorrectionIndex::get();
        let results = compute_all(&DesignInputs::default());
        let rows = result_rows(&results);
        let marked = rows
            .iter()
            .filter(|row| !index.marks(row.cell.as_deref()).is_empty())
            .count();
        // A marker means something: well under half of the results carry one.
        assert!(
            marked > 100 && marked < rows.len() / 2,
            "{marked} of {}",
            rows.len()
        );
        for marks in index.by_cell.values() {
            for mark in marks {
                assert!(marks_values(&REGISTRY[mark.id.index()]));
            }
        }
    }
}
