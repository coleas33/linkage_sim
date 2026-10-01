//! The results table (spec M4 "Layout": "every computed value with label, unit, cell;
//! searchable; CSV and JSON export").
//!
//! Every result, scalars and table rows, in the Python schema order ([`result_rows`]). The
//! layout of the results does not depend on the inputs, so the rows (path, metadata, cell,
//! marker, search text) are built once; a frame reads only the values of the rows on screen.
//! The exports write every result at full precision: CSV for spreadsheets, JSON with the
//! design that produced it. JSON has no infinity or NaN, so both write a non-finite number as
//! `+inf`, `-inf` or `NaN` (decision M41-15).

use std::sync::OnceLock;

use serde_json::{Map, Value as Json};

use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{
    ResultInfo, ResultNotes, end_effect_banner, result_info, result_tooltip,
};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};

/// The `format` of a results export.
pub const RESULTS_FORMAT: &str = "magcoupling-results";

/// The version of the results export.
pub const RESULTS_VERSION: u64 = 1;

/// The file names the exports suggest.
pub const CSV_FILE_NAME: &str = "magcoupling-results.csv";
pub const JSON_FILE_NAME: &str = "magcoupling-results.json";

/// The CSV header.
pub const CSV_HEADER: &str = "path,label,value,unit,cell";

/// The button labels.
pub const EXPORT_CSV: &str = "Export CSV";
pub const EXPORT_JSON: &str = "Export JSON";

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// One row of the table: what does not change with the inputs.
#[derive(Clone, Debug)]
pub struct TableEntry {
    pub path: String,
    pub info: &'static ResultInfo,
    /// The corrections' marker text, empty for none.
    pub marker: String,
    /// Label, path and cell, lowercase: what the search matches.
    haystack: String,
}

/// Every result's row, in schema order, built once.
pub fn table_entries() -> &'static [TableEntry] {
    static ENTRIES: OnceLock<Vec<TableEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        result_rows(&compute_all(&DesignInputs::default()))
            .into_iter()
            .map(|row| {
                let info = result_info(&row.path).expect("every result has its info");
                let haystack = format!(
                    "{}\n{}\n{}",
                    info.meta.label,
                    row.path,
                    info.cell.as_deref().unwrap_or("")
                )
                .to_lowercase();
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
                    path: row.path,
                    info,
                    haystack,
                }
            })
            .collect()
    })
}

/// The indices of the rows whose label, path or cell contains `query`, ignoring case and the
/// surrounding blanks; every row for a blank query.
pub fn search(entries: &[TableEntry], query: &str) -> Vec<usize> {
    let needle = query.trim().to_lowercase();
    (0..entries.len())
        .filter(|&i| needle.is_empty() || entries[i].haystack.contains(&needle))
        .collect()
}

/// A number at full precision: the shortest text that reads back as the same number
/// (`2.6884762950539796`, `5.27e-10`); `+inf`, `-inf` or `NaN` when not finite.
pub fn exact_number(x: f64) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None => format!("{x:?}"),
    }
}

/// A value at full precision; text as it is, an empty string for none.
fn exact_value(value: &Value) -> String {
    match value {
        Value::Num(x) => exact_number(*x),
        Value::Int(i) => i.to_string(),
        Value::Text(text) => text.clone(),
        Value::None => String::new(),
    }
}

/// A CSV field (RFC 4180): quoted, with quotes doubled, when it holds a comma, a quote or a
/// line break.
fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_owned()
    }
}

/// The CSV export: [`CSV_HEADER`], then one line per result in schema order.
pub fn results_csv(results: &DesignResults) -> String {
    let mut csv = String::from(CSV_HEADER);
    csv.push_str("\r\n");
    for row in result_rows(results) {
        let fields = [
            row.path.as_str(),
            row.meta.label,
            &exact_value(&row.value),
            row.meta.unit,
            row.cell.as_deref().unwrap_or(""),
        ];
        let line: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
        csv.push_str(&line.join(","));
        csv.push_str("\r\n");
    }
    csv
}

/// The JSON export: the design and every result with its label, unit and cell.
pub fn results_json(design: &Design, results: &DesignResults) -> String {
    let rows: Vec<Json> = result_rows(results)
        .into_iter()
        .map(|row| {
            let mut entry = Map::new();
            entry.insert("path".to_owned(), Json::String(row.path));
            entry.insert("label".to_owned(), Json::from(row.meta.label));
            entry.insert("value".to_owned(), json_value(&row.value));
            entry.insert("unit".to_owned(), Json::from(row.meta.unit));
            entry.insert("cell".to_owned(), row.cell.map_or(Json::Null, Json::String));
            Json::Object(entry)
        })
        .collect();
    let mut top = Map::new();
    top.insert("format".to_owned(), Json::from(RESULTS_FORMAT));
    top.insert("version".to_owned(), Json::from(RESULTS_VERSION));
    top.insert("design".to_owned(), design_json(design));
    top.insert("results".to_owned(), Json::Array(rows));
    let mut text = serde_json::to_string_pretty(&Json::Object(top))
        .expect("a JSON value of maps, strings and numbers always serializes");
    text.push('\n');
    text
}

/// The table's state: the search text and the rows it matches.
#[derive(Clone, Debug, Default)]
pub struct ResultsTable {
    query: String,
    /// The rows matching `matched_query`; `None` until the first frame.
    matches: Option<Vec<usize>>,
    matched_query: String,
}

/// What the user asked of the table this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableAction {
    ExportCsv,
    ExportJson,
}

impl ResultsTable {
    /// The search text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Draws the table: the end-effect banner when f_end ≤ 0, the search box and the export
    /// buttons, then the rows on screen. Returns an export asked for.
    pub fn ui(&mut self, ui: &mut egui::Ui, results: &DesignResults) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        if let Some(banner) = end_effect_banner(results) {
            ui.colored_label(ui.visuals().error_fg_color, banner);
        }
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text(SEARCH_HINT)
                    .desired_width(260.0),
            );
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            let shown = self.matches.as_ref().map_or(0, Vec::len);
            ui.weak(format!("{shown} of {} results", entries.len()));
            if ui.button(EXPORT_CSV).clicked() {
                action = Some(TableAction::ExportCsv);
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height);
                }
            });
        action
    }
}

/// A row's hover text: the hover hook's ([`result_tooltip`]) with the exact value.
pub fn row_tooltip(entry: &TableEntry, value: &Value) -> String {
    let marks = CorrectionIndex::get().marks(entry.info.cell.as_deref());
    let mut tooltip = result_tooltip(
        &entry.path,
        entry.info,
        ResultNotes {
            marks,
            ..ResultNotes::default()
        },
    );
    if let Value::Num(x) = value {
        tooltip.push_str(&format!("\nExact value: {}", exact_number(*x)));
    }
    tooltip
}

/// One table row: label, value with unit, workbook cell, marker (the path is in the hover
/// text, which is built only while the row is hovered).
fn row_ui(ui: &mut egui::Ui, entry: &TableEntry, results: &DesignResults, height: f32) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
        let cell = |ui: &mut egui::Ui, width: f32, text: &str| {
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            ui.allocate_ui_with_layout(egui::vec2(width, height), layout, |ui| {
                ui.set_min_width(width);
                ui.add(egui::Label::new(text).truncate());
            });
        };
        cell(ui, 260.0, entry.info.meta.label);
        cell(
            ui,
            130.0,
            &with_unit(format_value(&value), entry.info.meta.unit),
        );
        cell(ui, 130.0, entry.info.cell.as_deref().unwrap_or("Rust-only"));
        cell(ui, 60.0, &entry.marker);
    });
    row.response.on_hover_ui(|ui| {
        ui.label(row_tooltip(entry, &value));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A positive beta typed in for manual magnets with no grade (E20): no hot limit, +inf, and
    /// no torque at it, NaN (`tests/robustness.rs`).
    fn non_finite_design() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.0035;
        inputs
    }

    #[test]
    fn the_table_lists_every_result_once_in_schema_order() {
        let entries = table_entries();
        let rows = result_rows(&compute_all(&DesignInputs::default()));
        assert_eq!(entries.len(), rows.len());
        for (entry, row) in entries.iter().zip(&rows) {
            assert_eq!(entry.path, row.path);
            assert_eq!(entry.info.cell, row.cell);
        }
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout.marker, "E3 E7 E8");
    }

    #[test]
    fn the_search_matches_label_path_and_cell_ignoring_case() {
        let entries = table_entries();
        let paths = |query: &str| -> Vec<&str> {
            search(entries, query)
                .into_iter()
                .map(|i| entries[i].path.as_str())
                .collect()
        };
        assert_eq!(paths("calculator!c93"), ["model.pullout_Nm"]);
        assert_eq!(paths("  MODEL.PULLOUT_20C_NM "), ["model.pullout_20C_Nm"]);
        assert!(paths("pull-out torque").contains(&"model.pullout_Nm"));
        let row3: Vec<&str> = paths("gap_sweep[3].");
        let columns = result_rows(&compute_all(&DesignInputs::default()))
            .iter()
            .filter(|r| r.path.starts_with("gap_sweep[3]."))
            .count();
        assert_eq!(row3.len(), columns);
        assert_eq!(search(entries, "").len(), entries.len());
        assert_eq!(search(entries, "   ").len(), entries.len());
        assert!(search(entries, "no such result anywhere").is_empty());
    }

    #[test]
    fn a_row_tooltip_is_the_hover_hook_s_with_the_exact_value() {
        let entries = table_entries();
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        let value = Value::Num(2.6884762950539796);
        let tooltip = row_tooltip(pullout, &value);
        assert!(
            tooltip.starts_with("Pull-out torque at operating temperature\n"),
            "{tooltip}"
        );
        assert!(tooltip.contains("Corrected vs workbook:"), "{tooltip}");
        assert!(
            tooltip.ends_with("\nExact value: 2.6884762950539796"),
            "{tooltip}"
        );
        let claim = entries
            .iter()
            .find(|e| e.path == "housing.space_claim_check")
            .unwrap();
        let text = row_tooltip(claim, &Value::Text("Inside the space claim".to_owned()));
        assert!(!text.contains("Exact value"), "{text}");
    }

    #[test]
    fn exact_numbers_read_back_bit_for_bit() {
        for x in [
            2.6884762950539796,
            5.27e-10,
            -0.1031743945660739,
            50.0,
            1e300,
            0.0,
        ] {
            assert_eq!(exact_number(x).parse::<f64>().unwrap(), x, "{x}");
        }
        assert_eq!(exact_number(5.27e-10), "5.27e-10");
        assert_eq!(exact_number(f64::INFINITY), "+inf");
        assert_eq!(exact_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(exact_number(f64::NAN), "NaN");
    }

    #[test]
    fn the_csv_has_a_line_per_result_with_quoted_text() {
        let results = compute_all(&DesignInputs::default());
        let csv = results_csv(&results);
        let lines: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(
            lines.len(),
            result_rows(&results).len() + 2,
            "header, rows, final break"
        );
        assert_eq!(lines.last(), Some(&""));
        let pullout = lines
            .iter()
            .find(|l| l.starts_with("model.pullout_Nm,"))
            .unwrap();
        assert_eq!(
            *pullout,
            format!(
                "model.pullout_Nm,Pull-out torque at operating temperature,{},N·m,Calculator!C93",
                exact_number(results.model.pullout_Nm)
            )
        );
        let mass = lines
            .iter()
            .find(|l| l.starts_with("mass.total_g,"))
            .unwrap();
        assert!(
            mass.starts_with("mass.total_g,\"Preliminary rotating mass, including retainers\","),
            "{mass}"
        );
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_field("a\nb"), "\"a\nb\"");
        // A Rust-only result has an empty cell field.
        let claim = lines
            .iter()
            .find(|l| l.starts_with("housing.space_claim_check,"))
            .unwrap();
        assert!(claim.ends_with(",Inside the space claim,,"), "{claim}");
    }

    #[test]
    fn non_finite_results_export_as_text_in_csv_and_json() {
        let design = Design {
            inputs: non_finite_design(),
            ..Design::default()
        };
        let results = compute_all(&design.inputs);
        assert_eq!(results.temperature.demag.magnet_limit_C, f64::INFINITY);
        assert!(results.temperature.demag.torque_at_limit_Nm.is_nan());
        let csv = results_csv(&results);
        let line = |path: &str| {
            csv.split("\r\n")
                .find(|l| l.starts_with(&format!("{path},")))
                .unwrap()
                .to_owned()
        };
        assert!(line("temperature.demag.magnet_limit_C").contains(",+inf,"));
        assert!(line("temperature.demag.torque_at_limit_Nm").contains(",NaN,"));
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        let value = |path: &str| {
            json["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["path"] == path)
                .unwrap()["value"]
                .clone()
        };
        assert_eq!(
            value("temperature.demag.magnet_limit_C"),
            Json::from("+inf")
        );
        assert_eq!(
            value("temperature.demag.torque_at_limit_Nm"),
            Json::from("NaN")
        );
    }

    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
        let design = Design::default();
        let results = compute_all(&design.inputs);
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        assert_eq!(json["format"], RESULTS_FORMAT);
        assert_eq!(json["version"], RESULTS_VERSION);
        assert_eq!(json["design"], design_json(&design));
        let rows = json["results"].as_array().unwrap();
        assert_eq!(rows.len(), result_rows(&results).len());
        let pullout = rows
            .iter()
            .find(|r| r["path"] == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout["value"].as_f64(), Some(results.model.pullout_Nm));
        assert_eq!(pullout["unit"], "N·m");
        assert_eq!(pullout["cell"], "Calculator!C93");
        let claim = rows
            .iter()
            .find(|r| r["path"] == "housing.space_claim_check")
            .unwrap();
        assert_eq!(claim["cell"], Json::Null);
    }
}
