//! Metadata parity with the Python engine.
//!
//! Every ported input and result carries the Python label, unit, help, workbook
//! cell and choices exactly (the GUI shows them; a typo is a bug), every input
//! default equals the Python default, and no Python field of a ported group is
//! missing. `tests/data/python_schema.json` is written by
//! `reference/magcoupling-py/tools/gen_differential.py`.
//!
//! Where an applied correction rewords a help text (its registry entry's
//! `workbook_help`), Python still has the workbook text, so that field's help
//! is compared against the recorded workbook text instead.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{
    PORTED_INPUTS, PORTED_RESULTS, data_path, is_ported_input, is_ported_result, is_table_path,
    json_to_value, read_json, report, reworded_help,
};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with, headline};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{ResultRow, Value, input_rows, result_rows};

/// One Python schema row.
struct PyRow {
    kind: String,
    label: String,
    unit: String,
    help: String,
    cell: Option<String>,
    choices: Vec<(i64, String)>,
    default: Option<serde_json::Value>,
}

fn python_rows() -> BTreeMap<String, PyRow> {
    let doc = read_json(&data_path("python_schema.json"));
    let text = |row: &serde_json::Value, key: &str| row[key].as_str().expect(key).to_owned();
    doc["rows"]
        .as_array()
        .expect("a rows array")
        .iter()
        .map(|row| {
            let choices = row["choices"]
                .as_array()
                .expect("choices")
                .iter()
                .map(|pair| {
                    (
                        pair[0].as_i64().expect("choice code"),
                        pair[1].as_str().expect("choice text").to_owned(),
                    )
                })
                .collect();
            let py = PyRow {
                kind: text(row, "kind"),
                label: text(row, "label"),
                unit: text(row, "unit"),
                help: text(row, "help"),
                cell: row["cell"].as_str().map(str::to_owned),
                choices,
                default: row.get("default").cloned(),
            };
            (text(row, "path"), py)
        })
        .collect()
}

/// Compares the metadata both sides share; pushes one line per difference.
fn compare(
    failures: &mut Vec<String>,
    path: &str,
    rust: (&str, &str, &str, Option<&str>),
    py: &PyRow,
) {
    let (label, unit, help, cell) = rust;
    for (what, r, p) in [
        ("label", label, py.label.as_str()),
        ("unit", unit, py.unit.as_str()),
        ("help", help, py.help.as_str()),
    ] {
        if r != p {
            failures.push(format!("{path}: {what} rust={r:?} python={p:?}"));
        }
    }
    if cell != py.cell.as_deref() {
        failures.push(format!("{path}: cell rust={cell:?} python={:?}", py.cell));
    }
}

#[test]
fn ported_inputs_carry_the_python_metadata_and_defaults() {
    let python = python_rows();
    let reworded = reworded_help();
    let mut failures = Vec::new();
    for row in input_rows(&DesignInputs::defaults_with(Deviations::NONE)) {
        let Some(py) = python.get(&row.path) else {
            failures.push(format!("{}: not a Python input", row.path));
            continue;
        };
        let m = row.meta;
        if py.kind != "input" {
            failures.push(format!("{}: Python kind is {:?}", row.path, py.kind));
        }
        // Python keeps the workbook help where a correction rewords it.
        let help = reworded.get(row.path.as_str()).copied().unwrap_or(m.help);
        compare(
            &mut failures,
            &row.path,
            (m.label, m.unit, help, m.cell),
            py,
        );
        let choices: Vec<(i64, String)> =
            m.choices.iter().map(|&(c, t)| (c, t.to_owned())).collect();
        if choices != py.choices {
            failures.push(format!(
                "{}: choices rust={choices:?} python={:?}",
                row.path, py.choices
            ));
        }
        let py_default = json_to_value(py.default.as_ref().expect("inputs carry a default"));
        if !parity_close(&row.value, &py_default) {
            failures.push(format!(
                "{}: default rust={:?} python={py_default:?}",
                row.path, row.value
            ));
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn ported_results_carry_the_python_metadata() {
    let python = python_rows();
    let mut failures = Vec::new();
    for row in result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .filter(|r| !r.meta.rust_only)
    {
        // Python lists no metadata for table rows; `tables_match_the_python_layout` covers them.
        if is_table_path(&row.path) {
            continue;
        }
        let Some(py) = python.get(&row.path) else {
            failures.push(format!("{}: not a Python result", row.path));
            continue;
        };
        let m = row.meta;
        if py.kind != "result" {
            failures.push(format!("{}: Python kind is {:?}", row.path, py.kind));
        }
        compare(
            &mut failures,
            &row.path,
            (m.label, m.unit, m.help, m.cell),
            py,
        );
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_python_field_of_a_ported_group_is_ported() {
    let python = python_rows();
    let inputs = input_rows(&DesignInputs::default());
    let results = result_rows(&compute_all(&DesignInputs::default()));
    let rust_paths: BTreeSet<&str> = inputs
        .iter()
        .map(|r| r.path.as_str())
        .chain(results.iter().map(|r| r.path.as_str()))
        .collect();
    let missing: Vec<&String> = python
        .iter()
        .filter(|(path, row)| match row.kind.as_str() {
            "input" => is_ported_input(path),
            "result" => is_ported_result(path),
            _ => false,
        })
        .filter(|(path, _)| !rust_paths.contains(path.as_str()))
        .map(|(path, _)| path)
        .collect();
    assert!(missing.is_empty(), "Python fields not ported: {missing:?}");

    // The ratchet counts equal the Python engine's, counted as test_parity.py
    // does (cells only; inputs whose default is None skipped).
    let count = |group: &str, kind: &str| {
        python
            .iter()
            .filter(|(path, row)| {
                common::group_of(path) == group
                    && row.kind == kind
                    && row.cell.is_some()
                    && row.default != Some(serde_json::Value::Null)
            })
            .count()
    };
    for p in PORTED_INPUTS {
        assert_eq!(count(p.group, "input"), p.cells, "{}: input cells", p.group);
    }
    for p in PORTED_RESULTS {
        assert_eq!(
            count(p.group, "result"),
            p.cells,
            "{}: result cells",
            p.group
        );
    }
}

#[test]
fn tables_match_the_python_layout() {
    let doc = read_json(&data_path("python_schema.json"));
    let results = result_rows(&compute_all_with(
        &DesignInputs::defaults_with(Deviations::NONE),
        Deviations::NONE,
    ));
    let mut failures = Vec::new();
    for (table, layout) in doc["tables"].as_object().expect("a tables object") {
        if !is_ported_result(table) {
            continue;
        }
        let prefix = format!("{table}[");
        let rows: Vec<&ResultRow> = results
            .iter()
            .filter(|r| r.path.starts_with(&prefix))
            .collect();
        let fields: Vec<&str> = layout["fields"]
            .as_array()
            .expect("fields")
            .iter()
            .map(|f| f.as_str().expect("a name"))
            .collect();
        let n_rows = layout["rows"].as_u64().expect("rows") as usize;
        if rows.len() != n_rows * fields.len() {
            failures.push(format!(
                "{table}: {} values, Python has {n_rows} rows of {} fields",
                rows.len(),
                fields.len()
            ));
            continue;
        }
        for (i, chunk) in rows.chunks(fields.len()).enumerate() {
            for (row, name) in chunk.iter().zip(&fields) {
                let path = format!("{table}[{i}].{name}");
                if row.path != path {
                    failures.push(format!("{}: expected {path}", row.path));
                    continue;
                }
                let cell = layout["cells"]
                    .get(*name)
                    .map(|cells| cells[i].as_str().expect("a cell").to_owned());
                if row.cell != cell {
                    failures.push(format!("{path}: cell rust={:?} python={cell:?}", row.cell));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn headline_matches_the_python_api() {
    let doc = read_json(&data_path("python_schema.json"));
    let python: Vec<(String, Value)> = doc["headline"]
        .as_array()
        .expect("a headline array")
        .iter()
        .map(|pair| {
            (
                pair[0].as_str().expect("a key").to_owned(),
                json_to_value(&pair[1]),
            )
        })
        .collect();
    let rust = headline(&compute_all_with(
        &DesignInputs::defaults_with(Deviations::NONE),
        Deviations::NONE,
    ));
    let keys: Vec<&str> = rust.iter().map(|(k, _)| *k).collect();
    let py_keys: Vec<&str> = python.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, py_keys, "keys and order");
    for ((key, r), (_, p)) in rust.iter().zip(&python) {
        assert!(parity_close(r, p), "{key}: rust={r:?} python={p:?}");
    }
}

#[test]
fn schemas_list_fields_in_the_python_order() {
    // The GUI's results table and CSV export follow this order. Table paths are filtered:
    // results! emits tables after the scalars (Python declares clamps.table mid-struct).
    let doc = read_json(&data_path("python_schema.json"));
    let python = |kind: &str| -> Vec<String> {
        doc["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .filter(|r| r["kind"] == kind)
            .map(|r| r["path"].as_str().expect("a path").to_owned())
            .collect()
    };
    let inputs: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|r| r.path)
        .collect();
    let results: Vec<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .filter(|r| !is_table_path(&r.path))
        .filter(|r| !r.meta.rust_only)
        .map(|r| r.path)
        .collect();
    assert_eq!(inputs, python("input"));
    assert_eq!(results, python("result"));
}
