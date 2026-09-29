//! Field metadata checks, and the exported input schema.
//!
//! `tests/data/input_schema.json` is the Rust metadata of every input (types,
//! workbook defaults, choices, slider ranges). The Python differential
//! generator (`reference/magcoupling-py/tools/gen_differential.py`) reads its
//! ranges from it, so they are defined once, here in Rust. After changing any
//! input metadata, rewrite it with:
//!
//! ```text
//! MAGCOUPLING_BLESS=1 cargo test --test schema
//! ```
//!
//! then regenerate the differential data (see `magcoupling-rs/README.md`).

mod common;

use std::collections::BTreeSet;
use std::fs;

use common::{data_path, read_text, report, value_to_json};
use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, Value, input_rows, result_rows};
use serde_json::json;

const SCHEMA_FILE: &str = "input_schema.json";
const BLESS_VAR: &str = "MAGCOUPLING_BLESS";

fn schema_json() -> String {
    let inputs: Vec<serde_json::Value> = input_rows(&DesignInputs::defaults_with(Deviations::NONE))
        .iter()
        .map(|row| {
            let m = row.meta;
            json!({
                "path": row.path,
                "type": m.ty.name(),
                "label": m.label,
                "unit": m.unit,
                "help": m.help,
                "cell": m.cell,
                "default": value_to_json(&row.value),
                "choices": m.choices.iter().map(|(code, text)| json!([code, text])).collect::<Vec<_>>(),
                "range": m.range.map(|r| json!({"min": r.min, "max": r.max, "step": r.step, "log": r.log})),
                "assumption": m.assumption,
            })
        })
        .collect();
    let doc = json!({
        "about": "Rust input metadata of magcoupling-rs: types, workbook defaults, choices and slider \
                  ranges. Written by `MAGCOUPLING_BLESS=1 cargo test --test schema`; read by \
                  reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
        "inputs": inputs,
    });
    serde_json::to_string_pretty(&doc).expect("the schema serializes") + "\n"
}

#[test]
fn input_schema_json_is_current() {
    let path = data_path(SCHEMA_FILE);
    let fresh = schema_json();
    if std::env::var_os(BLESS_VAR).is_some() {
        fs::write(&path, &fresh).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let committed = if path.exists() {
        read_text(&path)
    } else {
        String::new()
    };
    assert!(
        committed == fresh,
        "tests/data/{SCHEMA_FILE} is stale: run `{BLESS_VAR}=1 cargo test --test schema`, \
         then regenerate the differential data"
    );
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(x) => Some(*x),
        Value::Int(i) => Some(*i as f64),
        Value::Text(_) | Value::None => None,
    }
}

#[test]
fn every_numeric_input_has_a_valid_slider_range() {
    let mut failures = Vec::new();
    for row in input_rows(&DesignInputs::default()) {
        let m = row.meta;
        let numeric = matches!(m.ty, FieldType::F64 | FieldType::I64 | FieldType::OptF64);
        if !numeric || !m.choices.is_empty() {
            continue;
        }
        let Some(r) = m.range else {
            failures.push(format!(
                "{}: numeric input without a slider range",
                row.path
            ));
            continue;
        };
        let p = &row.path;
        if !(r.min.is_finite() && r.max.is_finite() && r.min < r.max) {
            failures.push(format!("{p}: min {} must be below max {}", r.min, r.max));
        }
        if !(r.step > 0.0 && r.step <= r.max - r.min) {
            failures.push(format!(
                "{p}: step {} must be positive and fit the range",
                r.step
            ));
        }
        if r.log && r.min <= 0.0 {
            failures.push(format!("{p}: a log slider needs min > 0"));
        }
        if let Some(default) = as_f64(&row.value) {
            if !(r.min <= default && default <= r.max) {
                failures.push(format!(
                    "{p}: default {default} outside [{}, {}]",
                    r.min, r.max
                ));
            }
            if m.ty == FieldType::I64 {
                let integral = [r.min, r.max, r.step].iter().all(|x| x.fract() == 0.0);
                if !integral || ((default - r.min) / r.step).fract() != 0.0 {
                    failures.push(format!(
                        "{p}: integer range must be integral with the default on the step grid"
                    ));
                }
            }
        } else if row.value != Value::None {
            failures.push(format!("{p}: numeric input with default {:?}", row.value));
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_selector_is_an_integer_with_its_default_among_the_choices() {
    let mut failures = Vec::new();
    for row in input_rows(&DesignInputs::default()) {
        let m = row.meta;
        if m.choices.is_empty() {
            continue;
        }
        if m.ty != FieldType::I64 || m.range.is_some() {
            failures.push(format!(
                "{}: a selector is an i64 without a slider",
                row.path
            ));
        }
        let codes: BTreeSet<i64> = m.choices.iter().map(|&(c, _)| c).collect();
        if codes.len() != m.choices.len() {
            failures.push(format!("{}: duplicate choice codes", row.path));
        }
        match row.value {
            Value::Int(code) if codes.contains(&code) => {}
            ref other => failures.push(format!("{}: default {other:?} is not a choice", row.path)),
        }
        if m.choices.iter().any(|(_, text)| text.is_empty()) {
            failures.push(format!("{}: empty choice text", row.path));
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn assumptions_are_numeric_inputs_with_sliders() {
    for row in input_rows(&DesignInputs::default())
        .iter()
        .filter(|r| r.meta.assumption)
    {
        assert!(
            matches!(row.meta.ty, FieldType::F64 | FieldType::I64) && row.meta.range.is_some(),
            "{}: an assumption is a numeric input with a range",
            row.path
        );
    }
}

/// A workbook reference: sheet name, `!`, column letters, row number.
fn is_cell_reference(cell: &str) -> bool {
    let Some((sheet, address)) = cell.split_once('!') else {
        return false;
    };
    let letters = address
        .chars()
        .take_while(|c| c.is_ascii_uppercase())
        .count();
    let digits = &address[letters..];
    !sheet.is_empty()
        && sheet.chars().all(|c| c.is_ascii_alphabetic() || c == ' ')
        && (1..=2).contains(&letters)
        && !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && !digits.starts_with('0')
}

#[test]
fn every_field_has_a_label_and_well_formed_unique_path_and_cell() {
    let inputs = input_rows(&DesignInputs::default());
    let results = result_rows(&compute_all(&DesignInputs::default()));
    let fields = inputs
        .iter()
        .map(|r| (r.path.as_str(), r.meta.label, r.meta.cell))
        .chain(
            results
                .iter()
                .map(|r| (r.path.as_str(), r.meta.label, r.meta.cell)),
        );

    let mut failures = Vec::new();
    let mut paths = BTreeSet::new();
    let mut cells = BTreeSet::new();
    for (path, label, cell) in fields {
        if label.trim().is_empty() {
            failures.push(format!("{path}: empty label"));
        }
        if !paths.insert(path) {
            failures.push(format!("{path}: duplicate path"));
        }
        if let Some(cell) = cell {
            if !is_cell_reference(cell) {
                failures.push(format!("{path}: malformed cell {cell:?}"));
            }
            if !cells.insert(cell) {
                failures.push(format!(
                    "{path}: cell {cell} is already used by another field"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn cell_reference_check_rejects_malformed_references() {
    assert!(is_cell_reference("Temperature design!C106"));
    assert!(is_cell_reference("Gap sweep!AA6"));
    for bad in [
        "C5",
        "Calibration!5",
        "Calibration!C",
        "Calibration!C05",
        "Calibration!ABC5",
        "!C5",
    ] {
        assert!(!is_cell_reference(bad), "{bad}");
    }
}
