//! Helpers shared by the integration tests: test data paths, JSON conversion,
//! the workbook snapshot, and the list of ported modules.

#![allow(dead_code)] // each test crate uses a different subset

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use magcoupling::engine::meta::Value;

/// A top-level group of `DesignInputs`/`DesignResults` that is ported, with the
/// number of workbook cells it must check. The counts are a ratchet against
/// silently dropped fields; `tests/python_schema.rs` also checks them against
/// the Python engine's own schema.
pub struct Ported {
    pub group: &'static str,
    pub result_cells: usize,
    pub input_cells: usize,
}

/// Ported groups, in Python `compute_all` order. Add a line when a module lands.
pub const PORTED: &[Ported] = &[Ported {
    group: "calibration",
    result_cells: 23,
    input_cells: 16,
}];

/// The top-level group of a dotted path (`"calibration.br_T"` gives `"calibration"`).
pub fn group_of(path: &str) -> &str {
    path.split(['.', '[']).next().unwrap_or(path)
}

/// Whether a path belongs to a ported group.
pub fn is_ported(path: &str) -> bool {
    let group = group_of(path);
    PORTED.iter().any(|p| p.group == group)
}

/// A file under `magcoupling-rs/tests/data/`.
pub fn data_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(relative)
}

/// A file relative to the repository root.
pub fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

/// Reads a text file with its line endings normalized to `\n` (the checkout may
/// have converted them to CRLF).
pub fn read_text(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.replace("\r\n", "\n")
}

/// Parses a JSON file.
pub fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&read_text(path))
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// A JSON scalar as a [`Value`]: integers stay integers, as in Python.
pub fn json_to_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Num(n.as_f64().expect("a JSON number is an i64 or an f64")),
        },
        serde_json::Value::String(s) => Value::Text(s.clone()),
        other => panic!("expected a JSON scalar, got {other}"),
    }
}

/// A [`Value`] as JSON. Panics on a non-finite number (JSON has none).
pub fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Num(x) => serde_json::Value::from(
            serde_json::Number::from_f64(*x).unwrap_or_else(|| panic!("{x} has no JSON form")),
        ),
        Value::Int(i) => serde_json::Value::from(*i),
        Value::Text(s) => serde_json::Value::from(s.as_str()),
        Value::None => serde_json::Value::Null,
    }
}

/// The workbook snapshot, `"Sheet!Cell"` to value (`tests/data/reference_values.json`).
pub fn snapshot() -> BTreeMap<String, Value> {
    let json = read_json(&data_path("reference_values.json"));
    json.as_object()
        .expect("the snapshot is a JSON object")
        .iter()
        .map(|(cell, v)| (cell.clone(), json_to_value(v)))
        .collect()
}

/// Joins failure lines into one assertion message (at most 40 lines shown).
pub fn report(failures: &[String]) -> String {
    let shown: Vec<_> = failures.iter().take(40).cloned().collect();
    let more = failures.len().saturating_sub(shown.len());
    let tail = if more > 0 {
        format!("\n... and {more} more")
    } else {
        String::new()
    };
    format!("{} failure(s):\n{}{tail}", failures.len(), shown.join("\n"))
}
