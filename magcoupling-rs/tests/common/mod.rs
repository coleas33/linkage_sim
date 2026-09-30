//! Helpers shared by the integration tests: test data paths, JSON conversion,
//! the workbook snapshot, and the list of ported modules.

#![allow(dead_code)] // each test crate uses a different subset

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use magcoupling::engine::deviations::{DeviationStatus, REGISTRY};
use magcoupling::engine::meta::Value;

/// A ported input group of `DesignInputs` and its workbook input cells
/// (inputs whose default is `None` are not counted, as in `test_parity.py`).
pub struct PortedInputs {
    pub group: &'static str,
    pub cells: usize,
}

/// A ported result group of `DesignResults` and its workbook result cells.
pub struct PortedResults {
    pub group: &'static str,
    pub cells: usize,
    /// Cells of the group's tables: the screw table, the sweeps.
    pub table_cells: usize,
}

/// Ported input groups, in Python `DesignInputs` order. The counts are a
/// ratchet against silently dropped fields; `tests/python_schema.rs` checks
/// them against the Python engine's own schema.
pub const PORTED_INPUTS: &[PortedInputs] = &[
    PortedInputs {
        group: "coupling",
        cells: 24,
    },
    PortedInputs {
        group: "metal",
        cells: 47, // metal.measured_drag_Nm defaults to None: not counted
    },
    PortedInputs {
        group: "calibration",
        cells: 16,
    },
    PortedInputs {
        group: "materials",
        cells: 11,
    },
    PortedInputs {
        group: "temperature",
        cells: 37,
    },
    PortedInputs {
        group: "clamps",
        cells: 25,
    },
];

/// Ported result groups, in Python `DesignResults` order.
pub const PORTED_RESULTS: &[PortedResults] = &[
    PortedResults {
        group: "calibration",
        cells: 23,
        table_cells: 0,
    },
    PortedResults {
        group: "model",
        cells: 73,
        table_cells: 0,
    },
    PortedResults {
        group: "mass",
        cells: 6,
        table_cells: 0,
    },
    PortedResults {
        group: "retainers",
        cells: 9,
        table_cells: 0,
    },
    PortedResults {
        group: "metal",
        cells: 49,
        table_cells: 0,
    },
    PortedResults {
        group: "materials",
        cells: 7,
        table_cells: 0,
    },
    PortedResults {
        group: "temperature",
        cells: 130, // temperature.adhesive.selected_name has no cell: differential only
        table_cells: 0,
    },
    PortedResults {
        group: "clamps",
        cells: 33,
        table_cells: 165, // 33 celled screw-table fields x 5 sizes (the size name has no cell)
    },
    PortedResults {
        group: "gap_sweep",
        cells: 0,
        table_cells: 338,
    },
    PortedResults {
        group: "pole_sweep",
        cells: 0,
        table_cells: 156,
    },
];

/// The top-level group of a dotted path (`"calibration.br_T"` gives `"calibration"`).
pub fn group_of(path: &str) -> &str {
    path.split(['.', '[']).next().unwrap_or(path)
}

/// Whether an input path belongs to a ported input group.
pub fn is_ported_input(path: &str) -> bool {
    PORTED_INPUTS.iter().any(|p| p.group == group_of(path))
}

/// Whether a result path belongs to a ported result group.
pub fn is_ported_result(path: &str) -> bool {
    PORTED_RESULTS.iter().any(|p| p.group == group_of(path))
}

/// Whether a result path is a table value (`gap_sweep[3].x`): its cell is
/// synthesized from the table layout.
pub fn is_table_path(path: &str) -> bool {
    path.contains('[')
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

/// Help texts an applied correction rewords (its registry entry's
/// `workbook_help`): input path or table column path (`group.table[*].field`)
/// to the workbook (and Python) text.
pub fn reworded_help() -> BTreeMap<&'static str, &'static str> {
    REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
        .flat_map(|d| d.workbook_help.iter().copied())
        .collect()
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
