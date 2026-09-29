//! Static engine tables (magnet library, and later adhesives, alloys, screw
//! sizes, checklists) equal the Python engine's, value for value.
//! `tests/data/static_data.json` is written by
//! `reference/magcoupling-py/tools/gen_differential.py`.

mod common;

use common::{data_path, read_json};
use magcoupling::engine::library::{MAGNET_LIBRARY, lookup};

fn python_static() -> serde_json::Value {
    read_json(&data_path("static_data.json"))
}

fn text(row: &serde_json::Value, key: &str) -> String {
    row[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is text"))
        .to_owned()
}

fn number(row: &serde_json::Value, key: &str) -> f64 {
    row[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is a number"))
}

#[test]
fn magnet_library_equals_the_python_rows() {
    let doc = python_static();
    let rows = doc["magnet_library"]
        .as_array()
        .expect("a magnet_library array");
    assert_eq!(rows.len(), MAGNET_LIBRARY.len(), "row count");
    for (py, rs) in rows.iter().zip(MAGNET_LIBRARY.iter()) {
        let p = text(py, "part");
        assert_eq!(rs.part, p);
        assert_eq!(rs.vendor, text(py, "vendor"), "{p}");
        assert_eq!(rs.shape, text(py, "shape"), "{p}");
        assert_eq!(rs.grade, text(py, "grade"), "{p}");
        assert_eq!(rs.notes, text(py, "notes"), "{p}");
        for (key, value) in [
            ("length_mm", rs.length_mm),
            ("width_mm", rs.width_mm),
            ("thickness_mm", rs.thickness_mm),
            ("br_T", rs.br_T),
            ("tmax_C", rs.tmax_C),
        ] {
            // Static data: no arithmetic, so equality is exact.
            assert_eq!(value, number(py, key), "{p}.{key}");
        }
    }
}

#[test]
fn every_part_resolves_to_its_own_row_by_exact_text() {
    // By value: a const table has no guaranteed address, and parts are unique.
    for spec in &MAGNET_LIBRARY {
        assert_eq!(lookup(spec.part), Some(spec), "{}", spec.part);
    }
    let parts: std::collections::BTreeSet<&str> = MAGNET_LIBRARY.iter().map(|m| m.part).collect();
    assert_eq!(parts.len(), MAGNET_LIBRARY.len(), "part names are unique");
    for near_miss in ["", "b842sh", "B842SH ", " B842SH", "B842", "B842SH\n"] {
        let found = lookup(near_miss).map(|s| s.part);
        let expected = (near_miss == "B842").then_some("B842");
        assert_eq!(found, expected, "{near_miss:?}");
    }
}

#[test]
fn harmonics_equal_the_python_list() {
    let doc = python_static();
    let python: Vec<u64> = doc["harmonics"]
        .as_array()
        .expect("harmonics")
        .iter()
        .map(|n| n.as_u64().expect("an int"))
        .collect();
    let rust: Vec<u64> = magcoupling::engine::model::HARMONICS
        .iter()
        .map(|&n| u64::from(n))
        .collect();
    assert_eq!(rust, python);
}
