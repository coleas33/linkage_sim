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

#[test]
fn validation_items_equal_the_python_checklist() {
    let doc = python_static();
    let python: Vec<[String; 3]> = doc["validation_items"]
        .as_array()
        .expect("validation_items")
        .iter()
        .map(|row| [0, 1, 2].map(|i| row[i].as_str().expect("text").to_owned()))
        .collect();
    let rust: Vec<[String; 3]> = magcoupling::engine::metal_design::VALIDATION_ITEMS
        .iter()
        .map(|&(a, b, c)| [a.to_owned(), b.to_owned(), c.to_owned()])
        .collect();
    assert_eq!(rust, python);
}

#[test]
fn aluminium_alloys_equal_the_python_data() {
    use magcoupling::engine::materials::{AL6061, AL7075};
    let doc = python_static();
    let rows = doc["aluminium"].as_array().expect("aluminium");
    assert_eq!(rows.len(), 2);
    for (py, rs) in rows.iter().zip([AL7075, AL6061]) {
        assert_eq!(rs.name, text(py, "name"));
        for (key, value) in [
            ("yield_MPa", rs.yield_MPa),
            ("shear_MPa", rs.shear_MPa),
            ("head_pressure_limit_MPa", rs.head_pressure_limit_MPa),
            ("key_bearing_allow_MPa", rs.key_bearing_allow_MPa),
            ("conductivity_S_m", rs.conductivity_S_m),
        ] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
}

#[test]
fn adhesives_equal_the_python_candidates() {
    use magcoupling::engine::temperature::ADHESIVES;
    let doc = python_static();
    let rows = doc["adhesives"].as_array().expect("adhesives");
    assert_eq!(rows.len(), ADHESIVES.len());
    for (py, rs) in rows.iter().zip(ADHESIVES.iter()) {
        assert_eq!(rs.name, text(py, "name"));
        assert_eq!(
            (rs.role, rs.note),
            (text(py, "role").as_str(), text(py, "note").as_str()),
            "{}",
            rs.name
        );
        for (key, value) in [
            ("design_limit_C", rs.design_limit_C),
            ("cure_C", rs.cure_C),
            ("lap_shear_MPa", rs.lap_shear_MPa),
        ] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
}

#[test]
fn screw_sizes_and_machining_steps_equal_the_python_data() {
    use magcoupling::engine::clamps::{MACHINING_STEPS, SCREW_SIZES, TABLE_COLUMNS};
    let doc = python_static();
    let rows = doc["screw_sizes"].as_array().expect("screw_sizes");
    assert_eq!(rows.len(), SCREW_SIZES.len());
    for (py, rs) in rows.iter().zip(SCREW_SIZES.iter()) {
        assert_eq!(rs.name, text(py, "name"));
        for (key, value) in [
            ("d_mm", rs.d_mm),
            ("pitch_mm", rs.pitch_mm),
            ("As_mm2", rs.As_mm2),
            ("hole_mm", rs.hole_mm),
            ("head_mm", rs.head_mm),
            ("head_h_mm", rs.head_h_mm),
            ("hex_mm", rs.hex_mm),
        ] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
    let steps: Vec<&str> = doc["machining_steps"]
        .as_array()
        .expect("steps")
        .iter()
        .map(|s| s.as_str().expect("text"))
        .collect();
    assert_eq!(MACHINING_STEPS.as_slice(), steps.as_slice());
    let columns: Vec<&str> = doc["table_columns"]
        .as_array()
        .expect("columns")
        .iter()
        .map(|s| s.as_str().expect("text"))
        .collect();
    assert_eq!(TABLE_COLUMNS.as_slice(), columns.as_slice());
}

#[test]
fn sweep_variables_equal_the_python_lists() {
    use magcoupling::engine::sweeps::{GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES};
    let doc = python_static();
    let gaps: Vec<f64> = doc["sweeps"]["gap_sweep_corner_gaps_mm"]
        .as_array()
        .expect("gaps")
        .iter()
        .map(|g| g.as_f64().expect("a number"))
        .collect();
    let poles: Vec<i64> = doc["sweeps"]["pole_sweep_poles"]
        .as_array()
        .expect("poles")
        .iter()
        .map(|p| p.as_i64().expect("an int"))
        .collect();
    assert_eq!(GAP_SWEEP_CORNER_GAPS_MM.as_slice(), gaps.as_slice());
    assert_eq!(POLE_SWEEP_POLES.as_slice(), poles.as_slice());
}
