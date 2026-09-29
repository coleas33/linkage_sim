//! Workbook parity: every result with a workbook cell, and every default input,
//! equals the workbook snapshot (`tests/data/reference_values.json`) by the rule
//! of `reference/magcoupling-py/tests/test_parity.py`: numbers to 1e-9 relative
//! (1e-12 absolute), text exactly.
//!
//! Deviations are off here: the port must reproduce the workbook itself. The
//! corrected values are checked in `tests/deviations.rs`.

mod common;

use std::collections::BTreeMap;

use common::{
    PORTED_INPUTS, PORTED_RESULTS, data_path, group_of, read_json, repo_path, report, snapshot,
};
use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{Value, input_rows, result_rows};

#[test]
fn every_result_cell_matches_the_workbook() {
    let snapshot = snapshot();
    let results = compute_all_with(
        &DesignInputs::defaults_with(Deviations::NONE),
        Deviations::NONE,
    );
    let mut checked: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for row in result_rows(&results) {
        let Some(cell) = row.cell.as_deref() else {
            continue;
        };
        *checked.entry(group_of(&row.path).to_owned()).or_default() += 1;
        match snapshot.get(cell) {
            None => failures.push(format!(
                "{} names {cell}, which is not in the snapshot",
                row.path
            )),
            Some(want) if !parity_close(&row.value, want) => failures.push(format!(
                "{} ({cell}): rust={:?} workbook={want:?}",
                row.path, row.value
            )),
            Some(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
    let expected: BTreeMap<String, usize> = PORTED_RESULTS
        .iter()
        .map(|p| (p.group.to_owned(), p.cells + p.table_cells))
        .collect();
    assert_eq!(checked, expected, "result cells checked per ported group");
}

#[test]
fn every_default_input_matches_the_workbook() {
    let snapshot = snapshot();
    let inputs = DesignInputs::defaults_with(Deviations::NONE);
    let mut checked: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for row in input_rows(&inputs) {
        // As test_parity.py: an input whose default is None (not entered) has no cell value.
        let (Some(cell), false) = (row.meta.cell, row.value == Value::None) else {
            continue;
        };
        *checked.entry(group_of(&row.path).to_owned()).or_default() += 1;
        match snapshot.get(cell) {
            None => failures.push(format!(
                "{} names {cell}, which is not in the snapshot",
                row.path
            )),
            Some(want) if !parity_close(&row.value, want) => failures.push(format!(
                "{} ({cell}): default={:?} workbook={want:?}",
                row.path, row.value
            )),
            Some(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
    let expected: BTreeMap<String, usize> = PORTED_INPUTS
        .iter()
        .map(|p| (p.group.to_owned(), p.cells))
        .collect();
    assert_eq!(checked, expected, "default inputs checked per ported group");
}

#[test]
fn snapshot_copy_equals_the_vendored_snapshot() {
    let copy = read_json(&data_path("reference_values.json"));
    let vendored = read_json(&repo_path(
        "reference/magcoupling-py/tests/reference_values.json",
    ));
    assert!(
        copy == vendored,
        "tests/data/reference_values.json differs from reference/magcoupling-py/tests/reference_values.json; \
         copy the vendored file again"
    );
}
