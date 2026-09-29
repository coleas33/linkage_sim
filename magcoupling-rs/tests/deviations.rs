//! Deviation registry against the workbook snapshot and the audit report, and
//! the switch that turns corrections off.
//!
//! With every correction off the engine reproduces the workbook
//! (`tests/parity.rs`). Here: switching ONE correction on may change only the
//! cells that correction registers, each to its registered corrected value; and
//! switching all on changes nothing outside the registered cells.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{read_text, repo_path, report, snapshot};
use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::{
    DeviationClass, DeviationId, DeviationStatus, Deviations, REGISTRY, REPORT,
};
use magcoupling::engine::meta::{InputSet, Value, input_rows, result_rows};

/// Every value with a workbook cell (inputs and results) for `inputs` under `dev`.
fn cell_values_for(inputs: &DesignInputs, dev: Deviations) -> BTreeMap<String, Value> {
    let results = compute_all_with(inputs, dev);
    let mut cells = BTreeMap::new();
    for row in input_rows(inputs) {
        if let Some(cell) = row.meta.cell {
            cells.insert(cell.to_owned(), row.value);
        }
    }
    for row in result_rows(&results) {
        if let Some(cell) = row.cell {
            cells.insert(cell, row.value);
        }
    }
    cells
}

/// Every value with a workbook cell at the defaults `dev` implies.
fn cell_values(dev: Deviations) -> BTreeMap<String, Value> {
    cell_values_for(&DesignInputs::defaults_with(dev), dev)
}

/// The value at one workbook cell at the defaults `dev` implies.
fn at(cell: &str, dev: Deviations) -> Value {
    cell_values(dev)
        .remove(cell)
        .unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}

fn num(value: &Value) -> f64 {
    match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// `got` equals the audit report's figure `want`, which the report states to
/// within `half_step` (half a unit of its last digit).
fn assert_report(cell: &str, got: &Value, want: f64, half_step: f64) {
    let got = num(got);
    assert!(
        (got - want).abs() <= half_step,
        "{cell}: {got} is not the report's {want} (± {half_step})"
    );
}

/// With every correction off, the cell still holds the workbook snapshot value.
fn assert_workbook(cell: &str) {
    let snapshot = snapshot();
    assert!(
        parity_close(&at(cell, Deviations::NONE), &snapshot[cell]),
        "{cell}: the workbook-exact switch lost the snapshot value"
    );
}

/// Cells whose value differs between `a` and `b` by the parity rule.
fn changed_cells(a: &BTreeMap<String, Value>, b: &BTreeMap<String, Value>) -> BTreeSet<String> {
    a.iter()
        .filter(|(cell, va)| !parity_close(va, &b[*cell]))
        .map(|(cell, _)| cell.clone())
        .collect()
}

#[test]
fn every_registered_cell_is_in_the_snapshot() {
    let snapshot = snapshot();
    let failures: Vec<String> = REGISTRY
        .iter()
        .flat_map(|d| d.cells.iter().map(move |cell| (d.id, *cell)))
        .filter(|(_, cell)| !snapshot.contains_key(*cell))
        .map(|(id, cell)| format!("{id}: {cell} is not in the snapshot"))
        .collect();
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_entry_is_an_approved_row_of_the_audit_report() {
    let text = read_text(&repo_path(REPORT));
    for d in REGISTRY {
        let prefix = format!("| {} |", d.id);
        let row = text
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{REPORT} has no row for {}", d.id));
        assert!(
            row.contains("| approved (user, 2026-09-29) |"),
            "{} is not approved in the report",
            d.id
        );
        let documentation = row.contains("(documentation)");
        assert_eq!(
            documentation,
            d.class == DeviationClass::Documentation,
            "{}: class vs report",
            d.id
        );
    }
}

#[test]
fn registered_workbook_values_equal_the_snapshot() {
    let snapshot = snapshot();
    let defaults = DesignInputs::default();
    let mut failures = Vec::new();
    for d in REGISTRY {
        for change in d.changes_at_defaults {
            match snapshot.get(change.cell) {
                Some(want) if parity_close(&change.workbook.to_value(), want) => {}
                other => failures.push(format!(
                    "{}: {} workbook value vs snapshot {other:?}",
                    d.id, change.cell
                )),
            }
        }
        for &(path, workbook) in d.workbook_input_defaults {
            match defaults.get(path) {
                None => failures.push(format!("{}: {path} is not an input", d.id)),
                Some(declared) if parity_close(&declared, &workbook.to_value()) => {
                    failures.push(format!(
                        "{}: {path} declares the workbook default; a corrected default must differ",
                        d.id
                    ))
                }
                Some(_) => {}
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn applied_engine_entries_list_their_changes_at_defaults_or_are_default_neutral() {
    // An applied engine correction either changes some default output (listed)
    // or, like E7-E13, changes none at defaults; both are allowed, but a Planned
    // entry must not pretend to change anything (checked in the unit tests).
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for change in d.changes_at_defaults {
            assert!(
                d.cells.contains(&change.cell),
                "{}: {} changes but is not in cells",
                d.id,
                change.cell
            );
        }
    }
}

#[test]
fn each_deviation_alone_changes_exactly_its_registered_cells() {
    let workbook = cell_values(Deviations::NONE);
    let mut failures = Vec::new();
    for id in DeviationId::ALL {
        let d = &REGISTRY[id.index()];
        let corrected = cell_values(Deviations::only(id));
        let changed = changed_cells(&workbook, &corrected);
        let registered: BTreeSet<String> = d
            .changes_at_defaults
            .iter()
            .map(|c| c.cell.to_owned())
            .collect();
        if changed != registered {
            failures.push(format!(
                "{id}: changed {changed:?}, registered {registered:?}"
            ));
        }
        for change in d.changes_at_defaults {
            let got = &corrected[&change.cell.to_owned()];
            if !parity_close(got, &change.corrected.to_value()) {
                failures.push(format!(
                    "{id}: {} = {got:?}, registered {:?}",
                    change.cell, change.corrected
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn all_deviations_together_change_only_registered_cells() {
    let changed = changed_cells(
        &cell_values(Deviations::NONE),
        &cell_values(Deviations::ALL),
    );
    let registered: BTreeSet<String> = REGISTRY
        .iter()
        .flat_map(|d| d.changes_at_defaults.iter().map(|c| c.cell.to_owned()))
        .collect();
    let unexplained: Vec<_> = changed.difference(&registered).collect();
    assert!(
        unexplained.is_empty(),
        "cells changed by no registered deviation: {unexplained:?}"
    );
}

#[test]
fn e1_adhesive_shear_modulus_matches_the_report() {
    let e1 = Deviations::only(DeviationId::E1);
    assert_eq!(at("Temperature design!C96", e1), Value::Num(0.107));
    assert_report(
        "Temperature design!C104",
        &at("Temperature design!C104", e1),
        11.6,
        0.05,
    ); // was 46.1
    assert_report(
        "Temperature design!C105",
        &at("Temperature design!C105", e1),
        6.0,
        0.05,
    ); // was 26.7
    assert_report(
        "Temperature design!C201",
        &at("Temperature design!C201", e1),
        2.6,
        0.05,
    ); // was 11.4
    assert_eq!(
        at("Temperature design!C106", e1),
        Value::Text("Below the lap-shear strength".into())
    );
    assert_eq!(
        at("Temperature design!C202", e1),
        Value::Text("Below the fatigue endurance".into())
    );
    for cell in [
        "Temperature design!C96",
        "Temperature design!C104",
        "Temperature design!C105",
        "Temperature design!C106",
        "Temperature design!C201",
        "Temperature design!C202",
    ] {
        assert_workbook(cell);
    }
}

#[test]
fn reworded_help_is_recorded_for_real_fields() {
    let inputs = input_rows(&DesignInputs::default());
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for &(path, workbook) in d.workbook_help {
            if path.contains("[*]") {
                continue; // table columns: checked against the workbook headers in tests/schema.rs
            }
            let row = inputs
                .iter()
                .find(|r| r.path == path)
                .unwrap_or_else(|| panic!("{}: {path} is not an input", d.id));
            assert_ne!(
                row.meta.help, workbook,
                "{}: {path} help is not reworded",
                d.id
            );
        }
    }
}
