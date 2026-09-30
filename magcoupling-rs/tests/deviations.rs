//! Deviation registry against the workbook snapshot and the audit report, and
//! the switch that turns corrections off.
//!
//! With every correction off the engine reproduces the workbook
//! (`tests/parity.rs`). Here: switching ONE correction on may change only the
//! cells that correction registers, each to its registered corrected value; and
//! switching all on changes nothing outside the registered cells.
//!
//! A broad correction (more than 15 changed cells, decision D4) registers its
//! changes in a golden file (`Deviation::changes_file`) instead of by hand.
//! Rewrite the golden files from this run with
//! `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells`
//! and review the diff.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use common::{json_to_value, read_json, read_text, repo_path, report, snapshot, value_to_json};
use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::{
    Deviation, DeviationClass, DeviationId, DeviationStatus, Deviations, Probe, REGISTRY, REPORT,
};
use magcoupling::engine::meta::{InputSet, Value, input_rows, result_rows};

const BLESS_VAR: &str = "MAGCOUPLING_BLESS";

fn golden_path(file: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(file)
}

/// The cells a correction changes at defaults, with their workbook and corrected
/// values: hand-listed in the registry, or read from the correction's golden file.
fn registered_changes(d: &Deviation) -> BTreeMap<String, (Value, Value)> {
    match d.changes_file {
        None => d
            .changes_at_defaults
            .iter()
            .map(|c| {
                (
                    c.cell.to_owned(),
                    (c.workbook.to_value(), c.corrected.to_value()),
                )
            })
            .collect(),
        Some(file) if !golden_path(file).exists() => BTreeMap::new(),
        Some(file) => read_json(&golden_path(file))["changes"]
            .as_object()
            .expect("a changes object")
            .iter()
            .map(|(cell, pair)| {
                (
                    cell.clone(),
                    (json_to_value(&pair[0]), json_to_value(&pair[1])),
                )
            })
            .collect(),
    }
}

/// Bless mode: writes the golden file of a broad correction from this run.
fn bless_changes(
    d: &Deviation,
    workbook: &BTreeMap<String, Value>,
    corrected: &BTreeMap<String, Value>,
    changed: &BTreeSet<String>,
) {
    let file = d
        .changes_file
        .expect("only golden-file entries are blessed");
    let changes: serde_json::Map<String, serde_json::Value> = changed
        .iter()
        .map(|cell| {
            (
                cell.clone(),
                serde_json::json!([
                    value_to_json(&workbook[cell]),
                    value_to_json(&corrected[cell])
                ]),
            )
        })
        .collect();
    let doc = serde_json::json!({
        "about": format!("Every workbook cell correction {} changes at default inputs, as [workbook, corrected]. \
                          Written by `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells`; review the diff.", d.id),
        "id": d.id.to_string(),
        "changes": changes,
    });
    let path = golden_path(file);
    fs::create_dir_all(path.parent().expect("a parent directory"))
        .expect("create tests/data/deviations");
    fs::write(
        &path,
        serde_json::to_string_pretty(&doc).expect("serializes") + "\n",
    )
    .expect("write the golden file");
}

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

/// The value at one cell for a probe's inputs, applied to the defaults `dev` implies.
fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value {
    let mut inputs = DesignInputs::defaults_with(dev);
    for &(path, value) in probe.inputs {
        inputs
            .set(path, value.to_value())
            .unwrap_or_else(|e| panic!("probe {:?}: {e}", probe.label));
    }
    cell_values_for(&inputs, dev)
        .remove(cell)
        .unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
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
        for (cell, (workbook, _)) in registered_changes(d) {
            match snapshot.get(&cell) {
                Some(want) if parity_close(&workbook, want) => {}
                other => failures.push(format!(
                    "{}: {cell} workbook value {workbook:?} vs snapshot {other:?}",
                    d.id
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
    // Hand-listed changes must be cells the entry names; a golden-file entry
    // (a broad correction) changes cells far downstream of the ones it names.
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied && d.changes_file.is_none())
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
        if d.changes_file.is_some() && std::env::var_os(BLESS_VAR).is_some() {
            bless_changes(d, &workbook, &corrected, &changed);
        }
        let registered = registered_changes(d);
        let keys: BTreeSet<String> = registered.keys().cloned().collect();
        if changed != keys {
            failures.push(format!("{id}: changed {changed:?}, registered {keys:?}"));
        }
        for (cell, (_, want)) in &registered {
            match corrected.get(cell) {
                Some(got) if parity_close(got, want) => {}
                got => failures.push(format!("{id}: {cell} = {got:?}, registered {want:?}")),
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
        .flat_map(|d| registered_changes(d).into_keys())
        .collect();
    let unexplained: Vec<_> = changed.difference(&registered).collect();
    assert!(
        unexplained.is_empty(),
        "cells changed by no registered deviation: {unexplained:?}"
    );
}

#[test]
fn broad_corrections_use_golden_files_and_narrow_ones_list_their_cells() {
    // Decision D4: more than 15 changed cells go to a golden file.
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        match d.changes_file {
            Some(file) => {
                assert!(
                    golden_path(file).exists(),
                    "{}: {file} missing: bless it",
                    d.id
                );
                assert!(
                    d.changes_at_defaults.is_empty(),
                    "{}: golden file and hand list",
                    d.id
                );
                assert!(
                    registered_changes(d).len() > 15,
                    "{}: a golden file for 15 cells or fewer",
                    d.id
                );
            }
            None => assert!(
                d.changes_at_defaults.len() <= 15,
                "{}: more than 15 hand-listed cells",
                d.id
            ),
        }
    }
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
fn e2_clamp_screw_length_matches_the_report() {
    let e2 = Deviations::only(DeviationId::E2);
    assert_eq!(
        at("Shaft clamps!C48", e2),
        Value::Text("ISO 4762 M4 x 14, class 12.9".into())
    );
    assert_eq!(num(&at("Clamp screw sizes!E34", e2)), 14.0);
    assert_eq!(
        num(&at("Clamp screw sizes!E35", e2)),
        0.0,
        "the 14 mm screw protrudes from the 25 mm boss"
    );
    // Screw strength limits the preload, so capacity and safety factor do not change.
    assert_report(
        "Shaft clamps!C52",
        &at("Shaft clamps!C52", e2),
        7.665,
        0.0005,
    );
    assert_report("Shaft clamps!C53", &at("Shaft clamps!C53", e2), 2.04, 0.005);
    let note = |dev| {
        compute_all_with(&DesignInputs::defaults_with(dev), dev)
            .clamps
            .length_note
    };
    assert_eq!(
        note(e2),
        "No 2 mm length step of M4 both engages 8 mm of thread and stays inside the boss; M4 x 14 protrudes 0.34 mm"
    );
    assert_eq!(note(Deviations::NONE), "");
    for cell in [
        "Shaft clamps!C48",
        "Clamp screw sizes!E34",
        "Clamp screw sizes!E35",
    ] {
        assert_workbook(cell);
    }
}

#[test]
fn e3_library_remanence_matches_the_report() {
    let e3 = Deviations::only(DeviationId::E3);
    assert_eq!(at("Calculator!C21", e3), Value::Num(1.30));
    assert_report("Calculator!C93", &at("Calculator!C93", e3), 2.688, 0.0005); // pull-out, was 2.647
    assert_report("Metal design!C9", &at("Metal design!C9", e3), 2.285, 0.0005); // hot low, was 2.250
    assert_eq!(
        at("Metal design!C11", e3),
        Value::Text("Below hot minimum".into())
    );
    assert_report(
        "Metal design!C10",
        &at("Metal design!C10", e3),
        3.823,
        0.0005,
    ); // cold high, was 3.765
    assert_report(
        "Temperature design!C12",
        &at("Temperature design!C12", e3),
        93.06,
        0.005,
    ); // limit, was 92.55
    assert_eq!(
        at("Temperature design!C91", e3),
        Value::Text("OK: 7x margin".into())
    );
    assert_eq!(
        at("Gap sweep!AA9", e3),
        Value::Text("nominal: test needed".into())
    ); // 1.25 mm row
    assert_report("Gap sweep!X9", &at("Gap sweep!X9", e3), 2.500, 0.0005); // was 2.462
    assert_eq!(
        at("Shaft clamps!C48", e3),
        Value::Text("ISO 4762 M4 x 12, class 12.9".into()),
        "below 1.3025 T the clamp still fits"
    );
    for cell in [
        "Calculator!C21",
        "Calculator!C93",
        "Metal design!C9",
        "Temperature design!C12",
        "Temperature design!C91",
        "Gap sweep!AA9",
    ] {
        assert_workbook(cell);
    }
}

#[test]
fn e4_pole_sweep_hub_wall_matches_the_report() {
    let e4 = Deviations::only(DeviationId::E4);
    for cell in ["Pole sweep!C6", "Pole sweep!C7"] {
        assert_report(cell, &at(cell, e4), 9.250, 0.0005); // 6 and 8 poles, was 9.200
    }
    assert_eq!(
        at("Pole sweep!AA6", e4),
        Value::Text("outside OD envelope".into())
    ); // was "below hot minimum"
    assert_report("Pole sweep!G6", &at("Pole sweep!G6", e4), 43.01, 0.005); // cup OD, was 42.90
    assert_report("Pole sweep!X6", &at("Pole sweep!X6", e4), 0.954, 0.0005); // pull-out, was 0.958
    assert_eq!(
        at("Pole sweep!AA7", e4),
        at("Pole sweep!AA7", Deviations::NONE),
        "the 8-pole status does not change"
    );
    for cell in [
        "Pole sweep!C6",
        "Pole sweep!C7",
        "Pole sweep!AA6",
        "Pole sweep!G6",
        "Pole sweep!X6",
    ] {
        assert_workbook(cell);
    }
}

#[test]
fn e5_web_eddy_loss_matches_the_report() {
    let e5 = Deviations::only(DeviationId::E5);
    assert_eq!(at("Temperature design!C121", e5), Value::Num(4.14e-5));
    // The report row truncates the web loss to 0.365; its evidence gives 4 x 0.091401 = 0.3656 W.
    assert_report(
        "Temperature design!C125",
        &at("Temperature design!C125", e5),
        0.3656,
        0.00005,
    ); // web loss, was 0.091 W
    assert_report(
        "Temperature design!C130",
        &at("Temperature design!C130", e5),
        2.751,
        0.0005,
    ); // total, was 2.477 W
    assert_report(
        "Temperature design!C131",
        &at("Temperature design!C131", e5),
        0.0131,
        0.00005,
    ); // drag, was 0.0118
    assert_report(
        "Temperature design!C148",
        &at("Temperature design!C148", e5),
        74.17,
        0.005,
    ); // was 73.26 C
    assert_report(
        "Temperature design!C149",
        &at("Temperature design!C149", e5),
        92.51,
        0.005,
    ); // was 89.77 C
    // No text changes: the high case stays 0.04 C under the 92.55 C limit.
    assert_eq!(
        at("Temperature design!C19", e5),
        Value::Text("never: steady state stays below the limit".into())
    );
    for cell in [
        "Temperature design!C121",
        "Temperature design!C125",
        "Temperature design!C130",
        "Temperature design!C149",
    ] {
        assert_workbook(cell);
    }
}

#[test]
fn e6_cup_wall_at_the_flats_matches_the_report() {
    let e6 = Deviations::only(DeviationId::E6);
    assert_report("Calculator!C63", &at("Calculator!C63", e6), 2.723, 0.0005); // was 2.773
    assert_workbook("Calculator!C63");
}

#[test]
fn each_probe_shows_its_correction() {
    let mut failures = Vec::new();
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for probe in d.probes {
            for change in probe.expect {
                let workbook = at_probe(change.cell, probe, Deviations::NONE);
                if !parity_close(&workbook, &change.workbook.to_value()) {
                    failures.push(format!(
                        "{} {:?}: {} workbook {workbook:?}, registered {:?}",
                        d.id, probe.label, change.cell, change.workbook
                    ));
                }
                let corrected = at_probe(change.cell, probe, Deviations::only(d.id));
                if !parity_close(&corrected, &change.corrected.to_value()) {
                    failures.push(format!(
                        "{} {:?}: {} corrected {corrected:?}, registered {:?}",
                        d.id, probe.label, change.cell, change.corrected
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_applied_engine_correction_is_visible_somewhere() {
    // A correction that changes nothing at defaults must carry a probe, so no applied correction goes untested.
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied && d.class == DeviationClass::Engine)
    {
        let shows_at_defaults = !d.changes_at_defaults.is_empty() || d.changes_file.is_some();
        assert!(
            shows_at_defaults || !d.probes.is_empty(),
            "{} changes nothing at defaults and has no probe",
            d.id
        );
    }
}

#[test]
fn e7_pull_out_over_angle_matches_the_report() {
    let e7 = &REGISTRY[DeviationId::E7.index()];
    assert_eq!(e7.status, DeviationStatus::Applied);
    let six_poles = &e7.probes[0];
    let got = at_probe(
        "Calculator!C93",
        six_poles,
        Deviations::only(DeviationId::E7),
    );
    assert_report("Calculator!C93", &got, 0.911, 0.0005); // 6 poles, steel: was 0.861 N m
    let workbook = at_probe("Calculator!C93", six_poles, Deviations::NONE);
    assert_report("Calculator!C93", &workbook, 0.861, 0.0005);
}

/// Every celled value at the defaults is exactly equal (`Value ==`) with only `id`
/// on and with every correction off. Stronger than the parity rule of
/// each_deviation_alone_changes_exactly_its_registered_cells.
fn assert_bit_for_bit_at_defaults(id: DeviationId) {
    let workbook = cell_values(Deviations::NONE);
    let corrected = cell_values(Deviations::only(id));
    let differ: Vec<&String> = workbook
        .iter()
        .filter(|(cell, v)| corrected.get(*cell) != Some(v))
        .map(|(cell, _)| cell)
        .collect();
    assert!(differ.is_empty(), "{id} moved default cells: {differ:?}");
}

#[test]
fn e7_leaves_every_default_cell_bit_for_bit() {
    // At defaults every row peaks at half a pitch, so E7 keeps the workbook expression exactly.
    assert_bit_for_bit_at_defaults(DeviationId::E7);
}

#[test]
fn e8_leaves_flat_blocks_bit_for_bit() {
    // The defaults use flat blocks (coupling.faceted = 1), where the corner radius C55 is
    // the workbook's own corner expression and the pocket stays a polygon.
    assert_bit_for_bit_at_defaults(DeviationId::E8);
}

#[test]
fn e8_arc_mode_matches_the_report() {
    let e8 = &REGISTRY[DeviationId::E8.index()];
    assert_eq!(e8.status, DeviationStatus::Applied);
    let arcs = &e8.probes[0];
    let (workbook, corrected) = (Deviations::NONE, Deviations::only(DeviationId::E8));
    assert_report(
        "Calculator!C93",
        &at_probe("Calculator!C93", arcs, workbook),
        2.99,
        0.005,
    );
    assert_report(
        "Calculator!C93",
        &at_probe("Calculator!C93", arcs, corrected),
        2.65,
        0.005,
    );
    assert_report(
        "Metal design!C9",
        &at_probe("Metal design!C9", arcs, corrected),
        2.25,
        0.005,
    ); // hot low, was 2.54
    assert_report(
        "Metal design!C35",
        &at_probe("Metal design!C35", arcs, corrected),
        0.270,
        0.0005,
    ); // was -0.476
    assert_report(
        "Metal design!C175",
        &at_probe("Metal design!C175", arcs, corrected),
        26.69,
        0.005,
    ); // sleeve ID, was 27.44
    assert_report(
        "Calculator!C111",
        &at_probe("Calculator!C111", arcs, corrected),
        48.41,
        0.005,
    ); // cup mass, was 42.96
    assert_eq!(
        at_probe("Shaft clamps!C48", arcs, corrected),
        Value::Text("ISO 4762 M4 x 12, class 12.9".into())
    );
}

#[test]
fn e9_no_back_iron_matches_the_report() {
    let e9 = &REGISTRY[DeviationId::E9.index()];
    assert_eq!(e9.status, DeviationStatus::Applied);
    let no_iron = &e9.probes[0];
    let corrected = Deviations::only(DeviationId::E9);
    assert_report(
        "Calculator!C111",
        &at_probe("Calculator!C111", no_iron, corrected),
        20.90,
        0.005,
    ); // cup, was 60.75 g
    assert_report(
        "Calculator!C113",
        &at_probe("Calculator!C113", no_iron, corrected),
        10.59,
        0.005,
    ); // boss, was 30.78 g
    assert_report(
        "Calculator!C114",
        &at_probe("Calculator!C114", no_iron, corrected),
        96.8,
        0.05,
    ); // total, was 156.9 g
    assert_report(
        "Temperature design!C141",
        &at_probe("Temperature design!C141", no_iron, corrected),
        45.8,
        0.05,
    ); // was 74.2 J/K
    assert_eq!(
        at_probe("Materials!C22", no_iron, corrected),
        Value::Text("No back iron".into())
    );
}

#[test]
fn e9_leaves_every_default_cell_bit_for_bit() {
    // The defaults use the steel circuit (coupling.backiron = 1): steel cup and boss, wall advice.
    assert_bit_for_bit_at_defaults(DeviationId::E9);
}

#[test]
fn e10_gap_flux_density_matches_the_report() {
    let e10 = &REGISTRY[DeviationId::E10.index()];
    assert_eq!(e10.status, DeviationStatus::Applied);
    let mixed = &e10.probes[0];
    let corrected = Deviations::only(DeviationId::E10);
    assert_report(
        "Calculator!C103",
        &at_probe("Calculator!C103", mixed, Deviations::NONE),
        1.024,
        0.0005,
    );
    assert_report(
        "Calculator!C103",
        &at_probe("Calculator!C103", mixed, corrected),
        1.043,
        0.0005,
    );
    assert_report(
        "Calculator!C104",
        &at_probe("Calculator!C104", mixed, corrected),
        1.949,
        0.0005,
    ); // was 1.915
}

#[test]
fn e10_leaves_every_default_cell_bit_for_bit() {
    // The defaults use identical rings (B842SH inside and out). Then Br_i t_i + Br_o t_o is
    // 2 fl(Br t) and the workbook's (Br_i + Br_o)/2 (t_i + t_o) is fl(Br 2t): the same double.
    assert_bit_for_bit_at_defaults(DeviationId::E10);
}

#[test]
fn e11_fatigue_screen_matches_the_report() {
    let e11 = &REGISTRY[DeviationId::E11.index()];
    assert_eq!(e11.status, DeviationStatus::Applied);
    let low = &e11.probes[0];
    assert_eq!(
        at_probe("Temperature design!C91", low, Deviations::NONE),
        Value::Text("OK: 8x margin".into())
    );
    assert_eq!(
        at_probe(
            "Temperature design!C91",
            low,
            Deviations::only(DeviationId::E11)
        ),
        Value::Text("CHECK".into())
    ); // margin 3.77
}

#[test]
fn e11_leaves_every_default_cell_bit_for_bit() {
    // At the default endurance (C195 = 0.2) the input and the workbook's typed-in 0.2 are the same double.
    assert_bit_for_bit_at_defaults(DeviationId::E11);
}

#[test]
fn e11_screen_follows_the_endurance_input_and_nothing_else_moves() {
    // The report: the screen flips below an endurance of 0.106 (exactly 4 tau_b / lap shear =
    // 0.10608 at defaults). C195 already fed C196, C197 and C202, so only C91 may change.
    let path = "temperature.adhesive_life.fatigue_endurance";
    let screen = "Temperature design!C91";
    for (endurance, want) in [
        (0.106, "CHECK"),
        (0.107, "OK: 4x margin"),
        (0.6, "OK: 23x margin"),
    ] {
        let at = |dev: Deviations| {
            let mut inputs = DesignInputs::defaults_with(dev);
            inputs
                .set(path, Value::Num(endurance))
                .expect("a valid endurance");
            cell_values_for(&inputs, dev)
        };
        let (workbook, corrected) = (at(Deviations::NONE), at(Deviations::only(DeviationId::E11)));
        assert_eq!(
            workbook[screen],
            Value::Text("OK: 8x margin".into()),
            "{endurance}"
        );
        assert_eq!(corrected[screen], Value::Text(want.into()), "{endurance}");
        assert_eq!(
            changed_cells(&workbook, &corrected),
            BTreeSet::from([screen.to_owned()]),
            "{endurance}"
        );
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
