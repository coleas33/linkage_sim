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

use common::{
    cell_values_for, differential_files, json_to_value, load_cases, num, read_json, read_text,
    repo_path, report, snapshot, value_to_json,
};
use magcoupling::engine::api::{
    DesignInputs, DesignResults, compute_all, compute_all_with, headline,
};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::{
    A4_REPORT, ADDENDUM_REPORT, Approval, Deviation, DeviationClass, DeviationId, DeviationStatus,
    Deviations, Literal, Probe, REGISTRY, REPORT,
};
use magcoupling::engine::meta::{InputSet, NumOrText, Value, input_rows, result_rows};

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

/// `got` equals the audit report's figure `want`, which the report states to
/// within `half_step` (half a unit of its last digit).
fn assert_report(cell: &str, got: &Value, want: f64, half_step: f64) {
    let got = num(got);
    assert!(
        (got - want).abs() <= half_step,
        "{cell}: {got} is not the report's {want} (± {half_step})"
    );
}

/// The defaults `dev` implies with `overrides` applied; panics naming an invalid override.
fn inputs_with(overrides: &[(&str, Value)], dev: Deviations) -> DesignInputs {
    let mut inputs = DesignInputs::defaults_with(dev);
    for (path, value) in overrides {
        inputs
            .set(path, value.clone())
            .unwrap_or_else(|e| panic!("override {path} = {value:?}: {e}"));
    }
    inputs
}

/// A registry probe's inputs as overrides.
fn probe_overrides(probe: &Probe) -> Vec<(&'static str, Value)> {
    probe
        .inputs
        .iter()
        .map(|&(path, value)| (path, value.to_value()))
        .collect()
}

/// Every value with a workbook cell for a probe's inputs, applied to the defaults `dev` implies.
fn probe_cells(probe: &Probe, dev: Deviations) -> BTreeMap<String, Value> {
    cell_values_for(&inputs_with(&probe_overrides(probe), dev), dev)
}

/// The value at one cell for a probe's inputs, applied to the defaults `dev` implies.
fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value {
    probe_cells(probe, dev)
        .remove(cell)
        .unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}

/// The corrections a probe of `d` runs on, on both sides: those `d` refines (decision 15).
fn probe_base(d: &Deviation) -> Deviations {
    d.depends_on
        .iter()
        .fold(Deviations::NONE, |base, &id| base.with(id))
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
fn every_entry_is_approved_in_its_report() {
    let audit = read_text(&repo_path(REPORT));
    let addendum = read_text(&repo_path(ADDENDUM_REPORT));
    // Section 8 of the Addendum A report: the approval line and the numbered decisions.
    let section8 = addendum
        .split_once("\n## 8. Decisions for the user\n")
        .map(|(_, rest)| rest)
        .expect("the Addendum A report has a section 8");
    assert!(
        section8.contains("**Approved (user, 2026-09-30): option A on all 31 decisions.**"),
        "section 8 records the approval"
    );
    // Section 5 of the Addendum A-4 report: its approval line and numbered decisions.
    let a4 = read_text(&repo_path(A4_REPORT));
    let section5 = a4
        .split_once("\n## 5. Decisions for the user\n")
        .map(|(_, rest)| rest)
        .expect("the Addendum A-4 report has a section 5");
    assert!(
        section5.contains("**Approved (user, 2026-10-01): option A on all 10 decisions.**"),
        "section 5 records the approval"
    );
    for d in REGISTRY {
        match d.approval {
            Approval::AuditRow => {
                let prefix = format!("| {} |", d.id);
                let row = audit
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
            Approval::Addendum { decisions } => {
                for n in decisions {
                    let heading = format!("{n}. **");
                    assert!(
                        section8.lines().any(|line| line.starts_with(&heading)),
                        "{}: section 8 has no decision {n}",
                        d.id
                    );
                }
                // E15 to E18 have an audit row ("| E18 (candidate) |" too); its status
                // column names the first decision that approves the entry.
                let prefix = format!("| {} ", d.id);
                if let Some(row) = addendum.lines().find(|line| line.starts_with(&prefix)) {
                    let status = row.trim_end_matches('|').rsplit('|').next().unwrap_or("");
                    assert!(
                        status.contains("decision") && status.contains(&decisions[0].to_string()),
                        "{}: row status {status:?} does not name decision {}",
                        d.id,
                        decisions[0]
                    );
                }
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
            // E21 onward have no audit row: the A-4 report's tables name cells, not statuses.
            Approval::AddendumA4 { decisions } => {
                for n in decisions {
                    let heading = format!("{n}. **");
                    assert!(
                        section5.lines().any(|line| line.starts_with(&heading)),
                        "{}: section 5 has no decision {n}",
                        d.id
                    );
                }
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
        }
    }
}

#[test]
fn e15_to_e18_have_audit_rows_and_e19_e20_decisions_only() {
    let addendum = read_text(&repo_path(ADDENDUM_REPORT));
    for id in DeviationId::ALL.into_iter().skip(14) {
        let prefix = format!("| {id} ");
        let has_row = addendum.lines().any(|line| line.starts_with(&prefix));
        let expected = matches!(
            id,
            DeviationId::E15 | DeviationId::E16 | DeviationId::E17 | DeviationId::E18
        );
        assert_eq!(has_row, expected, "{id}");
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
        // Decision 15: a correction that refines others is probed on top of them.
        let base = probe_base(d);
        for probe in d.probes {
            for change in probe.expect {
                // Always run the workbook side (it must not panic); a workbook error value
                // (`#DIV/0!`, where Python raises) has no Rust counterpart to compare with.
                let workbook = at_probe(change.cell, probe, base);
                if !matches!(change.workbook, Literal::Error(_))
                    && !parity_close(&workbook, &change.workbook.to_value())
                {
                    failures.push(format!(
                        "{} {:?}: {} workbook {workbook:?}, registered {:?}",
                        d.id, probe.label, change.cell, change.workbook
                    ));
                }
                let corrected = at_probe(change.cell, probe, base.with(d.id));
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
fn addendum_entries_name_every_cell_their_probes_change() {
    // The Addendum A and A-4 entries (E15 onward) name every cell their probes change,
    // downstream cells included, so the registry alone says where each correction shows. The
    // M1 entries (E1 to E14) name the corrected and report-named cells only: their probes
    // change hundreds of sweep and downstream cells (E7, E8), and the broad ones keep golden
    // files.
    let mut failures = Vec::new();
    for d in REGISTRY
        .iter()
        .filter(|d| !matches!(d.approval, Approval::AuditRow))
    {
        let base = probe_base(d);
        for probe in d.probes {
            let changed = changed_cells(
                &probe_cells(probe, base),
                &probe_cells(probe, base.with(d.id)),
            );
            for cell in changed.iter().filter(|c| !d.cells.contains(&c.as_str())) {
                failures.push(format!(
                    "{} {:?}: {cell} changes but is not in cells",
                    d.id, probe.label
                ));
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

/// Brute-force maximum over the electrical angle x in [0, π/2] of
/// a1 sin x + a3 sin 3x + a5 sin 5x: a fine grid, then a ternary search around
/// the best grid point. Never above the true maximum.
fn max_over_angle(a: [f64; 3]) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
    let points = 200_001;
    let step = FRAC_PI_2 / (points - 1) as f64;
    let (best_i, best) = (0..points)
        .map(|i| (i, t(i as f64 * step)))
        .max_by(|p, q| p.1.total_cmp(&q.1))
        .expect("points > 0");
    let (mut lo, mut hi) = (
        (best_i as f64 - 1.0).max(0.0) * step,
        ((best_i + 1) as f64 * step).min(FRAC_PI_2),
    );
    for _ in 0..100 {
        let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if t(m1) < t(m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    best.max(t((lo + hi) / 2.0))
}

#[test]
fn e7_finds_the_peak_at_a_fill_of_exactly_0_4() {
    // Final review, E7: a ring at a fill of exactly 0.4 gives sin(5 * 0.4 * pi / 2) ~ 1e-16,
    // a vanishing fifth harmonic, where the textbook quadratic cancelled and E7 silently
    // kept half a pitch (a local minimum here). The harmonic sum with E7 on must be the
    // maximum of the torque-angle curve built from the workbook's half-pitch terms
    // (tau_n = a_n sin(n pi/2), so a = (tau1, -tau3, tau5)).
    use std::f64::consts::PI;
    let e7 = Deviations::only(DeviationId::E7);
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-9 * want.abs();

    // Calculator: 6 poles, no back iron, a manual inner block 0.4 of the inner pitch wide.
    let mut inputs = DesignInputs::defaults_with(e7);
    let c = &mut inputs.coupling;
    c.npole = 6;
    c.backiron = 0;
    c.magnets.part_inner = String::new(); // manual: the width below applies
    let pitch = 2.0 * PI * (c.inner_back_apothem_mm + c.magnets.manual_inner_thickness_mm / 2.0)
        / c.npole as f64;
    c.magnets.manual_inner_width_mm = 0.4 * pitch; // 4.915545 mm
    let off = compute_all_with(&inputs, Deviations::NONE).model;
    let on = compute_all_with(&inputs, e7).model;
    assert!((on.fill_inner - 0.4).abs() < 1e-15, "{}", on.fill_inner);
    let peak = max_over_angle([off.tau1_Pa, -off.tau3_Pa, off.tau5_Pa]);
    assert!(
        on.tau_Pa > 2.0 * off.tau_Pa,
        "{} vs half pitch {}",
        on.tau_Pa,
        off.tau_Pa
    ); // about 32,505 against 13,121 Pa
    assert!(close(on.tau_Pa, peak), "{} vs peak {peak}", on.tau_Pa);
    assert!(close(
        on.pullout_Nm / off.pullout_Nm,
        on.tau_Pa / off.tau_Pa
    ));
    // The no-iron circuit sum (C96) finds the same peak: it is the circuit backiron selects.
    assert!(close(on.pullout_noiron_Nm, on.pullout_Nm));

    // Calibration: 12 magnets (6 poles per ring), the block 0.4 of the inner pitch wide.
    let mut inputs = DesignInputs::defaults_with(e7);
    let cal = &mut inputs.calibration;
    cal.total_magnets = 12;
    let pitch = 2.0 * PI * (cal.apothem_mm + cal.magnet_thickness_mm / 2.0) / 6.0;
    cal.magnet_width_mm = 0.4 * pitch;
    let off = compute_all_with(&inputs, Deviations::NONE).calibration;
    let on = compute_all_with(&inputs, e7).calibration;
    assert!((on.fill_inner - 0.4).abs() < 1e-15, "{}", on.fill_inner);
    let peak = max_over_angle([off.tau1_Pa, -off.tau3_Pa, off.tau5_Pa]);
    let sum = on.tau1_Pa + on.tau3_Pa + on.tau5_Pa;
    assert!(
        sum > off.tau1_Pa + off.tau3_Pa + off.tau5_Pa,
        "{sum} not above half a pitch"
    );
    assert!(close(sum, peak), "{sum} vs peak {peak}");
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
fn e12_start_above_the_limit_matches_the_report() {
    let e12 = &REGISTRY[DeviationId::E12.index()];
    assert_eq!(e12.status, DeviationStatus::Applied);
    let hot = &e12.probes[0];
    let (workbook, corrected) = (Deviations::NONE, Deviations::only(DeviationId::E12));
    assert_report(
        "Temperature design!C150",
        &at_probe("Temperature design!C150", hot, workbook),
        -25.8,
        0.05,
    );
    assert_report(
        "Temperature design!C152",
        &at_probe("Temperature design!C152", hot, workbook),
        -71.2,
        0.05,
    );
    assert_report(
        "Temperature design!C151",
        &at_probe("Temperature design!C151", hot, workbook),
        -861.0,
        0.5,
    );
    assert_report(
        "Temperature design!C153",
        &at_probe("Temperature design!C153", hot, workbook),
        -0.0035,
        0.00005,
    );
    for cell in [
        "Temperature design!C19",
        "Temperature design!C23",
        "Temperature design!C150",
        "Temperature design!C151",
        "Temperature design!C152",
        "Temperature design!C153",
    ] {
        assert_eq!(num(&at_probe(cell, hot, corrected)), 0.0, "{cell}");
    }
}

#[test]
fn e12_leaves_every_default_cell_bit_for_bit() {
    // At defaults the hot-day start (65 C) is 27.55 C under the 92.55 C limit: the guard is off
    // and max(0, T_lim - T0) is T_lim - T0.
    assert_bit_for_bit_at_defaults(DeviationId::E12);
}

#[test]
fn e12_changes_its_six_cells_to_zero_above_the_limit_and_nothing_below() {
    // Cases checked against a patched-Python rerun (the guard and max(0, .) put into
    // temperature.compute in memory): each case above the limit changes exactly these six
    // fields and nothing else, and the case just below changes nothing.
    let cells = [
        "Temperature design!C19",
        "Temperature design!C23",
        "Temperature design!C150",
        "Temperature design!C151",
        "Temperature design!C152",
        "Temperature design!C153",
    ];
    let run = |overrides: &[(&str, Value)], dev: Deviations| {
        let mut inputs = DesignInputs::defaults_with(dev);
        for (path, value) in overrides {
            inputs.set(path, value.clone()).expect("a valid input");
        }
        cell_values_for(&inputs, dev)
    };
    let e12 = Deviations::only(DeviationId::E12);
    let rise = "temperature.duty.driving_rise_C";
    // Just below: a 92.5 C start under the 92.55 C limit.
    let below = [(rise, Value::Num(37.5))];
    assert!(changed_cells(&run(&below, Deviations::NONE), &run(&below, e12)).is_empty());
    // (label, inputs, the workbook's time to the limit C150 there)
    let above = [
        (
            "slider maximum: 115 C start",
            vec![(rise, Value::Num(60.0))],
            Value::Num(-176.76221972381924),
        ),
        (
            "DP460 at defaults: 65 C start above its 60 C limit",
            vec![("temperature.adhesive.selected", Value::Int(4))],
            Value::Num(-50.37429883884748),
        ),
        (
            // Python takes a negative bench drag; the steady state then sits below the limit
            // and the workbook reads "never": the guard is tested first.
            "95 C start, bench drag -0.01 N m",
            vec![
                (rise, Value::Num(40.0)),
                ("metal.measured_drag_Nm", Value::Num(-0.01)),
            ],
            Value::Text("never: steady state stays below the limit".into()),
        ),
    ];
    let wanted: BTreeSet<String> = cells.iter().map(|c| (*c).to_owned()).collect();
    for (label, overrides, workbook_c150) in &above {
        let (workbook, corrected) = (run(overrides, Deviations::NONE), run(overrides, e12));
        assert_eq!(
            &workbook["Temperature design!C150"], workbook_c150,
            "{label}"
        );
        assert_eq!(changed_cells(&workbook, &corrected), wanted, "{label}");
        for cell in cells {
            assert_eq!(corrected[cell], Value::Num(0.0), "{label}: {cell}");
        }
    }
}

#[test]
fn e13_zero_drag_matches_the_report() {
    let e13 = &REGISTRY[DeviationId::E13.index()];
    assert_eq!(e13.status, DeviationStatus::Applied);
    let (zero, neg_zero) = (&e13.probes[0], &e13.probes[1]);
    for cell in ["Temperature design!C156", "Temperature design!C157"] {
        assert_eq!(
            at_probe(cell, zero, Deviations::only(DeviationId::E13)),
            Value::Num(f64::INFINITY),
            "{cell}"
        );
        // +0.0 already gives +inf without the correction; -0.0 proves the guard does something:
        // without it p_use = ke = -0.0 and rev_s / -0.0 = -inf; the guard (-0.0 == 0.0) gives +inf.
        assert_eq!(
            at_probe(cell, neg_zero, Deviations::NONE),
            Value::Num(f64::NEG_INFINITY),
            "{cell}: E13 off"
        );
        assert_eq!(
            at_probe(cell, neg_zero, Deviations::only(DeviationId::E13)),
            Value::Num(f64::INFINITY),
            "{cell}: E13 on"
        );
    }
}

#[test]
fn e13_leaves_every_default_cell_bit_for_bit() {
    // At defaults no drag is measured: the estimated heating power (2.48 W) is not 0, so the
    // guard is off and the rotations per degree keep the workbook's rev_s / rate.
    assert_bit_for_bit_at_defaults(DeviationId::E13);
}

/// `got` is the Addendum A report's figure `want`, printed to 4 significant
/// figures (report section 5.1): within half a unit of the 4th figure.
fn assert_sig4(what: &str, got: &Value, want: f64) {
    let got = num(got);
    let half = 0.5 * 10f64.powi(want.abs().log10().floor() as i32 - 3);
    assert!(
        (got - want).abs() <= half * (1.0 + 1e-9),
        "{what}: {got} is not the report's {want} (4 s.f.)"
    );
}

/// Every celled value for the defaults with `overrides` applied, under `dev`.
fn cells_with(overrides: &[(&str, Value)], dev: Deviations) -> BTreeMap<String, Value> {
    cell_values_for(&inputs_with(overrides, dev), dev)
}

/// A report table of changed cells: (cell, before, after) at 4 significant figures.
type ReportRows<'a> = &'a [(&'a str, f64, f64)];

/// `before` -> `after` changes exactly the report's cells, to the report's figures.
fn assert_report_table(
    label: &str,
    overrides: &[(&str, Value)],
    before: Deviations,
    after: Deviations,
    rows: ReportRows,
) {
    let (b, a) = (cells_with(overrides, before), cells_with(overrides, after));
    let want: BTreeSet<String> = rows.iter().map(|(c, _, _)| (*c).to_owned()).collect();
    assert_eq!(changed_cells(&b, &a), want, "{label}: changed cells");
    for &(cell, was, now) in rows {
        assert_sig4(&format!("{label} {cell} before"), &b[cell], was);
        assert_sig4(&format!("{label} {cell} after"), &a[cell], now);
    }
}

const NO_BACK_IRON: [(&str, Value); 1] = [("coupling.backiron", Value::Int(0))];

/// The report's M2 basis: E1 to E14 on, the Addendum corrections off.
fn m2() -> Deviations {
    [
        DeviationId::E15,
        DeviationId::E16,
        DeviationId::E17,
        DeviationId::E18,
        DeviationId::E19,
        DeviationId::E20,
        DeviationId::E21,
        DeviationId::E24,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
}

#[test]
fn e15_heat_capacity_matches_the_report() {
    // Report 5.4, E15, all three columns: workbook + E9 (the probe basis, decision 15), M2
    // (every other correction on) and standalone (E9 off: only the hub moves).
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e15 = DeviationId::E15;
    let main: ReportRows = &[
        ("Temperature design!C20", 66.01, 65.92),
        ("Temperature design!C141", 45.78, 63.02),
        ("Temperature design!C143", 152.6, 210.1),
        ("Temperature design!C145", 0.005411, 0.003931),
        ("Temperature design!C154", 0.05411, 0.03931),
        ("Temperature design!C155", 0.1623, 0.1179),
        ("Temperature design!C156", 616.1, 848.0),
        ("Temperature design!C157", 205.4, 282.7),
        ("Temperature design!C158", 5087.0, 7002.0),
        ("Temperature design!C159", 457.8, 630.2),
        ("Temperature design!C160", 1.526e4, 2.101e4),
        ("Temperature design!C161", 65.32, 65.23),
        ("Temperature design!C171", 0.005411, 0.003931),
        ("Temperature design!C172", 0.01623, 0.01179),
        ("Temperature design!C180", 66.01, 65.92),
        ("Temperature design!C181", 36.54, 36.63),
        ("Temperature design!C182", 26.54, 26.63),
        ("Temperature design!C186", 1.63, 1.631),
        ("Temperature design!C189", 66.01, 65.92),
        ("Temperature design!C190", 53.99, 54.08),
        ("Temperature design!C192", 1.875, 1.875),
        ("Temperature design!C193", 0.1981, 0.1982),
        ("Temperature design!C196", 7.571, 7.569),
    ];
    assert_report_table("E15 on E9", &NO_BACK_IRON, e9, e9.with(e15), main);
    let m2_rows: ReportRows = &[
        ("Temperature design!C20", 66.12, 66.02),
        ("Temperature design!C141", 45.78, 63.02),
        ("Temperature design!C143", 152.6, 210.1),
        ("Temperature design!C145", 0.00601, 0.004366),
        ("Temperature design!C154", 0.0601, 0.04366),
        ("Temperature design!C155", 0.1803, 0.131),
        ("Temperature design!C156", 554.7, 763.5),
        ("Temperature design!C157", 184.9, 254.5),
        ("Temperature design!C158", 5087.0, 7002.0),
        ("Temperature design!C159", 457.8, 630.2),
        ("Temperature design!C160", 1.526e4, 2.101e4),
        ("Temperature design!C161", 65.36, 65.26),
        ("Temperature design!C171", 0.00601, 0.004366),
        ("Temperature design!C172", 0.01803, 0.0131),
        ("Temperature design!C180", 66.12, 66.02),
        ("Temperature design!C181", 36.93, 37.03),
        ("Temperature design!C182", 26.93, 27.03),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.12, 66.02),
        ("Temperature design!C190", 53.88, 53.98),
        ("Temperature design!C192", 1.874, 1.875),
        ("Temperature design!C193", 0.1981, 0.1981),
        ("Temperature design!C196", 7.573, 7.571),
    ];
    assert_report_table("E15 on M2", &NO_BACK_IRON, m2(), m2().with(e15), m2_rows);
    let standalone: ReportRows = &[
        ("Temperature design!C20", 65.89, 65.88),
        ("Temperature design!C141", 74.19, 77.98),
        ("Temperature design!C143", 247.3, 259.9),
        ("Temperature design!C145", 0.003339, 0.003177),
        ("Temperature design!C154", 0.03339, 0.03177),
        ("Temperature design!C155", 0.1002, 0.0953),
        ("Temperature design!C156", 998.3, 1049.0),
        ("Temperature design!C157", 332.8, 349.8),
        ("Temperature design!C158", 8243.0, 8664.0),
        ("Temperature design!C159", 741.9, 779.8),
        ("Temperature design!C160", 2.473e4, 2.599e4),
        ("Temperature design!C161", 65.2, 65.19),
        ("Temperature design!C171", 0.003339, 0.003177),
        ("Temperature design!C172", 0.01002, 0.00953),
        ("Temperature design!C180", 65.89, 65.88),
        ("Temperature design!C181", 36.66, 36.67),
        ("Temperature design!C182", 26.66, 26.67),
        ("Temperature design!C186", 1.631, 1.631),
        ("Temperature design!C189", 65.89, 65.88),
        ("Temperature design!C190", 54.11, 54.12),
        ("Temperature design!C192", 1.876, 1.876),
        ("Temperature design!C193", 0.1982, 0.1982),
        ("Temperature design!C196", 7.569, 7.569),
    ];
    assert_report_table(
        "E15 alone",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e15),
        standalone,
    );
    // C19 and C150 stay "never" in every column; C137 (the steel specific heat shown) does not move.
    for dev in [e9.with(e15), m2().with(e15), Deviations::only(e15)] {
        let c = cells_with(&NO_BACK_IRON, dev);
        assert_eq!(
            c["Temperature design!C19"],
            Value::Text("never: steady state stays below the limit".into())
        );
        assert_eq!(c["Temperature design!C137"], Value::Num(473.0));
    }
}

#[test]
fn e16_removed_disc_matches_the_report() {
    // Report 5.4, E16: the same four cells in the main and M2 columns; standalone (E9 off:
    // the web is steel) nothing moves. C147, C188, C47, C114 and C192 do not change.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e16 = DeviationId::E16;
    let rows: ReportRows = &[
        ("Metal design!C148", 101.5, 103.8),
        ("Metal design!C149", -4.689, -6.954),
        ("Metal design!C189", 3.453, 1.188),
        ("Metal design!C191", 101.5, 103.8),
    ];
    assert_report_table("E16 on E9", &NO_BACK_IRON, e9, e9.with(e16), rows);
    assert_report_table("E16 on M2", &NO_BACK_IRON, m2(), m2().with(e16), rows);
    assert_report_table(
        "E16 alone",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e16),
        &[],
    );
    // Full precision (report 5.4): the disc at 2.7 g/cm³.
    let c = cells_with(&NO_BACK_IRON, e9.with(e16));
    assert_eq!(c["Metal design!C189"], Value::Num(1.1875220230569419));
    assert_eq!(c["Metal design!C191"], Value::Num(103.76539290918065));
    assert_eq!(c["Metal design!C149"], Value::Num(-6.95366329618038));
}

#[test]
fn e17_aluminium_eddy_losses_match_the_report() {
    // Report 5.4, E17 (T1, end factor C114 = 0.7, the 4-s.f. free-space fields): 41 cells in
    // the main column (workbook + E9, decision 15) and the M2 column. C152 stays "never" and
    // no verdict text changes.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e17 = DeviationId::E17;
    let main: ReportRows = &[
        ("Temperature design!C17", 73.26, 74.73),
        ("Temperature design!C18", 89.77, 94.18),
        ("Temperature design!C20", 66.01, 66.19),
        ("Temperature design!C22", 0.6881, 0.8105),
        ("Temperature design!C123", 0.2259, 0.1942),
        ("Temperature design!C124", 1.353, 1.543),
        ("Temperature design!C125", 0.0914, 0.3737),
        ("Temperature design!C130", 2.477, 2.918),
        ("Temperature design!C131", 0.01183, 0.01393),
        ("Temperature design!C132", 2.477, 2.918),
        ("Temperature design!C134", 7.431, 8.754),
        ("Temperature design!C145", 0.005411, 0.006374),
        ("Temperature design!C146", 8.257, 9.726),
        ("Temperature design!C147", 24.77, 29.18),
        ("Temperature design!C148", 73.26, 74.73),
        ("Temperature design!C149", 89.77, 94.18),
        ("Temperature design!C154", 0.05411, 0.06374),
        ("Temperature design!C155", 0.1623, 0.1912),
        ("Temperature design!C156", 616.1, 523.0),
        ("Temperature design!C157", 205.4, 174.3),
        ("Temperature design!C161", 65.32, 65.38),
        ("Temperature design!C169", 0.2477, 0.2918),
        ("Temperature design!C170", 0.7431, 0.8754),
        ("Temperature design!C171", 0.005411, 0.006374),
        ("Temperature design!C172", 0.01623, 0.01912),
        ("Temperature design!C173", 14.86, 17.51),
        ("Temperature design!C175", 0.2294, 0.2702),
        ("Temperature design!C176", 0.6881, 0.8105),
        ("Temperature design!C177", 0.2477, 0.2918),
        ("Temperature design!C180", 66.01, 66.19),
        ("Temperature design!C181", 36.54, 36.36),
        ("Temperature design!C182", 26.54, 26.36),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.01, 66.19),
        ("Temperature design!C190", 53.99, 53.81),
        ("Temperature design!C192", 1.875, 1.874),
        ("Temperature design!C193", 0.1981, 0.198),
        ("Temperature design!C196", 7.571, 7.575),
    ];
    // C19, C150 and C151 are text ("never") before and numbers after: checked below.
    let text_cells = [
        "Temperature design!C19",
        "Temperature design!C150",
        "Temperature design!C151",
    ];
    let check = |label: &str,
                 before: Deviations,
                 after: Deviations,
                 rows: ReportRows,
                 t19: f64,
                 r151: f64| {
        let (b, a) = (
            cells_with(&NO_BACK_IRON, before),
            cells_with(&NO_BACK_IRON, after),
        );
        let mut want: BTreeSet<String> = rows.iter().map(|(c, _, _)| (*c).to_owned()).collect();
        want.extend(text_cells.iter().map(|c| (*c).to_owned()));
        assert_eq!(changed_cells(&b, &a), want, "{label}: changed cells");
        for &(cell, was, now) in rows {
            assert_sig4(&format!("{label} {cell} before"), &b[cell], was);
            assert_sig4(&format!("{label} {cell} after"), &a[cell], now);
        }
        for cell in ["Temperature design!C19", "Temperature design!C150"] {
            assert_eq!(
                b[cell],
                Value::Text("never: steady state stays below the limit".into()),
                "{label} {cell}"
            );
            assert_sig4(&format!("{label} {cell}"), &a[cell], t19);
        }
        assert_eq!(b["Temperature design!C151"], Value::Text("never".into()));
        assert_sig4(
            &format!("{label} C151"),
            &a["Temperature design!C151"],
            r151,
        );
        assert_eq!(
            a["Temperature design!C152"],
            Value::Text("never: steady state stays below the limit".into()),
            "{label}: the estimate never reaches the limit"
        );
    };
    check("E17 on E9", e9, e9.with(e17), main, 440.3, 1.468e4);
    let m2_rows: ReportRows = &[
        ("Temperature design!C17", 74.17, 74.73),
        ("Temperature design!C18", 92.51, 94.18),
        ("Temperature design!C20", 66.12, 66.19),
        ("Temperature design!C22", 0.7643, 0.8105),
        ("Temperature design!C123", 0.2259, 0.1942),
        ("Temperature design!C124", 1.353, 1.543),
        ("Temperature design!C125", 0.3656, 0.3737),
        ("Temperature design!C130", 2.751, 2.918),
        ("Temperature design!C131", 0.01314, 0.01393),
        ("Temperature design!C132", 2.751, 2.918),
        ("Temperature design!C134", 8.254, 8.754),
        ("Temperature design!C145", 0.00601, 0.006374),
        ("Temperature design!C146", 9.171, 9.726),
        ("Temperature design!C147", 27.51, 29.18),
        ("Temperature design!C148", 74.17, 74.73),
        ("Temperature design!C149", 92.51, 94.18),
        ("Temperature design!C154", 0.0601, 0.06374),
        ("Temperature design!C155", 0.1803, 0.1912),
        ("Temperature design!C156", 554.7, 523.0),
        ("Temperature design!C157", 184.9, 174.3),
        ("Temperature design!C161", 65.36, 65.38),
        ("Temperature design!C169", 0.2751, 0.2918),
        ("Temperature design!C170", 0.8254, 0.8754),
        ("Temperature design!C171", 0.00601, 0.006374),
        ("Temperature design!C172", 0.01803, 0.01912),
        ("Temperature design!C173", 16.51, 17.51),
        ("Temperature design!C175", 0.2548, 0.2702),
        ("Temperature design!C176", 0.7643, 0.8105),
        ("Temperature design!C177", 0.2751, 0.2918),
        ("Temperature design!C180", 66.12, 66.19),
        ("Temperature design!C181", 36.93, 36.87),
        ("Temperature design!C182", 26.93, 26.87),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.12, 66.19),
        ("Temperature design!C190", 53.88, 53.81),
        ("Temperature design!C192", 1.874, 1.874),
        ("Temperature design!C193", 0.1981, 0.198),
        ("Temperature design!C196", 7.573, 7.575),
    ];
    check("E17 on M2", m2(), m2().with(e17), m2_rows, 497.0, 1.657e4);

    // Full precision, workbook + E9 basis, with the 4-s.f. fields pinned (report 5.4, decision 12).
    let c = cells_with(&NO_BACK_IRON, e9.with(e17));
    for (cell, want) in [
        ("Temperature design!C123", 0.1942432063711941),
        ("Temperature design!C124", 1.5432858065214725),
        ("Temperature design!C125", 0.3736960055840505),
        ("Temperature design!C130", 2.917946124198304),
        ("Temperature design!C18", 94.17946124198303),
        ("Temperature design!C19", 440.3079945062734),
    ] {
        assert!(
            parity_close(&c[cell], &Value::Num(want)),
            "{cell}: {:?} vs {want}",
            c[cell]
        );
    }

    // Standalone (E9 off, amended): the cup and web stay steel; the aluminium hub sees half
    // the doubled steel-circuit field, C116 / 2 = 0.1035 T (report 5.6, correction 1).
    let alone = cells_with(&NO_BACK_IRON, Deviations::only(e17));
    let workbook = cells_with(&NO_BACK_IRON, Deviations::NONE);
    assert_sig4("E17 alone C123", &alone["Temperature design!C123"], 0.3392);
    assert_sig4("E17 alone C130", &alone["Temperature design!C130"], 2.590);
    assert_sig4("E17 alone C18", &alone["Temperature design!C18"], 90.90);
    for cell in ["Temperature design!C124", "Temperature design!C125"] {
        assert_eq!(
            alone[cell], workbook[cell],
            "{cell}: the steel cup and web keep the workbook formula"
        );
    }
}

#[test]
fn e15_to_e17_together_match_the_reports_headline_table() {
    // Report 5.2 at back iron = 0: the combined columns. No correction changes the torque,
    // the governing limit or the verdict C25.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let all3 = |base: Deviations| {
        base.with(DeviationId::E15)
            .with(DeviationId::E16)
            .with(DeviationId::E17)
    };
    let c = cells_with(&NO_BACK_IRON, all3(e9));
    assert_sig4("C18", &c["Temperature design!C18"], 94.18);
    assert_sig4("C19", &c["Temperature design!C19"], 606.1);
    assert_sig4("C151", &c["Temperature design!C151"], 2.020e4);
    assert_sig4("C20", &c["Temperature design!C20"], 66.09);
    let m = cells_with(&NO_BACK_IRON, all3(m2()));
    assert_sig4("M2 C19", &m["Temperature design!C19"], 684.1);
    assert_sig4("M2 C151", &m["Temperature design!C151"], 2.280e4);
    assert_sig4("M2 C20", &m["Temperature design!C20"], 66.09);
    assert_sig4("M2 C12", &m["Temperature design!C12"], 93.06);
    assert_sig4("M2 C23", &m["Temperature design!C23"], 0.04019);
    for (label, cells) in [("E9", &c), ("M2", &m)] {
        assert_sig4(&format!("{label} C93"), &cells["Calculator!C93"], 1.697);
        assert_eq!(
            cells["Temperature design!C152"],
            Value::Text("never: steady state stays below the limit".into())
        );
        assert_eq!(
            cells["Temperature design!C25"],
            Value::Text(
                "OK on temperature. Confirm drag torque and thermal cycling by test.".into()
            )
        );
    }
}

#[test]
fn e18_aluminium_hub_mismatch_matches_the_report() {
    // Report row E18. On the workbook basis (C96 = 0.55 GPa) the screens already read
    // "Above": only the numbers move. On the M2 basis (E1's 0.107 GPa) both verdicts flip.
    let e18 = DeviationId::E18;
    let workbook: ReportRows = &[
        ("Temperature design!C104", 46.13, 73.87),
        ("Temperature design!C105", 26.73, 45.1),
        ("Temperature design!C201", 11.37, 19.18),
    ];
    assert_report_table(
        "E18 workbook",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e18),
        workbook,
    );
    let (b, a) = (
        cells_with(&NO_BACK_IRON, m2()),
        cells_with(&NO_BACK_IRON, m2().with(e18)),
    );
    let flips = [
        (
            "Temperature design!C106",
            "Below the lap-shear strength",
            "Above the lap-shear strength at the block ends",
        ),
        (
            "Temperature design!C202",
            "Below the fatigue endurance",
            "Above the fatigue endurance: qualify by thermal cycling",
        ),
    ];
    let mut want: BTreeSet<String> = ["C104", "C105", "C201"]
        .iter()
        .map(|c| format!("Temperature design!{c}"))
        .collect();
    want.extend(flips.iter().map(|(c, _, _)| (*c).to_owned()));
    assert_eq!(changed_cells(&b, &a), want, "E18 on M2: changed cells");
    for (cell, was, now) in [
        ("Temperature design!C104", 11.68, 20.76),
        ("Temperature design!C105", 6.071, 11.03),
        ("Temperature design!C201", 2.563, 4.655),
    ] {
        assert_sig4(&format!("M2 {cell} before"), &b[cell], was);
        assert_sig4(&format!("M2 {cell} after"), &a[cell], now);
    }
    for (cell, was, now) in flips {
        assert_eq!(b[cell], Value::Text(was.into()), "{cell}");
        assert_eq!(a[cell], Value::Text(now.into()), "{cell}");
    }
    // C94 and C98 still show the steel inputs.
    assert_eq!(a["Temperature design!C94"], Value::Num(12.3e-6));
    assert_eq!(a["Temperature design!C98"], Value::Num(205.0));
    // What users see: every correction on, less E18, against every correction on.
    let (without, with) = (
        cells_with(&NO_BACK_IRON, Deviations::ALL.without(e18)),
        cells_with(&NO_BACK_IRON, Deviations::ALL),
    );
    assert_eq!(changed_cells(&without, &with), want, "E18 on ALL");
}

#[test]
fn e19_supermagnetman_arcs_follow_the_vendor_grid() {
    // Decision 2 A: the vendor's 60 C on all three arcs moves every rating-calibrated demag
    // cell by the rating change (the calibration offset absorbs it), and nothing else: 26
    // cells on the workbook basis (the E12 guard zeroes 6 of them on the M2 basis). The
    // temperature checks C107 and C108 stay "OK" at 50 C.
    let e19 = DeviationId::E19;
    for (part, stored) in [("M5044", 80.0), ("M5045", 100.0), ("M5026", 80.0)] {
        let rings = [
            ("coupling.magnets.part_inner", Value::Text(part.into())),
            ("coupling.magnets.part_outer", Value::Text(part.into())),
        ];
        for (basis, before, count) in [("workbook", Deviations::NONE, 26), ("M2", m2(), 20)] {
            let (b, a) = (
                cells_with(&rings, before),
                cells_with(&rings, before.with(e19)),
            );
            let changed = changed_cells(&b, &a);
            assert_eq!(changed.len(), count, "{part} {basis}: {changed:?}");
            assert_eq!(b["Calculator!C22"], Value::Num(stored), "{part}");
            assert_eq!(a["Calculator!C22"], Value::Num(60.0), "{part}");
            let shift = num(&b["Temperature design!C12"]) - num(&a["Temperature design!C12"]);
            assert!(
                (shift - (stored - 60.0)).abs() < 1e-9,
                "{part} {basis}: C12 shift {shift}"
            );
            for cell in ["Calculator!C107", "Calculator!C108"] {
                assert_eq!(a[cell], Value::Text("OK".into()), "{part} {cell}");
            }
        }
    }
    // M5045 maps to the grid's N50 (read by E20); its Br stays the workbook's 1.42 T.
    let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
    inputs.coupling.magnets.part_inner = "M5045".into();
    let off = compute_all_with(&inputs, Deviations::NONE).model;
    let on = compute_all_with(&inputs, Deviations::only(e19)).model;
    assert_eq!(
        (off.inner_grade.as_str(), on.inner_grade.as_str()),
        ("N50M", "N50")
    );
    assert_eq!((off.inner_br_T, on.inner_br_T), (1.42, 1.42));
}

/// The demagnetization results for `overrides` under `dev`.
fn demag_with(
    overrides: &[(&str, Value)],
    dev: Deviations,
) -> magcoupling::engine::temperature::DemagResults {
    compute_all_with(&inputs_with(overrides, dev), dev)
        .temperature
        .demag
}

fn both_rings(part: &str) -> [(&'static str, Value); 2] {
    [
        ("coupling.magnets.part_inner", Value::Text(part.into())),
        ("coupling.magnets.part_outer", Value::Text(part.into())),
    ]
}

#[test]
fn e20_each_part_uses_its_own_coercivity() {
    // Report 6.3, governing limit C12 with the part's own Hcj and beta (E20 alone: the
    // stored ratings, M5045 as the workbook's N50M). The N42SH parts do not move: the grade
    // keeps the workbook's 1592 kA/m and -0.005 /C (decisions 17 A, 18 A).
    let e20 = DeviationId::E20;
    for (part, workbook, corrected) in [
        ("B842", 23.06, -3.52),
        ("B822", 23.06, -3.52),
        ("B862", 23.06, -3.52),
        ("B882", 23.06, -3.52),
        ("B861", 23.06, -3.52),
        ("B881", 23.06, -3.52),
        ("B442", 23.06, -3.52),
        ("B842-N52", 30.73, 0.17),
        ("B882-N52", 30.73, 0.17),
        ("M5044", 29.18, -2.50),
        ("M5045", 49.18, 42.50),
        ("M5026", 29.18, -2.50),
        ("B842SH", 92.55, 92.55),
        ("BX042SH", 92.55, 92.55),
        ("BX082SH", 92.55, 92.55),
    ] {
        let rings = both_rings(part);
        let (b, a) = (
            cells_with(&rings, Deviations::NONE),
            cells_with(&rings, Deviations::only(e20)),
        );
        let limit = |c: &BTreeMap<String, Value>| num(&c["Temperature design!C12"]);
        assert!(
            (limit(&b) - workbook).abs() <= 0.005,
            "{part}: {}",
            limit(&b)
        );
        assert!(
            (limit(&a) - corrected).abs() <= 0.005,
            "{part}: {}",
            limit(&a)
        );
    }
    // B842 changes 24 cells (report 6.3) and shows the N42 grade's Hcj and beta.
    let rings = both_rings("B842");
    let changed = changed_cells(
        &cells_with(&rings, Deviations::NONE),
        &cells_with(&rings, Deviations::only(e20)),
    );
    assert_eq!(changed.len(), 24, "{changed:?}");
    let demag = demag_with(&rings, Deviations::only(e20));
    assert_eq!(
        (demag.hcj20_used_kA_m, demag.beta_used_per_C),
        (954.9, -0.0062)
    );
    let workbook = demag_with(&rings, Deviations::NONE);
    assert_eq!(
        (workbook.hcj20_used_kA_m, workbook.beta_used_per_C),
        (1592.0, -0.005)
    );
    // With E19 too, M5045 is N50 (the vendor grid): Hcj 875.4, beta -0.62 %/C.
    let m5045 = demag_with(&both_rings("M5045"), Deviations::ALL);
    assert_eq!(
        (m5045.hcj20_used_kA_m, m5045.beta_used_per_C),
        (875.4, -0.0062)
    );
    let n50m = demag_with(&both_rings("M5045"), Deviations::only(e20));
    assert_eq!(
        (n50m.hcj20_used_kA_m, n50m.beta_used_per_C),
        (1114.1, -0.00675)
    );
}

#[test]
fn e20_the_hcj_and_beta_inputs_override_the_grade_when_selected() {
    // Decision 19: C44 and C45 stay as overrides that win when set: the coercivity source 0.
    let mut rings = both_rings("B842").to_vec();
    rings.push(("temperature.demag.coercivity_source", Value::Int(0)));
    let e20 = Deviations::only(DeviationId::E20);
    assert!(
        changed_cells(
            &cells_with(&rings, Deviations::NONE),
            &cells_with(&rings, e20)
        )
        .is_empty()
    );
    rings.push(("temperature.demag.hcj20_kA_m", Value::Num(954.9)));
    rings.push(("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)));
    let typed = cells_with(&rings, e20);
    let graded = cells_with(&both_rings("B842"), e20);
    assert_eq!(
        typed["Temperature design!C12"], graded["Temperature design!C12"],
        "the grade's values typed into C44 and C45 give the same limit"
    );
    // Manual magnets without a grade always use the inputs (both rings manual: a library
    // ring beside a manual one is checked with its own grade and rating, A13).
    let manual = both_rings("");
    assert!(
        changed_cells(
            &cells_with(&manual, Deviations::NONE),
            &cells_with(&manual, e20)
        )
        .is_empty()
    );
}

#[test]
fn e20_ferrite_is_limited_on_the_cold_side() {
    // Spec, Addendum testing: "the demag check uses the part's own Hcj(T), including a
    // ferrite cold-case test", through the custom-dimension mode (no library part is ferrite).
    let e20 = &REGISTRY[DeviationId::E20.index()];
    let ferrite = &e20.probes[1];
    let overrides = probe_overrides(ferrite);
    let on = demag_with(&overrides, Deviations::only(DeviationId::E20));
    let off = demag_with(&overrides, Deviations::NONE);
    // The workbook takes |beta| and reports hot onsets near 240 C: "OK" for a magnet that
    // demagnetizes on every like-pole pass below about 100 C.
    assert!(off.onset_skipping_C > 200.0 && off.cold_check.starts_with("n/a"));
    assert_eq!((on.hcj20_used_kA_m, on.beta_used_per_C), (180.0, 0.0035));
    assert_eq!(on.onset_skipping_C, f64::INFINITY, "no knee on heating");
    assert_eq!(
        on.calibration_offset_C, 0.0,
        "the rating is not a cold-side knee rating"
    );
    assert_eq!(
        on.magnet_limit_C, 250.0,
        "the hot limit is the grade's rating"
    );
    let num_of = |x: NumOrText| match x {
        NumOrText::Num(v) => v,
        NumOrText::Text(t) => panic!("expected a number, got {t:?}"),
    };
    // The aligned field is below the knee at 20 C: its cold onset lies below 20 C. The
    // skipping field is past it: the magnet survives only above about 101 C.
    let aligned = num_of(on.cold_onset_aligned_C);
    let skipping = num_of(on.cold_onset_skipping_C);
    assert!(
        aligned < 20.0 && (aligned - (-57.82)).abs() < 0.005,
        "{aligned}"
    );
    assert!((skipping - 100.90).abs() < 0.005, "{skipping}");
    assert_eq!(num_of(on.cold_limit_C), skipping + 10.0);
    assert_eq!(on.cold_check, "Below the cold demagnetization limit");
}

#[test]
fn e20_mixed_rings_use_the_weaker_grade() {
    // A-1 plan decision A13: C52 to C55 are the outer blocks' reverse fields, and the workbook
    // checks only the inner ring's Br and rating against them. With E20 each ring is checked
    // with its own grade, Br and rating and the weaker ring governs, on either side: B842 (N42)
    // beside B842SH (N42SH) gives B842's limit in both orders, and the block shows B842.
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    let b842 = cells_with(&both_rings("B842"), Deviations::ALL);
    for (inner, outer, governing) in [
        ("B842SH", "B842", RING_OUTER),
        ("B842", "B842SH", RING_INNER),
    ] {
        let rings = [
            ("coupling.magnets.part_inner", Value::Text(inner.into())),
            ("coupling.magnets.part_outer", Value::Text(outer.into())),
        ];
        let cells = cells_with(&rings, Deviations::ALL);
        let limit = &cells["Temperature design!C12"];
        assert_report("Temperature design!C12", limit, -3.52, 0.005);
        assert_eq!(limit, &b842["Temperature design!C12"], "{inner}/{outer}");
        assert_eq!(
            cells["Temperature design!C25"],
            Value::Text("CHECK: see the rows above.".into()),
            "{inner}/{outer}"
        );
        let demag = demag_with(&rings, Deviations::ALL);
        assert_eq!(demag.demag_ring, governing, "{inner}/{outer}");
        assert_eq!(
            (demag.hcj20_used_kA_m, demag.beta_used_per_C),
            (954.9, -0.0062)
        );
        assert_eq!(
            demag.tmax_lib_C,
            NumOrText::Num(80.0),
            "the block shows B842"
        );
    }
    // Without E20 the workbook reads the inner ring only: B842SH inside reads OK.
    let stronger_inside = [
        ("coupling.magnets.part_inner", Value::Text("B842SH".into())),
        ("coupling.magnets.part_outer", Value::Text("B842".into())),
    ];
    let workbook = cells_with(&stronger_inside, Deviations::NONE);
    assert_report(
        "Temperature design!C12",
        &workbook["Temperature design!C12"],
        92.55,
        0.005,
    );
    assert_eq!(
        demag_with(&stronger_inside, Deviations::NONE).demag_ring,
        RING_INNER
    );
    // Identical rings tie: the inner ring's block, bit for bit (the default design).
    assert_eq!(
        demag_with(&both_rings("B842"), Deviations::ALL).demag_ring,
        RING_INNER
    );
}

#[test]
fn e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature() {
    // Review Focus 4: a ferrite grade with the reverse fields left at the stored NdFeB
    // values (354 to 863 kA/m, all above Y30's 162 kA/m knee). The cold onsets then lie
    // ABOVE the operating temperature: the magnet is demagnetized wherever it runs, and the
    // cold check and the verdict say so; nothing panics or reads as a cold-weather margin.
    let ferrite = [
        ("coupling.magnets.part_inner", Value::Text(String::new())),
        ("coupling.magnets.part_outer", Value::Text(String::new())),
        ("coupling.magnets.grade_inner", Value::Text("Y30".into())),
        ("coupling.magnets.grade_outer", Value::Text("Y30".into())),
    ];
    let demag = demag_with(&ferrite, Deviations::ALL);
    let cold = |x: NumOrText| match x {
        NumOrText::Num(v) => v,
        NumOrText::Text(t) => panic!("{t}"),
    };
    for onset in [
        demag.cold_onset_aligned_C,
        demag.cold_onset_pullout_C,
        demag.cold_onset_skipping_C,
        demag.cold_onset_single_ring_C,
    ] {
        assert!(cold(onset) > 50.0, "{onset:?}");
    }
    assert_eq!(demag.cold_check, "Below the cold demagnetization limit");
    assert_eq!(
        cells_with(&ferrite, Deviations::ALL)["Temperature design!C25"],
        Value::Text("CHECK: see the rows above.".into())
    );
}

#[test]
fn e20_leaves_every_default_cell_bit_for_bit() {
    // The default part's grade N42SH carries the workbook's own Hcj and beta.
    assert_bit_for_bit_at_defaults(DeviationId::E20);
}

/// The verdict C25's two texts.
const VERDICT_OK: &str = "OK on temperature. Confirm drag torque and thermal cycling by test.";
const VERDICT_CHECK: &str = "CHECK: see the rows above.";

/// The A-4 report's two bases for correction `id` (its section 1, "Conventions"): the
/// registry basis (the corrections `id` refines, then `id` too; decision 15) and the user
/// basis (every correction but `id`, then every correction).
fn a4_bases(id: DeviationId) -> [(&'static str, Deviations, Deviations); 2] {
    let registry = probe_base(&REGISTRY[id.index()]);
    [
        ("registry", registry, registry.with(id)),
        ("user", Deviations::ALL.without(id), Deviations::ALL),
    ]
}

/// Input overrides by path, applied to the defaults.
type Overrides = Vec<(&'static str, Value)>;

/// Manual dimensions with `inner` and `outer` picked as the rings' grades.
fn grade_rings(inner: &str, outer: &str) -> [(&'static str, Value); 4] {
    [
        ("coupling.magnets.part_inner", Value::Text(String::new())),
        ("coupling.magnets.part_outer", Value::Text(String::new())),
        ("coupling.magnets.grade_inner", Value::Text(inner.into())),
        ("coupling.magnets.grade_outer", Value::Text(outer.into())),
    ]
}

/// The library part `inner` on the inner ring and `outer` on the outer ring.
fn part_rings(inner: &str, outer: &str) -> [(&'static str, Value); 2] {
    [
        ("coupling.magnets.part_inner", Value::Text(inner.into())),
        ("coupling.magnets.part_outer", Value::Text(outer.into())),
    ]
}

/// Every ordered pair of different grades for manual dimensions (272 pairs, the A-4
/// harness's section 2.3), labelled "inner/outer".
fn grade_pairs() -> Vec<(String, Overrides)> {
    use magcoupling::engine::grades::GRADES;
    let mut pairs = Vec::new();
    for gi in &GRADES {
        for go in GRADES.iter().filter(|go| go.id != gi.id) {
            let rings = grade_rings(gi.id, go.id).to_vec();
            pairs.push((format!("{}/{}", gi.id, go.id), rings));
        }
    }
    pairs
}

/// Every ordered pair of different library parts (210 pairs, the A-4 harness's section 2.3),
/// labelled "inner/outer".
fn part_pairs() -> Vec<(String, Overrides)> {
    use magcoupling::engine::library::MAGNET_LIBRARY;
    let mut pairs = Vec::new();
    for pi in &MAGNET_LIBRARY {
        for po in MAGNET_LIBRARY.iter().filter(|po| po.part != pi.part) {
            let rings = part_rings(pi.part, po.part).to_vec();
            pairs.push((format!("{}/{}", pi.part, po.part), rings));
        }
    }
    pairs
}

/// Every result for `overrides` on the defaults `dev` implies, under `dev`.
fn results_with(overrides: &[(&str, Value)], dev: Deviations) -> DesignResults {
    compute_all_with(&inputs_with(overrides, dev), dev)
}

/// Bit for bit: a NaN equals a NaN of the same bits.
fn same_value(x: &Value, y: &Value) -> bool {
    match (x, y) {
        (Value::Num(p), Value::Num(q)) => p.to_bits() == q.to_bits(),
        (p, q) => p == q,
    }
}

/// Every result row, Rust-only ones included, is bit for bit the same in `a` and `b`.
fn assert_same_results(a: &DesignResults, b: &DesignResults, what: &str) {
    let (rows_a, rows_b) = (result_rows(a), result_rows(b));
    assert_eq!(rows_a.len(), rows_b.len(), "{what}");
    for (x, y) in rows_a.iter().zip(&rows_b) {
        assert!(
            same_value(&x.value, &y.value),
            "{what}: {} moved: {:?} -> {:?}",
            x.path,
            x.value,
            y.value
        );
    }
}

/// The workbook cells whose result differs, bit for bit, from `a` to `b`, in row order.
fn moved_cells(a: &DesignResults, b: &DesignResults) -> Vec<String> {
    result_rows(a)
        .into_iter()
        .zip(result_rows(b))
        .filter(|(x, y)| !same_value(&x.value, &y.value))
        .filter_map(|(x, _)| x.cell)
        .collect()
}

/// An A-4 correction keeps the defaults: every default cell bit for bit with only `id` on,
/// and every result row, Rust-only ones included, on both of its bases.
fn assert_a4_defaults_unchanged(id: DeviationId) {
    assert_bit_for_bit_at_defaults(id);
    for (basis, before, after) in a4_bases(id) {
        assert_same_results(&results_with(&[], before), &results_with(&[], after), basis);
    }
}

/// No differential case (3,391 today, every corpus file) moves from `before` to `after`, for
/// each `(before, after)` in `bases`: every result row, Rust-only ones included, bit for bit.
fn assert_no_differential_case_moves_between(bases: &[(Deviations, Deviations)], what: &str) {
    let all: Vec<(String, BTreeMap<String, Value>)> = differential_files()
        .into_iter()
        .flat_map(|file| {
            load_cases(file)
                .into_iter()
                .map(move |case| (format!("{file} case {}", case.id), case.inputs))
        })
        .collect();
    assert!(!all.is_empty());
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|s| {
        for chunk in all.chunks(all.len().div_ceil(threads)) {
            s.spawn(move || {
                for (label, overrides) in chunk {
                    for &(before, after) in bases {
                        let mut inputs = DesignInputs::defaults_with(before);
                        for (path, value) in overrides {
                            inputs
                                .set(path, value.clone())
                                .unwrap_or_else(|e| panic!("{label}: {e}"));
                        }
                        assert_same_results(
                            &compute_all_with(&inputs, before),
                            &compute_all_with(&inputs, after),
                            &format!("{what}, {label}"),
                        );
                    }
                }
            });
        }
    });
}

#[test]
fn e21_leaves_every_default_cell_bit_for_bit() {
    // Gated on E20, E21 alone is inert. On top of E20, and with every correction, the default
    // part's offset is positive (+10.43 C on the registry basis, +9.921 C with E3's 1.30 T),
    // so the clamp keeps every result, Rust-only ones included, bit for bit.
    assert_a4_defaults_unchanged(DeviationId::E21);
}

#[test]
fn e21_alone_moves_no_differential_case() {
    // A-4 report section 2.6, the double gate: E21 refines E20, so on its own it is NONE on
    // every corpus case, every result row bit for bit.
    let alone = (Deviations::NONE, Deviations::only(DeviationId::E21));
    assert_no_differential_case_moves_between(&[alone], "E21 alone");
}

#[test]
fn e21_without_e20_changes_nothing() {
    // N42's coercivity typed into C44 and C45 on the default part: without E20 the inputs act
    // for both rings and the offset is negative (-56.55 C), yet E21 alone changes nothing,
    // because it refines E20.
    let typed = [
        ("temperature.demag.hcj20_kA_m", Value::Num(954.9)),
        ("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)),
    ];
    let workbook = results_with(&typed, Deviations::NONE);
    assert!(workbook.temperature.demag.calibration_offset_C < 0.0);
    assert_same_results(
        &workbook,
        &results_with(&typed, Deviations::only(DeviationId::E21)),
        "E21 alone",
    );
}

#[test]
fn e21_clamps_exactly_the_negative_offsets() {
    // Every grade on both rings (manual dimensions), on both bases. A grade whose model
    // reference onset C49 is at or above its rating C47 moves nothing; below it, the offset
    // is 0 and the onsets are the uncalibrated model's (A-4 report section 2.2).
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::temperature::demag_onset_C;
    let e21 = DeviationId::E21;
    let d = DesignInputs::default().temperature.demag;
    for (basis, before, after) in a4_bases(e21) {
        let mut clamped = Vec::new();
        for g in &GRADES {
            let rings = grade_rings(g.id, g.id);
            let (b, a) = (demag_with(&rings, before), demag_with(&rings, after));
            if b.calibration_offset_C >= 0.0 {
                assert_same_results(
                    &results_with(&rings, before),
                    &results_with(&rings, after),
                    &format!("{basis} {}", g.id),
                );
                continue;
            }
            clamped.push(g.id);
            let model = demag_onset_C(
                d.h_rev_likepole_kA_m,
                g.hcj20_kA_m,
                g.beta_hcj_per_C,
                d.knee_fraction,
                g.alpha_br_per_C,
                0.0,
            );
            assert_eq!(a.calibration_offset_C, 0.0, "{basis} {}", g.id);
            assert_eq!(a.onset_skipping_C, model, "{basis} {}", g.id);
            assert_eq!(
                a.magnet_limit_C,
                model - d.design_margin_C,
                "{basis} {}",
                g.id
            );
            assert!(a.onset_skipping_C < b.onset_skipping_C, "{basis} {}", g.id);
            // Both rings carry the grade: each ring's applied offset is the block's.
            assert_eq!(
                (a.inner_calibration_offset_C, a.outer_calibration_offset_C),
                (0.0, 0.0),
                "{basis} {}",
                g.id
            );
        }
        assert_eq!(
            clamped,
            [
                "N52",
                "N50",
                "N50M",
                "N38UH",
                "N35EH",
                "SmCo_2_17_26",
                "SmCo_2_17_30"
            ],
            "{basis}"
        );
    }
}

#[test]
fn e21_clamps_exactly_the_parts_rated_above_their_model_onset() {
    // A-4 report section 2.2, the library parts on both rings: the N52 parts move on both
    // bases; the SuperMagnetMan arcs (N50 and N50M, stored ratings 80 and 100 C) move on the
    // registry basis only, since on the user basis E19's 60 C rating is below their model
    // onset (offset +12.98 C). Every other part is bit for bit unchanged, and no governing
    // limit rises.
    use magcoupling::engine::library::MAGNET_LIBRARY;
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let mut clamped = Vec::new();
        for spec in &MAGNET_LIBRARY {
            let rings = both_rings(spec.part);
            let what = format!("{basis} {}", spec.part);
            let (b, a) = (results_with(&rings, before), results_with(&rings, after));
            if b.temperature.demag.calibration_offset_C >= 0.0 {
                assert_same_results(&b, &a, &what);
                continue;
            }
            clamped.push(spec.part);
            assert_eq!(a.temperature.demag.calibration_offset_C, 0.0, "{what}");
            let limit = |r: &DesignResults| r.temperature.summary.governing_limit_C;
            assert!(limit(&a) < limit(&b), "{what}");
        }
        clamped.sort_unstable();
        let want: &[&str] = if basis == "registry" {
            &["B842-N52", "B882-N52", "M5026", "M5044", "M5045"]
        } else {
            &["B842-N52", "B882-N52"]
        };
        assert_eq!(clamped, want, "{basis}");
    }
}

/// A row of the A-4 report's table 2.3: (grade, C50 before, C60, C12 and C24 as [before,
/// after], C25 on both sides).
type ClampRow = (
    &'static str,
    f64,
    [f64; 2],
    [f64; 2],
    [f64; 2],
    &'static str,
);

#[test]
fn e21_changes_the_report_s_cells_to_the_report_s_figures() {
    // A-4 report section 2.3, the grade on both rings: the same figures on both bases.
    let rows: [ClampRow; 7] = [
        (
            "N52",
            -9.689,
            [0.1679, -9.521],
            [0.1679, -9.521],
            [59.77, 50.09],
            VERDICT_CHECK,
        ),
        (
            "N50",
            -6.138,
            [-3.383, -9.521],
            [-3.383, -9.521],
            [56.22, 50.09],
            VERDICT_CHECK,
        ),
        (
            "N50M",
            -7.535,
            [41.90, 34.37],
            [41.90, 34.37],
            [76.80, 69.27],
            VERDICT_CHECK,
        ),
        (
            "N38UH",
            -7.343,
            [131.9, 124.6],
            [120.0, 120.0],
            [149.9, 142.6],
            VERDICT_OK,
        ),
        (
            "N35EH",
            -3.574,
            [155.4, 151.9],
            [120.0, 120.0],
            [169.3, 165.8],
            VERDICT_OK,
        ),
        (
            "SmCo_2_17_26",
            -60.21,
            [161.9, 101.7],
            [120.0, 101.7],
            [265.2, 205.0],
            VERDICT_OK,
        ),
        (
            "SmCo_2_17_30",
            -0.4500,
            [46.27, 45.82],
            [46.27, 45.82],
            [169.9, 169.4],
            VERDICT_CHECK,
        ),
    ];
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        for (grade, offset, limit, governing, cure, verdict) in rows {
            let rings = grade_rings(grade, grade);
            let (b, a) = (cells_with(&rings, before), cells_with(&rings, after));
            let what = |cell: &str| format!("{basis} {grade} {cell}");
            assert_sig4(&what("C50"), &b["Temperature design!C50"], offset);
            assert_eq!(
                a["Temperature design!C50"],
                Value::Num(0.0),
                "{basis} {grade}"
            );
            for (cell, [want_b, want_a]) in [
                ("Temperature design!C60", limit),
                ("Temperature design!C12", governing),
                ("Temperature design!C24", cure),
            ] {
                assert_sig4(&what(cell), &b[cell], want_b);
                assert_sig4(&what(cell), &a[cell], want_a);
            }
            for c in [&b, &a] {
                assert_eq!(
                    c["Temperature design!C25"],
                    Value::Text(verdict.into()),
                    "{basis} {grade}"
                );
            }
        }
    }
}

#[test]
fn e21_typed_n42_coercivity_on_a_150_c_rating_reads_check() {
    // A-4 report section 2.5, the source-0 check: N42's Hcj and beta typed into C44 and C45 on
    // the default part (rated 150 C), with the coercivity source at 0. On the user basis the
    // clamp also turns the verdict from OK to CHECK.
    let typed = [
        ("temperature.demag.hcj20_kA_m", Value::Num(954.9)),
        ("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)),
        ("temperature.demag.coercivity_source", Value::Int(0)),
    ];
    // (basis, C50 before, C12 before, C24 before, C25 before); after: 0, 9.164, 60.51, CHECK.
    let want = [
        ("registry", -56.55, 65.71, 117.1, VERDICT_CHECK),
        ("user", -57.32, 66.48, 117.8, VERDICT_OK),
    ];
    for ((basis, before, after), (label, offset, governing, cure, verdict)) in
        a4_bases(DeviationId::E21).into_iter().zip(want)
    {
        assert_eq!(basis, label);
        let (b, a) = (cells_with(&typed, before), cells_with(&typed, after));
        assert_sig4(basis, &b["Temperature design!C50"], offset);
        assert_sig4(basis, &b["Temperature design!C12"], governing);
        assert_sig4(basis, &b["Temperature design!C24"], cure);
        assert_eq!(
            b["Temperature design!C25"],
            Value::Text(verdict.into()),
            "{basis}"
        );
        assert_eq!(a["Temperature design!C50"], Value::Num(0.0), "{basis}");
        assert_sig4(basis, &a["Temperature design!C12"], 9.164);
        assert_sig4(basis, &a["Temperature design!C24"], 60.51);
        assert_eq!(
            a["Temperature design!C25"],
            Value::Text(VERDICT_CHECK.into()),
            "{basis}"
        );
    }
}

#[test]
fn e21_binds_on_the_default_part_only_below_the_knee_threshold() {
    // A-4 report section 2.3: the default part's offset turns negative below a knee fraction
    // of 0.78349 (slider step 0.01): 0.79 moves nothing, 0.78 is the first clamped setting,
    // and at 0.70 the clamp turns the default design's verdict from OK to CHECK (C12 74.81 ->
    // 65.42 C, user basis).
    let knee = |k: f64| [("temperature.demag.knee_fraction", Value::Num(k))];
    let [_, (_, before, after)] = a4_bases(DeviationId::E21);
    assert_same_results(
        &results_with(&knee(0.79), before),
        &results_with(&knee(0.79), after),
        "knee 0.79",
    );
    let first = results_with(&knee(0.78), before).temperature.demag;
    assert!(
        first.calibration_offset_C < 0.0,
        "{}",
        first.calibration_offset_C
    );
    let (b, a) = (
        cells_with(&knee(0.70), before),
        cells_with(&knee(0.70), after),
    );
    assert_sig4("C12 before", &b["Temperature design!C12"], 74.81);
    assert_sig4("C12 after", &a["Temperature design!C12"], 65.42);
    assert_eq!(b["Temperature design!C25"], Value::Text(VERDICT_OK.into()));
    assert_eq!(
        a["Temperature design!C25"],
        Value::Text(VERDICT_CHECK.into())
    );
}

#[test]
fn e21_reorders_the_rings_where_the_clamp_moves_a_limit_and_ties_stay_inner() {
    // Probe 3's pair: B842 inside (N42, offset +12.68 C), B842-N52 outside (offset -9.689 C).
    // Clamped, the N52 ring has the lower limit: the block shows the outer ring. Each ring's
    // applied offset is a Rust-only result: the inner one is kept, the outer one is 0.
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    let rings = part_rings("B842", "B842-N52");
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let (b, a) = (demag_with(&rings, before), demag_with(&rings, after));
        assert_eq!(
            (b.demag_ring.as_str(), a.demag_ring.as_str()),
            (RING_INNER, RING_OUTER),
            "{basis}"
        );
        assert_eq!(a.inner_calibration_offset_C, b.inner_calibration_offset_C);
        assert_sig4(basis, &Value::Num(b.outer_calibration_offset_C), -9.689);
        assert_eq!(a.outer_calibration_offset_C, 0.0, "{basis}");
    }
    // N52 beside N50 is the one exact tie after the clamp (the A-4 report's "Ties"): both
    // skipping onsets are 0.4787157208430415 C, and the tie resolves to the inner ring.
    let tie = demag_with(&grade_rings("N52", "N50"), Deviations::ALL);
    assert_eq!(tie.onset_skipping_C, 0.4787157208430415);
    assert_eq!(tie.inner_magnet_limit_C, tie.outer_magnet_limit_C);
    assert_eq!(tie.demag_ring, RING_INNER);
}

#[test]
fn e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit() {
    // A-4 report sections 2.3 and 2.6, every ordered pair of different grades and of
    // different library parts (482 per basis): a pair in which neither ring's offset is
    // negative is bit for bit unchanged, Rust-only rows included; in every other pair the
    // clamp never raises the skipping onset C58, the magnet limit C60, the governing limit
    // C12 or the hot-day margin C15.
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let (mut moved, mut flips, mut verdicts) = (0, 0, 0);
        for (label, rings) in grade_pairs().into_iter().chain(part_pairs()) {
            let what = format!("{basis} {label}");
            let (b, a) = (results_with(&rings, before), results_with(&rings, after));
            let (db, da) = (&b.temperature.demag, &a.temperature.demag);
            if db.inner_calibration_offset_C >= 0.0 && db.outer_calibration_offset_C >= 0.0 {
                assert_same_results(&b, &a, &what);
                continue;
            }
            let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
            assert!(da.onset_skipping_C <= db.onset_skipping_C, "{what}: C58");
            assert!(da.magnet_limit_C <= db.magnet_limit_C, "{what}: C60");
            assert!(sa.governing_limit_C <= sb.governing_limit_C, "{what}: C12");
            assert!(sa.margin_hot_day_C <= sb.margin_hot_day_C, "{what}: C15");
            moved += usize::from(!moved_cells(&b, &a).is_empty());
            flips += usize::from(db.demag_ring != da.demag_ring);
            verdicts += usize::from(sb.verdict != sa.verdict);
        }
        // (pairs with a moved cell, governing-ring flips, verdict changes): the report's table,
        // but for six more pairs on the user basis: there E24, in every correction, reads the
        // single-ring onset of a ring E21 holds at 0 that does not govern, so C24 moves.
        let want = match basis {
            "registry" => (214, 71, 0),
            _ => (156, 39, 0),
        };
        assert_eq!((moved, flips, verdicts), want, "{basis}");
    }
}

#[test]
fn e24_leaves_every_default_cell_bit_for_bit() {
    // Identical rings: both single-ring onsets are the governing ring's, bit for bit.
    assert_a4_defaults_unchanged(DeviationId::E24);
}

#[test]
fn e24_alone_moves_no_differential_case() {
    // The double gate, as for E21 (A-4 report section 2.6): E24 refines E20, so on its own it
    // is NONE on every corpus case, every result row bit for bit.
    let alone = (Deviations::NONE, Deviations::only(DeviationId::E24));
    assert_no_differential_case_moves_between(&[alone], "E24 alone");
}

#[test]
fn e24_reproduces_the_report_s_cure_margins() {
    // A-4 report section 2.8 (with E21, so on the user basis): the shown C24 before E24 and
    // the lower ring's after it, in either ring order.
    let pairs: [(&str, &str, f64, f64); 7] = [
        ("N38UH", "SmCo_2_17_26", 205.0, 142.6),
        ("N35EH", "SmCo_2_17_26", 205.0, 165.8),
        ("SmCo_2_17_26", "SmCo_1_5_20", 205.0, 169.7),
        ("SmCo_2_17_30", "N38UH", 169.4, 142.6),
        ("SmCo_2_17_30", "N35EH", 169.4, 165.8),
        ("SmCo_2_17_30", "N42H", 169.4, 90.83),
        ("SmCo_2_17_30", "N42SH", 169.4, 121.4),
    ];
    let [_, (_, before, after)] = a4_bases(DeviationId::E24);
    for (a, b, shown, lower) in pairs {
        for (inner, outer) in [(a, b), (b, a)] {
            let rings = grade_rings(inner, outer);
            let what = format!("{inner}/{outer}");
            let cure = |dev| cells_with(&rings, dev)["Temperature design!C24"].clone();
            assert_sig4(&what, &cure(before), shown);
            assert_sig4(&what, &cure(after), lower);
        }
    }
    // The library parts: B842 (N42) beside B842-N52 (A-4 report probe 3), either order.
    for (inner, outer) in [("B842", "B842-N52"), ("B842-N52", "B842")] {
        let rings = part_rings(inner, outer);
        let what = format!("{inner}/{outer}");
        assert_sig4(
            &what,
            &cells_with(&rings, before)["Temperature design!C24"],
            50.09,
        );
        assert_sig4(
            &what,
            &cells_with(&rings, after)["Temperature design!C24"],
            47.83,
        );
    }
}

#[test]
fn e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict() {
    // Every ordered pair of grades and of library parts, alike or not, with each of the four
    // adhesives, both hubs, both bases: C24 is the lower of the two rings' single-ring onsets
    // less the cure temperature, never above the governing ring's, bit for bit where the
    // governing ring has the lower onset; no verdict moves at the default sliders (A-4 report
    // section 2.8: the lowest single-ring C24 of any grade is BCN-19's 12.26 C, above 10 C;
    // a slider setting that turns one is the next test's).
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::library::MAGNET_LIBRARY;
    // (label, rings, whether the rings are library parts)
    let mut configs: Vec<(String, Overrides, bool)> = Vec::new();
    for (label, rings) in grade_pairs() {
        configs.push((label, rings, false));
    }
    for g in &GRADES {
        let rings = grade_rings(g.id, g.id).to_vec();
        configs.push((format!("{} both rings", g.id), rings, false));
    }
    for (label, rings) in part_pairs() {
        configs.push((label, rings, true));
    }
    for spec in &MAGNET_LIBRARY {
        let rings = both_rings(spec.part).to_vec();
        configs.push((format!("{} both rings", spec.part), rings, true));
    }
    for adhesive in 1..=4 {
        for (hub, backiron) in HUBS {
            for (basis, before, after) in a4_bases(DeviationId::E24) {
                // How many grade configurations and part configurations C24 moves in.
                let mut moved = [0, 0];
                for (label, rings, part) in &configs {
                    let mut o = rings.clone();
                    o.push(("coupling.backiron", Value::Int(backiron)));
                    o.push(("temperature.adhesive.selected", Value::Int(adhesive)));
                    let what = format!("{label}, adhesive {adhesive}, {hub}, {basis}");
                    let (b, a) = (results_with(&o, before), results_with(&o, after));
                    let (d, cure) = (&a.temperature.demag, a.temperature.adhesive.cure_C);
                    let lower = d.inner_onset_single_ring_C.min(d.outer_onset_single_ring_C);
                    let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
                    assert_eq!(sa.cure_margin_C, lower - cure, "{what}");
                    assert!(sa.cure_margin_C <= sb.cure_margin_C, "{what}");
                    if d.onset_single_ring_C == lower {
                        assert_same_results(&b, &a, &what);
                    }
                    assert_eq!(sb.verdict, sa.verdict, "{what}");
                    if sa.cure_margin_C != sb.cure_margin_C {
                        moved[usize::from(*part)] += 1;
                    }
                }
                // The same with every adhesive and hub: the rings decide which onset is lower.
                let want = if basis == "registry" {
                    [20, 0]
                } else {
                    [26, 28]
                };
                assert_eq!(moved, want, "adhesive {adhesive}, {hub}, {basis}");
            }
        }
    }
    let bonded = grade_rings("Bonded_NdFeB_BCN19", "Bonded_NdFeB_BCN19");
    assert_sig4(
        "BCN-19",
        &cells_with(&bonded, Deviations::ALL)["Temperature design!C24"],
        12.26,
    );
}

#[test]
fn e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure() {
    // Decision A4-1: C25 is registered because E24 can turn the verdict, though at no default
    // slider setting (the test above). Recoma 30 beside N42H, in either order, bonded with
    // EA 9514 (cured at 120 C), with the knee fraction C46 at 0.95 and the design margin C51
    // at 0: Recoma 30 governs and its single-ring onset gives C24 = 75.40 C, so the verdict
    // reads OK; N42H's single-ring onset, 113.3 C, is below the cure, so with E24 C24 is
    // -6.719 C and the verdict reads CHECK. Only C24 and C25 move, on both bases.
    for (inner, outer) in [("SmCo_2_17_30", "N42H"), ("N42H", "SmCo_2_17_30")] {
        let mut o = grade_rings(inner, outer).to_vec();
        o.extend([
            ("temperature.adhesive.selected", Value::Int(2)),
            ("temperature.demag.knee_fraction", Value::Num(0.95)),
            ("temperature.demag.design_margin_C", Value::Num(0.0)),
        ]);
        for (basis, before, after) in a4_bases(DeviationId::E24) {
            let what = format!("{inner}/{outer}, {basis}");
            let (b, a) = (results_with(&o, before), results_with(&o, after));
            let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
            assert_sig4(&what, &Value::Num(sb.cure_margin_C), 75.40);
            assert_sig4(&what, &Value::Num(sa.cure_margin_C), -6.719);
            assert_eq!(
                (sb.verdict.as_str(), sa.verdict.as_str()),
                (VERDICT_OK, VERDICT_CHECK),
                "{what}"
            );
            assert_eq!(
                moved_cells(&b, &a),
                ["Temperature design!C24", "Temperature design!C25"],
                "{what}"
            );
        }
    }
}

#[test]
fn e24_with_e21_gives_the_lower_ring_s_margin_on_probe_4() {
    // A-4 report probe 4 (N38UH inside, Recoma 26 outside): E21 moves the block to Recoma 26,
    // whose C24 is 205.0 C; with E24 the margin is N38UH's, 142.57635668906815 C (the note
    // under decision 2 A).
    let rings = grade_rings("N38UH", "SmCo_2_17_26");
    let base = Deviations::NONE
        .with(DeviationId::E20)
        .with(DeviationId::E21);
    let cure = |dev| cells_with(&rings, dev)["Temperature design!C24"].clone();
    assert_eq!(cure(base), Value::Num(205.01249772124206));
    assert_eq!(
        cure(base.with(DeviationId::E24)),
        Value::Num(142.57635668906815)
    );
}

/// The two hubs of the A-4 harness: the default steel circuit (`coupling.backiron` 1) and no
/// back iron (0, an aluminium hub).
const HUBS: [(&str, i64); 2] = [("steel hub", 1), ("aluminium hub", 0)];

#[test]
fn e24_fails_a_cure_that_reaches_the_inner_ring_s_own_onset() {
    // BL-072's red test: inner N42H, outer Recoma 30 (SmCo_2_17_30) on manual dimensions at
    // gui::test_support::passing_design() (restated: a 1.8 mm face gap, a 2.0 mm cup wall, a
    // 1.6 N m requirement, a 1.0 N m gearbox input rating), bonded with 2214 Hi-Temp (cured
    // at 121 C). Recoma 30 has the lower magnet limit and governs, so before E24 the margin
    // reads its single-ring onset and passes; with E24 it reads the lower ring's, N42H's own
    // 112.83 C, which the cure exceeds: -8.17 C, below the 10 C minimum, and the verdict fails.
    let mut o = grade_rings("N42H", "SmCo_2_17_30").to_vec();
    o.extend([
        ("metal.face_gap_mm", Value::Num(1.8)),
        ("metal.cup_wall_corner_mm", Value::Num(2.0)),
        ("metal.required_min_Nm", Value::Num(1.6)),
        ("coupling.gearbox_input_rating_Nm", Value::Num(1.0)),
        ("temperature.adhesive.selected", Value::Int(3)),
    ]);
    let [_, (_, before, after)] = a4_bases(DeviationId::E24);
    let (b, a) = (results_with(&o, before), results_with(&o, after));
    assert_eq!(a.temperature.demag.demag_ring, "outer");
    assert!(b.temperature.summary.cure_margin_C >= 10.0);
    let own = a.temperature.demag.inner_onset_single_ring_C;
    assert_sig4("N42H's own onset", &Value::Num(own), 112.8);
    assert_eq!(
        a.temperature.summary.cure_margin_C,
        own - a.temperature.adhesive.cure_C
    );
    assert_sig4(
        "the cure margin",
        &Value::Num(a.temperature.summary.cure_margin_C),
        -8.170,
    );
    assert_eq!(a.temperature.summary.verdict, VERDICT_CHECK);
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
    // The default part is B842SH: the vendor grid applies to the arcs only.
    assert_bit_for_bit_at_defaults(DeviationId::E19);
}

#[test]
fn e18_leaves_every_default_cell_bit_for_bit() {
    // At defaults the hub is steel: the workbook's CTE and modulus stay.
    assert_bit_for_bit_at_defaults(DeviationId::E18);
}

#[test]
fn e17_leaves_every_default_cell_bit_for_bit() {
    // At defaults every loss term is steel: the skin-limited workbook formulas stay.
    assert_bit_for_bit_at_defaults(DeviationId::E17);
}

#[test]
fn e16_leaves_every_default_cell_bit_for_bit() {
    // At defaults the web is steel: the density E16 reads is C132 itself.
    assert_bit_for_bit_at_defaults(DeviationId::E16);
}

#[test]
fn e15_leaves_every_default_cell_bit_for_bit() {
    // At defaults (C6 = 1) every part is steel and the workbook expression stays.
    assert_bit_for_bit_at_defaults(DeviationId::E15);
}

#[test]
fn e14_the_22mm_boss_statement_is_true() {
    let e14 = &REGISTRY[DeviationId::E14.index()];
    assert_eq!(e14.status, DeviationStatus::Applied);
    let clamp = |length: f64| {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.clamps.boss_od_mm = 22.0;
        inputs.clamps.clamp_length_mm = length;
        compute_all_with(&inputs, Deviations::NONE).clamps
    };
    let c = clamp(14.0);
    assert_eq!(c.index, 0, "below 14.5 mm nothing fits");
    assert_eq!(c.table[2].works, 0, "M4 no longer fits at 22 mm");
    let c = clamp(14.5);
    assert!(
        c.recommended.starts_with("ISO 4762 M3 x "),
        "{}",
        c.recommended
    );
    assert_eq!(c.screws, NumOrText::Num(2.0), "two M3 need a 14.5 mm clamp");
    for length in [18.0, 25.0] {
        let c = clamp(length);
        assert!(
            c.recommended.starts_with("ISO 4762 M2.5 x "),
            "{length}: {}",
            c.recommended
        );
        assert_eq!(c.screws, NumOrText::Num(3.0), "from 18 mm up: three M2.5");
    }
}

#[test]
fn all_corrections_together_give_the_reviewed_headline() {
    // What users see (compute_all, every correction on) at the default design, D1 = 1.30 T.
    // Values from one Python rerun with E1, E2, E3 and E5 patched in together (E4, E6-E13 are
    // neutral for these cells at defaults).
    let res = compute_all(&DesignInputs::default());
    let h: BTreeMap<&str, Value> = headline(&res).into_iter().collect();
    let close = |key: &str, want: f64| {
        let got = num(&h[key]);
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "{key}: {got} vs {want}"
        );
    };
    close("pullout_at_op_temp_Nm", 2.6884762950539796);
    close("hot_low_with_variation_Nm", 2.2852048507958824);
    close("cold_high_with_variation_Nm", 3.823310370488638);
    close("governing_temp_limit_C", 93.05566428111358);
    close("running_clearance_mm", -0.10317439456607391);
    assert_eq!(h["hot_min_check"], Value::Text("Below hot minimum".into()));
    assert_eq!(h["clearance_check"], Value::Text("Below target".into()));
    assert_eq!(
        h["cup_wall_check"],
        Value::Text("Too thin: raise Metal design C122 to at least 2.0 mm".into())
    );
    assert_eq!(
        h["temperature_verdict"],
        Value::Text("OK on temperature. Confirm drag torque and thermal cycling by test.".into())
    );
    assert_eq!(
        h["clamp_screw"],
        Value::Text("ISO 4762 M4 x 14, class 12.9".into())
    );
    assert_eq!(
        res.temperature.mismatch.reading,
        "Below the lap-shear strength"
    );
    assert_eq!(
        res.temperature.adhesive_life.daily_screen,
        "Below the fatigue endurance"
    );
    assert_eq!(res.temperature.adhesive.fatigue_screen, "OK: 7x margin");
    assert!((res.temperature.thermal.steady_high_C - 92.51311161421395).abs() <= 1e-9 * 92.5);
    assert_eq!(
        res.temperature.thermal.time_to_limit_high,
        NumOrText::Text("never: steady state stays below the limit")
    );
    assert_eq!(
        res.clamps.length_note,
        "No 2 mm length step of M4 both engages 8 mm of thread and stays inside the boss; M4 x 14 protrudes 0.34 mm"
    );
}

#[test]
fn reworded_help_is_recorded_for_real_fields() {
    // An input path, a scalar result path (decision 14: E16's and E17's labels stay, their
    // help is reworded), or a table column (checked against the workbook in tests/schema.rs).
    let inputs = input_rows(&DesignInputs::default());
    let results = result_rows(&compute_all(&DesignInputs::default()));
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for &(path, workbook) in d.workbook_help {
            if path.contains("[*]") {
                continue; // table columns: checked against the workbook headers in tests/schema.rs
            }
            let help = inputs
                .iter()
                .find(|r| r.path == path)
                .map(|r| r.meta.help)
                .or_else(|| {
                    results
                        .iter()
                        .find(|r| r.path == path && !r.meta.rust_only)
                        .map(|r| r.meta.help)
                })
                .unwrap_or_else(|| panic!("{}: {path} is not an input or a result", d.id));
            assert_ne!(help, workbook, "{}: {path} help is not reworded", d.id);
        }
    }
}
