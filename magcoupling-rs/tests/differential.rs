//! Differential tests against the Python engine.
//!
//! The workbook snapshot covers only the default inputs. The Python generator
//! (`reference/magcoupling-py/tools/gen_differential.py`) runs the engine on
//! seeded input sets spanning every slider range and selector branch and
//! writes `tests/data/differential/<module>.json`. Every result of every case
//! must match here by the parity rule, deviations off. The helpers corpus
//! checks the Python/Excel rounding and formatting helpers the same way.
//!
//! Stale data is caught by `gen_differential.py --check` (gate 7).

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{PORTED_RESULTS, data_path, group_of, json_to_value, read_json, report};
use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::compat::{
    ceiling, floor_, fmt_fixed, fmt_num, parity_close, py_repr, text0,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{InputSet, Value, result_rows};

/// A generated case: inputs by path, and the Python results of its module.
struct Case {
    id: u64,
    tag: String,
    inputs: BTreeMap<String, Value>,
    results: BTreeMap<String, Value>,
}

fn load_cases(module: &str) -> Vec<Case> {
    let doc = read_json(&data_path(&format!("differential/{module}.json")));
    assert_eq!(doc["module"], module);
    let paths = |key: &str| -> Vec<String> {
        doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("{module}: {key} is an array"))
            .iter()
            .map(|p| p.as_str().expect("a path").to_owned())
            .collect()
    };
    let (input_paths, result_paths) = (paths("input_paths"), paths("result_paths"));
    doc["cases"]
        .as_array()
        .expect("a cases array")
        .iter()
        .map(|c| {
            let id = c["id"].as_u64().expect("a case id");
            let zip = |paths: &[String], key: &str| -> BTreeMap<String, Value> {
                let values = c[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("case {id}: {key}"));
                assert_eq!(
                    values.len(),
                    paths.len(),
                    "{module} case {id}: {key} length"
                );
                paths
                    .iter()
                    .cloned()
                    .zip(values.iter().map(json_to_value))
                    .collect()
            };
            Case {
                id,
                tag: c["tag"].as_str().expect("a case tag").to_owned(),
                inputs: zip(&input_paths, "inputs"),
                results: zip(&result_paths, "results"),
            }
        })
        .collect()
}

/// The Rust results of `module` for a case, deviations off.
fn rust_results(module: &str, case: &Case) -> Result<BTreeMap<String, Value>, String> {
    let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
    for (path, value) in &case.inputs {
        inputs
            .set(path, value.clone())
            .map_err(|e| format!("case {} ({}): {e}", case.id, case.tag))?;
    }
    let results = compute_all_with(&inputs, Deviations::NONE);
    Ok(result_rows(&results)
        .into_iter()
        .filter(|r| group_of(&r.path) == module)
        .map(|r| (r.path, r.value))
        .collect())
}

/// Compares every result of every case; returns the cases for coverage checks.
fn check_module(module: &str) -> Vec<Case> {
    let cases = load_cases(module);
    assert!(cases.len() >= 200, "{module}: only {} cases", cases.len());
    let mut failures = Vec::new();
    for case in &cases {
        let rust = match rust_results(module, case) {
            Ok(rust) => rust,
            Err(e) => {
                failures.push(e);
                continue;
            }
        };
        let paths: BTreeSet<&String> = rust.keys().chain(case.results.keys()).collect();
        for path in paths {
            match (rust.get(path), case.results.get(path)) {
                (Some(r), Some(p)) if parity_close(r, p) => {}
                (r, p) => failures.push(format!(
                    "case {} ({}): {path}: rust={r:?} python={p:?}",
                    case.id, case.tag
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{module}: {}", report(&failures));
    cases
}

/// What a text-producing result must reach across its module's cases.
#[derive(Clone, Copy, Debug)]
enum Reach {
    /// Exactly this text.
    Text(&'static str),
    /// A text starting with this.
    Prefix(&'static str),
    /// A number (the numeric side of a number-or-text result).
    Number,
}
use Reach::{Number, Prefix, Text};

/// Every branch of every ported module's text results, so each branch of the
/// Python source is compared at least once. `[*]` matches any table row.
/// Add the rows of a module when it lands.
const BRANCHES: &[(&str, &[Reach])] = &[
    (
        "calibration.fea_interp_Nm",
        &[Number, Text("outside range")],
    ),
    ("calibration.fea_interp_error", &[Number, Text("n.a.")]),
    ("model.inner_tmax_C", &[Number, Text("n/a")]),
    ("model.outer_tmax_C", &[Number, Text("n/a")]),
    (
        "model.inner_flat_check",
        &[
            Prefix("OK, "),
            Text("TOO NARROW: increase apothem or reduce poles"),
            Text("n/a (arcs)"),
        ],
    ),
    (
        "model.outer_flat_check",
        &[
            Prefix("OK, blocks "),
            Text("TOO NARROW: increase gap/apothem or reduce poles"),
            Text("n/a (arcs)"),
        ],
    ),
    (
        "model.verdict",
        &[Text("Below hot minimum"), Text("Nominal only: hot test")],
    ),
    (
        "model.cup_ring_check",
        &[Text("No back iron"), Text("Thickness OK"), Text("Too thin")],
    ),
    (
        "model.hub_check",
        &[Text("No back iron"), Text("Thickness OK"), Text("Too thin")],
    ),
    (
        "model.inner_temp_check",
        &[Text("unknown"), Text("OK"), Text("OVER the magnet rating")],
    ),
    (
        "model.outer_temp_check",
        &[Text("unknown"), Text("OK"), Text("OVER the magnet rating")],
    ),
    (
        "metal.hot_min_check",
        &[Text("Below hot minimum"), Text("Estimate covers hot min")],
    ),
    (
        "metal.clearance_check",
        &[Text("Below target"), Text("Meets assumed target")],
    ),
    ("metal.slip_loss_W", &[Number, Text("not measured")]),
    ("metal.slip_energy_J", &[Number, Text("not measured")]),
];

fn reached(value: &Value, reach: Reach) -> bool {
    match (reach, value) {
        (Text(t), Value::Text(v)) => v == t,
        (Prefix(p), Value::Text(v)) => v.starts_with(p),
        (Number, Value::Num(_) | Value::Int(_)) => true,
        _ => false,
    }
}

/// `gap_sweep[*].status` matches `gap_sweep[3].status`; other patterns match exactly.
fn matches_pattern(pattern: &str, path: &str) -> bool {
    match pattern.split_once("[*]") {
        None => pattern == path,
        Some((head, tail)) => path
            .strip_prefix(head)
            .and_then(|rest| rest.strip_prefix('['))
            .and_then(|rest| rest.split_once(']'))
            .is_some_and(|(index, rest)| {
                !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) && rest == tail
            }),
    }
}

#[test]
fn every_branch_is_reached() {
    let mut cache: BTreeMap<&str, Vec<Case>> = BTreeMap::new();
    let mut failures = Vec::new();
    for &(pattern, reaches) in BRANCHES {
        let module = group_of(pattern);
        let cases = cache.entry(module).or_insert_with(|| load_cases(module));
        let values: Vec<&Value> = cases
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|(path, _)| matches_pattern(pattern, path))
            .map(|(_, v)| v)
            .collect();
        if values.is_empty() {
            failures.push(format!(
                "{pattern}: no such result in differential/{module}.json"
            ));
            continue;
        }
        for &reach in reaches {
            if !values.iter().any(|v| reached(v, reach)) {
                failures.push(format!("{pattern}: never reaches {reach:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_varied_input_takes_two_values() {
    let mut failures = Vec::new();
    for p in PORTED_RESULTS {
        let cases = load_cases(p.group);
        let paths: BTreeSet<&String> = cases.iter().flat_map(|c| c.inputs.keys()).collect();
        for path in paths {
            let distinct: BTreeSet<String> = cases
                .iter()
                .map(|c| format!("{:?}", c.inputs[path]))
                .collect();
            if distinct.len() < 2 {
                failures.push(format!("{}: {path} never varies", p.group));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn branch_patterns_match_table_rows_only_by_index() {
    assert!(matches_pattern(
        "gap_sweep[*].status",
        "gap_sweep[12].status"
    ));
    assert!(!matches_pattern(
        "gap_sweep[*].status",
        "gap_sweep[].status"
    ));
    assert!(!matches_pattern(
        "gap_sweep[*].status",
        "gap_sweep[1].status_x"
    ));
    assert!(!matches_pattern(
        "gap_sweep[*].status",
        "pole_sweep[1].status"
    ));
    assert!(matches_pattern("model.verdict", "model.verdict"));
}

#[test]
fn every_ported_module_has_differential_data() {
    for p in PORTED_RESULTS {
        assert!(
            data_path(&format!("differential/{}.json", p.group)).exists(),
            "no differential data for {}: add it to MODULES in gen_differential.py",
            p.group
        );
    }
}

#[test]
fn calibration_matches_python_on_every_case() {
    let cases = check_module("calibration");

    // Coverage: both gap definitions, both sides of the interpolation span
    // and both inclusive span ends.
    let count = |f: &dyn Fn(&Case) -> bool| cases.iter().filter(|c| f(c)).count();
    let gap_definition = |c: &Case| c.inputs["calibration.gap_definition"].clone();
    assert!(count(&|c| gap_definition(c) == Value::Int(0)) >= 20);
    assert!(count(&|c| gap_definition(c) == Value::Int(1)) >= 20);
    let interp_is_text =
        |c: &Case| matches!(c.results["calibration.fea_interp_Nm"], Value::Text(_));
    assert!(count(&|c| interp_is_text(c)) >= 20);
    assert!(count(&|c| !interp_is_text(c)) >= 20);
    for end in [1.0, 1.5] {
        assert!(
            count(
                &|c| c.results["calibration.corner_gap_mm"] == Value::Num(end)
                    && !interp_is_text(c)
            ) >= 1,
            "no case on the inclusive span end {end}"
        );
    }
}

#[test]
fn model_matches_python_on_every_case() {
    let cases = check_module("model");
    let prototype = cases
        .iter()
        .find(|c| c.tag == "prototype circuit: measured calibration factor")
        .expect("the prototype probe");
    assert_ne!(
        prototype.results["model.f_cal"], prototype.inputs["calibration.f_cal_original"],
        "the prototype probe must select the measured factor"
    );
}

#[test]
fn retainers_matches_python_on_every_case() {
    check_module("retainers");
}

#[test]
fn mass_matches_python_on_every_case() {
    check_module("mass");
}

#[test]
fn metal_matches_python_on_every_case() {
    check_module("metal");
}

/// A corpus key and the Rust helper call it records (as the engine calls it).
type Rounding = (&'static str, fn(f64) -> f64);

const ROUNDINGS: [Rounding; 3] = [
    ("ceiling_0_1", |x| ceiling(x, 0.1)),
    ("ceiling_2", |x| ceiling(x, 2.0)),
    ("floor_1", |x| floor_(x, 1.0)),
];

#[test]
fn helpers_match_python_on_the_corpus() {
    let doc = read_json(&data_path("differential/helpers.json"));
    let entries = doc["entries"].as_array().expect("an entries array");
    assert!(
        entries.len() >= 1000,
        "only {} corpus entries",
        entries.len()
    );
    let mut failures = Vec::new();
    for e in entries {
        let x = e["x"].as_f64().expect("x is a number");
        let texts: [(&str, String); 5] = [
            ("repr", py_repr(x)),
            ("fmt_num", fmt_num(x)),
            ("fixed1", fmt_fixed(x, 1)),
            ("fixed2", fmt_fixed(x, 2)),
            ("text0", text0(x)),
        ];
        for (key, got) in texts {
            let want = e[key].as_str().expect("a text output");
            if got != want {
                failures.push(format!("{key}({x:e}): rust={got:?} python={want:?}"));
            }
        }
        for (key, f) in ROUNDINGS {
            if e[key].is_null() {
                continue;
            }
            let want = e[key].as_f64().expect("a numeric output");
            let got = f(x);
            // Bit-identical, including the sign of zero.
            if got.to_bits() != want.to_bits() {
                failures.push(format!("{key}({x:e}): rust={got:e} python={want:e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}
