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

use common::{PORTED, data_path, group_of, json_to_value, read_json, report};
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

fn scalar_map(json: &serde_json::Value) -> BTreeMap<String, Value> {
    json.as_object()
        .expect("a JSON object of path to value")
        .iter()
        .map(|(path, v)| (path.clone(), json_to_value(v)))
        .collect()
}

fn load_cases(module: &str) -> Vec<Case> {
    let doc = read_json(&data_path(&format!("differential/{module}.json")));
    assert_eq!(doc["module"], module);
    doc["cases"]
        .as_array()
        .expect("a cases array")
        .iter()
        .map(|c| Case {
            id: c["id"].as_u64().expect("a case id"),
            tag: c["tag"].as_str().expect("a case tag").to_owned(),
            inputs: scalar_map(&c["inputs"]),
            results: scalar_map(&c["results"]),
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

#[test]
fn every_ported_module_has_differential_data() {
    for p in PORTED {
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

    // Coverage: both gap definitions, both sides of the interpolation span,
    // both inclusive span ends, and every input actually varied.
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
    let inputs: BTreeSet<&String> = cases.iter().flat_map(|c| c.inputs.keys()).collect();
    for path in inputs {
        let distinct: BTreeSet<String> = cases
            .iter()
            .map(|c| format!("{:?}", c.inputs[path]))
            .collect();
        assert!(distinct.len() >= 2, "{path} never varies");
    }
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
