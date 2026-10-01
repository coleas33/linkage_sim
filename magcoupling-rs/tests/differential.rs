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

use common::{Case, FULL, PORTED_RESULTS, data_path, group_of, load_cases, read_json, report};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with};
use magcoupling::engine::compat::{
    ceiling, floor_, fmt_fixed, fmt_num, parity_close, py_repr, text0,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, Value, input_rows, result_rows};

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
        .filter(|r| !r.meta.rust_only)
        .filter(|r| module == FULL || group_of(&r.path) == module)
        .map(|r| (r.path, r.value))
        .collect())
}

/// Compares every result of every case; returns the cases for coverage checks.
fn check_module(module: &str) -> Vec<Case> {
    let cases = load_cases(module);
    let minimum = if module == FULL { 100 } else { 200 };
    assert!(
        cases.len() >= minimum,
        "{module}: only {} cases",
        cases.len()
    );
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
    /// Any text (a result with a single formatted form, e.g. clamps.layout_slit).
    AnyText,
}
use Reach::{AnyText, Number, Prefix, Text};

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
    (
        "materials.cup_wall_check",
        &[
            Text("OK"),
            Prefix("Too thin: raise Metal design C122 to at least "),
        ],
    ),
    (
        "temperature.summary.governing_note",
        &[
            Text("Magnets govern (skipping case)."),
            Text("Adhesive governs."),
        ],
    ),
    (
        "temperature.summary.torque_hot_day_note",
        &[
            Text("Meets it nominally (no variation allowance)"),
            Text("Below it"),
        ],
    ),
    (
        "temperature.summary.time_to_limit_high",
        &[Number, Text("never: steady state stays below the limit")],
    ),
    (
        "temperature.summary.verdict",
        &[
            Text("OK on temperature. Confirm drag torque and thermal cycling by test."),
            Text("CHECK: see the rows above."),
        ],
    ),
    ("temperature.demag.tmax_lib_C", &[Number, Text("n/a")]),
    (
        "temperature.adhesive.selected_name",
        &[
            Text("Loctite AA 326 + SF 7649"),
            Text("Loctite EA 9514"),
            Text("3M Scotch-Weld 2214 Hi-Temp"),
            Text("3M Scotch-Weld DP460"),
        ],
    ),
    (
        "temperature.adhesive.fatigue_screen",
        &[Prefix("OK: "), Text("CHECK")],
    ),
    (
        "temperature.mismatch.reading",
        &[
            Text("Above the lap-shear strength at the block ends"),
            Text("Below the lap-shear strength"),
        ],
    ),
    (
        "temperature.thermal.time_to_limit_high",
        &[Number, Text("never: steady state stays below the limit")],
    ),
    (
        "temperature.thermal.rotations_to_limit_high",
        &[Number, Text("never")],
    ),
    (
        "temperature.thermal.time_to_limit_est",
        &[Number, Text("never: steady state stays below the limit")],
    ),
    (
        "temperature.magnet_life.torque_hot_day_check",
        &[
            Text("Meets it nominally (no variation allowance)"),
            Text("Below it"),
        ],
    ),
    (
        "temperature.adhesive_life.hot_fatigue_screen",
        &[Text("OK"), Text("CHECK: get hot fatigue data")],
    ),
    (
        "temperature.adhesive_life.daily_screen",
        &[
            Text("Below the fatigue endurance"),
            Text("Above the fatigue endurance: qualify by thermal cycling"),
        ],
    ),
    (
        "clamps.recommended",
        &[
            Prefix("ISO 4762 "),
            Text("None: enlarge the boss or the clamp length"),
        ],
    ),
    ("clamps.screws", &[Number, Text("")]),
    ("clamps.tightening_Nm", &[Number, Text("")]),
    ("clamps.hex_mm", &[Number, Text("")]),
    ("clamps.capacity_Nm", &[Number, Text("")]),
    ("clamps.sf_coupling", &[Number, Text("")]),
    (
        "clamps.head_check",
        &[Text("OK"), Text("Use a hardened washer"), Text("")],
    ),
    (
        "clamps.vent_port",
        &[
            Text("Yes: the key fits the 4 mm limit"),
            Text("No: key too large"),
            Text(""),
        ],
    ),
    ("clamps.layout_offset_mm", &[Number, Text("")]),
    ("clamps.layout_pitch_mm", &[Number, Text("")]),
    ("clamps.layout_first_mm", &[Number, Text("")]),
    ("clamps.layout_cbore_dia_mm", &[Number, Text("")]),
    ("clamps.layout_cbore_depth_mm", &[Number, Text("")]),
    ("clamps.layout_grip_mm", &[Number, Text("")]),
    ("clamps.layout_tap_drill_mm", &[Number, Text("")]),
    ("clamps.layout_thread_avail_mm", &[Number, Text("")]),
    ("clamps.layout_slit", &[AnyText]),
    ("clamps.layout_relief", &[AnyText]),
    (
        "clamps.table[*].size",
        &[Text("M2.5"), Text("M3"), Text("M4"), Text("M5"), Text("M6")],
    ),
    (
        "clamps.table[*].head_check",
        &[Text("OK"), Text("Use a hardened washer")],
    ),
    (
        "gap_sweep[*].status",
        &[
            Text("inner flat too narrow"),
            Text("outer flat too narrow"),
            Text("outside OD envelope"),
            Text("below hot minimum"),
            Text("nominal: test needed"),
        ],
    ),
    // "inner flat too narrow" cannot occur in the pole sweep: its apothem is chosen so that
    // 2 a_i tan(pi/N) >= w_i + 0.1 tan(pi/N) > w_i.
    (
        "pole_sweep[*].status",
        &[
            Text("outer flat too narrow"),
            Text("outside OD envelope"),
            Text("below hot minimum"),
            Text("nominal: test needed"),
        ],
    ),
];

fn reached(value: &Value, reach: Reach) -> bool {
    match (reach, value) {
        (Text(t), Value::Text(v)) => v == t,
        (Prefix(p), Value::Text(v)) => v.starts_with(p),
        (Number, Value::Num(_) | Value::Int(_)) => true,
        (AnyText, Value::Text(_)) => true,
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

#[test]
fn materials_matches_python_on_every_case() {
    check_module("materials");
}

#[test]
fn temperature_matches_python_on_every_case() {
    check_module("temperature");
}

#[test]
fn clamps_matches_python_on_every_case() {
    check_module("clamps");
}

#[test]
fn gap_sweep_matches_python_on_every_case() {
    check_module("gap_sweep");
}

#[test]
fn pole_sweep_matches_python_on_every_case() {
    check_module("pole_sweep");
}

#[test]
fn full_run_matches_python_on_every_case() {
    check_module(FULL);
}

/// Selector inputs and their codes, from the Rust metadata. Rust-only selectors
/// are not in the differential data (the generator never passes them to Python).
fn selectors() -> Vec<(String, Vec<i64>)> {
    input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| !r.meta.choices.is_empty() && !r.meta.rust_only)
        .map(|r| (r.path, r.meta.choices.iter().map(|&(c, _)| c).collect()))
        .collect()
}

#[test]
fn every_selector_pair_is_covered_in_the_full_run() {
    let cases = load_cases(FULL);
    let selectors = selectors();
    let mut missing = Vec::new();
    for (i, (a, codes_a)) in selectors.iter().enumerate() {
        for (b, codes_b) in &selectors[i + 1..] {
            for &ca in codes_a {
                for &cb in codes_b {
                    let hit = cases
                        .iter()
                        .any(|c| c.inputs[a] == Value::Int(ca) && c.inputs[b] == Value::Int(cb));
                    if !hit {
                        missing.push(format!("{a} = {ca} with {b} = {cb}"));
                    }
                }
            }
        }
    }
    assert!(missing.is_empty(), "{}", report(&missing));
}

#[test]
fn every_selector_choice_appears_in_every_module_file() {
    let mut missing = Vec::new();
    for p in PORTED_RESULTS {
        let cases = load_cases(p.group);
        for (path, codes) in selectors() {
            if !cases[0].inputs.contains_key(&path) {
                continue; // this module does not vary that group
            }
            for code in codes {
                if !cases.iter().any(|c| c.inputs[&path] == Value::Int(code)) {
                    missing.push(format!("{}: {path} never takes {code}", p.group));
                }
            }
        }
    }
    assert!(missing.is_empty(), "{}", report(&missing));
}

#[test]
fn every_text_result_has_a_branches_entry() {
    // A result that can be text (a verdict, a sentinel, a built message) must list
    // its branches in BRANCHES, so every branch is compared with Python at least once.
    let results = result_rows(&compute_all(&DesignInputs::default()));
    let missing: Vec<String> = results
        .iter()
        .filter(|r| matches!(r.meta.ty, FieldType::Text | FieldType::NumOrText))
        .filter(|r| !r.meta.rust_only) // no Python counterpart, so no differential data to reach
        .filter(|r| {
            !BRANCHES
                .iter()
                .any(|(pattern, _)| matches_pattern(pattern, &r.path))
        })
        .map(|r| r.path.clone())
        .collect();
    assert!(
        missing.is_empty(),
        "text results without BRANCHES rows: {missing:?}"
    );
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
