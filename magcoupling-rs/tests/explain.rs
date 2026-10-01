//! Addendum A2 explanation layer: the drift guard, the registry's structure and the v1
//! scope.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
//! text exact), corrections on (`compute_all`, what users see), at: the defaults; every
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override), without which the tau7-tau11 records would compare
//! 0 with 0. Anti-vacuity checks require every `cases` arm of every record to be taken, every
//! numeric record and every value term of every record to take two values, and the E7
//! angles to leave half a pitch.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_2;

use common::{differential_files, load_cases, report};
use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::explain::markup::{Expr, Symbol};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, render};
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};

/// One input set the guard evaluates at.
struct Point {
    label: String,
    inputs: DesignInputs,
}

/// Rust-only inputs the differential generator never varies, applied on top of each case.
fn augmentations() -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    let text = |s: &str| Value::Text(s.to_owned());
    let mut v: Vec<(&'static str, Vec<(&'static str, Value)>)> = vec![("as generated", vec![])];
    for (label, code) in [
        ("harmonics 1", 1),
        ("harmonics 3", 3),
        ("harmonics 7", 7),
        ("harmonics 9", 9),
        ("harmonics 11", 11),
    ] {
        v.push((label, vec![("coupling.max_harmonic", Value::Int(code))]));
    }
    for (label, code) in [
        ("back iron 1018", 2),
        ("back iron 304", 7),
        ("back iron 6061", 8),
    ] {
        v.push((label, vec![("materials.parts.back_iron", Value::Int(code))]));
    }
    v.push((
        "grade mode",
        vec![
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("N52")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Y30")),
        ],
    ));
    v.push((
        "axial 8 mm",
        vec![("coupling.magnets.axial_length_mm", Value::Num(8.0))],
    ));
    v.push((
        "everything",
        vec![
            ("coupling.max_harmonic", Value::Int(11)),
            ("materials.parts.back_iron", Value::Int(8)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("N35EH")),
            ("coupling.magnets.axial_length_mm", Value::Num(20.0)),
        ],
    ));
    v
}

/// The defaults, then every differential case under every augmentation.
fn guard_points() -> Vec<Point> {
    let mut points = vec![Point {
        label: "defaults".into(),
        inputs: DesignInputs::default(),
    }];
    let augmentations = augmentations();
    for file in differential_files() {
        for case in load_cases(file) {
            let mut base = DesignInputs::default();
            for (path, value) in &case.inputs {
                base.set(path, value.clone())
                    .unwrap_or_else(|e| panic!("{file} case {}: {e}", case.id));
            }
            for (label, sets) in &augmentations {
                let mut inputs = base.clone();
                for (path, value) in sets {
                    inputs
                        .set(path, value.clone())
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                }
                points.push(Point {
                    label: format!("{file} case {} ({}), {label}", case.id, case.tag),
                    inputs,
                });
            }
        }
    }
    points
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) if x.is_nan() && y.is_nan() => true,
        _ => parity_close(a, b),
    }
}

/// What the guard learned across all points.
#[derive(Default)]
struct Coverage {
    arms: BTreeMap<String, BTreeSet<(usize, usize)>>,
    values: BTreeMap<String, BTreeSet<u64>>,
    angles_off_half_pitch: BTreeMap<String, usize>,
    /// Per equation (in `Registry::equations` order): each value term's distinct values,
    /// up to two.
    term_values: Vec<BTreeMap<String, Vec<Value>>>,
}

/// Adds `v` to `seen` unless it holds it already or holds two (two show the term varies;
/// every NaN is one value).
fn note_value(seen: &mut Vec<Value>, v: Value) {
    let same = |a: &Value| match (a, &v) {
        (Value::Num(x), Value::Num(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        (a, b) => a == b,
    };
    if seen.len() < 2 && !seen.iter().any(same) {
        seen.push(v);
    }
}

/// Evaluates every record at every point of `points`: the failures and the coverage.
fn guard(registry: &Registry, points: &[Point]) -> (Vec<String>, Coverage) {
    let mut failures = Vec::new();
    let mut cov = Coverage {
        term_values: vec![BTreeMap::new(); registry.equations().len()],
        ..Coverage::default()
    };
    for p in points {
        let results = compute_all(&p.inputs);
        let src = Design {
            inputs: &p.inputs,
            results: &results,
        };
        for (i, eq) in registry.equations().iter().enumerate() {
            let want = results
                .get(&eq.target)
                .expect("a record targets a result path");
            let mut trace = Trace::default();
            match registry.evaluate(eq, &src, Some(&mut trace)) {
                Ok(got) if same(&got, &want) => {}
                Ok(got) => failures.push(format!(
                    "{} at {}: record {got:?}, engine {want:?}",
                    eq.target, p.label
                )),
                Err(e) => failures.push(format!("{} at {}: {e}", eq.target, p.label)),
            }
            cov.arms
                .entry(eq.target.clone())
                .or_default()
                .extend(trace.arms);
            let terms = &mut cov.term_values[i];
            for term in trace.value_terms {
                match terms.get_mut(&term) {
                    Some(seen) => {
                        if seen.len() < 2 {
                            note_value(seen, src.value(&term).unwrap_or(Value::None));
                        }
                    }
                    None => {
                        let v = src.value(&term).unwrap_or(Value::None);
                        terms.insert(term, vec![v]);
                    }
                }
            }
            if let Value::Num(x) = want {
                cov.values
                    .entry(eq.target.clone())
                    .or_default()
                    .insert(x.to_bits());
                if eq.target.ends_with("angle_rad") && x != FRAC_PI_2 {
                    *cov.angles_off_half_pitch
                        .entry(eq.target.clone())
                        .or_default() += 1;
                }
            }
        }
    }
    (failures, cov)
}

impl Coverage {
    fn merge(&mut self, other: Coverage) {
        for (k, v) in other.arms {
            self.arms.entry(k).or_default().extend(v);
        }
        for (k, v) in other.values {
            self.values.entry(k).or_default().extend(v);
        }
        for (k, n) in other.angles_off_half_pitch {
            *self.angles_off_half_pitch.entry(k).or_default() += n;
        }
        if self.term_values.is_empty() {
            self.term_values = other.term_values;
        } else {
            for (mine, theirs) in self.term_values.iter_mut().zip(other.term_values) {
                for (term, values) in theirs {
                    let seen = mine.entry(term).or_default();
                    for v in values {
                        note_value(seen, v);
                    }
                }
            }
        }
    }
}

/// `cases` arms no input can reach, each with its reason (the anti-vacuity check skips them).
const UNREACHABLE_ARMS: &[(&str, usize, usize, &str)] = &[
    (
        "model.tau1_Pa",
        0,
        1,
        "the fundamental is in every harmonic set (the smallest choice is 1)",
    ),
    ("calibration.tau1_Pa", 0, 1, "as model.tau1_Pa"),
];

#[test]
fn every_record_reproduces_the_engine_everywhere() {
    let registry = Registry::build();
    let points = guard_points();
    // The defaults, then every differential case under every augmentation (no stale count:
    // the data files are regenerated).
    let cases: usize = differential_files()
        .into_iter()
        .map(|f| load_cases(f).len())
        .sum();
    assert!(cases > 0);
    assert_eq!(
        points.len(),
        1 + cases * augmentations().len(),
        "{} points",
        points.len()
    );
    // The points are independent: evaluate them on every core (about 1 s instead of 30 in a
    // debug build).
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = points.len().div_ceil(threads);
    let (mut failures, mut cov) = (Vec::new(), Coverage::default());
    std::thread::scope(|s| {
        let handles: Vec<_> = points
            .chunks(chunk)
            .map(|c| s.spawn(|| guard(&registry, c)))
            .collect();
        for h in handles {
            let (f, c) = h.join().expect("a guard thread");
            failures.extend(f);
            cov.merge(c);
        }
    });
    assert!(failures.is_empty(), "drift guard: {}", report(&failures));

    // Anti-vacuity: every arm of every `cases` taken somewhere.
    let mut untaken = Vec::new();
    for eq in registry.equations() {
        let mut ids = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Cases { id, arms, .. } = e {
                ids.insert(*id, arms.len());
            }
        });
        let taken = cov.arms.get(&eq.target).cloned().unwrap_or_default();
        for (id, n) in ids {
            for arm in 0..=n {
                let exempt = UNREACHABLE_ARMS
                    .iter()
                    .any(|&(t, i, a, _)| t == eq.target && i == id && a == arm);
                if taken.contains(&(id, arm)) {
                    assert!(
                        !exempt,
                        "{}: cases {id} arm {arm} is listed unreachable but is taken",
                        eq.target
                    );
                } else if !exempt {
                    untaken.push(format!("{}: cases {id} arm {arm}", eq.target));
                }
            }
        }
    }
    assert!(untaken.is_empty(), "arms never taken: {}", report(&untaken));
    // Every value term of every record takes two values somewhere: a term the guard sees at
    // one value only is indistinguishable there from a literal of that value, so a record
    // could lose a real dependency and still pass.
    let mut single = Vec::new();
    for (eq, terms) in registry.equations().iter().zip(&cov.term_values) {
        for (term, seen) in terms {
            if seen.len() < 2 {
                single.push(format!("{} reads {term} only as {:?}", eq.target, seen[0]));
            }
        }
    }
    assert!(
        single.is_empty(),
        "value terms seen at one value only: {}",
        report(&single)
    );
    // Every numeric record varies across the points (it is exercised, not a constant).
    let constant: Vec<&String> = cov
        .values
        .iter()
        .filter(|(_, v)| v.len() < 2)
        .map(|(k, _)| k)
        .collect();
    assert!(constant.is_empty(), "records that never vary: {constant:?}");
    // E7 leaves half a pitch for every angle record somewhere.
    for eq in registry
        .equations()
        .iter()
        .filter(|e| e.target.ends_with("angle_rad"))
    {
        let n = cov
            .angles_off_half_pitch
            .get(&eq.target)
            .copied()
            .unwrap_or(0);
        assert!(n > 0, "{}: never off half a pitch", eq.target);
    }
    // The harmonic 7 to 11 records are nonzero somewhere (the augmentations reach them).
    for n in [7, 9, 11] {
        for t in [format!("model.tau{n}_Pa"), format!("calibration.tau{n}_Pa")] {
            let values = &cov.values[&t];
            assert!(
                values.iter().any(|&b| f64::from_bits(b) != 0.0),
                "{t} is always 0"
            );
        }
    }
}

#[test]
fn input_and_result_paths_are_disjoint() {
    // `Design` looks a term up among the results, then the inputs: no path may be both.
    let inputs = DesignInputs::default();
    let ins: BTreeSet<String> = input_rows(&inputs).into_iter().map(|r| r.path).collect();
    let outs: BTreeSet<String> = result_rows(&compute_all(&inputs))
        .into_iter()
        .map(|r| r.path)
        .collect();
    assert!(
        ins.is_disjoint(&outs),
        "{:?}",
        ins.intersection(&outs).collect::<Vec<_>>()
    );
}

#[test]
fn the_registry_is_consistent() {
    let r = Registry::build();
    for eq in r.equations() {
        assert_eq!(
            r.equation_for(&eq.target).map(|e| &e.target),
            Some(&eq.target)
        );
        assert_eq!(r.term_kind(&eq.target), Some(TermKind::Explained));
        for t in &eq.terms {
            assert!(
                r.used_by(t).contains(&eq.target),
                "{t} used by {}",
                eq.target
            );
            assert!(r.term_kind(t).is_some(), "{t}");
            assert!(Symbol::parse(r.symbol(t).expect("every term has a symbol")).is_ok());
        }
        // The plain rendering names every term by its symbol, never by its path.
        let text = render::plain(&r, &eq.symbol, &eq.formula);
        assert!(!text.contains('['), "{}: {text}", eq.target);
    }
    // A stacked fraction as a factor is parenthesized in the plain line, so it reads as the
    // tree does.
    let f_end = r.equation_for("model.f_end").unwrap();
    assert_eq!(
        render::plain(&r, &f_end.symbol, &f_end.formula),
        "f_{end} = 1 − c_{end} · (τ_p/L)"
    );
    // No custom evals in the tracer: every displayed formula is the evaluated one.
    let custom: Vec<&str> = r
        .equations()
        .iter()
        .filter(|e| matches!(e.eval, Eval::Custom(_)))
        .map(|e| e.target.as_str())
        .collect();
    assert!(custom.is_empty(), "{custom:?}");
    assert!(r.is_leaf_input("metal.variation") && !r.is_leaf_input("metal.torque_hot_low_Nm"));
    assert_eq!(
        r.term_kind("metal.variation"),
        Some(TermKind::Input { assumption: true })
    );
    assert_eq!(r.family_symbol("model.b_i#").as_deref(), Some("B_{i,n}"));
    // The term list shows each term in the unit the formula reads it in.
    let inputs = DesignInputs::default();
    let results = compute_all(&inputs);
    let src = Design {
        inputs: &inputs,
        results: &results,
    };
    let rows = r.term_rows(r.equation_for("model.k3").unwrap(), &src);
    let rg = rows
        .iter()
        .find(|x| x.path == "model.gap_radius_mm")
        .unwrap();
    assert_eq!(rg.unit, "m");
    assert_eq!(
        rg.value,
        Some(Value::Num(results.model.gap_radius_mm * 1e-3))
    );
    assert_eq!(rg.symbol, "R_g");
    let n = rows.iter().find(|x| x.path == "coupling.npole").unwrap();
    assert_eq!(
        (n.unit.as_str(), n.kind),
        ("-", TermKind::Input { assumption: false })
    );
    let mut eleven = DesignInputs::default();
    eleven.set("coupling.max_harmonic", Value::Int(11)).unwrap();
    for (inputs, want) in [(DesignInputs::default(), 3), (eleven, 6)] {
        let results = compute_all(&inputs);
        let members = r.family_members(
            "model.b_i#",
            &Design {
                inputs: &inputs,
                results: &results,
            },
        );
        assert_eq!(members.len(), want);
        assert_eq!(members[0], "model.b_i1");
    }
}

/// The physics reviewer's sheet for a batch: every record's path, cell, symbol, rendered
/// formula and corrections, as a Markdown table on stdout. Run with
/// `cargo test --test explain review_sheet -- --ignored --nocapture`.
#[test]
#[ignore = "a review tool, not a check"]
fn review_sheet() {
    let r = Registry::build();
    println!("| Result | Cell | Formula | Corrections |\n|---|---|---|---|");
    for eq in r.equations() {
        let formula = render::plain(&r, &eq.symbol, &eq.formula).replace('|', "\\|");
        let cell = eq.cell.as_deref().unwrap_or("Rust-only");
        println!(
            "| `{}` | {cell} | {formula} | {:?} |",
            eq.target, eq.corrections
        );
    }
}

#[test]
fn every_explained_chain_drills_down_to_inputs() {
    // Every term below every record of an explained chain is an input or has a record:
    // nothing stops at a cell-only result.
    let r = Registry::build();
    let explained: Vec<_> = SCOPE
        .iter()
        .filter(|c| c.status == Status::Explained)
        .collect();
    assert!(explained.iter().any(|c| c.id == "torque"));
    for chain in explained {
        for path in chain.paths {
            for up in r.upstream(&path.replace("[]", "[0]")) {
                assert!(
                    matches!(
                        r.term_kind(&up),
                        Some(TermKind::Input { .. } | TermKind::Explained)
                    ),
                    "{}: {path} depends on {up}, which is neither an input nor explained",
                    chain.id
                );
            }
        }
    }
}

#[test]
fn explained_chains_have_every_record_and_scope_paths_exist() {
    let r = Registry::build();
    let results: BTreeSet<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .map(|x| x.path)
        .collect();
    let mut missing = Vec::new();
    for chain in SCOPE {
        for path in chain.paths {
            // A table column (`clamps.table[].x`) exists when its first row does.
            let first_row = path.replace("[]", "[0]");
            assert!(
                results.contains(&first_row),
                "{}: {path} is not a result path",
                chain.id
            );
            if chain.status == Status::Explained && r.equation_for(&first_row).is_none() {
                missing.push(format!("{}: {path}", chain.id));
            }
        }
    }
    assert!(missing.is_empty(), "{missing:?}");
    let union: BTreeSet<&str> = SCOPE.iter().flat_map(|c| c.paths.iter().copied()).collect();
    assert_eq!(
        union.len(),
        159,
        "decision 31: the chains and the dashboard (report section 7)"
    );
}
