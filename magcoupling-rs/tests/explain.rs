//! Addendum A2/A3 explanation layer: the drift guard, the registry's structure, the v1
//! scope, the teaching-note links and the A3 traceability test.
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
//!
//! **Traceability (A3).** For each input (the assumptions first, as the spec asks, then every
//! input), at several design points: nudging it changes no explained result outside its
//! static dependency set (the registry's graph), and changes every numeric result on its
//! active dependency path (the branch taken, the min/max winner; see `eval::Trace`).

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_2;

use common::{differential_files, load_cases, report};
use magcoupling::engine::api::{DesignInputs, DesignResults, compute_all};
use magcoupling::engine::assumptions::{self, ASSUMPTIONS};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::DeviationId;
use magcoupling::engine::explain::markup::{Expr, Symbol};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, notes, render};
use magcoupling::engine::library;
use magcoupling::engine::meta::{
    FieldType, InputMeta, InputSet, ResultSet, Value, input_rows, result_rows,
};

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
    // The three dependency queries agree, each as an exact set. `downstream` walks the
    // reverse edges (`used_by`), `upstream` walks the terms (the closure), and
    // `upstream_inputs` is the build's precomputed filter of that closure, so each half
    // checks a different one: the mirror catches a `downstream` that is too large (which
    // would make A3 soundness impossible to fail) or a closure that stops at the direct
    // terms; the intersection catches upstream inputs taken from the direct terms only
    // (which would leave results several hops down unstyled).
    let input_paths: BTreeSet<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|x| x.path)
        .collect();
    let upstream: BTreeMap<&str, BTreeSet<String>> = r
        .equations()
        .iter()
        .map(|e| (e.target.as_str(), r.upstream(&e.target)))
        .collect();
    for p in input_paths
        .iter()
        .map(String::as_str)
        .chain(upstream.keys().copied())
    {
        let mirror: BTreeSet<String> = upstream
            .iter()
            .filter(|(_, up)| up.contains(p))
            .map(|(&t, _)| t.to_owned())
            .collect();
        assert_eq!(r.downstream(p), mirror, "downstream of {p}");
    }
    for (&t, up) in &upstream {
        let want: BTreeSet<String> = up.intersection(&input_paths).cloned().collect();
        assert_eq!(r.upstream_inputs(t), Some(&want), "upstream inputs of {t}");
    }
}

/// The A4 accuracy gate for release (the M4 hands-on checklist runs it): every note the
/// "start here" order opens and every note an A5 warning links to has passed the physics
/// review. Red until the notes are reviewed; drafts never reach users meanwhile (`note_for`).
#[test]
#[ignore = "the M4 release gate: red until the notes are reviewed"]
fn release_notes_are_reviewed() {
    let mut ids: Vec<&str> = notes::START_HERE.iter().map(|(id, _)| *id).collect();
    ids.extend(
        magcoupling::engine::warnings::WARNING_RULES
            .iter()
            .map(|r| r.note_id),
    );
    let drafts: Vec<&str> = ids
        .into_iter()
        .filter(|id| {
            !matches!(
                notes::note(id).map(|n| n.review),
                Some(notes::Review::Reviewed { .. })
            )
        })
        .collect();
    assert!(drafts.is_empty(), "not yet reviewed: {drafts:?}");
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

#[test]
fn notes_link_to_records_and_each_equation_has_at_most_one() {
    let r = Registry::build();
    let results: BTreeSet<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .map(|x| x.path)
        .collect();
    for n in notes::NOTES {
        for entry in n.equations {
            let members: Vec<&String> =
                results.iter().filter(|p| notes::covers(entry, p)).collect();
            assert!(
                !members.is_empty(),
                "note {}: {entry} names no result",
                n.id
            );
            if n.sentences.is_empty() {
                continue; // a stub fixes an id; its links are checked when it is drafted
            }
            for m in members {
                assert!(
                    r.equation_for(m).is_some(),
                    "note {}: {m} has no record",
                    n.id
                );
            }
        }
    }
    for eq in r.equations() {
        let owners: Vec<&str> = notes::NOTES
            .iter()
            .filter(|n| n.equations.iter().any(|e| notes::covers(e, &eq.target)))
            .map(|n| n.id)
            .collect();
        assert!(owners.len() <= 1, "{}: notes {owners:?}", eq.target);
    }
    for (id, path) in notes::START_HERE {
        assert!(results.contains(*path), "start here {id}: {path}");
        let explained_chain = SCOPE
            .iter()
            .any(|c| c.status == Status::Explained && c.paths.contains(path));
        if explained_chain {
            assert!(
                r.equation_for(path).is_some(),
                "start here {id}: {path} has no record"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// A3 traceability
// ---------------------------------------------------------------------------------------

/// The nudged values of an input: a step both ways inside its slider for a number (around
/// the value when it is typed outside the slider), one step for a count, and for either the
/// slider's two ends; every other choice for a selector; another library part and "manual"
/// for a part name; set and unset for an optional number.
fn nudges(meta: &InputMeta, value: &Value) -> Vec<Value> {
    match (meta.ty, value) {
        (FieldType::I64, Value::Int(code)) if !meta.choices.is_empty() => meta
            .choices
            .iter()
            .map(|&(c, _)| c)
            .filter(|c| c != code)
            .map(Value::Int)
            .collect(),
        (FieldType::I64, Value::Int(n)) => {
            let r = meta.range.expect("a count has a slider");
            let step = r.step.max(1.0) as i64;
            let mut v = vec![if (*n + step) as f64 <= r.max {
                n + step
            } else {
                n - step
            }];
            // The slider's ends too: a threshold flips its verdict only under a large move.
            for end in [r.min as i64, r.max as i64] {
                if end != *n && !v.contains(&end) {
                    v.push(end);
                }
            }
            v.into_iter().map(Value::Int).collect()
        }
        (FieldType::F64 | FieldType::OptF64, Value::Num(x)) => {
            // Both ways inside the slider: a min() tie moves the result one way only. A value
            // typed outside the slider (hard ferrite's positive beta) is nudged where it is.
            let r = meta.range.expect("a number has a slider");
            let inside = |y: &f64| (r.min..=r.max).contains(y);
            let step = r.step.max(1e-3 * x.abs()).max(1e-12);
            let mut ys: Vec<f64> = if inside(x) {
                [x + step, x - step].into_iter().filter(inside).collect()
            } else {
                vec![x + step, x - step]
            };
            // The slider's ends too: a threshold (a minimum wall, a required torque) flips its
            // verdict only under a large move, and a slider narrower than the step (μ0's) is
            // still nudged.
            for end in [r.min, r.max] {
                if end != *x && !ys.contains(&end) {
                    ys.push(end);
                }
            }
            let mut v: Vec<Value> = ys.into_iter().map(Value::Num).collect();
            if meta.ty == FieldType::OptF64 {
                v.push(Value::None);
            }
            v
        }
        (FieldType::OptF64, Value::None) => {
            let r = meta.range.expect("a number has a slider");
            vec![Value::Num((r.min + r.max) / 2.0)]
        }
        (FieldType::Text, Value::Text(s)) if meta.name.starts_with("part_") => {
            let other = library::MAGNET_LIBRARY
                .iter()
                .map(|m| m.part)
                .find(|p| p != s)
                .expect("two parts");
            vec![Value::Text(other.to_owned()), Value::Text(String::new())]
        }
        (FieldType::Text, Value::Text(s)) if meta.name.starts_with("grade_") => {
            let g = if s == "N52" { "Y30" } else { "N52" };
            vec![Value::Text(g.to_owned()), Value::Text(String::new())]
        }
        (FieldType::Text, _) => vec![Value::Text("Loctite AA 326 + SF 7649".into())],
        other => panic!("{}: no nudge for {other:?}", meta.name),
    }
}

/// Design points for the traceability test: the defaults; the measured prototype's own
/// circuit (6061 back iron, so the bench correction is the calibration factor) at a test
/// temperature off 20 °C; a grade-mode ring with eleven harmonics at six poles; and a dozen
/// full-run cases under four of the drift guard's augmentations.
fn trace_points() -> Vec<(String, DesignInputs)> {
    let design = |sets: &[(&str, Value)]| {
        let mut inputs = DesignInputs::default();
        for (p, v) in sets {
            inputs.set(p, v.clone()).unwrap();
        }
        inputs
    };
    let text = |s: &str| Value::Text(s.to_owned());
    let mut pts = vec![
        ("defaults".to_owned(), DesignInputs::default()),
        (
            "prototype circuit, test at 35 °C".to_owned(),
            design(&[
                ("materials.parts.back_iron", Value::Int(8)),
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
        (
            "grade, 11 harmonics, 6 poles, test at 35 °C".to_owned(),
            design(&[
                ("coupling.max_harmonic", Value::Int(11)),
                ("coupling.magnets.part_inner", text("")),
                ("coupling.magnets.grade_inner", text("N52")),
                ("coupling.npole", Value::Int(6)),
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
    ];
    let augmentations: Vec<_> = augmentations()
        .into_iter()
        .filter(|(l, _)| {
            [
                "as generated",
                "harmonics 11",
                "back iron 6061",
                "grade mode",
            ]
            .contains(l)
        })
        .collect();
    for case in load_cases("full").into_iter().take(12) {
        for (label, sets) in &augmentations {
            let mut inputs = DesignInputs::default();
            for (path, value) in case
                .inputs
                .iter()
                .map(|(p, v)| (p.as_str(), v))
                .chain(sets.iter().map(|(p, v)| (*p, v)))
            {
                inputs.set(path, value.clone()).unwrap();
            }
            pts.push((format!("full case {}, {label}", case.id), inputs));
        }
    }
    pts
}

/// Any change at all: what soundness forbids for an independent result.
fn changed(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()),
        _ => a != b,
    }
}

/// A change above rounding noise (1e-10 relative): what sensitivity requires, so that an
/// algebraic cancellation cannot pass on its last-bit wobble.
fn moved_beyond_rounding(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => (x - y).abs() > 1e-10 * x.abs().max(y.abs()),
        _ => a != b,
    }
}

/// Each term's users along value edges at a design point: the records whose value moves
/// with that term there (each record's trace, `eval::Trace::value_terms`).
fn value_edges(
    r: &Registry,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> BTreeMap<String, Vec<String>> {
    let src = Design { inputs, results };
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for eq in r.equations() {
        let mut t = Trace::default();
        r.evaluate(eq, &src, Some(&mut t))
            .expect("the drift guard evaluates every record");
        for term in t.value_terms {
            edges.entry(term).or_default().push(eq.target.clone());
        }
    }
    edges
}

/// The explained results on `input`'s active path: reachable from it along value edges.
fn active_downstream(value_edges: &BTreeMap<String, Vec<String>>, input: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![input.to_owned()];
    while let Some(p) = stack.pop() {
        for user in value_edges.get(&p).into_iter().flatten() {
            if seen.insert(user.clone()) {
                stack.push(user.clone());
            }
        }
    }
    seen
}

/// Checks both sides of traceability for every input in `paths`; returns failures.
///
/// Soundness (nothing independent moves) holds at every point and nudge. Sensitivity (the
/// active path moves) is judged across the points together: an input need move a result at
/// one point where it is on its active path, so a point where a factor happens to vanish
/// (the test temperature at 20 °C in `1 + α (ϑ − 20)`) does not fail it. What never moves at
/// any point although the graph says it depends is an algebraic cancellation, which must be
/// listed in [`CANCELLATIONS`] with its reason (and a listed pair that does move fails).
fn check_traceability(paths: &[String], sensitivity: bool) -> Vec<String> {
    let r = Registry::build();
    let points = trace_points();
    let mut failures = Vec::new();
    let mut must_move: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut moved_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    let mut effective: BTreeSet<String> = BTreeSet::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = points
            .iter()
            .map(|(label, base)| s.spawn(|| trace_at(&r, label, base, paths)))
            .collect();
        for h in handles {
            let t = h.join().expect("a traceability thread");
            failures.extend(t.failures);
            for (k, at) in t.must_move {
                must_move.entry(k).or_insert(at);
            }
            moved_pairs.extend(t.moved);
            effective.extend(t.effective);
        }
    });
    // Not vacuous: every input checked had a nudge `set` accepted that changed some result
    // (explained or not) at some design point, so soundness was put to the test for it;
    // an input no result reads is listed in INERT_INPUTS instead.
    for input in paths {
        let inert = INERT_INPUTS.iter().any(|&(i, _)| i == input);
        match (effective.contains(input), inert) {
            (false, false) => failures.push(format!(
                "{input}: no accepted nudge changed any result at any design point"
            )),
            (true, true) => failures.push(format!(
                "INERT_INPUTS lists {input}, but a nudge changed a result: remove the entry"
            )),
            _ => {}
        }
    }
    if !sensitivity {
        return failures;
    }
    for ((input, result), at) in &must_move {
        let key = (input.clone(), result.clone());
        let listed = CANCELLATIONS
            .iter()
            .any(|&(i, r, _)| i == input && r == result);
        if !moved_pairs.contains(&key) && !listed {
            failures.push(format!("{input} never moves {result}, which is on its active path (first at {at}): a spurious edge, or a cancellation to list"));
        }
    }
    for &(input, result, _) in CANCELLATIONS {
        if paths.iter().any(|p| p == input)
            && moved_pairs.contains(&(input.to_owned(), result.to_owned()))
        {
            failures.push(format!(
                "CANCELLATIONS lists {input} → {result}, but it moves: remove the entry"
            ));
        }
    }
    failures
}

/// Inputs no result reads (the workbook shows them, or only the drawing reads them): no
/// nudge can change anything, so the non-vacuity check of [`check_traceability`] skips them
/// (and fails if a listed one ever changes a result).
const INERT_INPUTS: &[(&str, &str)] = &[
    (
        "materials.steel.density_g_cm3",
        "Materials C19 is shown for reference: the mass model reads Metal design C132 (the input's help says so)",
    ),
    (
        "clamps.key_width_mm",
        "only the drawing reads it (Python drawing.py); the key pressure divides by the contact height",
    ),
];

/// Pairs (input, result).
type Pairs = BTreeSet<(String, String)>;

/// What one design point of [`check_traceability`] found.
struct PointTrace {
    /// The soundness failures.
    failures: Vec<String>,
    /// The pairs on an active path that did not move, with where.
    must_move: BTreeMap<(String, String), String>,
    /// The pairs that moved above rounding.
    moved: Pairs,
    /// The inputs an accepted nudge changed some result of (explained or not).
    effective: BTreeSet<String>,
}

/// One design point of [`check_traceability`].
fn trace_at(r: &Registry, label: &str, base: &DesignInputs, paths: &[String]) -> PointTrace {
    let metas: BTreeMap<String, &'static InputMeta> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|x| (x.path, x.meta))
        .collect();
    let mut failures = Vec::new();
    let mut must_move: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut moved_pairs = Pairs::new();
    let mut effective = BTreeSet::new();
    {
        let base_results = compute_all(base);
        let base_rows = result_rows(&base_results);
        let edges = value_edges(r, base, &base_results);
        for path in paths {
            let meta = metas[path];
            let value = base.get(path).expect("an input path");
            let statics = r.downstream(path);
            let active = active_downstream(&edges, path);
            let _ = &statics;
            for nudged in nudges(meta, &value) {
                let mut inputs = base.clone();
                if inputs.set(path, nudged.clone()).is_err() {
                    continue; // outside the choices this design allows
                }
                let results = compute_all(&inputs);
                let mut any = false;
                for eq in r.equations() {
                    let (b, a) = (
                        base_results.get(&eq.target).unwrap(),
                        results.get(&eq.target).unwrap(),
                    );
                    let moved = changed(&b, &a);
                    any |= moved;
                    if moved && !statics.contains(&eq.target) {
                        failures.push(format!("{label}: {path} → {nudged:?} moves {} ({b:?} → {a:?}), which the graph says does not depend on it", eq.target));
                    }
                    let key = (path.clone(), eq.target.clone());
                    if moved_beyond_rounding(&b, &a) {
                        moved_pairs.insert(key);
                    } else if matches!(b, Value::Num(x) if x.is_finite())
                        && active.contains(&eq.target)
                    {
                        must_move
                            .entry(key)
                            .or_insert_with(|| format!("{label}, {nudged:?}"));
                    }
                }
                if !any && !effective.contains(path) {
                    // No explained result moved: did anything (a result outside the scope)?
                    any = result_rows(&results)
                        .iter()
                        .zip(&base_rows)
                        .any(|(a, b)| changed(&a.value, &b.value));
                }
                if any {
                    effective.insert(path.clone());
                }
            }
        }
    }
    PointTrace {
        failures,
        must_move,
        moved: moved_pairs,
        effective,
    }
}

/// Dependencies the graph names that cancel algebraically: the formula reads the input on
/// the way, but the result never moves with it. Each is a fact worth knowing (a teaching
/// note may cite it); `check_traceability` fails if a listed pair ever moves.
const CANCELLATIONS: &[(&str, &str, &str)] = &[
    (
        "calibration.f_cal_original",
        "calibration.f_cal_updated",
        "f_cal,1 = f_cal,0 · T_meas / T_model and T_model = T_2D f_end f_cal,0: the assumed factor cancels, so the bench correction is T_meas / (T_2D f_end)",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.pullout_angle_rad",
        "alpha scales every harmonic amplitude by the same Br(T) ratio, and a common scale leaves the peak angle where it is",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.iron_circuit_angle_rad",
        "as model.pullout_angle_rad",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.free_circuit_angle_rad",
        "as model.pullout_angle_rad",
    ),
    (
        "calibration.alpha_br_per_C",
        "calibration.pullout_angle_rad",
        "as model.pullout_angle_rad, through the prototype's Br at the test temperature",
    ),
];

#[test]
fn each_assumption_moves_what_depends_on_it_and_nothing_else() {
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two).
    let paths: Vec<String> = ASSUMPTIONS
        .iter()
        .flat_map(|a| a.paths.iter().map(|p| (*p).to_owned()))
        .collect();
    assert_eq!(paths.len(), 15);
    let failures = check_traceability(&paths, true);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn no_input_moves_a_result_the_graph_says_is_independent_of_it() {
    // Soundness for every input: the graph is complete. (Sensitivity is asserted for the
    // assumptions, as the spec asks: for all inputs the E8 corner geometry alone cancels
    // dozens of structural paths, e.g. the inner width reaches A_o through r_corner twice
    // with opposite signs.)
    let paths: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|x| x.path)
        .collect();
    let failures = check_traceability(&paths, false);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn a_modified_assumption_styles_its_term_and_what_it_flows_into() {
    let r = Registry::build();
    let mut inputs = DesignInputs::default();
    assert!(!assumptions::any_modified(&inputs));
    let style = |p: &str, i: &DesignInputs| r.term_style(p, i).expect("a known path");
    assert!(!style("metal.variation", &inputs).changed_from_default);
    inputs.set("metal.variation", Value::Num(0.2)).unwrap();
    assert!(assumptions::any_modified(&inputs), "the banner shows");
    let v = style("metal.variation", &inputs);
    assert_eq!(v.kind, TermKind::Input { assumption: true });
    assert!(v.changed_from_default);
    assert!(style("metal.torque_hot_low_Nm", &inputs).affected_by_modified_assumption);
    assert!(
        !style("model.pullout_Nm", &inputs).affected_by_modified_assumption,
        "variation does not reach the pull-out"
    );
    assert_eq!(
        r.modified_assumptions_upstream("metal.torque_cold_high_Nm", &inputs)
            .iter()
            .map(|a| a.id)
            .collect::<Vec<_>>(),
        ["production_variation"]
    );
    // Two hops: T_cold,high,MD reads only T_cold,high (a result), which reads the variation.
    assert!(style("metal.cold_high_Nm", &inputs).affected_by_modified_assumption);
    assert_eq!(
        r.modified_assumptions_upstream("metal.cold_high_Nm", &inputs)
            .iter()
            .map(|a| a.id)
            .collect::<Vec<_>>(),
        ["production_variation"]
    );
    // Several hops: T_pull's own terms (T_2D, f_end, f_cal) are all results, so α reaches
    // it only through them.
    let mut alpha = DesignInputs::default();
    alpha
        .set("calibration.alpha_br_per_C", Value::Num(-0.002))
        .unwrap();
    assert!(style("model.pullout_Nm", &alpha).affected_by_modified_assumption);
    assert_eq!(
        r.modified_assumptions_upstream("model.pullout_Nm", &alpha)
            .iter()
            .map(|a| a.id)
            .collect::<Vec<_>>(),
        ["br_temperature_coefficient"]
    );
    // A design input changed from its default gets the dot but is not an assumption.
    inputs.set("coupling.npole", Value::Int(12)).unwrap();
    let n = style("coupling.npole", &inputs);
    assert_eq!(n.kind, TermKind::Input { assumption: false });
    assert!(n.changed_from_default);
    assumptions::reset_to_workbook_defaults(&mut inputs);
    assert!(!assumptions::any_modified(&inputs), "the banner clears");
    assert!(!style("metal.torque_hot_low_Nm", &inputs).affected_by_modified_assumption);
    assert!(!style("metal.cold_high_Nm", &inputs).affected_by_modified_assumption);
    assert!(
        style("coupling.npole", &inputs).changed_from_default,
        "the reset keeps design inputs"
    );
}

#[test]
fn the_panel_labels_selector_codes_and_names_upstream_corrections() {
    let r = Registry::build();
    // A selector input shows its choice labels; a result holding a code borrows its input's.
    assert_eq!(
        r.choices("coupling.backiron"),
        [(1, "steel circuit"), (0, "no back iron")]
    );
    assert_eq!(
        r.choices("materials.circuit_backiron"),
        r.choices("coupling.backiron")
    );
    assert_eq!(
        r.choices("materials.parts.back_iron")[0],
        (1, "4140 annealed")
    );
    assert!(r.choices("coupling.npole").is_empty() && r.choices("model.pullout_Nm").is_empty());
    // The "corrected vs workbook" marker: the pull-out's own record names no correction, but
    // E7 acts upstream of it (the pull-out angle) and E8 in the corner gap.
    assert!(
        r.equation_for("model.pullout_Nm")
            .unwrap()
            .corrections
            .is_empty()
    );
    let pullout = r.corrections_upstream("model.pullout_Nm");
    assert!(
        pullout.contains(&DeviationId::E7) && pullout.contains(&DeviationId::E8),
        "{pullout:?}"
    );
    // A record's own corrections come first, each once.
    assert_eq!(
        r.corrections_upstream("model.pullout_angle_rad")[0],
        DeviationId::E7
    );
    for (i, c) in pullout.iter().enumerate() {
        assert!(!pullout[..i].contains(c), "{c:?} twice in {pullout:?}");
    }
    assert!(
        r.corrections_upstream("coupling.npole").is_empty(),
        "an input"
    );
}
