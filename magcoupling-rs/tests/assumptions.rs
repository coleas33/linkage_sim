//! Addendum A3: the assumptions registry (`src/engine/assumptions.rs`) against the input
//! metadata, and its engine support ("assumptions modified", reset to workbook defaults).
//!
//! The spec's traceability test ("changing each assumption changes every dependent result
//! and no independent one, using the equation registry's dependency graph") needs the A2
//! equation registry and belongs to plan A-3. `each_assumption_moves_a_result_at_the_default_design`
//! is the smoke version: each assumption moves at least one result. Of the three documented
//! overrides (`src/engine/assumptions.rs`), only C45's shows at the default design: its
//! library parts still read C22, and it has no 1018 back iron.

use std::collections::BTreeSet;

use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::assumptions::{
    ASSUMPTIONS, any_modified, modified, reset_to_workbook_defaults, states,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputMeta, InputSet, Value, input_rows, result_rows};

/// The metadata of the input at `path`.
fn meta(path: &str) -> &'static InputMeta {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|r| r.path == path)
        .unwrap_or_else(|| panic!("{path}: not an input"))
        .meta
}

/// A value of the input at `path` other than its default, inside its slider or among its
/// choices: the range end farther from the default, or the first other choice.
fn other_value(path: &str) -> Value {
    let m = meta(path);
    let default = DesignInputs::default().get(path).expect("an input");
    if !m.choices.is_empty() {
        let code = m
            .choices
            .iter()
            .map(|&(c, _)| c)
            .find(|&c| Value::Int(c) != default)
            .expect("a second choice");
        return Value::Int(code);
    }
    let r = m.range.expect("a numeric assumption has a slider");
    let x = match default {
        Value::Num(x) => x,
        other => panic!("{path}: {other:?}"),
    };
    Value::Num(if (r.max - x).abs() >= (x - r.min).abs() {
        r.max
    } else {
        r.min
    })
}

#[test]
fn every_flagged_input_is_in_exactly_one_row() {
    let flagged: BTreeSet<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| r.meta.assumption)
        .map(|r| r.path)
        .collect();
    let mut listed = BTreeSet::new();
    for a in &ASSUMPTIONS {
        assert!(!a.paths.is_empty(), "{}", a.id);
        for &path in a.paths {
            assert!(listed.insert(path.to_owned()), "{path} is in two rows");
            assert!(meta(path).assumption, "{path} is not flagged .assumption()");
        }
    }
    assert_eq!(listed, flagged);
}

#[test]
fn the_rows_are_the_spec_v1_set() {
    // Spec A3, in its order: harmonics included, end-effect coefficient, calibration factor,
    // production variation, Br and Hcj temperature coefficients, demag knee fraction, demag
    // margin, back-iron design flux density, thermal conductance, driving rise, slip-event
    // duration, clamp friction coefficient, preload fraction of proof load.
    let ids: Vec<&str> = ASSUMPTIONS.iter().map(|a| a.id).collect();
    assert_eq!(
        ids,
        [
            "harmonics",
            "end_effect",
            "calibration_factor",
            "production_variation",
            "br_temperature_coefficient",
            "hcj_temperature_coefficient",
            "knee_fraction",
            "demag_margin",
            "backiron_design_flux_density",
            "thermal_conductance",
            "driving_rise",
            "slip_event_duration",
            "clamp_friction",
            "preload_fraction",
        ]
    );
    // The end-effect coefficient is the Calculator's and the Calibration's (report 6.5);
    // the clamp friction is the shaft-to-bore input, not the adapter joint's C66.
    let paths = |id: &str| ASSUMPTIONS.iter().find(|a| a.id == id).unwrap().paths;
    assert_eq!(paths("end_effect"), ["coupling.c_end", "calibration.c_end"]);
    assert_eq!(paths("clamp_friction"), ["clamps.friction"]);
    assert_eq!(paths("harmonics"), ["coupling.max_harmonic"]);
}

#[test]
fn every_row_has_a_label_rationale_source_and_one_unit() {
    let mut ids = BTreeSet::new();
    for a in &ASSUMPTIONS {
        assert!(ids.insert(a.id), "{}: duplicate id", a.id);
        for (what, text) in [
            ("label", a.label),
            ("rationale", a.rationale),
            ("source", a.source),
        ] {
            assert!(!text.trim().is_empty(), "{}: empty {what}", a.id);
        }
        let unit = meta(a.paths[0]).unit;
        for &path in a.paths {
            let m = meta(path);
            assert_eq!(m.unit, unit, "{}: {path} has another unit", a.id);
            // The source cites the workbook cell of every workbook input it sets; a Rust-only
            // input (the harmonic set) has none and cites the spec.
            match m.cell {
                Some(cell) => assert!(a.source.contains(cell), "{}: source omits {cell}", a.id),
                None => assert!(a.source.contains("Addendum A3"), "{}", a.id),
            }
        }
    }
}

#[test]
fn the_workbook_defaults_are_the_shipped_defaults() {
    // No approved correction changes an assumption's default (E1, E3 and E5 correct other
    // inputs), so "reset to workbook defaults" is `DesignInputs::default()` for every row; the
    // Rust-only harmonic set's workbook value is 1, 3, 5.
    let shipped = DesignInputs::default();
    let workbook = DesignInputs::defaults_with(Deviations::NONE);
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            assert_eq!(shipped.get(path), workbook.get(path), "{path}");
        }
    }
    assert_eq!(shipped.get("coupling.max_harmonic"), Some(Value::Int(5)));
}

#[test]
fn nothing_is_modified_at_the_defaults() {
    let inputs = DesignInputs::default();
    assert!(!any_modified(&inputs));
    assert!(modified(&inputs).is_empty());
    let panel = states(&inputs);
    assert_eq!(panel.len(), ASSUMPTIONS.len());
    for (s, a) in panel.iter().zip(&ASSUMPTIONS) {
        assert_eq!(s.assumption, a);
        assert!(!s.modified, "{}", a.id);
        assert_eq!(s.values, s.defaults, "{}", a.id);
        assert_eq!(s.unit, meta(a.paths[0]).unit);
        assert_eq!(s.values.len(), a.paths.len());
    }
}

#[test]
fn changing_one_assumption_flags_exactly_its_row() {
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let mut inputs = DesignInputs::default();
            inputs.set(path, other_value(path)).unwrap();
            let ids: Vec<&str> = modified(&inputs).iter().map(|m| m.id).collect();
            assert_eq!(ids, [a.id], "{path}");
            assert!(any_modified(&inputs), "{path}");
            let s = states(&inputs)
                .into_iter()
                .find(|s| s.assumption.id == a.id)
                .unwrap();
            assert!(s.modified && s.values != s.defaults, "{path}");
        }
    }
    // A design input is not an assumption: the banner stays off.
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = 12;
    inputs.metal.face_gap_mm = 1.2;
    assert!(!any_modified(&inputs));
}

#[test]
fn reset_restores_every_assumption_and_keeps_the_design_inputs() {
    let mut design = DesignInputs::default();
    design.coupling.npole = 12;
    design.metal.face_gap_mm = 1.2;
    design.coupling.magnets.part_inner = "B842".to_owned();
    design.materials.parts.back_iron = 2;
    let mut inputs = design.clone();
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            inputs.set(path, other_value(path)).unwrap();
        }
    }
    assert_eq!(modified(&inputs).len(), ASSUMPTIONS.len());
    reset_to_workbook_defaults(&mut inputs);
    assert!(!any_modified(&inputs));
    assert_eq!(inputs, design, "only the assumptions change");
}

#[test]
fn each_assumption_moves_a_result_at_the_default_design() {
    // The smoke version of A3's traceability test (the dependency-graph version is plan A-3's).
    // Documented override (A-1 plan decisions A2 and A9): with correction E20 and the coercivity
    // source at 1 (the default), the Hcj temperature coefficient (C45) acts only for a magnet
    // without a grade, and every library part has one; with the source at 0 it moves results.
    // The other two overrides (a 1018 back iron's own flux density; a grade-mode ring's own
    // alpha(Br) in place of C22, decision A2-7) do not apply here: the default design has
    // library rings and no 1018 back iron, so the design flux density and C22 move results.
    let values = |inputs: &DesignInputs| -> Vec<Value> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .map(|r| r.value)
            .collect()
    };
    let base = values(&DesignInputs::default());
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let mut inputs = DesignInputs::default();
            inputs.set(path, other_value(path)).unwrap();
            let moved = values(&inputs) != base;
            if path == "temperature.demag.beta_hcj_per_C" {
                assert!(!moved, "{path}: the grade's beta governs (E20)");
                inputs.temperature.demag.coercivity_source = 0;
                let mut source_0 = DesignInputs::default();
                source_0.temperature.demag.coercivity_source = 0;
                assert!(values(&inputs) != values(&source_0), "{path} with source 0");
            } else {
                assert!(moved, "{path} moves no result");
            }
        }
    }
    // Every assumption is numeric (tests/schema.rs); the harmonic set is the one selector.
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let m = meta(path);
            assert!(matches!(m.ty, FieldType::F64 | FieldType::I64), "{path}");
        }
    }
}
