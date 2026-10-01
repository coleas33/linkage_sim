//! Inputs no parity or differential case holds (the plan's Review Focus): a
//! selector code outside its choices set on the struct, a measured drag of
//! exactly zero, extreme typed values. The engine must never panic.

use magcoupling::compute_all;
use magcoupling::engine::api::{DesignInputs, compute_all_with, headline};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, NumOrText, SetErrorKind, Value, input_rows};

#[test]
fn an_invalid_adhesive_code_selects_no_adhesive() {
    for code in [0, 5, -1, i64::MIN, i64::MAX] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.temperature.adhesive.selected = code; // bypasses set(), which refuses it
        let t = compute_all_with(&inputs, Deviations::NONE).temperature;
        // Python's candidates[selected - 1] would silently take DP460 for code 0.
        assert_eq!(t.adhesive.selected_name, "#N/A", "{code}");
        assert!(
            t.adhesive.lap_shear_MPa.is_nan() && t.summary.adhesive_limit_C.is_nan(),
            "{code}"
        );
    }
}

#[test]
fn zero_measured_drag_does_not_panic() {
    // Python raises ZeroDivisionError (rotations per degree, C156/C157); the limit is +inf.
    for dev in [Deviations::NONE, Deviations::ALL] {
        let mut inputs = DesignInputs::defaults_with(dev);
        inputs.metal.measured_drag_Nm = Some(0.0);
        let t = compute_all_with(&inputs, dev).temperature;
        assert_eq!(t.thermal.rev_per_C_est, f64::INFINITY);
        assert_eq!(t.thermal.rev_per_C_high, f64::INFINITY);
        assert!(t.thermal.heat_capacity_J_K.is_finite() && t.summary.governing_limit_C.is_finite());
        assert_eq!(
            t.thermal.time_to_limit_high,
            NumOrText::Text("never: steady state stays below the limit")
        );
    }
}

#[test]
fn an_invalid_screw_class_is_nan_not_a_panic() {
    for code in [0, 4, i64::MIN] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.clamps.screw_class = code; // bypasses set()
        let c = compute_all_with(&inputs, Deviations::NONE).clamps;
        assert!(c.screw_proof_MPa.is_nan(), "{code}");
        assert!(c.table.iter().all(|r| r.preload_N.is_nan()), "{code}");
        assert!(
            c.recommended.ends_with("class #N/A") || c.recommended.starts_with("None"),
            "{code}: {}",
            c.recommended
        );
    }
}

#[test]
fn compute_all_never_panics_on_extreme_inputs() {
    // Review Focus 3 and 5: typed values far outside the sliders (set() accepts them,
    // as Python does). Results may be inf or NaN; nothing may panic (no integer overflow).
    // Once with the default steel back iron and once with none (backiron 0): the free-space
    // circuit is where E17's inputs (b_hub_free_T, b_cup_free_T, web_integral_free_T2m2) act.
    // compute_all applies every correction.
    for backiron in [1, 0] {
        let mut base = DesignInputs::default();
        base.coupling.backiron = backiron;
        let mut tried = 0;
        for row in input_rows(&base) {
            let values: Vec<Value> = match (row.meta.ty, row.meta.choices.is_empty()) {
                (FieldType::F64 | FieldType::OptF64, _) => {
                    let r = row
                        .meta
                        .range
                        .expect("every numeric input has a range (tests/schema.rs)");
                    [0.0, -1.0, r.min / 10.0, r.max * 10.0, 1e300, -1e300]
                        .map(Value::Num)
                        .to_vec()
                }
                // 2: Review Focus 3's `coupling.npole = 2` (tan(pi/2) is huge but finite).
                (FieldType::I64, true) => {
                    [0, 2, -2, 3, i64::MAX, i64::MIN].map(Value::Int).to_vec()
                }
                _ => continue, // selectors: next test; text: any text is valid (manual magnet)
            };
            for value in values {
                let mut inputs = base.clone();
                inputs
                    .set(&row.path, value.clone())
                    .unwrap_or_else(|e| panic!("{e}"));
                let _ = compute_all(&inputs);
                tried += 1;
            }
        }
        assert!(
            tried > 800,
            "backiron {backiron}: only {tried} extreme cases"
        );
    }
}

#[test]
fn compute_all_never_panics_on_non_finite_struct_literals() {
    // Decision D3: set() refuses NaN and infinities, but a struct literal, a design file or
    // a share link can hold one. compute_all must not panic on it (gate 4 is a debug build,
    // overflow checks on), with the corrections off or on; validate() names the one path.
    type Put = fn(&mut DesignInputs, f64);
    let cases: [(&str, Put); 5] = [
        ("metal.face_gap_mm", |i, x| i.metal.face_gap_mm = x),
        ("clamps.boss_od_mm", |i, x| i.clamps.boss_od_mm = x),
        ("temperature.thermal.conductance_W_K", |i, x| {
            i.temperature.thermal.conductance_W_K = x
        }),
        ("metal.measured_drag_Nm", |i, x| {
            i.metal.measured_drag_Nm = Some(x)
        }),
        ("coupling.magnets.axial_length_mm", |i, x| {
            i.coupling.magnets.axial_length_mm = Some(x)
        }),
    ];
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (path, put) in cases {
            for dev in [Deviations::NONE, Deviations::ALL] {
                let mut inputs = DesignInputs::defaults_with(dev);
                put(&mut inputs, x);
                let _ = compute_all_with(&inputs, dev);
                let errors = inputs.validate().expect_err(path);
                assert_eq!(errors.len(), 1, "{path} {x}: {errors:?}");
                assert_eq!(
                    (errors[0].path.as_str(), &errors[0].kind),
                    (path, &SetErrorKind::NotFinite),
                    "{x}"
                );
            }
        }
        // A non-finite measured drag together with a pole count far outside its slider.
        for dev in [Deviations::NONE, Deviations::ALL] {
            let mut inputs = DesignInputs::defaults_with(dev);
            inputs.metal.measured_drag_Nm = Some(x);
            inputs.coupling.npole = i64::MAX;
            let _ = compute_all_with(&inputs, dev);
        }
    }
}

#[test]
fn compute_all_never_panics_on_selector_codes_outside_the_choices() {
    // Review Focus 1: codes set on the struct, bypassing set(). validate() names every one.
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 7;
    inputs.coupling.faceted = -1;
    inputs.calibration.gap_definition = 9;
    inputs.temperature.adhesive.selected = 0;
    inputs.clamps.clamp_type = 3;
    inputs.clamps.alloy = 0;
    inputs.clamps.screw_class = i64::MIN;
    let res = compute_all(&inputs);
    assert_eq!(res.temperature.adhesive.selected_name, "#N/A");
    let errors = inputs.validate().expect_err("seven invalid codes");
    let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "coupling.backiron",
            "coupling.faceted",
            "calibration.gap_definition",
            "temperature.adhesive.selected",
            "clamps.clamp_type",
            "clamps.alloy",
            "clamps.screw_class",
        ]
    );
    assert!(DesignInputs::default().validate().is_ok());
    let mut nan = DesignInputs::default();
    nan.metal.face_gap_mm = f64::NAN;
    assert_eq!(
        nan.validate().expect_err("NaN")[0].path,
        "metal.face_gap_mm"
    );
}

#[test]
fn an_invalid_harmonic_set_is_nan_not_another_set() {
    // A Rust-only selector set on the struct (bypassing set()): the Calculator, the
    // Calibration prototype and every sweep row read NaN, never another harmonic set, with
    // the corrections off and on; validate() names the path.
    for dev in [Deviations::NONE, Deviations::ALL] {
        let mut inputs = DesignInputs::defaults_with(dev);
        inputs.coupling.max_harmonic = 4;
        let res = compute_all_with(&inputs, dev);
        assert!(res.model.pullout_Nm.is_nan() && res.metal.torque_hot_low_Nm.is_nan());
        assert!(res.calibration.model_torque_Nm.is_nan());
        assert!(res.gap_sweep.iter().all(|r| r.tau_Pa.is_nan()));
        assert!(res.pole_sweep.iter().all(|r| r.pullout_op_Nm.is_nan()));
        let errors = inputs.validate().expect_err("an invalid code");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].path, "coupling.max_harmonic");
        assert_eq!(errors[0].kind, SetErrorKind::NotAChoice { code: 4 });
    }
}

#[test]
fn an_invalid_coercivity_source_uses_the_inputs() {
    // A Rust-only selector set on the struct (bypassing set()): any code but 1 means the Hcj
    // and beta inputs for both rings, as the catch-all else of a two-way IF (Global
    // Constraints); validate() names it.
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B842".into();
    let graded = compute_all(&inputs).temperature.demag;
    inputs.temperature.demag.coercivity_source = 7;
    let res = compute_all(&inputs).temperature.demag;
    assert_eq!(graded.hcj20_used_kA_m, 954.9);
    assert_eq!(res.hcj20_used_kA_m, inputs.temperature.demag.hcj20_kA_m);
    let errors = inputs.validate().expect_err("an invalid code");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, "temperature.demag.coercivity_source");
    assert_eq!(errors[0].kind, SetErrorKind::NotAChoice { code: 7 });
}

#[test]
fn a_positive_beta_without_a_rating_has_no_hot_limit() {
    // E20: a positive beta typed into C45 (coercivity source 0) for manual magnets without a
    // grade: no knee on heating and no rating, so no magnet limit (+inf: the adhesive governs
    // C12) and no torque at it (NaN, where the workbook formula would give +inf).
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.part_inner = String::new();
    inputs.coupling.magnets.part_outer = String::new();
    inputs.temperature.demag.coercivity_source = 0;
    inputs.temperature.demag.beta_hcj_per_C = 0.0035;
    let t = compute_all(&inputs).temperature;
    assert_eq!(t.demag.magnet_limit_C, f64::INFINITY);
    assert!(t.demag.torque_at_limit_Nm.is_nan());
    assert_eq!(t.summary.governing_limit_C, t.summary.adhesive_limit_C);
    assert_eq!(t.summary.governing_note, "Adhesive governs.");
    assert_eq!(
        t.summary.verdict,
        "OK on temperature. Confirm drag torque and thermal cycling by test."
    );
}

#[test]
fn sizing_never_panics_on_extreme_designs() {
    // Inverse sizing on designs no slider reaches: it must return an outcome or an error for
    // every free variable, never panic (gate 4 is a debug build, overflow checks on).
    use magcoupling::engine::sizing::{FreeVariable, solve};
    let mut designs = Vec::new();
    let mut d = DesignInputs::default();
    d.coupling.npole = i64::MAX;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.metal.face_gap_mm = 1e300;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.coupling.magnets.part_inner = String::new();
    d.coupling.magnets.manual_inner_thickness_mm = 0.0;
    d.coupling.backiron = 0;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.metal.variation = 1.0; // the hot low torque is 0
    designs.push(d);
    for design in &designs {
        for variable in FreeVariable::ALL {
            let _ = solve(design, variable, 1.0);
        }
    }
}

#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
    // Spec: "milliseconds per call". Debug build, generous bound (a smoke check, not a benchmark).
    // A GUI frame recomputes and reads the dashboard numbers: compute_all plus headline.
    let inputs = DesignInputs::default();
    let start = std::time::Instant::now();
    for _ in 0..200 {
        std::hint::black_box(headline(&compute_all(std::hint::black_box(&inputs))));
    }
    let per_call = start.elapsed() / 200;
    assert!(
        per_call < std::time::Duration::from_millis(5),
        "{per_call:?} per call"
    );
}
