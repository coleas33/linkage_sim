//! BL-059: the gearbox check (`gearbox.cold_high_check`, `src/engine/gearbox.rs`), end to end
//! through `compute_all` and on `gearbox::compute` itself: cold-high at the gearbox input
//! (`metal.cold_high_input_Nm`) against the gearbox input torque rating
//! (`coupling.gearbox_input_rating_Nm`), owner decision 1A. A Rust-only result, tested here as
//! the space claim is in `tests/sizing.rs`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::explain::probes::probe_designs;
use magcoupling::engine::gearbox::{
    self, ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN, WITHIN_THE_GEARBOX_RATING,
};
use magcoupling::engine::meta::{
    InputSet, SetErrorKind, SliderRange, Value, input_rows, result_rows,
};

/// The failing verdict quoting a cold-high at the gearbox input and a rating, as the badge
/// quotes them (to 0.01 N·m).
fn above(cold_high_input: &str, rating: &str) -> String {
    format!("{ABOVE_THE_GEARBOX_RATING}: {cold_high_input} N·m against {rating} N·m")
}

/// The two numbers a failing verdict quotes: (cold-high at the gearbox input, the rating).
fn quoted(text: &str) -> (f64, f64) {
    let rest = text
        .strip_prefix(ABOVE_THE_GEARBOX_RATING)
        .and_then(|t| t.strip_prefix(": "))
        .unwrap_or_else(|| panic!("not a failing verdict: {text}"));
    let (torque, rating) = rest
        .split_once(" N·m against ")
        .unwrap_or_else(|| panic!("{text}"));
    let rating = rating
        .strip_suffix(" N·m")
        .unwrap_or_else(|| panic!("{text}"));
    (
        torque.parse().unwrap_or_else(|_| panic!("{text}")),
        rating.parse().unwrap_or_else(|_| panic!("{text}")),
    )
}

/// The default design with `edit` applied.
fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    edit(&mut inputs);
    inputs
}

/// The default design with both rings on their custom (manual) dimensions, `grade` on both.
fn custom_dimensions(grade: &str) -> DesignInputs {
    design(|i| {
        i.coupling.magnets.part_inner.clear();
        i.coupling.magnets.part_outer.clear();
        i.coupling.magnets.grade_inner = grade.to_owned();
        i.coupling.magnets.grade_outer = grade.to_owned();
    })
}

/// The rating's slider.
fn rating_range() -> SliderRange {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|row| row.path == "coupling.gearbox_input_rating_Nm")
        .and_then(|row| row.meta.range)
        .expect("the gearbox input rating has a slider")
}

/// The check's verdict with the rating at `rating` [N·m], the rest the defaults.
fn verdict_at_rating(rating: f64) -> String {
    compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = rating))
        .gearbox
        .cold_high_check
}

#[test]
fn the_defaults_fail_with_both_torques_quoted() {
    // Spec section 3: cold-high 3.8233 N·m at the gearbox output, 0.8049 N·m at its input
    // (÷ 5 ÷ 0.95), against the 0.6 N·m input rating.
    let inputs = DesignInputs::default();
    let r = compute_all(&inputs);
    let c = &inputs.coupling;
    assert_eq!(
        (c.gearbox_input_rating_Nm, c.gear_ratio, c.gear_efficiency),
        (0.6, 5.0, 0.95)
    );
    assert!(
        (r.metal.torque_cold_high_Nm - 3.8233).abs() < 5e-5,
        "{}",
        r.metal.torque_cold_high_Nm
    );
    assert!(
        (r.metal.cold_high_input_Nm - 0.8049).abs() < 5e-5,
        "{}",
        r.metal.cold_high_input_Nm
    );
    assert_eq!(r.gearbox.cold_high_check, above("0.80", "0.60"));
}

#[test]
fn a_rating_one_step_under_cold_high_quotes_the_torque_a_hundredth_above_it() {
    // At 0.80 N·m, one slider step, the rating sits under cold-high at the input (0.8049 N·m),
    // which rounds to 0.80 too: the badge quotes it at 0.81, never "0.80 against 0.80".
    let range = rating_range();
    let steps = (0.80 - range.min) / range.step;
    assert!(
        (steps - steps.round()).abs() < 1e-9,
        "0.80 is on the slider"
    );
    let x = compute_all(&DesignInputs::default())
        .metal
        .cold_high_input_Nm;
    assert!(x > 0.80 && x < 0.805, "{x}");
    assert_eq!(verdict_at_rating(0.80), above("0.81", "0.80"));
}

#[test]
fn each_probe_design_passes_exactly_when_cold_high_at_the_input_is_within_the_rating() {
    // The five probe designs of the notes (spec section 3), in `probe_designs()`' order: (name,
    // cold-high at the gearbox output and at its input [N·m], the verdict at the 0.6 N·m
    // rating). A change to a probe design moves its numbers here.
    let within = WITHIN_THE_GEARBOX_RATING.to_owned();
    let want: [(&str, f64, f64, String); 5] = [
        ("default", 3.8233, 0.8049, above("0.80", "0.60")),
        ("8 magnets per ring", 2.7992, 0.5893, within.clone()),
        ("304 back iron", 2.4138, 0.5082, within.clone()),
        ("Y30 ferrite, custom dimensions", 0.3381, 0.0712, within),
        ("20 mm ring radius", 3.6111, 0.7602, above("0.76", "0.60")),
    ];
    let designs = probe_designs();
    assert_eq!(designs.len(), want.len());
    for ((name, inputs), (want_name, cold_high, at_input, verdict)) in designs.into_iter().zip(want)
    {
        assert_eq!(name, want_name);
        let r = compute_all(&inputs);
        let c = &inputs.coupling;
        assert_eq!(c.gearbox_input_rating_Nm, 0.6, "{name}");
        assert!(
            (r.metal.torque_cold_high_Nm - cold_high).abs() < 5e-5,
            "{name}: {}",
            r.metal.torque_cold_high_Nm
        );
        assert!(
            (r.metal.cold_high_input_Nm - at_input).abs() < 5e-5,
            "{name}: {}",
            r.metal.cold_high_input_Nm
        );
        assert_eq!(r.gearbox.cold_high_check, verdict, "{name}");
        // The input basis is the workbook's reference at the output (rating × ratio ×
        // efficiency, 2.85 N·m): away from equality the two agree.
        assert_eq!(
            r.model.gearbox_reference_Nm,
            c.gearbox_input_rating_Nm * c.gear_ratio * c.gear_efficiency,
            "{name}"
        );
        assert_eq!(
            r.gearbox.cold_high_check == WITHIN_THE_GEARBOX_RATING,
            r.metal.torque_cold_high_Nm <= r.model.gearbox_reference_Nm,
            "{name}"
        );
    }
}

#[test]
fn n33ah_fails_the_input_basis_that_an_output_basis_without_the_efficiency_passes() {
    // Owner decision 1A: cold-high at the input against the input rating is cold-high against
    // rating × ratio × efficiency = 2.85 N·m at the output. Without the efficiency the output
    // limit would be 0.6 × 5 = 3.0 N·m, which this design (2.9401 N·m) would pass.
    let inputs = custom_dimensions("N33AH");
    let r = compute_all(&inputs);
    let c = &inputs.coupling;
    assert!(
        (r.metal.torque_cold_high_Nm - 2.9401).abs() < 5e-5,
        "{}",
        r.metal.torque_cold_high_Nm
    );
    assert!(
        (r.metal.cold_high_input_Nm - 0.6190).abs() < 5e-5,
        "{}",
        r.metal.cold_high_input_Nm
    );
    assert_eq!(r.gearbox.cold_high_check, above("0.62", "0.60"));
    assert!(r.metal.torque_cold_high_Nm <= c.gearbox_input_rating_Nm * c.gear_ratio);
    assert!(r.metal.torque_cold_high_Nm > r.model.gearbox_reference_Nm);
}

#[test]
fn exactly_at_the_rating_passes_and_one_float_above_fails() {
    // Equality passes, as the hot minimum check passes at the requirement.
    let rating: f64 = 0.6;
    assert_eq!(
        gearbox::compute(rating, rating).cold_high_check,
        WITHIN_THE_GEARBOX_RATING
    );
    assert_eq!(
        gearbox::compute(rating.next_up(), rating).cold_high_check,
        above("0.61", "0.60")
    );
    // Through the engine (the rating moves no torque): the rating at the defaults' cold-high
    // at the input passes; one float below it, so cold-high is the rating's next float up, fails.
    let x = compute_all(&DesignInputs::default())
        .metal
        .cold_high_input_Nm;
    let r = compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = x));
    assert_eq!(r.metal.cold_high_input_Nm, x);
    assert_eq!(r.gearbox.cold_high_check, WITHIN_THE_GEARBOX_RATING);
    let below = x.next_down();
    assert_eq!(below.next_up(), x);
    let r = compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = below));
    assert_eq!(r.metal.cold_high_input_Nm, x);
    assert_eq!(r.gearbox.cold_high_check, above("0.81", "0.80"));
}

#[test]
fn a_torque_that_is_not_a_number_reads_unknown() {
    for (cold_high_input, rating) in [
        (f64::NAN, 0.6),
        (0.5, f64::NAN),
        (f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NAN),
    ] {
        assert_eq!(
            gearbox::compute(cold_high_input, rating).cold_high_check,
            GEARBOX_CHECK_UNKNOWN,
            "{cold_high_input} against {rating}"
        );
    }
    // Through the engine: set() refuses NaN, but a struct, a design file or a share link can
    // hold one. A NaN rating, or a NaN ratio (cold-high at the input is then NaN), reads
    // unknown, and validate() names the input.
    type Put = fn(&mut DesignInputs);
    let cases: [(&str, Put); 2] = [
        ("coupling.gearbox_input_rating_Nm", |i| {
            i.coupling.gearbox_input_rating_Nm = f64::NAN
        }),
        ("coupling.gear_ratio", |i| i.coupling.gear_ratio = f64::NAN),
    ];
    for (path, put) in cases {
        let mut inputs = DesignInputs::default();
        put(&mut inputs);
        assert_eq!(
            compute_all(&inputs).gearbox.cold_high_check,
            GEARBOX_CHECK_UNKNOWN,
            "{path}"
        );
        let errors = inputs.validate().expect_err(path);
        assert_eq!(errors.len(), 1, "{path}: {errors:?}");
        assert_eq!(
            (errors[0].path.as_str(), &errors[0].kind),
            (path, &SetErrorKind::NotFinite)
        );
    }
}

#[test]
fn an_infinite_cold_high_at_the_gearbox_input_fails() {
    assert_eq!(
        gearbox::compute(f64::INFINITY, 0.6).cold_high_check,
        above("inf", "0.60")
    );
    // Through the engine: an efficiency of 0 set on the struct (its slider starts at 0.5)
    // divides cold-high by 0 at the input.
    let r = compute_all(&design(|i| i.coupling.gear_efficiency = 0.0));
    assert_eq!(r.metal.cold_high_input_Nm, f64::INFINITY);
    assert_eq!(r.gearbox.cold_high_check, above("inf", "0.60"));
}

#[test]
fn a_failing_check_never_quotes_two_equal_numbers() {
    // A rating of 0.604 quotes as 0.60, and cold-high at the input 0.6041 rounds to 0.60 too:
    // the badge quotes the torque 0.01 N·m above the rating as quoted.
    assert_eq!(
        gearbox::compute(0.6041, 0.604).cold_high_check,
        above("0.61", "0.60")
    );
    // Over the whole slider: at each step, just above it (quoted as the step) and half-way to
    // the next, a cold-high one float above the rating quotes above the rating.
    let range = rating_range();
    let steps = ((range.max - range.min) / range.step).round() as u32;
    for k in 0..=steps {
        let step = range.min + f64::from(k) * range.step;
        for rating in [step, step + 0.004, step + 0.005] {
            let text = gearbox::compute(rating.next_up(), rating).cold_high_check;
            let (torque, quoted_rating) = quoted(&text);
            assert!(torque > quoted_rating, "{rating}: {text}");
        }
    }
}

#[test]
fn the_rating_slider_s_ends_fail_and_pass_the_defaults() {
    let range = rating_range();
    assert_eq!((range.min, range.max, range.step), (0.05, 5.0, 0.01));
    assert_eq!(verdict_at_rating(range.min), above("0.80", "0.05"));
    assert_eq!(verdict_at_rating(range.max), WITHIN_THE_GEARBOX_RATING);
}

#[test]
fn the_rating_moves_only_the_gearbox_check_and_the_workbook_s_reference() {
    // The rating's two readers: this check and the workbook's reference (rating × ratio ×
    // efficiency). Every other result stays bit for bit (NaN equal to NaN).
    let bits = |inputs: &DesignInputs| -> Vec<(String, String)> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .map(|row| {
                let bits = match row.value {
                    Value::Num(x) => format!("{:016x}", x.to_bits()),
                    other => format!("{other:?}"),
                };
                (row.path, bits)
            })
            .collect()
    };
    let base = bits(&DesignInputs::default());
    let raised = bits(&design(|i| i.coupling.gearbox_input_rating_Nm = 1.0));
    assert_eq!(base.len(), raised.len());
    let moved: Vec<&str> = base
        .iter()
        .zip(&raised)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.0.as_str())
        .collect();
    assert_eq!(
        moved,
        ["model.gearbox_reference_Nm", "gearbox.cold_high_check"]
    );
}

#[test]
fn the_check_is_a_rust_only_text_result_with_no_cell_after_the_housing_group() {
    let rows = result_rows(&compute_all(&DesignInputs::default()));
    let gearbox: Vec<&str> = rows
        .iter()
        .filter(|row| row.path.starts_with("gearbox."))
        .map(|row| row.path.as_str())
        .collect();
    assert_eq!(gearbox, ["gearbox.cold_high_check"]);
    let at = rows
        .iter()
        .position(|row| row.path == "gearbox.cold_high_check")
        .expect("the check");
    let row = &rows[at];
    assert!(row.meta.rust_only);
    assert_eq!((row.meta.cell, row.meta.unit), (None, ""));
    assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
    assert!(matches!(row.value, Value::Text(_)), "{:?}", row.value);
    // Schema order: the group follows the housing group, before the tables.
    assert!(
        rows[at - 1].path.starts_with("housing."),
        "{}",
        rows[at - 1].path
    );
    assert!(
        rows[at + 1].path.starts_with("gap_sweep["),
        "{}",
        rows[at + 1].path
    );
}

/// A slider's two ends and seven steps spread evenly between them, bottom to top, each value
/// once: fewer steps between on a slider of fewer than eight steps.
fn ends_and_steps_between(range: SliderRange) -> Vec<f64> {
    const PARTS: u32 = 8;
    let steps = ((range.max - range.min) / range.step).round();
    // How many steps up from the bottom each value is: whole numbers, rising, so a repeat is
    // the one before it.
    let mut ups: Vec<f64> = (0..=PARTS)
        .map(|k| (steps * f64::from(k) / f64::from(PARTS)).round())
        .collect();
    ups.dedup();
    ups.into_iter()
        .map(|up| {
            if up == steps {
                range.max
            } else {
                range.min + up * range.step
            }
        })
        .collect()
}

#[test]
fn ends_and_steps_between_lists_each_slider_value_once() {
    // Whole-branch review fix wave 2, m-3: on a slider of fewer than eight steps the seven
    // evenly spread steps fall on the same values, and on the ends; each is listed once, bottom
    // to top, the ends exact.
    let slider = |min, max, step| SliderRange {
        min,
        max,
        step,
        log: false,
    };
    assert_eq!(
        ends_and_steps_between(slider(4.0, 10.0, 2.0)),
        [4.0, 6.0, 8.0, 10.0]
    );
    // 0.2 + 4 × 0.1 is not 0.6 in binary: the top is the slider's own end, once.
    assert_ne!(0.2 + 4.0 * 0.1, 0.6);
    assert_eq!(
        ends_and_steps_between(slider(0.2, 0.6, 0.1)),
        [0.2, 0.2 + 0.1, 0.2 + 2.0 * 0.1, 0.2 + 3.0 * 0.1, 0.6]
    );
    // Eight steps or more: the two ends and seven steps between.
    assert_eq!(
        ends_and_steps_between(slider(0.0, 16.0, 1.0)),
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    );
}

#[test]
fn the_band_is_the_same_at_every_magnet_size_gap_count_and_back_iron() {
    // Whole-branch review MF-3: the premise of the gearbox Problem's second sentence, "No magnet
    // size, gap or count passes both" (spec section 3). Cold-high over hot-low, the band, depends
    // only on the two temperatures, each ring's Br temperature coefficient and the variation
    // allowance, so every design that just meets the requirement puts the requirement times
    // the same band on the gearbox output. An engine change that made cold-high and hot-low
    // scale apart with the geometry (a torque not proportional to Br_i Br_o, say) would make
    // that sentence false with the tests that take the band from the same design still green:
    // this one fails then. Over the defaults with the magnet length, the ring radius, the face
    // gap, the magnet count and the back iron each at its slider's ends and steps between (each
    // choice of the back iron), and the probe designs that keep the defaults' Br temperature
    // coefficients, wherever the end-effect model is in range the band is the defaults' up to
    // float noise.
    let band = |inputs: &DesignInputs| -> Option<f64> {
        let r = compute_all(inputs);
        (r.model.end_effect_check == "OK" && r.metal.torque_hot_low_Nm > 0.0)
            .then(|| r.metal.torque_cold_high_Nm / r.metal.torque_hot_low_Nm)
    };
    let defaults = DesignInputs::default();
    let at_defaults = band(&defaults).expect("the defaults are in the end-effect range");
    let assert_same_band = |name: &str, inputs: &DesignInputs| -> bool {
        let Some(here) = band(inputs) else {
            return false;
        };
        assert!(
            ((here - at_defaults) / at_defaults).abs() <= 1e-12,
            "{name}: band {here} against the defaults' {at_defaults}"
        );
        true
    };
    let rows = input_rows(&defaults);
    for path in [
        "coupling.magnets.axial_length_mm",
        "coupling.inner_back_apothem_mm",
        "metal.face_gap_mm",
        "coupling.npole",
        "coupling.backiron",
    ] {
        let row = rows
            .iter()
            .find(|row| row.path == path)
            .unwrap_or_else(|| panic!("{path} is an input"));
        let values: Vec<Value> = match row.meta.range {
            Some(range) => ends_and_steps_between(range)
                .into_iter()
                .map(|x| match row.value {
                    Value::Int(_) => Value::Int(x.round() as i64),
                    _ => Value::Num(x),
                })
                .collect(),
            None => row
                .meta
                .choices
                .iter()
                .map(|&(code, _)| Value::Int(code))
                .collect(),
        };
        assert!(values.len() >= 2, "{path}: {values:?}");
        let count = values.len();
        let mut checked = 0;
        for value in values {
            let mut inputs = defaults.clone();
            inputs
                .set(path, value.clone())
                .unwrap_or_else(|e| panic!("{path} = {value:?}: {e:?}"));
            if assert_same_band(&format!("{path} = {value:?}"), &inputs) {
                checked += 1;
            }
        }
        // Most of each slider is in the end-effect range: the band is checked, not skipped.
        assert!(
            2 * checked > count,
            "{path}: {checked} of {count} designs in range"
        );
    }
    // The probe designs that keep the defaults' Br temperature coefficients: all but the
    // ferrite, whose grade sets its own.
    let alphas = |inputs: &DesignInputs| {
        let m = compute_all(inputs).model;
        (m.inner_alpha_br_per_C, m.outer_alpha_br_per_C)
    };
    let mut kept = Vec::new();
    for (name, inputs) in probe_designs() {
        if alphas(&inputs) == alphas(&defaults) {
            assert!(assert_same_band(name, &inputs), "{name}: in range");
            kept.push(name);
        }
    }
    assert_eq!(
        kept,
        [
            "default",
            "8 magnets per ring",
            "304 back iron",
            "20 mm ring radius"
        ]
    );
}
