//! BL-064: the shaft joint check (`shaft_joint.check`, `src/engine/shaft_joint.rs`), end to end
//! through `compute_all` and on the pure `joint` function. Both rotors are held on their shafts
//! by a keyed shrink fit: the inner hub on the wheel shaft, the cup on the gearbox output shaft.
//! The fit alone must hold the safety factor times cold-high at its loosest, the part must stay
//! within yield ÷ 1.2 at its tightest, it must be fitted within its allowed temperature, and the
//! key alone carries the same torque as a backup (owner decisions 1A, 2A, 7A, R-15 and its ring
//! ruling). A Rust-only result group, tested here as the gearbox check is in `tests/gearbox.rs`.
//! The numbers are the spec's (sections 3.5 to 3.7), which its scratch crate computed.

use magcoupling::compute_all;
use magcoupling::engine::api::{DesignInputs, shaft_joint_links};
use magcoupling::engine::compat::py_min;
use magcoupling::engine::explain::probes::probe_designs;
use magcoupling::engine::meta::{InputMeta, InputSet, SliderRange, Value, input_rows, result_rows};
use magcoupling::engine::shaft_joint::{
    self, BORE_OVER_ITS_LIMIT, CANNOT_BE_FITTED, CAUSES, CUP, Cause, FIT_SLIPS, Floor, HubFitLimit,
    INNER_HUB, JointCommon, JointSpec, JointTerms, KEY_HAS_NO_BEARING_AREA, KEY_OVER_ITS_LIMIT,
    NoWall, ROOM_C, SHAFT_CTE_PER_C, SHAFT_JOINT_UNKNOWN, SHAFT_JOINTS_PASS, ShaftJointInputs,
    ShaftJointLinks, UNKNOWN_BAND_REVERSED, UNKNOWN_IMPOSSIBLE_VALUE, UNKNOWN_NO_WALL,
    UNKNOWN_NOT_A_NUMBER, UNKNOWN_SAFETY_FACTOR, YIELD_FACTOR, hub_fit_allowed, impossible_values,
    shaft_shrink_um,
};
use magcoupling::engine::temperature::CURE_MARGIN_MIN_C;

/// The default design with `edit` applied.
fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    edit(&mut inputs);
    inputs
}

/// `gui::test_support::passing_design()`, restated here (the GUI's test support is not reachable
/// from an engine test): the defaults with a 1.8 mm face gap, a 2.0 mm cup wall, a 1.6 N·m
/// requirement and a 1.0 N·m gearbox input rating. Its shaft joints pass (9.48 and 16.90
/// against 6.72 N·m).
fn passing() -> DesignInputs {
    design(|i| {
        i.metal.face_gap_mm = 1.8;
        i.metal.cup_wall_corner_mm = 2.0;
        i.metal.required_min_Nm = 1.6;
        i.coupling.gearbox_input_rating_Nm = 1.0;
    })
}

/// `base` with both rings on manual dimensions, grade id `inner` inside and `outer` outside.
fn manual(mut base: DesignInputs, inner: &str, outer: &str) -> DesignInputs {
    let m = &mut base.coupling.magnets;
    m.part_inner.clear();
    m.part_outer.clear();
    m.grade_inner = inner.to_owned();
    m.grade_outer = outer.to_owned();
    base
}

/// The shaft joint check's verdict.
fn check(inputs: &DesignInputs) -> String {
    compute_all(inputs).shaft_joint.check
}

/// The engine values the check read, rebuilt from the design (`api::shaft_joint_links`).
fn links(inputs: &DesignInputs) -> ShaftJointLinks {
    shaft_joint_links(inputs, &compute_all(inputs))
}

/// What the two joints are at `inputs`: what both share and each one's own values.
fn specs(inputs: &DesignInputs) -> (JointCommon, [JointSpec; 2]) {
    shaft_joint::joint_specs(&inputs.shaft_joint, &links(inputs))
}

/// Both joints' terms at `inputs`, inner hub first.
fn terms(inputs: &DesignInputs) -> [JointTerms; 2] {
    shaft_joint::joints(&inputs.shaft_joint, &links(inputs))
}

/// `x` within `tolerance` of `want`.
fn assert_near(x: f64, want: f64, tolerance: f64, what: &str) {
    assert!((x - want).abs() <= tolerance, "{what}: {x}, want {want}");
}

/// The badge's slip text.
fn slips(joint: &str, holding: &str, required: &str) -> String {
    format!("{FIT_SLIPS} ({joint}): holds {holding} N·m against {required} N·m")
}

/// The badge's stress text.
fn over_stress(joint: &str, stress: &str, limit: &str) -> String {
    format!("{BORE_OVER_ITS_LIMIT} ({joint}): {stress} MPa against {limit} MPa")
}

/// The badge's fitting text.
fn cannot_fit(joint: &str, needed: &str, allowed: &str) -> String {
    format!("{CANNOT_BE_FITTED} ({joint}): needs {needed} °C against {allowed} °C")
}

/// The badge's key text.
fn key_over(joint: &str, pressure: &str, limit: &str) -> String {
    format!("{KEY_OVER_ITS_LIMIT} ({joint}): {pressure} MPa against {limit} MPa")
}

/// The two numbers a failing badge quotes: (the failing value, the limit or need).
fn quoted(text: &str) -> (f64, f64) {
    let (_, rest) = text
        .split_once("): ")
        .unwrap_or_else(|| panic!("no numbers: {text}"));
    let rest = rest
        .strip_prefix("holds ")
        .or_else(|| rest.strip_prefix("needs "))
        .unwrap_or(rest);
    let (value, limit) = rest
        .split_once(" against ")
        .unwrap_or_else(|| panic!("{text}"));
    let number = |s: &str| -> f64 {
        s.split(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("{text}"))
    };
    (number(value), number(limit))
}

/// The metadata of the input at `path`.
fn input_meta(path: &str) -> &'static InputMeta {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|row| row.path == path)
        .unwrap_or_else(|| panic!("{path} is an input"))
        .meta
}

/// An input's slider.
fn slider(path: &str) -> SliderRange {
    input_meta(path)
        .range
        .unwrap_or_else(|| panic!("{path} has a slider"))
}

/// Every value of `path`'s slider, bottom to top.
fn slider_values(path: &str) -> Vec<f64> {
    let range = slider(path);
    let steps = ((range.max - range.min) / range.step).round() as u32;
    (0..=steps)
        .map(|k| range.min + f64::from(k) * range.step)
        .collect()
}

/// The help of the result at `path`.
fn result_help(path: &str) -> &'static str {
    result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .find(|row| row.path == path)
        .map(|row| row.meta.help)
        .unwrap_or_else(|| panic!("{path} is a result"))
}

#[test]
fn the_check_is_a_rust_only_group_of_fifteen_inputs_and_twenty_four_results() {
    // Spec section 4.1, in its order: no workbook cell, no Python twin; the inputs after the
    // clamp group, the results after the gearbox group and before the sweeps.
    let inputs: Vec<_> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
        .collect();
    let input_paths: Vec<&str> = inputs.iter().map(|row| row.path.as_str()).collect();
    assert_eq!(
        input_paths,
        [
            "shaft_joint.hub_interference_min_um",
            "shaft_joint.hub_interference_max_um",
            "shaft_joint.cup_interference_min_um",
            "shaft_joint.cup_interference_max_um",
            "shaft_joint.safety_factor",
            "shaft_joint.friction",
            "shaft_joint.smoothing_um",
            "shaft_joint.rotor_over_shaft_C",
            "shaft_joint.key_width_mm",
            "shaft_joint.key_contact_mm",
            "shaft_joint.hub_fit_max_C",
            "shaft_joint.cup_fit_max_C",
            "shaft_joint.wheel_shaft_fit_C",
            "shaft_joint.output_shaft_fit_C",
            "shaft_joint.joining_clearance_um",
        ]
    );
    // Units, defaults and sliders (min, max, step), spec section 4.1.
    let want: [(&str, f64, [f64; 3]); 15] = [
        ("µm", 13.0, [0.0, 40.0, 0.5]),
        ("µm", 18.0, [0.0, 40.0, 0.5]),
        ("µm", 15.0, [0.0, 40.0, 0.5]),
        ("µm", 22.0, [0.0, 40.0, 0.5]),
        ("-", 2.0, [1.0, 4.0, 0.1]),
        ("-", 0.14, [0.05, 0.30, 0.01]),
        ("µm", 5.0, [0.0, 15.0, 0.1]),
        ("°C", 20.0, [0.0, 60.0, 1.0]),
        ("mm", 3.0, [1.0, 6.0, 0.5]),
        ("mm", 1.2, [0.5, 3.0, 0.1]),
        ("°C", 100.0, [20.0, 130.0, 1.0]),
        ("°C", 300.0, [20.0, 400.0, 5.0]),
        ("°C", -196.0, [-196.0, 40.0, 1.0]),
        ("°C", 20.0, [-20.0, 40.0, 1.0]),
        ("µm", 10.0, [0.0, 30.0, 0.5]),
    ];
    for (row, (unit, default, [min, max, step])) in inputs.iter().zip(want) {
        assert!(row.meta.rust_only, "{}", row.path);
        assert_eq!(row.meta.cell, None, "{}", row.path);
        assert_eq!(row.meta.unit, unit, "{}", row.path);
        assert_eq!(row.value, Value::Num(default), "{}", row.path);
        let range = row.meta.range.expect("a slider");
        assert_eq!(
            (range.min, range.max, range.step),
            (min, max, step),
            "{}",
            row.path
        );
        assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
    }
    let all_inputs = input_rows(&DesignInputs::default());
    let first = all_inputs
        .iter()
        .position(|row| row.path.starts_with("shaft_joint."))
        .expect("the group");
    assert!(all_inputs[first - 1].path.starts_with("clamps."));
    assert_eq!(first + 15, all_inputs.len(), "the last input group");

    let rows = result_rows(&compute_all(&DesignInputs::default()));
    let results: Vec<_> = rows
        .iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
        .collect();
    let units: Vec<(&str, &str)> = results
        .iter()
        .map(|row| (row.path.as_str(), row.meta.unit))
        .collect();
    assert_eq!(
        units,
        [
            ("shaft_joint.required_Nm", "N·m"),
            ("shaft_joint.part_modulus_GPa", "GPa"),
            ("shaft_joint.part_cte_per_C", "1/°C"),
            ("shaft_joint.part_yield_MPa", "MPa"),
            ("shaft_joint.stress_limit_MPa", "MPa"),
            ("shaft_joint.hub_holding_Nm", "N·m"),
            ("shaft_joint.cup_holding_Nm", "N·m"),
            ("shaft_joint.hub_bore_stress_MPa", "MPa"),
            ("shaft_joint.cup_bore_stress_MPa", "MPa"),
            ("shaft_joint.hub_key_pressure_MPa", "MPa"),
            ("shaft_joint.cup_key_pressure_MPa", "MPa"),
            ("shaft_joint.hub_needed_um", "µm"),
            ("shaft_joint.cup_needed_um", "µm"),
            ("shaft_joint.hub_elastic_max_um", "µm"),
            ("shaft_joint.cup_elastic_max_um", "µm"),
            ("shaft_joint.wheel_shaft_shrink_um", "µm"),
            ("shaft_joint.output_shaft_shrink_um", "µm"),
            ("shaft_joint.hub_fittable_max_um", "µm"),
            ("shaft_joint.cup_fittable_max_um", "µm"),
            ("shaft_joint.hub_fit_temp_C", "°C"),
            ("shaft_joint.cup_fit_temp_C", "°C"),
            ("shaft_joint.hub_fit_allowed_C", "°C"),
            ("shaft_joint.cup_fit_allowed_C", "°C"),
            ("shaft_joint.check", ""),
        ]
    );
    for row in &results {
        assert!(row.meta.rust_only, "{}", row.path);
        assert_eq!(row.meta.cell, None, "{}", row.path);
        assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
        let text = row.path == "shaft_joint.check";
        assert_eq!(matches!(row.value, Value::Text(_)), text, "{}", row.path);
    }
    let at = rows
        .iter()
        .position(|row| row.path == "shaft_joint.required_Nm")
        .expect("the group");
    assert!(
        rows[at - 1].path.starts_with("gearbox."),
        "{}",
        rows[at - 1].path
    );
    assert!(
        rows[at + 24].path.starts_with("gap_sweep["),
        "{}",
        rows[at + 24].path
    );
}

#[test]
fn the_verdict_texts_are_the_spec_s() {
    // Spec section 4.2: one pass, one unknown prefix with three reasons, the safety factor's
    // (ruling C-4) and the impossible value's (its extension, the whole-branch review's MF-1),
    // five failing prefixes.
    assert_eq!(
        SHAFT_JOINTS_PASS,
        "Both shaft joints hold and can be fitted"
    );
    assert_eq!(SHAFT_JOINT_UNKNOWN, "Shaft joint check unknown");
    for (text, reason) in [
        (UNKNOWN_NOT_A_NUMBER, "a value is not a number"),
        (UNKNOWN_NO_WALL, "no wall around the bore"),
        (
            UNKNOWN_BAND_REVERSED,
            "the minimum interference is above the maximum",
        ),
        (UNKNOWN_SAFETY_FACTOR, "the safety factor is not above zero"),
        (
            UNKNOWN_IMPOSSIBLE_VALUE,
            "a value is outside what it can physically be",
        ),
    ] {
        assert_eq!(text, format!("{SHAFT_JOINT_UNKNOWN}: {reason}"));
    }
    assert_eq!(FIT_SLIPS, "The fit slips");
    assert_eq!(BORE_OVER_ITS_LIMIT, "Bore stress above its limit");
    assert_eq!(
        CANNOT_BE_FITTED,
        "Cannot be fitted at the allowed temperature"
    );
    assert_eq!(KEY_OVER_ITS_LIMIT, "Key backup above its limit");
    assert_eq!(KEY_HAS_NO_BEARING_AREA, "The key has no bearing area");
    assert_eq!((INNER_HUB, CUP), ("inner hub", "cup"));
    // No text names the paused clamp, a path or a cell.
    for text in [
        SHAFT_JOINTS_PASS,
        UNKNOWN_NOT_A_NUMBER,
        UNKNOWN_NO_WALL,
        UNKNOWN_BAND_REVERSED,
        UNKNOWN_SAFETY_FACTOR,
        UNKNOWN_IMPOSSIBLE_VALUE,
        FIT_SLIPS,
        BORE_OVER_ITS_LIMIT,
        CANNOT_BE_FITTED,
        KEY_OVER_ITS_LIMIT,
        KEY_HAS_NO_BEARING_AREA,
    ] {
        assert!(!text.to_lowercase().contains("clamp"), "{text}");
        assert!(!text.contains('_') && !text.contains('!'), "{text}");
    }
}

#[test]
fn the_safety_factor_help_says_a_slip_can_reverse_the_torque_not_that_each_does() {
    // G-3 review MF-1: the swing runs at the pole-pair slip frequency for the slip duration.
    // At the defaults that is 16.7 cycles per event; at the slip speed's slider bottom it is
    // under one, so a slip can reverse the torque many times, and need not.
    let cycles =
        |inputs: &DesignInputs| compute_all(inputs).metal.slip_freq_Hz * inputs.metal.slip_event_s;
    assert_near(
        cycles(&DesignInputs::default()),
        16.6667,
        5e-5,
        "the defaults",
    );
    let slowest = design(|i| i.metal.slip_rpm = slider("metal.slip_rpm").min);
    assert_near(cycles(&slowest), 0.83333, 5e-6, "the slowest slip");
    let help = input_meta("shaft_joint.safety_factor").help;
    assert!(!help.contains("each slip"), "{help}");
    assert!(
        help.contains("A slip can reverse the torque many times."),
        "{help}"
    );
}

#[test]
fn the_holding_helps_put_the_rotor_allowance_at_the_hot_end_alone() {
    // G-3 review MF-2 (spec 3.3): the loosest fit is the cold end or the hot end, whichever
    // loosens it more, and only the hot end has the rotor warmer than its shaft. 4140 (the
    // defaults) loosens hot, so the allowance lowers its holding. 416 (pick 4) loosens cold up
    // to the default allowance, so there the holding is bit for bit the same at 0 and 20 °C;
    // at the slider's top the hot end takes over.
    let holding = |back_iron: i64, over: f64| {
        let r = compute_all(&design(|i| {
            i.materials.parts.back_iron = back_iron;
            i.shaft_joint.rotor_over_shaft_C = over;
        }))
        .shaft_joint;
        [r.hub_holding_Nm, r.cup_holding_Nm]
    };
    let shipped = DesignInputs::default().materials.parts.back_iron;
    let [hub_cool, _] = holding(shipped, 0.0);
    let [hub_warm, _] = holding(shipped, 20.0);
    assert_near(hub_cool, 14.41, 5e-3, "4140 with no allowance");
    assert!(
        hub_warm < hub_cool,
        "4140 at the default allowance: {hub_warm}"
    );
    let top = slider("shaft_joint.rotor_over_shaft_C").max;
    let [no_allowance, at_default, at_top] = [0.0, 20.0, top].map(|over| holding(4, over));
    assert_eq!(no_allowance.map(f64::to_bits), at_default.map(f64::to_bits));
    assert_near(at_default[0], 14.0443, 5e-5, "416");
    assert!(
        at_top[0] < at_default[0] && at_top[1] < at_default[1],
        "{at_top:?}"
    );
    for path in ["shaft_joint.hub_holding_Nm", "shaft_joint.cup_holding_Nm"] {
        let help = result_help(path);
        assert!(
            help.contains(
                "cold, or hot with the rotor warmer than its shaft, whichever loosens it more."
            ),
            "{path}: {help}"
        );
    }
}

#[test]
fn the_bare_cup_s_limit_help_names_no_steel_as_the_cup_can_be_aluminium() {
    // G-3 review MF-3: with no back iron, or the 6061 pick, the cup is the library's 6061-T6
    // (pinned by `the_304_and_aluminium_windows_are_empty`), so the help of its fitting limit
    // names the cup's own material, never a steel.
    let help = input_meta("shaft_joint.cup_fit_max_C").help;
    assert!(!help.contains("steel"), "{help}");
    assert!(
        help.contains("the treatment temperatures of its material and finish"),
        "{help}"
    );
}

#[test]
fn the_shaft_shrink_helps_count_from_room_temperature_and_type_no_number() {
    // G-3 review MF-4 (a, b): each shaft's shrink is zero at room temperature, `ROOM_C` (pinned
    // in the NIST shrink test below), which the helps name in words and never type.
    for (path, shaft) in [
        ("shaft_joint.wheel_shaft_shrink_um", "wheel"),
        ("shaft_joint.output_shaft_shrink_um", "gearbox"),
    ] {
        let help = result_help(path);
        assert!(!help.chars().any(|c| c.is_ascii_digit()), "{path}: {help}");
        assert!(
            help.contains(&format!(
                "How much the {shaft} shaft's diameter shrinks from room temperature"
            )),
            "{path}: {help}"
        );
        assert!(
            help.contains("negative above room temperature"),
            "{path}: {help}"
        );
    }
}

#[test]
fn per_micrometre_each_joint_grips_and_stresses_as_the_spec_says() {
    // Spec section 3.4, the defaults: contact pressure, holding torque (friction 0.14, the 3 mm
    // key's arc taken off) and bore von Mises stress per µm of effective interference.
    let [hub, cup] = terms(&DesignInputs::default());
    assert_near(hub.pressure_per_um, 7.738, 5e-4, "inner hub pressure");
    assert_near(hub.holding_per_um, 2.0009, 5e-5, "inner hub holding");
    assert_near(hub.stress_per_um, 17.930, 5e-4, "inner hub stress");
    assert_near(cup.pressure_per_um, 8.132, 5e-4, "cup pressure");
    assert_near(cup.holding_per_um, 2.5073, 5e-5, "cup holding");
    assert_near(cup.stress_per_um, 17.879, 5e-4, "cup stress");
}

#[test]
fn the_defaults_pass_with_the_numbers_of_the_spec() {
    // Spec section 3.5: 4140 on a Ø10 bore, the hub 13-18 µm on a -196 °C wheel shaft, the cup
    // 15-22 µm on a 20 °C gearbox shaft.
    let r = compute_all(&DesignInputs::default());
    let s = &r.shaft_joint;
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
    assert_eq!(s.required_Nm, 2.0 * r.metal.torque_cold_high_Nm);
    assert_near(s.required_Nm, 7.6466, 5e-5, "required");
    assert_eq!(
        (s.part_modulus_GPa, s.part_cte_per_C, s.part_yield_MPa),
        (205.0, 12.3e-6, 415.0)
    );
    assert_eq!(s.stress_limit_MPa, 415.0 / YIELD_FACTOR);
    let pins: [(&str, f64, f64, f64); 18] = [
        ("hub holding", s.hub_holding_Nm, 9.4844, 5e-5),
        ("cup holding", s.cup_holding_Nm, 16.899, 5e-4),
        ("hub bore stress", s.hub_bore_stress_MPa, 241.70, 5e-3),
        ("cup bore stress", s.cup_bore_stress_MPa, 312.53, 5e-3),
        ("hub key", s.hub_key_pressure_MPa, 127.44, 5e-3),
        ("cup key", s.cup_key_pressure_MPa, 101.95, 5e-3),
        ("hub needed", s.hub_needed_um, 12.082, 5e-4),
        ("cup needed", s.cup_needed_um, 11.310, 5e-4),
        ("hub elastic", s.hub_elastic_max_um, 23.808, 5e-4),
        ("cup elastic", s.cup_elastic_max_um, 23.863, 5e-4),
        ("wheel shaft shrink", s.wheel_shaft_shrink_um, 18.988, 5e-4),
        ("gearbox shaft shrink", s.output_shaft_shrink_um, 0.0, 0.0),
        ("hub fittable", s.hub_fittable_max_um, 18.828, 5e-4),
        ("cup fittable", s.cup_fittable_max_um, 24.440, 5e-4),
        ("hub fit temperature", s.hub_fit_temp_C, 93.267, 5e-4),
        ("cup fit temperature", s.cup_fit_temp_C, 280.16, 5e-3),
        ("hub allowed", s.hub_fit_allowed_C, 100.0, 0.0),
        ("cup allowed", s.cup_fit_allowed_C, 300.0, 0.0),
    ];
    for (what, x, want, tolerance) in pins {
        assert_near(x, want, tolerance, what);
    }
    // The windows: 12.08-18.83 µm on the hub (the fitting edge binds, the elastic one is
    // 23.81), 11.31-23.86 µm on the cup (the stress edge binds, the fitting one is 24.44).
    assert!(s.hub_fittable_max_um < s.hub_elastic_max_um);
    assert!(s.cup_elastic_max_um < s.cup_fittable_max_um);
    // The bands sit inside them: 0.92 µm above the hub's needed minimum and 0.83 µm under its
    // fitting edge (one 0.5 µm step up on the maximum still fits, two do not).
    let i = &DesignInputs::default().shaft_joint;
    assert_near(
        i.hub_interference_min_um - s.hub_needed_um,
        0.918,
        5e-4,
        "hub",
    );
    assert_near(
        s.hub_fittable_max_um - i.hub_interference_max_um,
        0.828,
        5e-4,
        "hub",
    );
    for (maximum, passes) in [(18.5, true), (19.0, false)] {
        let r = compute_all(&design(|d| d.shaft_joint.hub_interference_max_um = maximum));
        assert_eq!(
            r.shaft_joint.check == SHAFT_JOINTS_PASS,
            passes,
            "{maximum}"
        );
    }
}

#[test]
fn the_bl067_corner_reads_cold_high_and_passes() {
    // Review MF-4: with the operating temperature (-40 °C) below the minimum (20 °C) and a
    // 0.1 variation, the check reads cold-high (3.1823 N·m), not the 3.657 N·m operating
    // pull-out with its allowance (3.3246 × 1.1). BL-067's fix of which pull-out the checks
    // compare moves this pin on purpose.
    let corner = design(|i| {
        i.coupling.op_temp_C = -40.0;
        i.metal.min_temp_C = 20.0;
        i.metal.variation = 0.1;
    });
    let r = compute_all(&corner);
    let s = &r.shaft_joint;
    assert_near(r.metal.torque_cold_high_Nm, 3.1823, 5e-5, "cold-high");
    assert_near(r.model.pullout_Nm * 1.1, 3.657, 5e-4, "operating pull-out");
    assert_near(s.required_Nm, 6.3647, 5e-5, "required");
    assert_near(s.hub_holding_Nm, 9.484, 5e-4, "hub holding");
    assert_near(s.cup_holding_Nm, 16.899, 5e-4, "cup holding");
    assert_near(s.hub_bore_stress_MPa, 233.09, 5e-3, "hub stress");
    assert_near(s.cup_bore_stress_MPa, 303.95, 5e-3, "cup stress");
    assert_near(s.hub_key_pressure_MPa, 106.08, 5e-3, "hub key");
    assert_near(s.cup_key_pressure_MPa, 84.86, 5e-3, "cup key");
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
}

#[test]
fn each_probe_design_reads_its_verdict() {
    // Spec section 3.6, in `probe_designs()` order: (name, the holding torques, the required
    // torque [N·m], the verdict). The allowed hub temperature is your 100 °C limit at each
    // (owner R-15: the inner ring governs at each, so the onset read is the governing one).
    let want: [(&str, f64, f64, f64, String); 5] = [
        (
            "default",
            9.4844,
            16.899,
            7.6466,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "8 magnets per ring",
            9.4844,
            16.899,
            5.5985,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "304 back iron",
            0.0,
            2.9274,
            4.8277,
            slips(INNER_HUB, "0.00", "4.83"),
        ),
        (
            "Y30 ferrite, custom dimensions",
            9.4844,
            16.899,
            0.6761,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "20 mm ring radius",
            11.774,
            16.899,
            7.2222,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
    ];
    let designs = probe_designs();
    assert_eq!(designs.len(), want.len());
    for ((name, inputs), (want_name, hub, cup, required, verdict)) in designs.into_iter().zip(want)
    {
        assert_eq!(name, want_name);
        let s = compute_all(&inputs).shaft_joint;
        assert_near(s.hub_holding_Nm, hub, 5e-4, name);
        assert_near(s.cup_holding_Nm, cup, 5e-4, name);
        assert_near(s.required_Nm, required, 5e-4, name);
        assert_eq!(s.hub_fit_allowed_C, 100.0, "{name}");
        assert_eq!(s.check, verdict, "{name}");
    }
}

#[test]
fn a_band_minimum_too_low_slips_at_its_own_joint() {
    let hub = check(&design(|i| i.shaft_joint.hub_interference_min_um = 10.0));
    assert_eq!(hub, slips(INNER_HUB, "3.48", "7.65"));
    let cup = check(&design(|i| i.shaft_joint.cup_interference_min_um = 10.0));
    assert_eq!(cup, slips(CUP, "4.36", "7.65"));
}

#[test]
fn a_band_maximum_too_high_is_over_the_stress_limit_and_cannot_be_fitted() {
    // The stress cause is the badge; the fitting cause follows it (spec section 4.3).
    let limit = "345.8";
    for (joint, maximum, stress) in [
        (INNER_HUB, 26.0, "385.1"),
        (INNER_HUB, 28.0, "421.0"),
        (CUP, 26.0, "384.0"),
        (CUP, 28.0, "419.8"),
    ] {
        let inputs = design(|i| {
            if joint == INNER_HUB {
                i.shaft_joint.hub_interference_max_um = maximum;
            } else {
                i.shaft_joint.cup_interference_max_um = maximum;
            }
        });
        assert_eq!(
            check(&inputs),
            over_stress(joint, stress, limit),
            "{joint} {maximum}"
        );
        let joints = terms(&inputs);
        let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
            .into_iter()
            .map(|(cause, j)| (cause, j.name))
            .collect();
        assert_eq!(
            failing,
            [(Cause::BoreStress, joint), (Cause::CannotBeFitted, joint)],
            "{joint} {maximum}"
        );
    }
}

#[test]
fn the_badge_takes_causes_in_order_then_the_inner_hub_before_the_cup() {
    // Spec section 4.3: within a cause the inner hub is named first; an earlier cause on the
    // cup outranks a later one on the inner hub.
    let both = design(|i| {
        i.shaft_joint.hub_interference_max_um = 28.0;
        i.shaft_joint.cup_interference_max_um = 28.0;
    });
    let joints = terms(&both);
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(
        failing,
        [
            (Cause::BoreStress, INNER_HUB),
            (Cause::BoreStress, CUP),
            (Cause::CannotBeFitted, INNER_HUB),
            (Cause::CannotBeFitted, CUP),
        ]
    );
    assert_eq!(check(&both), over_stress(INNER_HUB, "421.0", "345.8"));
    let cup_slips_hub_overstressed = design(|i| {
        i.shaft_joint.hub_interference_max_um = 26.0;
        i.shaft_joint.cup_interference_min_um = 10.0;
    });
    assert_eq!(
        check(&cup_slips_hub_overstressed),
        slips(CUP, "4.36", "7.65")
    );
    assert_eq!(
        CAUSES,
        [
            Cause::Slips,
            Cause::BoreStress,
            Cause::CannotBeFitted,
            Cause::Key
        ]
    );
}

#[test]
fn the_304_and_aluminium_windows_are_empty() {
    // Spec section 3.6: neither hub has a working interference at Ø10. 304: holding needs
    // 16.30 µm, the bore stays within its limit only up to 11.72. Aluminium (the 6061 pick, or
    // no back iron with any pick, one outside the choices too): 27.74 against 25.22.
    let r304 = compute_all(&design(|i| i.materials.parts.back_iron = 7)).shaft_joint;
    assert_near(r304.hub_needed_um, 16.3031, 5e-5, "304 needed");
    assert_near(r304.hub_elastic_max_um, 11.7235, 5e-5, "304 elastic");
    assert_near(r304.cup_needed_um, 15.7920, 5e-5, "304 cup needed");
    assert_near(r304.cup_elastic_max_um, 11.7439, 5e-5, "304 cup elastic");
    for aluminium in [
        design(|i| i.materials.parts.back_iron = 8),
        design(|i| i.coupling.backiron = 0),
        design(|i| {
            i.coupling.backiron = 0;
            i.materials.parts.back_iron = 2;
        }),
        design(|i| {
            i.coupling.backiron = 0;
            i.materials.parts.back_iron = 99;
        }),
    ] {
        let s = compute_all(&aluminium).shaft_joint;
        assert_eq!(
            (s.part_modulus_GPa, s.part_cte_per_C, s.part_yield_MPa),
            (68.9, 23.6e-6, 241.0)
        );
        assert_near(s.hub_needed_um, 27.74, 5e-3, "aluminium needed");
        assert_near(
            py_min(s.hub_elastic_max_um, s.hub_fittable_max_um),
            25.22,
            5e-3,
            "upper",
        );
        assert!(s.hub_needed_um > py_min(s.hub_elastic_max_um, s.hub_fittable_max_um));
        assert_eq!((s.hub_holding_Nm, s.cup_holding_Nm), (0.0, 0.0));
        assert_eq!(s.check, slips(INNER_HUB, "0.00", "4.83"));
    }
}

#[test]
fn the_end_that_loosens_the_fit_follows_the_part_s_expansion() {
    // Synthesis correction 1, spec section 3.3: a part expanding less than the shaft (416,
    // 10.08e-6 /°C) loosens cold, so its minimum is taken at the minimum magnet temperature;
    // one expanding more (304, 16.9e-6 /°C) loosens hot, at the adhesive's design limit with
    // the rotor 20 °C over its shaft. R-14 puts this hot-end form to the physics review: the
    // bond-limited form would take 416's minimum hot.
    for (code, cold) in [(4, true), (7, false)] {
        let inputs = design(|i| i.materials.parts.back_iron = code);
        let r = compute_all(&inputs);
        let (common, [hub, _]) = specs(&inputs);
        let alpha = r.shaft_joint.part_cte_per_C;
        let d = inputs.coupling.bore_mm;
        let mismatch = |t: f64| (SHAFT_CTE_PER_C - alpha) * (t - ROOM_C) * d * 1000.0;
        let cold_end = mismatch(common.cold_C);
        let hot_end = mismatch(common.hot_C) - alpha * common.rotor_over_shaft_C * d * 1000.0;
        assert_eq!(cold_end < hot_end, cold, "{code}");
        let loosest = if cold { cold_end } else { hot_end };
        let [terms_hub, _] = terms(&inputs);
        assert_eq!(
            terms_hub.effective_min_um,
            (hub.interference_min_um - common.smoothing_um + loosest).max(0.0),
            "{code}"
        );
    }
}

#[test]
fn an_uncooled_wheel_shaft_or_dry_ice_cannot_fit_the_inner_hub_nor_a_cooler_bare_cup() {
    for (shaft, needed) in [(20.0, "247.6"), (-78.5, "163.5")] {
        let inputs = design(|i| i.shaft_joint.wheel_shaft_fit_C = shaft);
        assert_eq!(
            check(&inputs),
            cannot_fit(INNER_HUB, needed, "100.0"),
            "{shaft}"
        );
    }
    let cup = design(|i| i.shaft_joint.cup_fit_max_C = 250.0);
    assert_eq!(check(&cup), cannot_fit(CUP, "280.2", "250.0"));
}

#[test]
fn the_allowed_hub_temperature_is_the_lowest_of_three() {
    // Owner R-15: the bonded inner hub's allowed fitting temperature is the lowest of your
    // limit, the selected adhesive's design limit and the inner ring's own single-ring onset
    // less the cure margin (CURE_MARGIN_MIN_C), with the limit that binds.
    let allowed = |inputs: &DesignInputs| {
        let r = compute_all(inputs);
        let (value, limit) = hub_fit_allowed(
            inputs.shaft_joint.hub_fit_max_C,
            r.temperature.adhesive.design_limit_C,
            r.temperature.demag.inner_onset_single_ring_C,
        );
        assert_eq!(value.to_bits(), r.shaft_joint.hub_fit_allowed_C.to_bits());
        (r, limit)
    };
    // The defaults: your 100 °C (the onset 143.3565 °C less 10 is 133.3565 °C).
    let (r, limit) = allowed(&DesignInputs::default());
    assert_eq!(limit, Some(HubFitLimit::Yours));
    assert_eq!(r.shaft_joint.hub_fit_allowed_C, 100.0);
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C,
        133.3565,
        5e-5,
        "the onset's cap",
    );
    // DP460 (code 4) caps it at its 60 °C design limit; a typed 150 °C (above the slider's
    // 130 °C top) is capped at AA 326's 120 °C.
    let (r, limit) = allowed(&design(|i| i.temperature.adhesive.selected = 4));
    assert_eq!(
        (r.shaft_joint.hub_fit_allowed_C, limit),
        (60.0, Some(HubFitLimit::Adhesive))
    );
    let (r, limit) = allowed(&design(|i| i.shaft_joint.hub_fit_max_C = 150.0));
    assert_eq!(
        (r.shaft_joint.hub_fit_allowed_C, limit),
        (120.0, Some(HubFitLimit::Adhesive))
    );
    // `breaking(n42_parts)`: the library's N42 blocks in both rings, onset 69.83 °C, so
    // 59.83 °C allowed; the badge read a pass before R-15.
    let mut n42_parts = passing();
    n42_parts.coupling.magnets.part_inner = "B842".to_owned();
    n42_parts.coupling.magnets.part_outer = "B842".to_owned();
    let (r, limit) = allowed(&n42_parts);
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        69.8326,
        5e-5,
        "N42 onset",
    );
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        59.8326,
        5e-5,
        "N42 allowed",
    );
    assert_near(r.shaft_joint.hub_fit_temp_C, 93.267, 5e-4, "N42 needed");
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "59.8"));
    // N50M on manual dimensions in both rings: its onset, 98.80 °C, is above the 93.27 °C
    // need; only the onset less 10 °C, 88.80 °C, is below it.
    let (r, limit) = allowed(&manual(passing(), "N50M", "N50M"));
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        98.8029,
        5e-5,
        "N50M onset",
    );
    assert!(r.temperature.demag.inner_onset_single_ring_C > r.shaft_joint.hub_fit_temp_C);
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        88.8029,
        5e-5,
        "N50M allowed",
    );
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "88.8"));
    // Ferrite's onset is +inf: the cap never binds (the Y30 probe and `breaking(ferrite)`).
    for ferrite in [
        probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the Y30 probe")
            .1,
        manual(passing(), "Y30", "Y30"),
    ] {
        let (r, limit) = allowed(&ferrite);
        assert_eq!(r.temperature.demag.inner_onset_single_ring_C, f64::INFINITY);
        assert_eq!(
            (r.shaft_joint.hub_fit_allowed_C, limit),
            (100.0, Some(HubFitLimit::Yours))
        );
    }
    // 2214 Hi-Temp (code 3, 177 °C) with a typed 150 °C: the onset less 10 °C binds.
    let (r, limit) = allowed(&design(|i| {
        i.temperature.adhesive.selected = 3;
        i.shaft_joint.hub_fit_max_C = 150.0;
    }));
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        133.3565,
        5e-5,
        "2214 allowed",
    );
    assert_eq!(r.shaft_joint.check, SHAFT_JOINTS_PASS);
}

#[test]
fn the_hub_fitting_slider_tops_out_below_the_inner_onset_less_the_cure_margin() {
    // Review SF-3: at the defaults the slider's 130 °C top is below the inner ring's own
    // single-ring onset less the cure margin (133.36 °C), so the slider alone never reaches it.
    let r = compute_all(&DesignInputs::default());
    let top = slider("shaft_joint.hub_fit_max_C").max;
    assert_eq!(top, 130.0);
    assert!(top < r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C);
}

#[test]
fn the_cap_reads_the_inner_ring_s_own_onset_with_mixed_rings() {
    // Owner R-15 ring: the bonded hub's fitting heat reaches only the inner magnets (the cup's
    // are bonded after the cup is fitted), so the cap reads the inner ring's own onset, whichever
    // ring governs. Each pin asserts the governing ring and both onsets, so a cap read from the
    // governing ring, or a mistyped part name or grade id, fails it.
    let demag_pins = |inputs: &DesignInputs, ring: &str, governing: f64, inner: f64| {
        let r = compute_all(inputs);
        let demag = &r.temperature.demag;
        assert_eq!(demag.demag_ring, ring);
        assert_near(
            demag.onset_single_ring_C,
            governing,
            5e-5,
            "governing onset",
        );
        assert_near(demag.inner_onset_single_ring_C, inner, 5e-5, "inner onset");
        r
    };
    // Inner B842SH with outer B842: the outer N42 governs at 69.83 °C; the hub passes, your
    // limit binding (window 11.62-18.83 µm).
    let mut inputs = passing();
    inputs.coupling.magnets.part_outer = "B842".to_owned();
    let r = demag_pins(&inputs, "outer", 69.8326, 143.3565);
    assert_eq!(r.shaft_joint.hub_fit_allowed_C, 100.0);
    assert_near(r.shaft_joint.hub_needed_um, 11.617, 5e-4, "needed");
    assert_near(r.shaft_joint.hub_fittable_max_um, 18.828, 5e-4, "fittable");
    assert_eq!(r.shaft_joint.check, SHAFT_JOINTS_PASS);
    // Inner B842 with outer B842SH: the inner B842's own 69.83 °C less 10.
    let mut inputs = passing();
    inputs.coupling.magnets.part_inner = "B842".to_owned();
    let r = demag_pins(&inputs, "inner", 69.8326, 69.8326);
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "59.8"));
    // 2214 Hi-Temp, the hub input at its 130 °C slider top and a 22 µm hub maximum, inner N42H
    // with outer SmCo_2_17_30 (grade ids) on manual dimensions: the outer SmCo governs with
    // 191.88 °C, the inner N42H's own onset is 112.83 °C. The same at `passing_design()` and at
    // the defaults.
    for base in [passing(), DesignInputs::default()] {
        let mut inputs = manual(base, "N42H", "SmCo_2_17_30");
        inputs.temperature.adhesive.selected = 3;
        inputs.shaft_joint.hub_fit_max_C = 130.0;
        inputs.shaft_joint.hub_interference_max_um = 22.0;
        let r = demag_pins(&inputs, "outer", 191.8772, 112.8297);
        assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "125.8", "102.8"));
    }
}

#[test]
fn a_long_magnet_makes_the_cup_slip_first() {
    // The hub grows with the magnets under the axial override, the boss does not: at 40 mm the
    // cup slips and its key follows; the cup starts to slip near a magnet length of 26.5 mm.
    let long = design(|i| i.coupling.magnets.axial_length_mm = Some(40.0));
    let verdict = check(&long);
    assert!(
        verdict.starts_with(&format!("{FIT_SLIPS} ({CUP})")),
        "{verdict}"
    );
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&terms(&long))
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(failing, [(Cause::Slips, CUP), (Cause::Key, CUP)]);
    for (length, slips_at_cup) in [(26.0, false), (27.0, true)] {
        let inputs = design(|i| i.coupling.magnets.axial_length_mm = Some(length));
        let [_, cup] = terms(&inputs);
        assert_eq!(cup.fails(Cause::Slips), slips_at_cup, "{length}");
    }
}

#[test]
fn exactly_at_each_limit_passes_and_the_next_float_fails() {
    // On `joint`, from the inner hub at the defaults: the need, the stress limit, the allowed
    // temperature and the key's limit each set exactly to the term compared with it.
    let (common, [hub, _]) = specs(&DesignInputs::default());
    let base = shaft_joint::joint(&hub, &common);
    let with = |common: JointCommon, spec: JointSpec| shaft_joint::joint(&spec, &common);
    // Slip: holding = required passes; a required torque one float above slips.
    let at = with(
        JointCommon {
            required_Nm: base.holding_Nm,
            ..common
        },
        hub,
    );
    assert!(!at.fails(Cause::Slips));
    let over = with(
        JointCommon {
            required_Nm: base.holding_Nm.next_up(),
            ..common
        },
        hub,
    );
    assert!(over.fails(Cause::Slips));
    // Stress: stress = limit passes; a limit one float below fails.
    let at = with(
        JointCommon {
            stress_limit_MPa: base.bore_stress_MPa,
            ..common
        },
        hub,
    );
    assert!(!at.fails(Cause::BoreStress));
    let under = with(
        JointCommon {
            stress_limit_MPa: base.bore_stress_MPa.next_down(),
            ..common
        },
        hub,
    );
    assert!(under.fails(Cause::BoreStress));
    // Fitting: needed = allowed passes; an allowed temperature one float below fails.
    let at = with(
        common,
        JointSpec {
            fit_allowed_C: base.fit_temp_C,
            ..hub
        },
    );
    assert!(!at.fails(Cause::CannotBeFitted));
    let under = with(
        common,
        JointSpec {
            fit_allowed_C: base.fit_temp_C.next_down(),
            ..hub
        },
    );
    assert!(under.fails(Cause::CannotBeFitted));
    // Key: with no interference (no bore stress) the limit set to the key pressure passes, the
    // key margin is 0 to rounding; a limit one float below fails.
    let unstressed = JointSpec {
        interference_max_um: 0.0,
        ..hub
    };
    let at = with(
        JointCommon {
            stress_limit_MPa: base.key_pressure_MPa,
            ..common
        },
        unstressed,
    );
    assert!(!at.fails(Cause::Key) && !at.fails(Cause::BoreStress));
    assert!(
        at.key_margin_mm.abs() <= 1e-12 * at.key_bearing_mm,
        "{}",
        at.key_margin_mm
    );
    let under = with(
        JointCommon {
            stress_limit_MPa: base.key_pressure_MPa.next_down(),
            ..common
        },
        unstressed,
    );
    assert!(under.fails(Cause::Key));
}

#[test]
fn a_failing_badge_never_quotes_two_equal_numbers() {
    // The BL-059 rounding (spec section 4.2): the limit half-up to its decimals, the failing
    // value pushed one last digit past it.
    let (common, [hub, _]) = specs(&DesignInputs::default());
    let base = shaft_joint::joint(&hub, &common);
    // A stress 0.04 MPa over the limit.
    let t = shaft_joint::joint(
        &hub,
        &JointCommon {
            stress_limit_MPa: base.bore_stress_MPa - 0.04,
            ..common
        },
    );
    assert_eq!(
        t.text(Cause::BoreStress),
        over_stress(INNER_HUB, "241.8", "241.7")
    );
    // A fitting temperature 0.04 °C over the allowed.
    let t = shaft_joint::joint(
        &JointSpec {
            fit_allowed_C: base.fit_temp_C - 0.04,
            ..hub
        },
        &common,
    );
    assert_eq!(
        t.text(Cause::CannotBeFitted),
        cannot_fit(INNER_HUB, "93.3", "93.2")
    );
    // A holding 0.004 N·m under the need.
    let t = shaft_joint::joint(
        &hub,
        &JointCommon {
            required_Nm: base.holding_Nm + 0.004,
            ..common
        },
    );
    assert_eq!(t.text(Cause::Slips), slips(INNER_HUB, "9.48", "9.49"));
    // A need under 0.005 N·m quotes as 0.00: the holding (0 here) is quoted 0.00, the need 0.01.
    let t = shaft_joint::joint(
        &JointSpec {
            interference_min_um: 0.0,
            ..hub
        },
        &JointCommon {
            required_Nm: 0.004,
            ..common
        },
    );
    assert!(t.fails(Cause::Slips));
    assert_eq!(t.text(Cause::Slips), slips(INNER_HUB, "0.00", "0.01"));
    // The key 0.04 MPa over its limit (no interference, so no bore stress).
    let t = shaft_joint::joint(
        &JointSpec {
            interference_max_um: 0.0,
            ..hub
        },
        &JointCommon {
            stress_limit_MPa: base.key_pressure_MPa - 0.04,
            ..common
        },
    );
    assert_eq!(t.text(Cause::Key), key_over(INNER_HUB, "127.5", "127.4"));
    // On the sliders: the hub's allowed temperature 93 °C against the 93.27 °C need.
    let slider_step = design(|i| i.shaft_joint.hub_fit_max_C = 93.0);
    assert_eq!(check(&slider_step), cannot_fit(INNER_HUB, "93.3", "93.0"));
    // Every limit around each value, across two last digits: the value quoted is above the
    // limit quoted (below the need quoted, for holding).
    for k in 0..=200 {
        let offset = f64::from(k) * 0.001;
        let stress = shaft_joint::joint(
            &hub,
            &JointCommon {
                stress_limit_MPa: base.bore_stress_MPa - offset - f64::EPSILON,
                ..common
            },
        );
        let (q, r) = quoted(&stress.text(Cause::BoreStress));
        assert!(q > r, "{offset}");
        let fit = shaft_joint::joint(
            &JointSpec {
                fit_allowed_C: base.fit_temp_C - offset - 1e-9,
                ..hub
            },
            &common,
        );
        let (q, r) = quoted(&fit.text(Cause::CannotBeFitted));
        assert!(q > r, "{offset}");
        let holding = shaft_joint::joint(
            &hub,
            &JointCommon {
                required_Nm: base.holding_Nm + offset * 0.1 + 1e-9,
                ..common
            },
        );
        let (q, r) = quoted(&holding.text(Cause::Slips));
        assert!(q < r, "{offset}");
    }
}

#[test]
fn a_value_that_is_not_a_number_reads_unknown() {
    // Spec section 4.4: NaN in any of the 15 inputs or any of the 15 engine values the check
    // reads, tested on the values themselves before any min or max. `set` refuses NaN; a struct
    // holds one.
    type PutInput = fn(&mut ShaftJointInputs);
    let inputs_nan: [PutInput; 15] = [
        |i| i.hub_interference_min_um = f64::NAN,
        |i| i.hub_interference_max_um = f64::NAN,
        |i| i.cup_interference_min_um = f64::NAN,
        |i| i.cup_interference_max_um = f64::NAN,
        |i| i.safety_factor = f64::NAN,
        |i| i.friction = f64::NAN,
        |i| i.smoothing_um = f64::NAN,
        |i| i.rotor_over_shaft_C = f64::NAN,
        |i| i.key_width_mm = f64::NAN,
        |i| i.key_contact_mm = f64::NAN,
        |i| i.hub_fit_max_C = f64::NAN,
        |i| i.cup_fit_max_C = f64::NAN,
        |i| i.wheel_shaft_fit_C = f64::NAN,
        |i| i.output_shaft_fit_C = f64::NAN,
        |i| i.joining_clearance_um = f64::NAN,
    ];
    for (k, put) in inputs_nan.into_iter().enumerate() {
        let mut inputs = DesignInputs::default();
        put(&mut inputs.shaft_joint);
        assert_eq!(check(&inputs), UNKNOWN_NOT_A_NUMBER, "input {k}");
    }
    type PutLink = fn(&mut ShaftJointLinks);
    let links_nan: [PutLink; 15] = [
        |l| l.cold_high_Nm = f64::NAN,
        |l| l.bore_mm = f64::NAN,
        |l| l.keyway_depth_mm = f64::NAN,
        |l| l.hub_length_mm = f64::NAN,
        |l| l.inner_back_apothem_mm = f64::NAN,
        |l| l.bond_inner_mm = f64::NAN,
        |l| l.boss_length_mm = f64::NAN,
        |l| l.web_mm = f64::NAN,
        |l| l.boss_od_mm = f64::NAN,
        |l| l.min_temp_C = f64::NAN,
        |l| l.adhesive_limit_C = f64::NAN,
        |l| l.inner_onset_single_ring_C = f64::NAN,
        |l| l.part_modulus_GPa = f64::NAN,
        |l| l.part_cte_per_C = f64::NAN,
        |l| l.part_yield_MPa = f64::NAN,
    ];
    let defaults = DesignInputs::default();
    let base = links(&defaults);
    assert_eq!(
        shaft_joint::compute(&defaults.shaft_joint, &base).check,
        SHAFT_JOINTS_PASS
    );
    for (k, put) in links_nan.into_iter().enumerate() {
        let mut links = base;
        put(&mut links);
        assert!(links.values()[k].is_nan(), "link {k} is the {k}th value");
        assert_eq!(
            shaft_joint::compute(&defaults.shaft_joint, &links).check,
            UNKNOWN_NOT_A_NUMBER,
            "link {k}"
        );
    }
    // Through the engine: no adhesive (code 0: a NaN design limit), a NaN single-ring reverse
    // field (a NaN inner onset), and a back-iron pick outside the choices on the steel circuit
    // (NaN part values).
    let no_adhesive = design(|i| i.temperature.adhesive.selected = 0);
    assert_eq!(check(&no_adhesive), UNKNOWN_NOT_A_NUMBER);
    let nan_field = compute_all(&design(|i| {
        i.temperature.demag.h_rev_single_ring_kA_m = f64::NAN;
    }));
    assert!(
        nan_field
            .temperature
            .demag
            .inner_onset_single_ring_C
            .is_nan()
    );
    assert!(nan_field.shaft_joint.hub_fit_allowed_C.is_nan());
    assert_eq!(nan_field.shaft_joint.check, UNKNOWN_NOT_A_NUMBER);
    let unknown_pick = compute_all(&design(|i| i.materials.parts.back_iron = 99)).shaft_joint;
    assert!(
        unknown_pick.part_modulus_GPa.is_nan()
            && unknown_pick.part_cte_per_C.is_nan()
            && unknown_pick.part_yield_MPa.is_nan()
    );
    assert_eq!(unknown_pick.check, UNKNOWN_NOT_A_NUMBER);
}

#[test]
fn a_value_below_what_its_input_can_physically_be_reads_unknown() {
    // The whole-branch review's MF-1 (ruling C-4 extended): a friction, smoothing or joining
    // clearance below zero, a key width not above zero or a shaft below absolute zero is a typed
    // value no slider reaches; judged as if real, it gives false passes and fixes that make a
    // joint worse. Each reads unknown just past its floor and is judged at it;
    // `impossible_values` names it by its field, with its value and floor; and its slider's
    // bottom is a value it can be (the Problem asks for at least that).
    let absolute_zero = -273.15;
    // (the field, its floor, the lowest value it can be, a value just past that)
    let cases = [
        ("friction", Floor::Zero, 0.0, -1e-9),
        ("smoothing_um", Floor::Zero, 0.0, -1e-9),
        ("key_width_mm", Floor::AboveZero, 1e-9, 0.0),
        (
            "wheel_shaft_fit_C",
            Floor::AbsoluteZero,
            absolute_zero,
            absolute_zero - 1e-9,
        ),
        (
            "output_shaft_fit_C",
            Floor::AbsoluteZero,
            absolute_zero,
            absolute_zero - 1e-9,
        ),
        ("joining_clearance_um", Floor::Zero, 0.0, -1e-9),
    ];
    for (field, floor, lowest, past) in cases {
        let at = |x: f64| {
            let mut inputs = DesignInputs::default();
            inputs
                .shaft_joint
                .set(field, Value::Num(x))
                .unwrap_or_else(|e| panic!("{e}"));
            inputs
        };
        assert!(floor.allows(lowest) && !floor.allows(past), "{field}");
        let bottom = slider(&format!("shaft_joint.{field}")).min;
        assert!(floor.allows(bottom), "{field}'s slider bottom, {bottom}");
        let judged = at(lowest);
        assert_eq!(
            impossible_values(&judged.shaft_joint),
            [],
            "{field} at {lowest}"
        );
        assert_ne!(
            check(&judged),
            UNKNOWN_IMPOSSIBLE_VALUE,
            "{field} at {lowest}"
        );
        let typed = at(past);
        let listed = impossible_values(&typed.shaft_joint);
        assert_eq!(listed.len(), 1, "{field} at {past}");
        assert_eq!(
            (listed[0].field, listed[0].value, listed[0].floor),
            (field, past, floor)
        );
        assert_eq!(
            typed.shaft_joint.get(listed[0].field),
            Some(Value::Num(past)),
            "{field}"
        );
        assert_eq!(check(&typed), UNKNOWN_IMPOSSIBLE_VALUE, "{field} at {past}");
    }
    // Several at once: each named, in the inputs' order, whatever order they were typed in.
    let both = design(|i| {
        i.shaft_joint.joining_clearance_um = -2.0;
        i.shaft_joint.friction = -0.14;
    });
    let fields: Vec<&str> = impossible_values(&both.shaft_joint)
        .iter()
        .map(|value| value.field)
        .collect();
    assert_eq!(fields, ["friction", "joining_clearance_um"]);
    // A NaN is at or above no floor, and the verdict reads it as not a number first; then a
    // safety factor not above zero; then the impossible value, before any judged term (a
    // friction of -0.14 holds less than nothing, so judged it would read the fit slips).
    let nan = design(|i| {
        i.shaft_joint.friction = -0.14;
        i.shaft_joint.key_width_mm = f64::NAN;
    });
    assert_eq!(impossible_values(&nan.shaft_joint).len(), 2);
    assert_eq!(check(&nan), UNKNOWN_NOT_A_NUMBER);
    let no_factor = design(|i| {
        i.shaft_joint.friction = -0.14;
        i.shaft_joint.safety_factor = 0.0;
    });
    assert_eq!(check(&no_factor), UNKNOWN_SAFETY_FACTOR);
    let negative = design(|i| i.shaft_joint.friction = -0.14);
    let [hub, cup] = terms(&negative);
    assert!(hub.holding_Nm < 0.0 && cup.holding_Nm < 0.0);
    assert!(hub.determinate() && hub.fails(Cause::Slips));
    assert_eq!(check(&negative), UNKNOWN_IMPOSSIBLE_VALUE);
    // The engine's one test of whether the check can be made, which the guide reads too: its
    // reason's verdict is the check's, and the defaults have none.
    for (name, inputs) in [
        ("not a number", nan),
        ("safety factor", no_factor),
        ("impossible", negative),
        ("two impossible", both.clone()),
    ] {
        let links = links(&inputs);
        let joints = shaft_joint::joints(&inputs.shaft_joint, &links);
        let reason = shaft_joint::cannot_be_made(&inputs.shaft_joint, &links, &joints)
            .unwrap_or_else(|| panic!("{name}: the check cannot be made"));
        assert_eq!(reason.verdict(), check(&inputs), "{name}");
    }
    let links_both = links(&both);
    assert_eq!(
        shaft_joint::cannot_be_made(
            &both.shaft_joint,
            &links_both,
            &shaft_joint::joints(&both.shaft_joint, &links_both)
        ),
        Some(shaft_joint::CannotBeMade::ImpossibleValues(
            impossible_values(&both.shaft_joint)
        ))
    );
    let defaults = DesignInputs::default();
    let links = links(&defaults);
    let joints = shaft_joint::joints(&defaults.shaft_joint, &links);
    assert_eq!(
        shaft_joint::cannot_be_made(&defaults.shaft_joint, &links, &joints),
        None
    );
}

#[test]
fn a_nan_hub_limit_makes_the_allowed_temperature_nan_not_the_adhesive_s() {
    // Review 2, SF-3: `f64::min(NaN, 120)` is 120, so a NaN hub input would otherwise show an
    // allowed 120 °C and a fittable 21.29 µm while the verdict reads unknown.
    assert_eq!(f64::NAN.min(120.0), 120.0);
    let s = compute_all(&design(|i| i.shaft_joint.hub_fit_max_C = f64::NAN)).shaft_joint;
    assert!(s.hub_fit_allowed_C.is_nan() && s.hub_fittable_max_um.is_nan());
    assert_eq!(s.check, UNKNOWN_NOT_A_NUMBER);
}

#[test]
fn the_allowed_temperature_helper_drops_no_nan_and_names_the_limit_that_binds() {
    // Owner R-15, on `hub_fit_allowed`: py_min carries a NaN only in its first argument, so a
    // NaN anywhere is tested first.
    assert_eq!(py_min(100.0, f64::NAN), 100.0);
    for (input, adhesive, onset) in [
        (f64::NAN, 120.0, 143.0),
        (100.0, f64::NAN, 143.0),
        (100.0, 120.0, f64::NAN),
    ] {
        let (allowed, limit) = hub_fit_allowed(input, adhesive, onset);
        assert!(
            allowed.is_nan() && limit.is_none(),
            "{input} {adhesive} {onset}"
        );
    }
    let cap = |onset: f64| onset - CURE_MARGIN_MIN_C;
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 143.0),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 60.0, 143.0),
        (60.0, Some(HubFitLimit::Adhesive))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 69.0),
        (cap(69.0), Some(HubFitLimit::MagnetOnset))
    );
    // Ties name the earlier limit: an input equal to the adhesive's limit, and an onset exactly
    // the input plus the margin, name your limit; the onset tied with the adhesive names it.
    assert_eq!(
        hub_fit_allowed(120.0, 120.0, 143.0),
        (120.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 100.0 + CURE_MARGIN_MIN_C),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(130.0, 120.0, 120.0 + CURE_MARGIN_MIN_C),
        (120.0, Some(HubFitLimit::Adhesive))
    );
    // An onset of +inf (ferrite) never binds; one below 30 °C allows less than 20 °C.
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, f64::INFINITY),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 25.0),
        (cap(25.0), Some(HubFitLimit::MagnetOnset))
    );
    // Through the engine: the single-ring field at its slider's top (1500 kA/m) puts the inner
    // onset at -2.449 °C, so -12.45 °C allowed, quoted by the same never-equal rule.
    let top = slider("temperature.demag.h_rev_single_ring_kA_m").max;
    assert_eq!(top, 1500.0);
    let r = compute_all(&design(|i| {
        i.temperature.demag.h_rev_single_ring_kA_m = top
    }));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        -2.449,
        5e-4,
        "onset",
    );
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "-12.4"));
}

#[test]
fn a_reversed_band_reads_unknown_unless_the_other_joint_fails() {
    // Review SF-8: a reversed band is not judged and reads unknown, naming the joint; its terms
    // are still computed. A judged joint's failure outranks it.
    let reversed = design(|i| i.shaft_joint.hub_interference_min_um = 20.0);
    assert_eq!(
        check(&reversed),
        format!("{UNKNOWN_BAND_REVERSED} ({INNER_HUB})")
    );
    let [hub, cup] = terms(&reversed);
    assert!(hub.band_reversed && !hub.determinate() && cup.determinate());
    assert_near(hub.holding_Nm, 23.49, 5e-3, "the terms are computed");
    assert!(shaft_joint::failing(&[hub, cup]).is_empty());
    let and_cup_slips = design(|i| {
        i.shaft_joint.hub_interference_min_um = 20.0;
        i.shaft_joint.cup_interference_min_um = 10.0;
    });
    assert_eq!(check(&and_cup_slips), slips(CUP, "4.36", "7.65"));
}

#[test]
fn a_joint_with_no_wall_reads_nan_and_is_not_judged() {
    // Review SF-8, SF-9: bore 16 mm in a 12 mm boss leaves the cup no wall (Q >= 1). The inner
    // hub is judged and slips; the cup's Lamé rows read NaN, its key, fittable and fitting rows
    // numbers.
    let no_cup_wall = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.metal.boss_od_mm = 12.0;
    });
    let s = compute_all(&no_cup_wall).shaft_joint;
    assert_eq!(s.check, slips(INNER_HUB, "4.57", "7.65"));
    assert!(
        s.cup_holding_Nm.is_nan()
            && s.cup_bore_stress_MPa.is_nan()
            && s.cup_needed_um.is_nan()
            && s.cup_elastic_max_um.is_nan()
    );
    assert!(
        s.cup_key_pressure_MPa.is_finite()
            && s.cup_fittable_max_um.is_finite()
            && s.cup_fit_temp_C.is_finite()
    );
    let [hub, cup] = terms(&no_cup_wall);
    assert_eq!(
        (hub.no_wall, cup.no_wall),
        (None, Some(NoWall::BoreAsWideAsPart))
    );
    // With the hub's minimum at 15 µm it holds: the cup's missing wall is the verdict.
    let mut holding = no_cup_wall.clone();
    holding.shaft_joint.hub_interference_min_um = 15.0;
    assert_eq!(check(&holding), format!("{UNKNOWN_NO_WALL} ({CUP})"));
    // Bore 16 mm with a 2.1 mm keyway: the hub's 2.10 mm wall is cut through; at 2.05 mm it is
    // not, and the hub slips.
    let cut = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.coupling.keyway_depth_mm = 2.1;
    });
    let [hub, cup] = terms(&cut);
    assert_eq!(
        (hub.no_wall, cup.no_wall),
        (Some(NoWall::KeywayThroughWall), None)
    );
    assert_eq!(check(&cut), format!("{UNKNOWN_NO_WALL} ({INNER_HUB})"));
    let uncut = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.coupling.keyway_depth_mm = 2.05;
    });
    assert_eq!(check(&uncut), slips(INNER_HUB, "4.57", "7.65"));
    // A part diameter typed at or below zero has no wall either (ruling C-3, plan deviation
    // PD-1): keyless, with the boss typed at -22 mm, the inner hub passes and the cup is not
    // judged, where `q >= 1` would have judged it with the numbers of a +22 mm boss.
    let negative_boss = design(|i| {
        i.coupling.keyway_depth_mm = 0.0;
        i.metal.boss_od_mm = -22.0;
    });
    assert_eq!(check(&negative_boss), format!("{UNKNOWN_NO_WALL} ({CUP})"));
    // The inner hub's half (plan review 3A): its part diameter, 2 · (apothem − bondline), typed
    // at 0 mm (the apothem typed at the bondline) and at -40.4 mm (the apothem typed -10.15 mm),
    // each keyless and at the default 1.7 mm keyway. The hub has no wall and is not judged, and
    // the cup holds, where `q >= 1` would have judged the keyless -40.4 mm hub with the numbers
    // of a +40.4 mm one (a pass) and read the keyed one as cut through by its keyway.
    for apothem in [DesignInputs::default().metal.bond_inner_mm, -10.15] {
        for keyway in [0.0, 1.7] {
            let typed = design(|i| {
                i.coupling.inner_back_apothem_mm = apothem;
                i.coupling.keyway_depth_mm = keyway;
            });
            let what = format!("apothem {apothem} mm, keyway {keyway} mm");
            assert_eq!(
                check(&typed),
                format!("{UNKNOWN_NO_WALL} ({INNER_HUB})"),
                "{what}"
            );
            let joints = terms(&typed);
            assert_eq!(joints[0].no_wall, Some(NoWall::BoreAsWideAsPart), "{what}");
            assert!(shaft_joint::failing(&joints).is_empty(), "{what}");
            let s = compute_all(&typed).shaft_joint;
            assert!(
                s.hub_holding_Nm.is_nan()
                    && s.hub_bore_stress_MPa.is_nan()
                    && s.hub_needed_um.is_nan()
                    && s.hub_elastic_max_um.is_nan(),
                "{what}"
            );
        }
    }
}

#[test]
fn the_key_alone_is_a_backup_with_its_own_edges() {
    // Keyway depth 0.3 mm (on its slider's 0.05 mm grid): the contact height is capped at
    // 0.3 mm. Both fits hold, both stresses and fitting temperatures pass, so the key is the
    // badge's first cause; the cup's key (407.8 MPa) follows.
    let shallow = design(|i| i.coupling.keyway_depth_mm = 0.3);
    assert_eq!(check(&shallow), key_over(INNER_HUB, "509.8", "345.8"));
    let joints = terms(&shallow);
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(failing, [(Cause::Key, INNER_HUB), (Cause::Key, CUP)]);
    assert_near(joints[1].key_pressure_MPa, 407.82, 5e-3, "cup key");
    // The contact height is capped at the keyway depth: 152.9 / 122.3 MPa at 1.0 mm.
    let s = compute_all(&design(|i| i.coupling.keyway_depth_mm = 1.0)).shaft_joint;
    assert_near(s.hub_key_pressure_MPa, 152.93, 5e-3, "hub key");
    assert_near(s.cup_key_pressure_MPa, 122.35, 5e-3, "cup key");
    // No keyway: no key term and no arc taken off (10.49 / 18.68 N·m); the key rows read 0 and
    // both key terms pass with margin +inf.
    let keyless = design(|i| i.coupling.keyway_depth_mm = 0.0);
    let s = compute_all(&keyless).shaft_joint;
    assert_near(s.hub_holding_Nm, 10.486, 5e-3, "keyless hub");
    assert_near(s.cup_holding_Nm, 18.683, 5e-3, "keyless cup");
    assert_eq!((s.hub_key_pressure_MPa, s.cup_key_pressure_MPa), (0.0, 0.0));
    for joint in terms(&keyless) {
        assert!(!joint.fails(Cause::Key));
        assert_eq!(joint.margin(Cause::Key), f64::INFINITY);
    }
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
    // A contact height typed -0.5 mm: no bearing area, +inf (never -305.9 MPa, a pass).
    let negative = design(|i| i.shaft_joint.key_contact_mm = -0.5);
    let s = compute_all(&negative).shaft_joint;
    assert_eq!(s.hub_key_pressure_MPa, f64::INFINITY);
    assert_eq!(
        check(&negative),
        format!("{KEY_HAS_NO_BEARING_AREA} ({INNER_HUB})")
    );
    // A contact height typed 0 (shared, so both keys) with the boss at 0: neither key has a
    // bearing area, and each key margin is -inf until the contact is above zero (review 2,
    // SF-5).
    let no_contact = design(|i| {
        i.shaft_joint.key_contact_mm = 0.0;
        i.metal.boss_length_mm = 0.0;
    });
    for joint in terms(&no_contact) {
        assert!(!joint.key_has_bearing_area(), "{}", joint.name);
        assert_eq!(
            joint.margin(Cause::Key),
            f64::NEG_INFINITY,
            "{}",
            joint.name
        );
    }
    // Boss length 0 and 0.5 mm (the cup's 2.5 + 0.5 = 3 mm, the key's width): no bearing
    // length, so no bearing area, never a negative pressure that passes.
    for boss in [0.0, 0.5] {
        let [_, cup] = terms(&design(|i| i.metal.boss_length_mm = boss));
        assert!(!cup.key_has_bearing_area(), "{boss}");
        assert_eq!(cup.key_pressure_MPa, f64::INFINITY, "{boss}");
        assert!(cup.fails(Cause::Key), "{boss}");
        assert_eq!(
            cup.text(Cause::Key),
            format!("{KEY_HAS_NO_BEARING_AREA} ({CUP})")
        );
    }
    // The inner hub at its slider's bottom length, 3.0 mm (L = b): holding 2.189 N·m, the key
    // +inf; both causes, the slip in the badge.
    let short_hub = design(|i| i.metal.hub_length_mm = 3.0);
    let s = compute_all(&short_hub).shaft_joint;
    assert_near(s.hub_holding_Nm, 2.189, 5e-4, "short hub");
    assert_eq!(s.hub_key_pressure_MPa, f64::INFINITY);
    assert_eq!(s.check, slips(INNER_HUB, "2.19", "7.65"));
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&terms(&short_hub))
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(
        failing,
        [(Cause::Slips, INNER_HUB), (Cause::Key, INNER_HUB)]
    );
}

#[test]
fn each_margin_is_finite_where_its_term_can_fail_and_its_sign_is_the_verdict() {
    // Review MF-2: the key margin (bearing length less the length needed) is finite at boss
    // lengths 0 to 40 mm and its sign matches the key verdict, at contact heights 0.5, 0.8,
    // 1.2 and 1.7 mm. Review SF-1: the slip margin's sign matches the holding against the
    // required torque over the hub minimum 0 to 40 µm.
    let (common, [hub, cup]) = specs(&DesignInputs::default());
    let web = DesignInputs::default().metal.web_mm;
    for contact in [0.5, 0.8, 1.2, 1.7] {
        let common = JointCommon {
            key_contact_mm: contact,
            ..common
        };
        for boss in slider_values("metal.boss_length_mm") {
            let t = shaft_joint::joint(
                &JointSpec {
                    length_mm: boss + web,
                    ..cup
                },
                &common,
            );
            let margin = t.margin(Cause::Key);
            assert!(margin.is_finite(), "{contact} {boss}");
            assert_eq!(
                t.fails(Cause::Key),
                margin < 0.0,
                "{contact} {boss}: {margin}"
            );
        }
    }
    for minimum in slider_values("shaft_joint.hub_interference_min_um") {
        let t = shaft_joint::joint(
            &JointSpec {
                interference_min_um: minimum,
                ..hub
            },
            &common,
        );
        let margin = t.margin(Cause::Slips);
        assert!(margin.is_finite(), "{minimum}");
        assert_eq!(t.fails(Cause::Slips), margin < 0.0, "{minimum}: {margin}");
    }
    // The stress and fitting margins are the limit less the term.
    let t = shaft_joint::joint(&hub, &common);
    assert_eq!(
        t.margin(Cause::BoreStress),
        t.stress_limit_MPa - t.bore_stress_MPa
    );
    assert_eq!(
        t.margin(Cause::CannotBeFitted),
        t.fit_allowed_C - t.fit_temp_C
    );
}

#[test]
fn the_effective_interference_is_floored_at_zero() {
    // The 304 inner hub loses more than its minimum at the hot end: no grip, not a negative one.
    let [hub, _] = terms(&design(|i| i.materials.parts.back_iron = 7));
    assert_eq!(hub.effective_min_um, 0.0);
    assert_eq!(hub.holding_Nm, 0.0);
}

#[test]
fn a_shaft_above_room_temperature_grows_and_a_cooled_one_shrinks_as_nist_s_steel_does() {
    // Spec section 3.7, Ø10, diametral [µm]: the NIST Fe-2.25Ni fit below 265 K, the linear
    // 11.5e-6 /°C above.
    for (shaft, want) in [
        (40.0, -2.30),
        (-20.0, 4.55),
        (-40.0, 6.66),
        (-78.5, 10.35),
        (-196.0, 18.99),
    ] {
        assert_near(
            shaft_shrink_um(shaft, 10.0),
            want,
            5e-3,
            &format!("{shaft} °C"),
        );
    }
    assert_eq!(shaft_shrink_um(ROOM_C, 10.0), 0.0);
    // Continuous at 265 K and strictly falling as the shaft warms over the wheel shaft's slider.
    let edge = 265.0 - 273.15; // °C
    assert!((shaft_shrink_um(edge, 10.0) - shaft_shrink_um(edge - 1e-9, 10.0)).abs() < 1e-9);
    let values = slider_values("shaft_joint.wheel_shaft_fit_C");
    for pair in values.windows(2) {
        assert!(
            shaft_shrink_um(pair[1], 10.0) < shaft_shrink_um(pair[0], 10.0),
            "{pair:?}"
        );
    }
    // Through the engine: the gearbox shaft at 40 °C grows 2.30 µm; the cup still fits at
    // 298.9 °C against 300 °C.
    let s = compute_all(&design(|i| i.shaft_joint.output_shaft_fit_C = 40.0)).shaft_joint;
    assert_near(s.output_shaft_shrink_um, -2.30, 5e-3, "grows");
    assert_near(s.cup_fit_temp_C, 298.86, 5e-3, "cup fit temperature");
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
}

#[test]
fn an_infinite_cold_high_slips() {
    // +inf fails where it is compared: the required torque +inf slips at the inner hub first.
    let defaults = DesignInputs::default();
    let infinite = ShaftJointLinks {
        cold_high_Nm: f64::INFINITY,
        ..links(&defaults)
    };
    assert_eq!(
        shaft_joint::compute(&defaults.shaft_joint, &infinite).check,
        slips(INNER_HUB, "9.48", "inf")
    );
}

#[test]
fn the_joints_rebuilt_from_the_results_are_the_engine_s() {
    // `api::shaft_joint_links` reads every value from the design's inputs and results; the check
    // recomputed from it is the engine's, at the probe designs and at the edge designs above.
    let mut designs: Vec<DesignInputs> = probe_designs().into_iter().map(|(_, d)| d).collect();
    designs.extend([
        passing(),
        design(|i| i.coupling.magnets.axial_length_mm = Some(40.0)),
        design(|i| {
            i.coupling.bore_mm = 16.0;
            i.metal.boss_od_mm = 12.0;
        }),
        design(|i| i.coupling.backiron = 0),
        design(|i| i.temperature.adhesive.selected = 0),
    ]);
    for inputs in designs {
        let r = compute_all(&inputs);
        let again = shaft_joint::compute(&inputs.shaft_joint, &shaft_joint_links(&inputs, &r));
        // NaN equal to NaN: compare the debug texts.
        assert_eq!(format!("{again:?}"), format!("{:?}", r.shaft_joint));
    }
}

#[test]
fn the_shaft_joint_inputs_move_no_result_outside_the_check() {
    // Spec section 7 item 3, in the spirit of the paused clamp's guard: each `shaft_joint.*`
    // input at both slider ends, at the five probe designs, changes only `shaft_joint.*` results
    // (NaN equal to NaN).
    let outside = |inputs: &DesignInputs| -> Vec<(String, String)> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .filter(|row| !row.path.starts_with("shaft_joint."))
            .map(|row| (row.path, format!("{:?}", row.value)))
            .collect()
    };
    let mut moved_inside = 0;
    for (name, design) in probe_designs() {
        let base = outside(&design);
        let base_check = compute_all(&design).shaft_joint;
        for row in input_rows(&design)
            .into_iter()
            .filter(|row| row.path.starts_with("shaft_joint."))
        {
            let range = row.meta.range.expect("a slider");
            for value in [range.min, range.max] {
                let mut edited = design.clone();
                edited
                    .set(&row.path, Value::Num(value))
                    .expect("a value the input accepts");
                assert_eq!(outside(&edited), base, "{name}: {} at {value}", row.path);
                moved_inside += usize::from(compute_all(&edited).shaft_joint != base_check);
            }
        }
    }
    // Not vacuous: the inputs move the check's own results.
    assert!(moved_inside > 100, "{moved_inside}");
}
