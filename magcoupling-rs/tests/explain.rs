//! Addendum A2/A3 explanation layer: the drift guard, the registry's structure, the v1
//! scope, the teaching-note links and the A3 traceability test.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
//! text exact), corrections on (`compute_all`, what users see), at: the defaults; every
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override, the coercivity from the inputs with two ferrite betas,
//! the sleeve and cap materials, E17's free-space fields), without which the tau7-tau11
//! records would compare 0 with 0 and the ferrite and material branches would never run.
//! Anti-vacuity checks require every `cases` arm of every record to be taken, every numeric
//! record and every value term of every record to take two values, and the E7 angles to
//! leave half a pitch.
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
use magcoupling::engine::explain::notes::{Change, Sense, Step};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, note_text, notes, probes, render};
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library;
use magcoupling::engine::material_library::{BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, chosen};
use magcoupling::engine::meta::{
    FieldType, InputMeta, InputSet, NumOrText, ResultSet, SliderRange, Value, input_rows,
    result_rows,
};
use magcoupling::engine::model::{ModelResults, blocks_fit};
use magcoupling::engine::size_controls::with_ring_radius;
use magcoupling::engine::temperature::{NO_COLD_ONSET, RING_INNER, RING_OUTER, demag_onset_C};

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
    // The sleeve, liner and cap picks (A5): each supplies its library values in place of the inputs.
    v.push((
        "sleeve Ti, cap acetal",
        vec![
            ("materials.parts.sleeve_liner", Value::Int(2)),
            ("materials.parts.cap_housing", Value::Int(3)),
        ],
    ));
    // E17's free-space fields (Rust-only inputs) off their defaults, with an aluminium back
    // iron so the free-space branches of the hub, cup and web losses read them.
    v.push((
        "E17 fields",
        vec![
            ("materials.parts.back_iron", Value::Int(8)),
            ("temperature.slip_loss.b_hub_free_T", Value::Num(0.1)),
            ("temperature.slip_loss.b_cup_free_T", Value::Num(0.12)),
            (
                "temperature.slip_loss.web_integral_free_T2m2",
                Value::Num(1e-5),
            ),
        ],
    ));
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
    // E20's cold side from the inputs: hard ferrite's positive beta for both rings, first on
    // rated magnets (both rings in the grade mode, bonded NdFeB and ferrite, so the rings' Br
    // coefficients and densities differ and either ring can hold the higher cold limit), then
    // on manual magnets with no grade and so no rating (no hot limit).
    v.push((
        "ferrite inputs",
        vec![
            ("temperature.demag.coercivity_source", Value::Int(0)),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.0035)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("Bonded_NdFeB_BCN19")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Y30")),
        ],
    ));
    v.push((
        "unrated ferrite",
        vec![
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.part_outer", text("")),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.0035)),
        ],
    ));
    // A second positive beta from the inputs with the rings swapped (ferrite inside, bonded
    // NdFeB outside), so every cold-side record sees two betas, and the outer ring a second
    // Br coefficient and density.
    v.push((
        "ferrite inputs, beta 0.002",
        vec![
            ("temperature.demag.coercivity_source", Value::Int(0)),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.002)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("Y30")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Bonded_NdFeB_BCN19")),
        ],
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

/// Input sets `set` accepts that pin what no differential case or augmentation is built to
/// reach, each with what it pins, applied on top of the defaults (records hold over every input
/// `set` accepts, except a clamp count outside i64, which the engine holds as its saturating
/// cast: `saturation_artifact`).
fn edge_points() -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    let text = |s: &str| Value::Text(s.to_owned());
    // The clamp comparisons the differential data never lands on exactly (the engine's own unit
    // tests pin the same edges, clamps.rs). Each point sets one input to a value of the default
    // design's M4 column (row 2 of the clamp table, the size that works there), so the
    // comparison holds with equality: the records' `<=`, `<` and `>=` must be the engine's and
    // not a strict or a loose twin. Read from the defaults, those values cannot go stale (the
    // span point's length is a dyadic constant, explained there).
    let defaults = compute_all(&DesignInputs::default());
    let default_num = |path: &str| match defaults.get(path) {
        Some(Value::Num(x)) => x,
        other => panic!("{path}: {other:?}"),
    };
    let m4 = |field: &str| default_num(&format!("clamps.table[2].{field}"));
    // The temperature chain's points check that they sit on their edge: `on_edge` applies the
    // sets to the defaults and asserts that each listed result holds exactly the given value
    // there, so a change to the defaults or the engine cannot leave a point pinning nothing.
    let on_edge = |label: &'static str, sets: Vec<(&'static str, Value)>, ties: &[(&str, f64)]| {
        let results = compute_all(&defaults_with(label, &sets));
        for &(path, want) in ties {
            assert_eq!(
                results.get(path),
                Some(Value::Num(want)),
                "edge point {label}: {path} is not on its edge"
            );
        }
        (label, sets)
    };
    let num = Value::Num;
    let onset = default_num("temperature.demag.onset_skipping_C");
    let hot_day_start = default_num("temperature.duty.hot_day_start_C");
    vec![
        (
            // The outer ring (manual, ferrite's beta) has a 0/0 cold onset, so its cold limit is
            // NaN, while the inner Y30 ring's is finite and below the minimum temperature: the
            // inner ring holds the cold side, yet the outer ring's check fails (min >= NaN is
            // false), so `cold_check` must state both rings' checks, not the higher limit's.
            "outer cold limit undefined (NaN), inner Y30",
            vec![
                ("temperature.demag.h_rev_likepole_kA_m", Value::Num(0.0)),
                ("temperature.demag.hcj20_kA_m", Value::Num(0.0)),
                ("temperature.demag.beta_hcj_per_C", Value::Num(0.0035)),
                ("coupling.magnets.part_inner", text("")),
                ("coupling.magnets.grade_inner", text("Y30")),
                ("coupling.magnets.part_outer", text("")),
            ],
        ),
        (
            // head_fits is `offset + head / 2 <= R`: the head's edge on the boss wall (a 23.5 mm
            // boss at the defaults) still fits, and its grip takes the square-root branch at 0.
            "clamp M4 head edge exactly on the boss wall",
            vec![(
                "clamps.boss_od_mm",
                Value::Num(2.0 * (m4("offset_mm") + m4("head_mm") / 2.0)),
            )],
        ),
        (
            // thread_avail is the square root only while `offset < R`: with the screw axis on
            // the wall (a 16.5 mm boss at the defaults) it is 0, not the root's -slit / 2.
            "clamp M4 screw axis exactly on the boss wall",
            vec![("clamps.boss_od_mm", Value::Num(2.0 * m4("offset_mm")))],
        ),
        (
            // geometry_ok needs `wall >= minimum wall`.
            "clamp M4 wall outside the hole exactly the minimum",
            vec![("clamps.wall_out_mm", Value::Num(m4("wall_out_mm")))],
        ),
        (
            // geometry_ok needs `grip >= minimum grip`.
            "clamp M4 grip exactly the minimum",
            vec![("clamps.grip_min_mm", Value::Num(m4("grip_mm")))],
        ),
        (
            // geometry_ok needs `thread available >= engagement`, engagement = factor x d. M4's d
            // is 4 mm, a power of two, so dividing by it and multiplying back is exact.
            "clamp M4 thread available exactly the engagement required",
            vec![(
                "clamps.engagement_x_d",
                Value::Num(m4("thread_avail_mm") / m4("d_mm")),
            )],
        ),
        (
            // screws_fit counts screws only while the span `length - 2 margin - (head + 0.5)` is
            // not negative: a span of exactly 0 fits one. M4's head is 7 mm and the margin 1 mm,
            // so 9.5 mm (every value dyadic: exact), the engine's own unit test's length.
            "clamp M4 span exactly zero",
            vec![("clamps.clamp_length_mm", Value::Num(9.5))],
        ),
        // The dashboard's two verdict comparisons, on the default design, each input set to the
        // value it is compared with (both inputs, read from the defaults like the clamp values).
        (
            // materials.cup_wall_check says OK at `wall >= need` (materials.rs): a cup wall of
            // exactly the back-iron thickness needed is not too thin.
            "cup wall exactly the back iron needed",
            vec![(
                "metal.cup_wall_corner_mm",
                Value::Num(default_num("model.backiron_needed_mm")),
            )],
        ),
        (
            // metal.clearance_check says Below target only at `min run < target`
            // (metal_design.rs, whose own unit test pins the same edge): a residual target of
            // exactly the minimum running clearance is met.
            "residual target exactly the minimum running clearance",
            vec![(
                "metal.residual_target_mm",
                Value::Num(default_num("metal.min_running_clearance_mm")),
            )],
        ),
        // The temperature chain's comparisons (the engine's own unit tests pin the same edges,
        // metal_design.rs and temperature.rs), each on the default design with an input set from
        // the defaults so that the comparison holds with equality. Where a magnet limit must
        // equal a value X, the design margin is set to onset - X: the magnet limit is the
        // skipping onset minus the margin, the onset does not read the margin, and both
        // subtractions are exact (Sterbenz: the onset, about 103 °C, lies within a factor 2 of
        // each X used), so the limit is exactly X.
        on_edge(
            // metal.hot_min_check says Below hot minimum only at `hot low < required minimum`;
            // the hot low torque does not read the required minimum.
            "required minimum exactly the hot low torque",
            vec![(
                "metal.required_min_Nm",
                num(default_num("metal.torque_hot_low_Nm")),
            )],
            &[(
                "metal.torque_hot_low_Nm",
                default_num("metal.torque_hot_low_Nm"),
            )],
        ),
        on_edge(
            // torque_hot_day_check meets it at `hot-day torque >= required minimum`.
            "required minimum exactly the hot-day torque",
            vec![(
                "metal.required_min_Nm",
                num(default_num("temperature.magnet_life.torque_hot_day_Nm")),
            )],
            &[(
                "temperature.magnet_life.torque_hot_day_Nm",
                default_num("temperature.magnet_life.torque_hot_day_Nm"),
            )],
        ),
        // The rating checks say OK at `operating temperature <= rating`, one point per ring
        // (a differential case lands there too today: the slider's maximum is the 150 °C
        // rating; these points do not depend on it).
        on_edge(
            "operating temperature exactly the inner ring's rating",
            vec![("coupling.op_temp_C", num(default_num("model.inner_tmax_C")))],
            &[("model.inner_tmax_C", default_num("model.inner_tmax_C"))],
        ),
        on_edge(
            "operating temperature exactly the outer ring's rating",
            vec![("coupling.op_temp_C", num(default_num("model.outer_tmax_C")))],
            &[("model.outer_tmax_C", default_num("model.outer_tmax_C"))],
        ),
        on_edge(
            // governing_note says the magnets govern at `magnet limit <= adhesive limit`.
            "magnet limit exactly the adhesive limit",
            vec![(
                "temperature.demag.design_margin_C",
                num(onset - default_num("temperature.adhesive.design_limit_C")),
            )],
            &[(
                "temperature.demag.magnet_limit_C",
                default_num("temperature.adhesive.design_limit_C"),
            )],
        ),
        // The verdict's four terms, each alone on its edge with the other three holding
        // (temperature.rs `verdict_*`). With heat flowing the peak lies above the hot-day
        // start, so a hot-day margin of 0 leaves the magnet or the adhesive margin at most 0;
        // only a negative bench drag (`set` takes it, as Python does) separates the two.
        on_edge(
            // `margin_hot > 0`: the drag cools the coupling, so the peak stays below the hot-day
            // start, which the governing magnet limit equals.
            "hot-day margin exactly 0 (a negative bench drag)",
            vec![
                ("metal.measured_drag_Nm", num(-1.0)),
                (
                    "temperature.demag.design_margin_C",
                    num(onset - hot_day_start),
                ),
            ],
            &[("temperature.summary.margin_hot_day_C", 0.0)],
        ),
        on_edge(
            // `magnet margin > 0`: the peak does not read the margin.
            "magnet margin exactly 0",
            vec![(
                "temperature.demag.design_margin_C",
                num(onset - default_num("temperature.magnet_life.peak_C")),
            )],
            &[("temperature.magnet_life.margin_limit_C", 0.0)],
        ),
        on_edge(
            // `adhesive margin > 0`: DP460's 60 °C limit governs, and the hot-day ambient that
            // puts the peak exactly on it is searched (the peak follows the ambient one for one).
            "adhesive margin exactly 0",
            vec![
                ("temperature.adhesive.selected", Value::Int(4)),
                (
                    "temperature.duty.hot_ambient_C",
                    num(exact_input(
                        &[("temperature.adhesive.selected", Value::Int(4))],
                        "temperature.duty.hot_ambient_C",
                        (1.0, 60.0),
                        "temperature.adhesive_life.margin_C",
                        0.0,
                    )),
                ),
            ],
            &[("temperature.adhesive_life.margin_C", 0.0)],
        ),
        on_edge(
            // `cure margin >= 10`: EA 9514 cures at 120 °C, and the single-ring reverse field
            // that puts its onset exactly at 130 °C is searched.
            "cure margin exactly 10",
            vec![
                ("temperature.adhesive.selected", Value::Int(2)),
                (
                    "temperature.demag.h_rev_single_ring_kA_m",
                    num(exact_input(
                        &[("temperature.adhesive.selected", Value::Int(2))],
                        "temperature.demag.h_rev_single_ring_kA_m",
                        (1.0, 1000.0),
                        "temperature.summary.cure_margin_C",
                        10.0,
                    )),
                ),
            ],
            &[("temperature.summary.cure_margin_C", 10.0)],
        ),
        // calibration.fea_interp_Nm interpolates the two FEA torques only for a corner gap in
        // [1, 1.5] mm. With the gap defined as the spacing (code 0) the corner gap is the
        // spacing, so each end exactly (differential cases land there too today).
        on_edge(
            "Calibration corner gap exactly 1 mm (the FEA range's low end)",
            vec![
                ("calibration.gap_definition", Value::Int(0)),
                ("calibration.spacing_mm", num(1.0)),
            ],
            &[("calibration.corner_gap_mm", 1.0)],
        ),
        on_edge(
            "Calibration corner gap exactly 1.5 mm (the FEA range's high end)",
            vec![
                ("calibration.gap_definition", Value::Int(0)),
                ("calibration.spacing_mm", num(1.5)),
            ],
            &[("calibration.corner_gap_mm", 1.5)],
        ),
        on_edge(
            // The time to the limit (E12): `start >= limit => 0` at equality, tested before
            // `steady <= limit`. A bench drag of 0 puts both steady states at the start too, so a
            // strict `>` would read "never" (with heat flowing it would give -tau ln(1 - 0) =
            // -0 s, which the parity rule takes for 0: that point would pin nothing).
            "hot-day start exactly the governing limit, no slip heating",
            vec![
                ("metal.measured_drag_Nm", num(0.0)),
                (
                    "temperature.demag.design_margin_C",
                    num(onset - hot_day_start),
                ),
            ],
            &[
                ("temperature.summary.governing_limit_C", hot_day_start),
                ("temperature.thermal.steady_high_C", hot_day_start),
                ("temperature.thermal.steady_est_C", hot_day_start),
            ],
        ),
        // `steady <= limit => "never"` at equality, with the start below the limit: the
        // governing magnet limit set to each steady state in turn (neither reads the margin).
        on_edge(
            "governing limit exactly the high-loss steady state",
            vec![(
                "temperature.demag.design_margin_C",
                num(onset - default_num("temperature.thermal.steady_high_C")),
            )],
            &[(
                "temperature.summary.governing_limit_C",
                default_num("temperature.thermal.steady_high_C"),
            )],
        ),
        on_edge(
            "governing limit exactly the estimated steady state",
            vec![(
                "temperature.demag.design_margin_C",
                num(onset - default_num("temperature.thermal.steady_est_C")),
            )],
            &[(
                "temperature.summary.governing_limit_C",
                default_num("temperature.thermal.steady_est_C"),
            )],
        ),
        // The clamp counts outside i64 (`saturation_artifact`), one point per kind, each
        // reaching every screw size the exemption's liveness assertion names.
        (
            // A friction coefficient of 0 holds nothing: the torque per screw is 0 and the screws
            // needed are +inf in the record, i64::MAX in the engine's cast.
            "clamp friction exactly 0",
            vec![("clamps.friction", Value::Num(0.0))],
        ),
        (
            // Nothing to hold either (required torque 0) is 0 / 0: NaN in the record, 0 in the
            // engine's cast.
            "clamp friction exactly 0 and safety factor 0",
            vec![
                ("clamps.friction", Value::Num(0.0)),
                ("clamps.safety_factor", Value::Num(0.0)),
            ],
        ),
        (
            // A negative required torque over a torque per screw of 0 is -inf in the record,
            // i64::MIN in the engine's cast.
            "clamp friction exactly 0 and safety factor -1",
            vec![
                ("clamps.friction", Value::Num(0.0)),
                ("clamps.safety_factor", Value::Num(-1.0)),
            ],
        ),
        (
            // A friction coefficient of almost 0: every size's screws needed are finite but
            // past 2^63 (about 1e29), i64::MAX in the engine's cast.
            "clamp friction 1e-30",
            vec![("clamps.friction", Value::Num(1e-30))],
        ),
        (
            // The same with a negative required torque: past -2^63, i64::MIN in the cast.
            "clamp friction 1e-30 and safety factor -1",
            vec![
                ("clamps.friction", Value::Num(1e-30)),
                ("clamps.safety_factor", Value::Num(-1.0)),
            ],
        ),
        (
            // An absurdly long clamp: every size's screws that fit are past 2^63 (about 1e299),
            // i64::MAX in the engine's cast (saturating_add keeps it there).
            "clamp length 1e300",
            vec![("clamps.clamp_length_mm", Value::Num(1e300))],
        ),
    ]
}

/// The defaults with `sets` applied (`label` names the point in a refusal).
fn defaults_with(label: &str, sets: &[(&str, Value)]) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    for (path, value) in sets {
        inputs
            .set(path, value.clone())
            .unwrap_or_else(|e| panic!("{label}: {e}"));
    }
    inputs
}

/// The value in `[lo, hi]` of the input `path` (set on top of the defaults with `base`) at
/// which the result `out` is exactly `target`, for `out` monotone in it with `target` between
/// its values at the ends. It bisects on the floats themselves (positive floats order as their
/// bit patterns), so it ends on two adjacent floats, one of which must give `target` exactly;
/// it panics if neither does, since an edge point off its edge would pin nothing.
fn exact_input(
    base: &[(&'static str, Value)],
    path: &'static str,
    (lo, hi): (f64, f64),
    out: &str,
    target: f64,
) -> f64 {
    assert!(0.0 < lo && lo < hi, "{path}: a positive bracket");
    let at = |x: f64| {
        let mut sets = base.to_vec();
        sets.push((path, Value::Num(x)));
        match compute_all(&defaults_with(path, &sets)).get(out) {
            Some(Value::Num(y)) => y,
            other => panic!("{out}: {other:?}"),
        }
    };
    let below = at(lo) < target;
    assert_ne!(
        below,
        at(hi) < target,
        "{out} = {target} is not between {path} = {lo} and {hi}"
    );
    let (mut a, mut b) = (lo.to_bits(), hi.to_bits());
    while b - a > 1 {
        let m = a + (b - a) / 2;
        if (at(f64::from_bits(m)) < target) == below {
            a = m;
        } else {
            b = m;
        }
    }
    let (a, b) = (f64::from_bits(a), f64::from_bits(b));
    [a, b]
        .into_iter()
        .find(|&x| at(x) == target)
        .unwrap_or_else(|| {
            panic!("no {path} gives {out} = {target} exactly: {a} and {b} straddle it")
        })
}

/// The defaults, the edge points, then every differential case under every augmentation.
fn guard_points() -> Vec<Point> {
    let mut points = vec![Point {
        label: "defaults".into(),
        inputs: DesignInputs::default(),
    }];
    for (label, sets) in edge_points() {
        points.push(Point {
            label: format!("edge point: {label}"),
            inputs: defaults_with(label, &sets),
        });
    }
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

/// The pairs the guard accepts although they differ: a clamp count the engine stores as an
/// integer when the record's count lies outside i64. The record states the count, the screws
/// needed `ceilto(required / torque per screw, 1)` or the screws that fit `floorto(span /
/// pitch, 1) + 1`: +inf, -inf or NaN (0 / 0) when a screw holds no torque, finite but past
/// 2^63 when it holds almost none or the clamp is absurdly long. The engine's `as i64`
/// (clamps.rs `need` and `fit`, the only float-to-integer casts behind a result) saturates
/// (README, the port's Python-to-Rust table: Python's `int(inf)` raises, outside every slider
/// range), so it holds i64::MAX, i64::MIN or 0. None of these is a physical count. Keyed by
/// path, by the record lying outside i64 and by the engine holding exactly the record's
/// saturating cast, so an in-range record, any other engine value or any other path still
/// fails. Returns the kind's name.
fn saturation_artifact(target: &str, record: &Value, engine: &Value) -> Option<&'static str> {
    // 2^63, the first f64 past i64::MAX; i64::MIN is exactly -2^63, so it is in range.
    const PAST_I64: f64 = 9_223_372_036_854_775_808.0;
    let (row, column) = target.strip_prefix("clamps.table[")?.split_once("].")?;
    row.parse::<usize>().ok()?;
    if column != "screws_needed" && column != "screws_fit" {
        return None;
    }
    let (Value::Num(r), Value::Int(e)) = (record, engine) else {
        return None;
    };
    let outside = r.is_nan() || *r >= PAST_I64 || *r < -PAST_I64;
    if !outside || *e != *r as i64 {
        return None;
    }
    Some(match *r {
        r if r.is_nan() => "NaN",
        f64::INFINITY => "+inf",
        f64::NEG_INFINITY => "-inf",
        r if r > 0.0 => "past i64::MAX",
        _ => "past i64::MIN",
    })
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
    /// The "target: kind" entries `saturation_artifact` accepted (the exemption is live).
    saturated: BTreeSet<String>,
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
                Ok(got) => match saturation_artifact(&eq.target, &got, &want) {
                    Some(kind) => {
                        cov.saturated.insert(format!("{}: {kind}", eq.target));
                    }
                    None => failures.push(format!(
                        "{} at {}: record {got:?}, engine {want:?}",
                        eq.target, p.label
                    )),
                },
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
        self.saturated.extend(other.saturated);
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
    // The defaults, the edge points, then every differential case under every augmentation
    // (no stale count: the data files are regenerated).
    let cases: usize = differential_files()
        .into_iter()
        .map(|f| load_cases(f).len())
        .sum();
    assert!(cases > 0);
    assert_eq!(
        points.len(),
        1 + edge_points().len() + cases * augmentations().len(),
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
    // The saturation exemption is live and exactly as wide as stated: each of the five sizes'
    // screws needed met each kind (the friction edge points) and its screws that fit met
    // "past i64::MAX" (the long clamp); no other record met any.
    let kinds = ["+inf", "-inf", "NaN", "past i64::MAX", "past i64::MIN"];
    let want: BTreeSet<String> = (0..5)
        .flat_map(|n| {
            kinds
                .map(|kind| format!("clamps.table[{n}].screws_needed: {kind}"))
                .into_iter()
                .chain([format!("clamps.table[{n}].screws_fit: past i64::MAX")])
        })
        .collect();
    assert_eq!(cov.saturated, want, "the saturation exemption");
    // And no wider: an in-range count, an engine value that is not the record's cast, another
    // column, another path or a target that is not a table row still fails.
    for (target, record, engine) in [
        ("clamps.table[0].screws_needed", 3.5, 3),
        ("clamps.table[0].screws_needed", f64::INFINITY, 0),
        ("clamps.table[0].screws_fit", -1e300, i64::MAX),
        ("clamps.table[0].preload_N", f64::INFINITY, i64::MAX),
        ("clamps.screws", f64::INFINITY, i64::MAX),
        ("clamps.table[].screws_needed", f64::INFINITY, i64::MAX),
    ] {
        let (record, engine) = (Value::Num(record), Value::Int(engine));
        assert_eq!(
            saturation_artifact(target, &record, &engine),
            None,
            "{target}: record {record:?}, engine {engine:?}"
        );
    }

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

/// The A4 accuracy gate for release (spec section 4, test 7), on since slice 2: every note a
/// user can open by id has passed the physics review: each "start here" step's, each A5
/// warning's and, with the `gui` feature, each check guide's and each dashboard row's Why note
/// (gate 7 runs that part, gate 4 the rest). A note hidden with the paused shaft-clamp section
/// (`notes::shown`, ruling 11) cannot be opened, so it is out of the release. `note_for` keeps
/// a draft off the screen; this keeps a draft out of a release.
#[test]
fn release_notes_are_reviewed() {
    let mut linked: Vec<&str> = notes::START_HERE.iter().map(|(id, _)| *id).collect();
    linked.extend(
        magcoupling::engine::warnings::WARNING_RULES
            .iter()
            .map(|r| r.note_id),
    );
    #[cfg(feature = "gui")]
    {
        use magcoupling::gui::guidance::{
            BANNER_NOTE, CHECK_GUIDES, Learn, ROW_LEARN, SIZE_TAB_NOTE, TORQUE_BAND_NOTE,
        };
        let guides = CHECK_GUIDES.iter().map(|g| g.learn);
        let rows = ROW_LEARN.iter().map(|&(_, learn)| learn);
        let before = linked.len();
        linked.extend(guides.chain(rows).filter_map(|learn| match learn {
            Learn::Note(id) => Some(id),
            Learn::Assumption(_) | Learn::None(_) => None,
        }));
        assert!(
            linked.len() > before,
            "the guides and the rows link notes (the widened gate is not vacuous)"
        );
        // The Size tab's and the end-effect banner's Why links (spec 3.7), after the check
        // above, which counts the guides' and the rows' links alone.
        linked.extend([SIZE_TAB_NOTE, TORQUE_BAND_NOTE, BANNER_NOTE]);
    }
    linked.sort_unstable();
    linked.dedup();
    linked.retain(|id| notes::shown(id));
    let drafts: Vec<&str> = linked
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

/// Spec section 4 (S2-6): every note shown ships as reviewed by the slice-2 physics review, its
/// record names its own section of the committed review report, and the report holds that
/// section exactly once (a re-review goes under the note's section as a `###` subsection).
#[test]
fn every_note_names_its_committed_review_section() {
    const REPORT: &str = "docs/analyses/2026-10-14-magcoupler-notes-review.md";
    let report = include_str!("../../docs/analyses/2026-10-14-magcoupler-notes-review.md");
    let iso_date = |date: &str| {
        date.len() == 10
            && date.bytes().enumerate().all(|(i, b)| {
                if i == 4 || i == 7 {
                    b == b'-'
                } else {
                    b.is_ascii_digit()
                }
            })
    };
    for n in notes::NOTES.iter().filter(|n| notes::shown(n.id)) {
        let notes::Review::Reviewed {
            reviewer,
            date,
            record,
        } = n.review
        else {
            panic!("{}: not reviewed", n.id);
        };
        assert_eq!(reviewer, "physics reviewer (session model)", "{}", n.id);
        assert!(iso_date(date), "{}: {date}", n.id);
        assert_eq!(record, format!("{REPORT}#{}", n.id), "{}", n.id);
        let heading = format!("## {}", n.id);
        let sections = report
            .lines()
            .filter(|line| line.trim_end() == heading)
            .count();
        assert_eq!(sections, 1, "{}: its section of {REPORT}", n.id);
    }
}

/// Spec 3.3 (R4), section 6 test 5 and ruling 11: the full Start here order is fourteen
/// steps, the torque note first and the clamp note last; while the shaft-clamp section is
/// paused `START_HERE` shows the first thirteen. Each note is a step once; with the six
/// warning notes that is every note shown, in both switch states. (That each step's note
/// explains its equation is `notes.rs`'s `ids_are_unique_and_every_link_resolves`.)
#[test]
fn start_here_is_fourteen_steps_and_thirteen_while_paused() {
    const FULL: [&str; 14] = [
        "torque_from_shear",
        "harmonics",
        "back_iron_factor",
        "pullout_angle",
        "end_effect",
        "calibration",
        "br_temperature",
        "hot_low_cold_high",
        "when_slipping",
        "slip_heating",
        "thermal_time_constant",
        "demagnetization",
        "ferrite_cold_demag",
        "clamp_preload",
    ];
    let ids = |steps: &[(&'static str, &'static str)]| -> Vec<&'static str> {
        steps.iter().map(|&(id, _)| id).collect()
    };
    assert_eq!(ids(notes::START_HERE_ALL), FULL);
    assert_eq!(ids(notes::start_here_for(false)), FULL);
    assert_eq!(ids(notes::start_here_for(true)), FULL[..13].to_vec());
    let warnings: Vec<&str> = magcoupling::engine::warnings::WARNING_RULES
        .iter()
        .map(|r| r.note_id)
        .collect();
    for paused in [false, true] {
        let order = ids(notes::start_here_for(paused));
        let mut reached: BTreeSet<&str> = order.iter().copied().collect();
        assert_eq!(
            reached.len(),
            order.len(),
            "each note once (paused: {paused})"
        );
        reached.extend(warnings.iter().copied());
        let shown: BTreeSet<&str> = notes::NOTES
            .iter()
            .map(|n| n.id)
            .filter(|id| notes::shown_when(id, paused))
            .collect();
        assert_eq!(
            reached, shown,
            "every note shown is a step or a warning's (paused: {paused})"
        );
    }
}

/// The note, help and guide checks this slice wrote red and ignored run for good (spec section
/// 6, tests 1, 7 and 9): in the files that hold them only the review tools stay ignored, the
/// tests that print or write a file for a reviewer instead of checking anything.
#[test]
fn only_the_review_tools_are_ignored() {
    const TOOLS: [&str; 7] = [
        "review_sheet",
        "guidance_strings_for_review",
        "notes_for_review",
        "notes_signoff_sheet",
        "overlay_strings_for_review",
        "help_for_review",
        "notes_before_snapshot",
    ];
    let sources = [
        (
            "src/engine/explain/notes.rs",
            include_str!("../src/engine/explain/notes.rs"),
        ),
        (
            "src/gui/plain_text.rs",
            include_str!("../src/gui/plain_text.rs"),
        ),
        (
            "src/gui/explorer.rs",
            include_str!("../src/gui/explorer.rs"),
        ),
        (
            "src/gui/guidance.rs",
            include_str!("../src/gui/guidance.rs"),
        ),
        (
            "src/gui/dashboard.rs",
            include_str!("../src/gui/dashboard.rs"),
        ),
        ("tests/explain.rs", include_str!("explain.rs")),
    ];
    // Spelled in two pieces, so that this file's own text does not hold the attribute.
    let attribute = concat!("#[", "ignore");
    let mut checks = Vec::new();
    for (file, text) in sources {
        for (at, _) in text.match_indices(attribute) {
            let name = text[at..]
                .split_once("fn ")
                .and_then(|(_, rest)| rest.split('(').next())
                .unwrap_or("?")
                .trim();
            if !TOOLS.contains(&name) {
                checks.push(format!("{file}: {name}"));
            }
        }
    }
    assert!(checks.is_empty(), "ignored checks: {checks:?}");
}

/// The docs state the counts the code has (F36: the note count went stale before): the
/// README's and docs/ai/03-structure.yaml's note count, the README's Start here step count
/// and, while the shaft-clamp section is paused (ruling 11), how many notes show; once the
/// section is back the docs drop that phrase.
#[test]
fn the_docs_count_the_notes_and_the_steps() {
    use magcoupling::engine::clamps;
    const PAUSED_PHRASE: &str = "shown while the clamp section is paused";
    let readme = include_str!("../README.md");
    let structure = include_str!("../../docs/ai/03-structure.yaml");
    let all = notes::NOTES.len();
    let shown = notes::NOTES.iter().filter(|n| notes::shown(n.id)).count();
    let notes_claim = format!("the {all} A4 teaching notes");
    let steps_claim = format!("its {} steps", notes::START_HERE.len());
    let structure_claim = format!("NOTES ({all} notes");
    assert!(
        readme.contains(&notes_claim),
        "README.md should say {notes_claim:?}"
    );
    assert!(
        readme.contains(&steps_claim),
        "README.md should say {steps_claim:?}"
    );
    assert!(
        structure.contains(&structure_claim),
        "docs/ai/03-structure.yaml should say {structure_claim:?}"
    );
    let paused_claim = format!("{all} notes, {shown} {PAUSED_PHRASE}");
    for (doc, text) in [
        ("README.md", readme),
        ("docs/ai/03-structure.yaml", structure),
    ] {
        if clamps::PAUSED {
            assert!(
                text.contains(&paused_claim),
                "{doc} should say {paused_claim:?}"
            );
        } else {
            assert!(
                !text.contains(PAUSED_PHRASE),
                "{doc} still says the clamp section is paused"
            );
        }
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

/// The terms below `path` (transitively) that are neither inputs nor explained: where a
/// drill-down from `path` would stop at a cell-only result. Empty when it reaches inputs.
fn cell_only_below(r: &Registry, path: &str) -> Vec<String> {
    r.upstream(path)
        .into_iter()
        .filter(|up| {
            !matches!(
                r.term_kind(up),
                Some(TermKind::Input { .. } | TermKind::Explained)
            )
        })
        .collect()
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
            let stops = cell_only_below(&r, &path.replace("[]", "[0]"));
            assert!(
                stops.is_empty(),
                "{}: {path} depends on {stops:?}, neither inputs nor explained",
                chain.id
            );
        }
    }
}

#[test]
fn the_demagnetization_block_and_the_governing_limit_drill_down_to_inputs() {
    // Batch 2: both rings' blocks (E20), the block the sheet shows, the cold side and the
    // governing limit reach inputs; the chains that read them flip once their other terms do.
    let r = Registry::build();
    for path in [
        "temperature.summary.governing_limit_C",
        "temperature.demag.h_ref_kA_m",
        "temperature.demag.t_ref_model_C",
        "temperature.demag.calibration_offset_C",
        "temperature.demag.onset_aligned_C",
        "temperature.demag.onset_pullout_C",
        "temperature.demag.onset_skipping_C",
        "temperature.demag.onset_single_ring_C",
        "temperature.demag.magnet_limit_C",
        "temperature.demag.torque_at_limit_Nm",
        "temperature.demag.torque_at_service_Nm",
        "temperature.demag.cold_check",
        "temperature.demag.cold_onset_skipping_C",
        "temperature.duty.hot_day_start_C",
        "temperature.adhesive.cure_C",
    ] {
        assert!(r.equation_for(path).is_some(), "{path} has no record");
        let stops = cell_only_below(&r, path);
        assert!(
            stops.is_empty(),
            "{path} depends on {stops:?}, neither inputs nor explained"
        );
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
        169,
        "decision 31: the chains and the dashboard (report section 7), and plan A-3 decision G1's 10 new geometry callouts"
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
    // The whole order, the paused clamp step's too: it must still resolve when the section
    // comes back.
    for (id, path) in notes::START_HERE_ALL {
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
// Slice-2 notes: what the note text claims about the engine
// ---------------------------------------------------------------------------------------

/// The pull-out of `inputs` after `edit`.
fn pullout_after(inputs: &DesignInputs, edit: impl FnOnce(&mut DesignInputs)) -> f64 {
    let mut edited = inputs.clone();
    edit(&mut edited);
    compute_all(&edited).model.pullout_Nm
}

/// The probe design named `name` (spec 3.2, `probes::probe_designs`).
fn probe_inputs(name: &str) -> DesignInputs {
    probes::probe_designs()
        .into_iter()
        .find(|(design, _)| *design == name)
        .map(|(_, inputs)| inputs)
        .unwrap_or_else(|| panic!("no probe design {name:?}"))
}

/// `torque_from_shear`'s claims about the engine. Its length entry: on the example design a
/// millimetre of magnet length adds more pull-out than a millimetre of ring radius. Its R_g²
/// watch-out: a larger ring at the same magnet count can lower the pull-out (the 8-magnet probe
/// design, 1 mm larger). At all five probe designs (ruling 10 c): two fewer magnets keep the gap
/// radius R_g and grow the cup (the magnet-count sentence); one slider step of a wider gap lowers
/// σ, and magnetic steel behind the magnets (4140) gives a higher σ than 304 on the same design
/// (the σ sentence); the 2D torque rises with each two more magnets while the blocks still fit
/// (the magnet-count entry).
#[test]
fn torque_from_shear_claims_hold_on_the_engine() {
    let example = DesignInputs::default();
    let results = compute_all(&example);
    let (base, length) = (results.model.pullout_Nm, results.model.inner_length_mm);
    let per_mm_length = pullout_after(&example, |i| {
        i.coupling.magnets.axial_length_mm = Some(length + 1.0);
    }) - base;
    let per_mm_radius = pullout_after(&example, |i| i.coupling.inner_back_apothem_mm += 1.0) - base;
    assert!(
        per_mm_radius > 0.0 && per_mm_length > per_mm_radius,
        "per mm: length {per_mm_length}, ring radius {per_mm_radius}"
    );
    let eight = probe_inputs("8 magnets per ring");
    let before = compute_all(&eight).model.pullout_Nm;
    let after = pullout_after(&eight, |i| i.coupling.inner_back_apothem_mm += 1.0);
    assert!(
        after < before,
        "8 magnets, ring radius +1 mm: {before} -> {after}"
    );

    let back_iron = |id: &str| {
        BACK_IRON_CHOICES
            .iter()
            .find(|&&(_, choice)| choice == id)
            .map(|&(code, _)| code)
            .unwrap_or_else(|| panic!("{id} is a back-iron choice"))
    };
    let (steel, stainless) = (
        back_iron("4140_annealed"),
        back_iron(probes::PROBE_BACK_IRON),
    );
    let most_magnets = slider("coupling.npole").max as i64;
    let mut rises = 0;
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design).model;
        // Two fewer magnets keep the gap radius and grow the cup.
        let mut fewer = design.clone();
        fewer.coupling.npole -= 2;
        let fewer = compute_all(&fewer).model;
        assert!(
            close(fewer.gap_radius_mm, base.gap_radius_mm),
            "{name}: R_g {} -> {}",
            base.gap_radius_mm,
            fewer.gap_radius_mm
        );
        assert!(
            fewer.cup_od_mm > base.cup_od_mm,
            "{name}: cup {} -> {}",
            base.cup_od_mm,
            fewer.cup_od_mm
        );
        // A wider gap lowers σ.
        let wider = compute_all(&stepped(&design, "metal.face_gap_mm", Step::Raise)).model;
        assert!(
            wider.tau_Pa < base.tau_Pa,
            "{name}: σ {} -> {} one gap step wider",
            base.tau_Pa,
            wider.tau_Pa
        );
        // Magnetic steel behind the magnets gives a higher σ than 304 on the same design.
        let tau_with = |code: i64| {
            let mut d = design.clone();
            d.materials.parts.back_iron = code;
            d.coupling.backiron = 1;
            compute_all(&d).model.tau_Pa
        };
        let (with_steel, with_304) = (tau_with(steel), tau_with(stainless));
        assert!(
            with_steel > with_304,
            "{name}: σ {with_steel} with 4140 against {with_304} with 304"
        );
        // The 2D torque rises with each two more magnets while the blocks still fit.
        let mut previous = base.torque_2d_Nm;
        let mut count = design.coupling.npole + 2;
        while count <= most_magnets {
            let mut more = design.clone();
            more.coupling.npole = count;
            let model = compute_all(&more).model;
            if !blocks_fit(&more.coupling, &model) {
                break;
            }
            assert!(
                model.torque_2d_Nm > previous,
                "{name}: T_2D {previous} -> {} at {count} magnets",
                model.torque_2d_Nm
            );
            previous = model.torque_2d_Nm;
            rises += 1;
            count += 2;
        }
    }
    assert!(rises >= 1, "the magnet-count pin compares no fitting count");
}

/// `harmonics` states two things about the engine. Its harmonic-set entry, "keep at least 1, 3,
/// 5": at the five probe designs summing past the fifth moves the pull-out under 2 % (designs
/// off the probes where it moves more: `harmonics_claims_hold_off_the_probe_designs`). Its
/// watch-out: at low fill the third harmonic can change the pull-out by more than a tenth and
/// move its angle (the 20 mm ring radius design, both fills under one half).
#[test]
fn harmonics_claims_hold_at_the_probe_designs() {
    let with_set = |inputs: &DesignInputs, set: i64| {
        let mut edited = inputs.clone();
        edited.coupling.max_harmonic = set;
        compute_all(&edited).model
    };
    for (design, inputs) in probes::probe_designs() {
        let five = with_set(&inputs, 5).pullout_Nm;
        for set in [7, 9, 11] {
            let more = with_set(&inputs, set).pullout_Nm;
            assert!(
                (more / five - 1.0).abs() < 0.02,
                "{design}, harmonics to {set}: {more} against {five}"
            );
        }
    }
    let low_fill = probe_inputs("20 mm ring radius");
    let (first, third) = (with_set(&low_fill, 1), with_set(&low_fill, 3));
    assert!(
        third.fill_inner < 0.5 && third.fill_outer < 0.5,
        "{} {}",
        third.fill_inner,
        third.fill_outer
    );
    assert!(
        (third.pullout_Nm / first.pullout_Nm - 1.0).abs() > 0.1,
        "{} against {}",
        third.pullout_Nm,
        first.pullout_Nm
    );
    assert!(
        (third.pullout_angle_rad - FRAC_PI_2).abs() > 0.1,
        "{}",
        third.pullout_angle_rad
    );
}

/// `harmonics` off the five probe designs, on designs the Size tab reaches from the example
/// design with its library parts (T12 task review, I1 and I2; its text re-review 1, MF-1).
/// - The harmonic-set entry. At the fewest magnets per ring the slider allows, on the probes'
///   20 mm ring, each block is narrower than the space beside it (both fills under one half:
///   narrow, widely spaced blocks), and summing past the fifth harmonic moves the pull-out by
///   more than a twentieth. So the entry names that case and states no bound on the change.
/// - The σ line. With the harmonic set at 1, σ equals the fundamental's peak A_1, at the five
///   probe designs and on a 17.5 mm ring, so A_1 is what the fundamental alone gives. Beside
///   the summed σ, A_1 shows the set's real effect: σ − A_1 has the sign of the 2D torque's
///   change from the set 1 to the design's set. The fundamental's stress at the summed curve's
///   pull-out angle, σ_1, is no such baseline: it lies below σ on the example design but above
///   it on a 17.5 mm ring, and on the 20 mm ring radius design σ_1 < σ < A_1, so σ_1 beside σ
///   would show the harmonics adding stress where they take it away. So the sentence that draws
///   σ draws A_1 as "the fundamental alone" and words neither as a share of the other.
#[test]
fn harmonics_claims_hold_off_the_probe_designs() {
    let example = DesignInputs::default();
    let harmonics = notes::note("harmonics").expect("the harmonics note");

    let fewest = slider("coupling.npole").min as i64;
    let mut few = example.clone();
    few.coupling.npole = fewest;
    let spaced = with_ring_radius(&few, probes::PROBE_RING_RADIUS_MM)
        .expect("the Size tab reaches the probes' ring radius at the fewest magnets");
    let model = compute_all(&spaced).model;
    assert!(blocks_fit(&spaced.coupling, &model), "{fewest} magnets fit");
    assert!(
        model.fill_inner < 0.5 && model.fill_outer < 0.5,
        "narrow, widely spaced: fills {} {}",
        model.fill_inner,
        model.fill_outer
    );
    let to_fifth = pullout_after(&spaced, |i| i.coupling.max_harmonic = 5);
    let to_eleventh = pullout_after(&spaced, |i| i.coupling.max_harmonic = 11);
    assert!(
        (to_eleventh / to_fifth - 1.0).abs() > 0.05,
        "{fewest} magnets: {to_fifth} -> {to_eleventh}"
    );
    let entry = harmonics
        .change
        .iter()
        .find(|c| c.input == "coupling.max_harmonic")
        .expect("the harmonic-set entry");
    assert!(
        entry.text.contains("narrow, widely spaced blocks") && !entry.text.contains('%'),
        "names the case, bounds no change: {:?}",
        entry.text
    );

    let split = |inputs: &DesignInputs| {
        let m = compute_all(inputs).model;
        (m.tau1_Pa, m.tau_Pa)
    };
    let (first, summed) = split(&example);
    assert!(first < summed, "example design: {first} against {summed}");
    let larger_ring = with_ring_radius(&example, 17.5).expect("the Size tab reaches 17.5 mm");
    let (first, summed) = split(&larger_ring);
    assert!(first > summed, "17.5 mm ring: {first} against {summed}");

    let mut designs = probes::probe_designs();
    designs.push(("17.5 mm ring", larger_ring));
    for (design, inputs) in &designs {
        let model = compute_all(inputs).model;
        let mut fundamental_only = inputs.clone();
        fundamental_only.coupling.max_harmonic = 1;
        let alone = compute_all(&fundamental_only).model;
        assert!(
            (alone.tau_Pa / model.amp1_Pa - 1.0).abs() < 1e-12,
            "{design}: the set 1 gives σ = {} against A_1 = {}",
            alone.tau_Pa,
            model.amp1_Pa
        );
        assert_eq!(
            (model.tau_Pa - model.amp1_Pa).signum(),
            (model.torque_2d_Nm - alone.torque_2d_Nm).signum(),
            "{design}: σ {} against A_1 {}, T_2D {} against {} at the set 1",
            model.tau_Pa,
            model.amp1_Pa,
            model.torque_2d_Nm,
            alone.torque_2d_Nm
        );
    }
    let low_fill = compute_all(&probe_inputs("20 mm ring radius")).model;
    assert!(
        low_fill.tau1_Pa < low_fill.tau_Pa && low_fill.tau_Pa < low_fill.amp1_Pa,
        "20 mm ring radius: σ_1 {} < σ {} < A_1 {}",
        low_fill.tau1_Pa,
        low_fill.tau_Pa,
        low_fill.amp1_Pa
    );

    let line = harmonics
        .sentences
        .iter()
        .find(|s| s.contains("{model.amp1_Pa}") && s.contains("{model.tau_Pa}"))
        .expect("the sentence that draws A_1 and σ");
    assert!(
        line.contains("the fundamental alone"),
        "names A_1 as the fundamental alone: {line:?}"
    );
    let (alone_at, summed_at) = (
        line.find("{model.amp1_Pa}").expect("A_1"),
        line.find("{model.tau_Pa}").expect("σ"),
    );
    let between = &line[alone_at.min(summed_at)..alone_at.max(summed_at)];
    for share in [" of ", "total", "share", "part"] {
        assert!(
            !between.contains(share),
            "{share:?} between A_1 and σ: {line:?}"
        );
    }
}

/// `harmonics`' outer-part entry, on the example design. B862, a wider outer block of equal
/// thickness, fits and raises the outer fill and the pull-out; the cold-high torque the shaft
/// joint and the gearbox carry rises, and its grade lowers the magnet limit. B861, as wide but
/// thinner, also fits yet lowers the pull-out, so the entry needs "equally thick".
#[test]
fn harmonics_outer_part_entry_holds_on_the_example_design() {
    let example = DesignInputs::default();
    let base = compute_all(&example);
    let picked = |part: &str| {
        let mut inputs = example.clone();
        inputs.coupling.magnets.part_outer = part.to_owned();
        let results = compute_all(&inputs);
        assert!(blocks_fit(&inputs.coupling, &results.model), "{part} fits");
        assert!(
            results.model.outer_width_mm > base.model.outer_width_mm,
            "{part} is wider"
        );
        results
    };
    let (equal, thinner) = (picked("B862"), picked("B861"));
    assert_eq!(
        equal.model.outer_thickness_mm,
        base.model.outer_thickness_mm
    );
    assert!(thinner.model.outer_thickness_mm < base.model.outer_thickness_mm);
    for (what, after, before) in [
        ("outer fill", equal.model.fill_outer, base.model.fill_outer),
        ("pull-out", equal.model.pullout_Nm, base.model.pullout_Nm),
        (
            "cold-high torque",
            equal.metal.torque_cold_high_Nm,
            base.metal.torque_cold_high_Nm,
        ),
    ] {
        assert!(
            after > before,
            "B862 raises the {what}: {before} -> {after}"
        );
    }
    let limit = |r: &DesignResults| r.temperature.summary.magnet_limit_C;
    assert!(
        limit(&equal) < limit(&base),
        "B862's grade lowers the magnet limit: {} -> {}",
        limit(&base),
        limit(&equal)
    );
    assert!(
        thinner.model.pullout_Nm < base.model.pullout_Nm,
        "B861 lowers the pull-out: {} -> {}",
        base.model.pullout_Nm,
        thinner.model.pullout_Nm
    );
}

/// `back_iron_factor`'s two choice entries: among the back-iron materials only 304 and
/// aluminium switch the torque model to free space, and with no back iron it is free space
/// whatever material is picked.
#[test]
fn back_iron_factor_claims_hold_on_the_engine() {
    use magcoupling::engine::material_library::BACK_IRON_CHOICES;
    let circuit = |code: i64, backiron: i64| {
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.back_iron = code;
        inputs.coupling.backiron = backiron;
        compute_all(&inputs).materials.circuit_backiron
    };
    let free_space: Vec<&str> = BACK_IRON_CHOICES
        .iter()
        .filter(|&&(code, _)| circuit(code, 1) == 0)
        .map(|&(_, id)| id)
        .collect();
    assert_eq!(free_space, ["304_annealed", "6061_T6"]);
    for &(code, id) in &BACK_IRON_CHOICES {
        assert_eq!(circuit(code, 0), 0, "{id} with no back iron");
    }
}

/// `pullout_angle`'s grade entry. On the example design a same-size, stronger-grade block on
/// the inner ring (B842-N52 for B842SH: the same width, thickness and length) raises the
/// pull-out in proportion to that ring's Br, raises the cold-high torque the shaft joint and
/// the gearbox carry, and is rated for less heat. "Same-size" is needed: M5026, a stronger
/// grade that fits but is thinner, lowers the pull-out and the cold-high torque. "Unless
/// calibration changes": on the 304 back iron design the same-size pick ends the bench factor
/// (the default factor applies), so the pull-out moves by the Br ratio times the factor ratio.
#[test]
fn pullout_angle_claims_hold_on_the_engine() {
    use magcoupling::engine::meta::NumOrText;
    let with_inner = |inputs: &DesignInputs, part: &str| {
        let mut picked = inputs.clone();
        picked.coupling.magnets.part_inner = part.to_owned();
        let results = compute_all(&picked);
        (picked, results)
    };
    let stronger_inner =
        |inputs: &DesignInputs| (compute_all(inputs), with_inner(inputs, "B842-N52").1);
    let example = DesignInputs::default();
    assert_eq!(example.coupling.magnets.part_inner, "B842SH");
    let (base, n52) = stronger_inner(&example);
    for (what, picked_mm, base_mm) in [
        ("width", n52.model.inner_width_mm, base.model.inner_width_mm),
        (
            "thickness",
            n52.model.inner_thickness_mm,
            base.model.inner_thickness_mm,
        ),
        (
            "length",
            n52.model.inner_length_mm,
            base.model.inner_length_mm,
        ),
    ] {
        assert_eq!(picked_mm, base_mm, "B842-N52 has B842SH's {what}");
    }
    assert!(n52.model.inner_br_T > base.model.inner_br_T);
    assert_eq!(n52.model.f_cal, base.model.f_cal);
    let (torque, br) = (
        n52.model.pullout_Nm / base.model.pullout_Nm,
        n52.model.inner_br_T / base.model.inner_br_T,
    );
    assert!((torque / br - 1.0).abs() < 1e-12, "{torque} against {br}");
    assert!(
        n52.metal.torque_cold_high_Nm > base.metal.torque_cold_high_Nm,
        "cold-high: {} -> {}",
        base.metal.torque_cold_high_Nm,
        n52.metal.torque_cold_high_Nm
    );
    match (n52.model.inner_tmax_C, base.model.inner_tmax_C) {
        (NumOrText::Num(rated), NumOrText::Num(base_rated)) => {
            assert!(rated < base_rated, "{rated} against {base_rated}")
        }
        other => panic!("{other:?}"),
    }

    let (arc_inputs, arc) = with_inner(&example, "M5026");
    assert!(blocks_fit(&arc_inputs.coupling, &arc.model), "M5026 fits");
    assert!(
        arc.model.inner_br_T > base.model.inner_br_T,
        "M5026 is a stronger grade: {} -> {}",
        base.model.inner_br_T,
        arc.model.inner_br_T
    );
    assert!(
        arc.model.inner_thickness_mm < base.model.inner_thickness_mm,
        "M5026 is thinner: {} -> {}",
        base.model.inner_thickness_mm,
        arc.model.inner_thickness_mm
    );
    for (what, after, before) in [
        ("pull-out", arc.model.pullout_Nm, base.model.pullout_Nm),
        (
            "cold-high torque",
            arc.metal.torque_cold_high_Nm,
            base.metal.torque_cold_high_Nm,
        ),
    ] {
        assert!(
            after < before,
            "M5026 lowers the {what}: {before} -> {after}"
        );
    }

    let bench = probe_inputs("304 back iron");
    let (base, n52) = stronger_inner(&bench);
    let (bench_factor, default_factor) = (
        base.calibration.f_cal_updated,
        bench.calibration.f_cal_original,
    );
    assert_ne!(bench_factor, default_factor);
    assert_eq!(base.model.f_cal, bench_factor, "B842SH: the bench factor");
    assert_eq!(
        n52.model.f_cal, default_factor,
        "B842-N52: the default factor"
    );
    let (torque, br, factor) = (
        n52.model.pullout_Nm / base.model.pullout_Nm,
        n52.model.inner_br_T / base.model.inner_br_T,
        n52.model.f_cal / base.model.f_cal,
    );
    assert!(
        (torque / (br * factor) - 1.0).abs() < 1e-12,
        "304: {torque} against {br} x {factor}"
    );
}

// Slice 2 note claims: the end-effect and calibration notes (plan T13). Each lever moves by
// one slider step (`probes::step_input`; a blank axial length starts from the length in use).

/// `design` with the input at `path` one slider step `step`'s way (`probes::step_input`): the
/// note levers the claims tests move are free at every probe design.
fn stepped(design: &DesignInputs, path: &str, step: Step) -> DesignInputs {
    probes::step_input(design, path, step)
        .unwrap_or_else(|why| panic!("{path} is free at every probe design: {why:?}"))
}

/// Whether `a` and `b` agree to a part in 10¹² (an engine relation the note states, computed
/// two ways).
fn close(a: f64, b: f64) -> bool {
    (a / b - 1.0).abs() < 1e-12
}

/// The slider of the input at `path`, from its metadata.
fn slider(path: &str) -> SliderRange {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|row| row.path == path)
        .and_then(|row| row.meta.range)
        .unwrap_or_else(|| panic!("{path} has no slider"))
}

/// Which rings a grade pick puts on custom dimensions ([`on_grade`]).
#[derive(Clone, Copy, Debug)]
enum Rings {
    Inner,
    Outer,
    Both,
}

/// `design` with `rings` on custom dimensions and the grade `grade`, as the part and grade
/// picks set them; a ring left out keeps its part.
fn on_grade(design: &DesignInputs, rings: Rings, grade: &str) -> DesignInputs {
    let mut d = design.clone();
    let m = &mut d.coupling.magnets;
    if matches!(rings, Rings::Inner | Rings::Both) {
        m.part_inner.clear();
        m.grade_inner = grade.to_owned();
    }
    if matches!(rings, Rings::Outer | Rings::Both) {
        m.part_outer.clear();
        m.grade_outer = grade.to_owned();
    }
    d
}

/// `end_effect`'s two entries at the five probe designs: longer magnets lose a smaller share
/// at the ends; a larger end coefficient lowers every torque but leaves the bench prototype's
/// own end factor. The trade-offs the entries name hold too: cold-high, and with it the
/// shaft-joint load (cold-high itself) and the gearbox load (cold-high through the gear ratio
/// and efficiency at its input).
#[test]
fn the_end_effect_note_s_levers_move_the_end_factor_as_it_says() {
    let note = notes::note("end_effect").expect("the end-effect note");
    let levers: Vec<&str> = note.change.iter().map(|c| c.input).collect();
    // Magnet length first (lengthen first), then the assumption that only evidence should move.
    assert_eq!(
        levers,
        ["coupling.magnets.axial_length_mm", "coupling.c_end"]
    );
    assert!(note.change.iter().all(|c| matches!(c.step, Step::Raise)));
    assert_eq!(note.change[0].aim, Some(("model.f_end", Sense::Up)));
    assert_eq!(note.change[1].aim, Some(("model.f_end", Sense::Down)));
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        // Entry 1: a smaller share lost at the ends, more torque, and the trade-off it names:
        // cold-high rises, and with it the shaft-joint load (cold-high itself) and the gearbox
        // load (cold-high through the gear ratio and efficiency at its input).
        let r = compute_all(&stepped(
            &design,
            "coupling.magnets.axial_length_mm",
            Step::Raise,
        ));
        assert!(r.model.f_end > base.model.f_end, "{name}: f_end");
        assert!(
            r.model.pullout_Nm > base.model.pullout_Nm,
            "{name}: pull-out"
        );
        assert!(
            r.metal.torque_cold_high_Nm > base.metal.torque_cold_high_Nm,
            "{name}: cold-high"
        );
        assert!(
            r.metal.cold_high_input_Nm > base.metal.cold_high_input_Nm,
            "{name}: gearbox input"
        );
        // Entry 2: every torque falls, cold-high and the shaft-joint and gearbox loads
        // included; the bench prototype's own end factor does not move.
        let r = compute_all(&stepped(&design, "coupling.c_end", Step::Raise));
        assert!(r.model.pullout_Nm < base.model.pullout_Nm, "{name}");
        assert!(
            r.metal.torque_hot_low_Nm < base.metal.torque_hot_low_Nm,
            "{name}: hot-low"
        );
        assert!(
            r.metal.torque_cold_high_Nm < base.metal.torque_cold_high_Nm,
            "{name}"
        );
        assert!(
            r.metal.cold_high_input_Nm < base.metal.cold_high_input_Nm,
            "{name}: gearbox input"
        );
        assert_eq!(r.calibration.f_end, base.calibration.f_end, "{name}");
    }
}

/// `end_effect`'s length sentences and its second watch-out at the five probe designs: every
/// extra millimetre of overlap adds the same torque, and at or below c_end·τ_p the end-effect
/// check flags the overlap (no prediction) while just above it reads OK.
#[test]
fn the_end_effect_note_s_length_claims_hold_at_every_probe_design() {
    use magcoupling::engine::model::END_EFFECT_OUT_OF_RANGE;
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        let at = |length: f64| {
            let mut d = design.clone();
            d.coupling.magnets.axial_length_mm = Some(length);
            compute_all(&d)
        };
        // Sentence 4: every extra millimetre of overlap adds the same torque.
        let l = base.model.active_length_mm;
        let t: Vec<f64> = [l, l + 1.0, l + 2.0]
            .iter()
            .map(|&x| at(x).model.pullout_Nm)
            .collect();
        assert!(t[1] > t[0], "{name}: {t:?}");
        assert!(
            ((t[2] - t[1]) - (t[1] - t[0])).abs() <= 1e-9 * (t[1] - t[0]),
            "{name}: {t:?}"
        );
        // Watch-out 2: at or below c_end·τ_p there is no prediction and the check flags it;
        // just above it, a prediction.
        let lost = design.coupling.c_end * base.model.pole_pitch_mm;
        assert_eq!(
            at(0.999 * lost).model.end_effect_check,
            END_EFFECT_OUT_OF_RANGE,
            "{name}"
        );
        assert_eq!(at(1.001 * lost).model.end_effect_check, "OK", "{name}");
    }
}

/// `end_effect`'s second watch-out says that at short overlap f_end is least reliable and can
/// read too high or too low, draws no pole-pitch boundary and names no design (T13 text
/// re-reviews 1 and 2, MF-1 each). Against 3D, no single direction holds across designs the
/// inputs reach with a prediction (the check reads OK):
/// - low: the example design at the magnet length slider's floor (2.0 mm) reads −19.9 % without
///   steel (−20.7 % on rings) and −1.3 to −4.0 % with its own steel behind both rings;
/// - high: the example design with that steel reads +12.8 to +15.6 % at 2.5 mm, and the narrower
///   B822 block on both rings of the example ring reads +3.2 to +3.8 % with 10 per ring at 2.5 mm
///   and +8.7 to +9.3 % with 12 at the floor.
///
/// The ranges span the reviewers' methods: planar or ring 3D, with ideal steel flush with the
/// magnets or as infinite plates. So a one-direction claim ("reads low", even scoped to the
/// example design) is false somewhere, and false in the optimistic direction.
///
/// No fraction of a pole pitch bounds the error either. In all ten circuits re-review 2 checked,
/// the largest error lies at a short overlap, 0.16 to 0.42 τ_p. On the example design with steel
/// it lies above τ_p/3 (+21.5 % at 3.66 mm), so "below a third of a pole pitch" would put the
/// largest optimistic error on its safe side.
///
/// The engine has no 3D model: the figures are the reviewers' (the record's `## end_effect`).
/// This test pins the engine side, that each of those designs has a short overlap and still gets
/// a prediction. "Short" is the test's own measure, below a third of a pole pitch; the note draws
/// no such line.
#[test]
fn the_end_effect_note_s_short_overlap_watch_out_names_both_directions() {
    let example = DesignInputs::default();
    let example_width = compute_all(&example).model.inner_width_mm;
    let floor = slider("coupling.magnets.axial_length_mm").min;
    let at = |design: &DesignInputs, length: f64| {
        let mut d = design.clone();
        d.coupling.magnets.axial_length_mm = Some(length);
        d
    };
    // A short overlap, with a prediction: the blocks fit, the overlap is below a third of a pole
    // pitch (this test's measure of short, not the note's), the end factor is positive and the
    // end-effect check reads OK.
    let predicted_below_a_third = |design: &DesignInputs, name: &str| {
        let model = compute_all(design).model;
        assert!(
            blocks_fit(&design.coupling, &model),
            "{name}: the blocks fit"
        );
        assert!(
            model.active_length_mm < model.pole_pitch_mm / 3.0,
            "{name}: overlap {} mm against a third of the pole pitch {} mm",
            model.active_length_mm,
            model.pole_pitch_mm
        );
        assert!(model.f_end > 0.0, "{name}: f_end {}", model.f_end);
        assert_eq!(model.end_effect_check, "OK", "{name}: a prediction");
        model
    };
    // The low side: the example design itself at the slider's floor.
    predicted_below_a_third(
        &at(&example, floor),
        &format!("the example design at {floor} mm"),
    );
    // The high side: the narrower B822 block on both rings of the example ring.
    for (per_ring, length) in [(10, 2.5), (12, floor)] {
        let mut narrow = at(&example, length);
        narrow.coupling.npole = per_ring;
        narrow.coupling.magnets.part_inner = "B822".to_owned();
        narrow.coupling.magnets.part_outer = "B822".to_owned();
        let name = format!("{per_ring} B822 per ring at {length} mm");
        let model = predicted_below_a_third(&narrow, &name);
        assert!(
            model.inner_width_mm < example_width,
            "{name}: block width {} mm against the example's {example_width} mm",
            model.inner_width_mm
        );
    }
    let note = notes::note("end_effect").expect("the end-effect note");
    let short = note
        .watch_out
        .iter()
        .find(|w| w.to_lowercase().contains("short overlap") || w.contains("c_end·τ_p"))
        .expect("the short-overlap watch-out");
    let (claim, _) = short
        .split_once(';')
        .expect("the claim, then the no-prediction clause");
    // Whole words, so "below" does not count as "low".
    let says = |word: &str| {
        claim
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| w.eq_ignore_ascii_case(word))
    };
    assert!(
        says("high") && says("low"),
        "names both directions: {short:?}"
    );
    assert!(
        !claim.to_lowercase().contains("on the example design"),
        "scopes the claim to no one design: {short:?}"
    );
    assert!(
        !claim.to_lowercase().contains("pole pitch"),
        "draws no pole-pitch boundary: {short:?}"
    );
}

/// `calibration`'s three entries and its rule at the five probe designs. A prototype match
/// (the rule's own terms: no steel behind the magnets, the prototype's magnets per ring, B842SH
/// on both rings) ignores the assumed factor; elsewhere every torque scales with it. The two
/// bench entries are entered from the test, so they claim no aim: the measured factor rises
/// with a larger measurement or a warmer test, and only a match uses it.
#[test]
fn the_calibration_note_s_levers_and_rule_hold_at_every_probe_design() {
    let note = notes::note("calibration").expect("the calibration note");
    let levers: Vec<&str> = note.change.iter().map(|c| c.input).collect();
    assert_eq!(
        levers,
        [
            "calibration.f_cal_original",
            "calibration.measured_torque_Nm",
            "calibration.test_temp_C",
        ]
    );
    assert!(matches!(note.change[0].step, Step::Lower));
    assert!(
        note.change[1..]
            .iter()
            .all(|c| matches!(c.step, Step::Enter))
    );
    // A prototype match ignores the assumed factor, and the bench values are entered, not
    // stepped, so no entry claims a direction a test could check at every design: each is
    // checked here, design by design.
    assert!(note.change.iter().all(|c| c.aim.is_none()));
    // A prototype match by the rule's own terms (sentence 3).
    let is_match = |d: &DesignInputs, r: &DesignResults| {
        r.materials.circuit_backiron == 0
            && d.coupling.npole as f64 == r.calibration.poles_per_ring
            && d.coupling.magnets.part_inner == "B842SH"
            && d.coupling.magnets.part_outer == "B842SH"
    };
    let (mut matching, mut other) = (0, 0);
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        let matched = is_match(&design, &base);
        // Entry 1: every torque outside a match falls in proportion, cold-high included; a
        // match ignores it.
        let lower = stepped(&design, "calibration.f_cal_original", Step::Lower);
        let r = compute_all(&lower);
        if matched {
            matching += 1;
            assert_eq!(base.model.f_cal, base.calibration.f_cal_updated, "{name}");
            assert!(
                (r.model.pullout_Nm / base.model.pullout_Nm - 1.0).abs() < 1e-12,
                "{name}: a match ignores the assumed factor"
            );
        } else {
            other += 1;
            assert_eq!(
                base.model.f_cal, design.calibration.f_cal_original,
                "{name}"
            );
            let ratio = lower.calibration.f_cal_original / design.calibration.f_cal_original;
            assert!(ratio < 1.0, "{name}: {ratio}");
            for (before, after) in [
                (base.model.pullout_Nm, r.model.pullout_Nm),
                (base.metal.torque_hot_low_Nm, r.metal.torque_hot_low_Nm),
                (base.metal.torque_cold_high_Nm, r.metal.torque_cold_high_Nm),
            ] {
                assert!(
                    (after / before - ratio).abs() < 1e-12,
                    "{name}: {before} -> {after}"
                );
            }
        }
        // Entry 2: the measured factor rises with the measurement, and only a match uses it.
        let r = compute_all(&stepped(
            &design,
            "calibration.measured_torque_Nm",
            Step::Raise,
        ));
        assert!(
            r.calibration.f_cal_updated > base.calibration.f_cal_updated,
            "{name}"
        );
        assert_eq!(
            r.model.f_cal != base.model.f_cal,
            matched,
            "{name}: only a match uses it"
        );
        // Entry 3: a warmer test weakens the modelled prototype, so the measured factor rises.
        let r = compute_all(&stepped(&design, "calibration.test_temp_C", Step::Raise));
        assert!(
            r.calibration.model_torque_Nm < base.calibration.model_torque_Nm,
            "{name}: a weaker modelled prototype"
        );
        assert!(
            r.calibration.f_cal_updated > base.calibration.f_cal_updated,
            "{name}"
        );
    }
    assert!(
        matching >= 1 && other >= 1,
        "the probe designs hold a match and others"
    );
}

/// `calibration`'s match sentences and its watch-out, on the 304 back iron probe design (the
/// one prototype match): the rule ignores ring radius, gap and magnet length; another magnet
/// count leaves the match for the assumed factor; and "matching" the prototype inputs to that
/// count makes a match again, with a factor far from the bench one.
#[test]
fn the_calibration_note_s_match_claims_hold() {
    let matched = probe_inputs("304 back iron");
    let bench = compute_all(&matched);
    assert_eq!(bench.model.f_cal, bench.calibration.f_cal_updated);
    // Sentence 4: the rule ignores ring radius, gap and magnet length.
    let mut resized = matched.clone();
    resized.coupling.inner_back_apothem_mm = 20.0;
    resized.metal.face_gap_mm = 2.0;
    resized.coupling.magnets.axial_length_mm = Some(20.0);
    assert_eq!(
        compute_all(&resized).model.f_cal,
        bench.calibration.f_cal_updated
    );
    // Sentence 5: leaving the match (another magnet count) switches to the assumed factor.
    let mut left = matched.clone();
    left.coupling.npole = 8;
    assert_eq!(
        compute_all(&left).model.f_cal,
        left.calibration.f_cal_original
    );
    // The watch-out: "matching" the prototype inputs to the new count makes a match again,
    // with a factor far from the bench one: the prototype model at another count, not a
    // measurement.
    let mut faked = left.clone();
    faked.calibration.total_magnets = 2 * faked.coupling.npole;
    let fake = compute_all(&faked);
    assert_eq!(fake.model.f_cal, fake.calibration.f_cal_updated);
    assert!(
        fake.model.f_cal / bench.model.f_cal - 1.0 > 0.4,
        "{} against the bench {}",
        fake.model.f_cal,
        bench.model.f_cal
    );
}

// Slice 2 note claims: the torque-band and temperature notes (plan T14). Each lever moves by
// one slider step ([`stepped`]).

/// `hot_low_cold_high`'s four entries at the five probe designs: each moves hot-low or
/// cold-high the way it says, and the edge it says stays stays.
#[test]
fn the_torque_band_note_s_levers_move_hot_low_and_cold_high_as_it_says() {
    let note = notes::note("hot_low_cold_high").expect("the torque-band note");
    let levers: Vec<&str> = note.change.iter().map(|c| c.input).collect();
    assert_eq!(
        levers,
        [
            "coupling.magnets.axial_length_mm",
            "metal.variation",
            "coupling.op_temp_C",
            "metal.min_temp_C",
        ]
    );
    assert!(matches!(note.change[0].step, Step::Raise));
    assert!(
        note.change[1..]
            .iter()
            .all(|c| matches!(c.step, Step::Lower))
    );
    let hot_low_up = Some(("metal.torque_hot_low_Nm", Sense::Up));
    assert_eq!(note.change[0].aim, hot_low_up);
    assert_eq!(note.change[1].aim, hot_low_up);
    assert_eq!(note.change[2].aim, hot_low_up);
    assert_eq!(
        note.change[3].aim,
        Some(("metal.torque_cold_high_Nm", Sense::Up))
    );
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        let (hot, cold) = (base.metal.torque_hot_low_Nm, base.metal.torque_cold_high_Nm);
        // Entry 1: longer magnets raise both edges and, with cold-high, the shaft-joint and
        // gearbox loads (the trade-off it names; the gearbox input sees cold-high through the
        // ratio and the efficiency).
        let r = compute_all(&stepped(
            &design,
            "coupling.magnets.axial_length_mm",
            Step::Raise,
        ));
        assert!(
            r.metal.torque_hot_low_Nm > hot && r.metal.torque_cold_high_Nm > cold,
            "{name}: length"
        );
        assert!(
            r.metal.cold_high_input_Nm > base.metal.cold_high_input_Nm,
            "{name}: gearbox input"
        );
        // Entry 2: a smaller allowance narrows the band both ways.
        let r = compute_all(&stepped(&design, "metal.variation", Step::Lower));
        assert!(
            r.metal.torque_hot_low_Nm > hot && r.metal.torque_cold_high_Nm < cold,
            "{name}: allowance"
        );
        // Entry 3: a cooler operating temperature raises hot-low; cold-high stays.
        let r = compute_all(&stepped(&design, "coupling.op_temp_C", Step::Lower));
        assert!(r.metal.torque_hot_low_Nm > hot, "{name}: operating");
        assert!(
            close(r.metal.torque_cold_high_Nm, cold),
            "{name}: cold-high stays"
        );
        // Entry 4: a colder minimum raises cold-high, and with it the shaft-joint and gearbox
        // loads; hot-low stays.
        let r = compute_all(&stepped(&design, "metal.min_temp_C", Step::Lower));
        assert!(r.metal.torque_cold_high_Nm > cold, "{name}: minimum");
        assert!(
            r.metal.cold_high_input_Nm > base.metal.cold_high_input_Nm,
            "{name}: minimum, gearbox input"
        );
        assert_eq!(r.metal.torque_hot_low_Nm, hot, "{name}: hot-low stays");
    }
}

/// `hot_low_cold_high`'s sentences at the five probe designs: what each edge of the band is,
/// what carries cold-high, and the 20 °C pull-out at which hot-low passes.
#[test]
fn the_torque_band_note_s_sentences_hold_at_every_probe_design() {
    // Sentence 0 answers the rows that link the note (spec 3.1: the hot-low, hot-minimum and
    // cold-high rows), so it names both edges of the band and the temperature each edge uses;
    // and production scatter, which glosses sentence 1's "variation allowance" before its first
    // use (task review fix round 1). "Operating temperature" keeps hot-low off the hot-day start
    // (trap T-E), and "expected" keeps cold-high an expectation, not a bound (trap T-O; text
    // re-review 1's SF-1).
    let note = notes::note("hot_low_cold_high").expect("the torque-band note");
    let first = note.sentences[0].to_lowercase();
    for named in [
        "hot-low",
        "cold-high",
        "production scatter",
        "operating temperature",
        "minimum temperature",
        "expected",
    ] {
        assert!(
            first.contains(named),
            "sentence 0 names {named}: {:?}",
            note.sentences[0]
        );
    }
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        let m = &base.metal;
        let v = design.metal.variation;
        // Sentence 0: hot-low, the weakest coupling expected, lies below cold-high, the
        // strongest.
        assert!(
            m.torque_hot_low_Nm < m.torque_cold_high_Nm,
            "{name}: hot-low below cold-high"
        );
        // Sentence 1: hot-low is the pull-out at the operating temperature less the allowance.
        assert!(
            close(m.torque_hot_low_Nm, base.model.pullout_Nm * (1.0 - v)),
            "{name}: hot-low"
        );
        // Sentence 2: cold-high is the pull-out at the minimum temperature plus the allowance.
        let mut cold_running = design.clone();
        cold_running.coupling.op_temp_C = design.metal.min_temp_C;
        assert!(
            close(
                m.torque_cold_high_Nm,
                compute_all(&cold_running).model.pullout_Nm * (1.0 + v)
            ),
            "{name}: cold-high"
        );
        // Sentence 3: the gearbox output carries cold-high itself (the coupling sits on it,
        // ruling 8), so its input sees cold-high through the ratio and the efficiency;
        // whatever holds the hub on the shaft carries the same torque, and no joint is sized
        // while the clamp section is paused (ruling 11).
        let c = &design.coupling;
        assert!(
            close(
                m.cold_high_input_Nm * c.gear_ratio * c.gear_efficiency,
                m.torque_cold_high_Nm
            ),
            "{name}"
        );
        // Sentence 4: hot-low passes exactly when the 20 °C pull-out reaches the requirement
        // taken back to 20 °C with the allowance; the requirement just either side of hot-low.
        for scale in [0.99, 1.01] {
            let mut d = design.clone();
            d.metal.required_min_Nm = scale * m.torque_hot_low_Nm;
            let r = compute_all(&d);
            let passes = r.metal.torque_hot_low_Nm >= d.metal.required_min_Nm;
            assert_eq!(passes, scale < 1.0, "{name} at {scale}");
            assert_eq!(
                passes,
                r.model.pullout_20C_Nm >= r.metal.required_20C_Nm,
                "{name} at {scale}"
            );
        }
    }
}

/// `hot_low_cold_high`'s watch-outs on the example design.
#[test]
fn the_torque_band_note_s_watch_outs_hold() {
    // Watch-out 1: hot-low reads the operating temperature, not the hot-day start, and the
    // hot-day torque is compared with the requirement without the allowance: on the example
    // design hot-low fails while the hot-day torque meets the requirement.
    let defaults = DesignInputs::default();
    let r = compute_all(&defaults);
    let required = defaults.metal.required_min_Nm;
    assert!(r.metal.torque_hot_low_Nm < required);
    assert!(r.temperature.magnet_life.torque_hot_day_Nm >= required);
    assert_ne!(
        r.temperature.duty.hot_day_start_C,
        defaults.coupling.op_temp_C
    );
    let hotter_day = stepped(&defaults, "temperature.duty.hot_ambient_C", Step::Raise);
    assert_eq!(
        compute_all(&hotter_day).metal.torque_hot_low_Nm,
        r.metal.torque_hot_low_Nm
    );
    // Watch-out 2: cold-high at the gearbox input (the relation: the sentences test) is above
    // the input rating on the example design (the GUI test
    // `nothing_checks_a_gearbox_torque_against_its_rating_yet` holds the "nothing checks it").
    let c = &defaults.coupling;
    assert!(r.metal.cold_high_input_Nm > c.gearbox_input_rating_Nm);
}

/// `br_temperature`'s four entries at the five probe designs. The coefficient input moves only
/// rings without a picked grade, so it is checked design by design.
#[test]
fn the_temperature_note_s_levers_hold_at_every_probe_design() {
    let note = notes::note("br_temperature").expect("the temperature note");
    let levers: Vec<&str> = note.change.iter().map(|c| c.input).collect();
    assert_eq!(
        levers,
        [
            "coupling.op_temp_C",
            "calibration.alpha_br_per_C",
            "coupling.magnets.grade_inner",
            "temperature.duty.hot_ambient_C",
        ]
    );
    assert!(matches!(note.change[0].step, Step::Raise));
    assert!(matches!(note.change[1].step, Step::Raise));
    assert!(matches!(note.change[2].step, Step::Choose));
    assert!(matches!(note.change[3].step, Step::Raise));
    assert_eq!(note.change[0].aim, Some(("model.pullout_Nm", Sense::Down)));
    // The coefficient moves only library rings, and which way the torque goes depends on the
    // temperature: no single direction to check, so it is checked here, design by design.
    assert_eq!(note.change[1].aim, None);
    assert_eq!(note.change[2].aim, None);
    assert_eq!(
        note.change[3].aim,
        Some(("temperature.magnet_life.torque_hot_day_Nm", Sense::Down))
    );
    // Entry 4 names every temperature margin a hotter day lowers, not the hot-day one alone
    // (spec 3.1's trade-off rule; the in-task review's SF-2, task review fix round 1).
    assert!(
        note.change[3].text.contains("the temperature margins"),
        "names the margins it lowers: {:?}",
        note.change[3].text
    );
    let (mut library, mut graded, mut hot_onsets) = (0, 0, 0);
    for (name, design) in probes::probe_designs() {
        let base = compute_all(&design);
        // Entry 1: hotter running lowers the pull-out and hot-low, never the 20 °C row.
        let r = compute_all(&stepped(&design, "coupling.op_temp_C", Step::Raise));
        assert!(r.model.pullout_Nm < base.model.pullout_Nm, "{name}");
        assert!(
            r.metal.torque_hot_low_Nm < base.metal.torque_hot_low_Nm,
            "{name}"
        );
        assert!(
            close(r.model.pullout_20C_Nm, base.model.pullout_20C_Nm),
            "{name}"
        );
        // Entry 2 and watch-out 1: the coefficient input moves only rings without a picked
        // grade; there, nearer zero, less is lost hot (the pull-out at the warmer operating
        // temperature rises) and less is gained cold (the cold torque falls).
        let r = compute_all(&stepped(&design, "calibration.alpha_br_per_C", Step::Raise));
        let inner = r.model.inner_alpha_br_per_C != base.model.inner_alpha_br_per_C;
        let outer = r.model.outer_alpha_br_per_C != base.model.outer_alpha_br_per_C;
        assert_eq!(
            inner, outer,
            "{name}: the probe designs have both rings of one kind"
        );
        if inner {
            library += 1;
            assert!(design.coupling.op_temp_C > 20.0, "{name}: runs warm");
            assert!(r.model.pullout_Nm > base.model.pullout_Nm, "{name}");
            assert!(r.metal.torque_cold_Nm < base.metal.torque_cold_Nm, "{name}");
        } else {
            graded += 1;
            assert_eq!(r.model.pullout_Nm, base.model.pullout_Nm, "{name}");
            assert_eq!(r.metal.torque_cold_Nm, base.metal.torque_cold_Nm, "{name}");
        }
        // Entry 4: a hotter day lowers the hot-day torque and the temperature margins, not the
        // pull-out at the operating temperature. The margins it lowers are the three the
        // temperature verdict reads (hot-day, magnet limit, adhesive), since the hot-day start
        // and the peak follow the ambient, and the skipping onset's wherever the magnets have a
        // hot onset; it raises none (the service and cure margins read no ambient).
        let r = compute_all(&stepped(
            &design,
            "temperature.duty.hot_ambient_C",
            Step::Raise,
        ));
        assert!(
            r.temperature.magnet_life.torque_hot_day_Nm
                < base.temperature.magnet_life.torque_hot_day_Nm,
            "{name}"
        );
        // The entry's second clause: the hot-day start is this ambient plus the driving rise (an
        // exact sum in the engine; text re-review 1's SF-2).
        let duty = &design.temperature.duty;
        assert_eq!(
            base.temperature.summary.hot_day_start_C,
            duty.hot_ambient_C + duty.driving_rise_C,
            "{name}: the hot-day start"
        );
        let (t, t0) = (&r.temperature, &base.temperature);
        for (margin, after, before) in [
            (
                "hot-day",
                t.summary.margin_hot_day_C,
                t0.summary.margin_hot_day_C,
            ),
            (
                "magnet limit",
                t.magnet_life.margin_limit_C,
                t0.magnet_life.margin_limit_C,
            ),
            (
                "adhesive",
                t.adhesive_life.margin_C,
                t0.adhesive_life.margin_C,
            ),
        ] {
            assert!(after < before, "{name}: the {margin} margin falls");
        }
        // Ferrite has no hot skipping onset (the Y30 design's is infinite, drawn as "no
        // prediction"), so its margin stays infinite; elsewhere it falls.
        let (after, before) = (t.magnet_life.margin_onset_C, t0.magnet_life.margin_onset_C);
        if before.is_finite() {
            hot_onsets += 1;
            assert!(after < before, "{name}: the skipping onset margin falls");
        } else {
            assert_eq!(after, before, "{name}: no hot onset, no margin to lose");
        }
        for (margin, after, before) in [
            (
                "service",
                t.summary.margin_service_C,
                t0.summary.margin_service_C,
            ),
            ("cure", t.summary.cure_margin_C, t0.summary.cure_margin_C),
        ] {
            assert!(after <= before, "{name}: the {margin} margin does not rise");
        }
        assert_eq!(r.model.pullout_Nm, base.model.pullout_Nm, "{name}");
    }
    assert!(
        library >= 1 && graded >= 1,
        "library and graded rings among the probes"
    );
    assert!(hot_onsets >= 1, "a hot skipping onset among the probes");
}

#[test]
fn only_a_ring_with_a_picked_grade_keeps_its_own_coefficient() {
    // Watch-out 1 of the temperature note, on a mixed design: the inner ring on custom
    // dimensions with Y30, the outer on its library part; the input moves the outer ring only.
    let mixed = on_grade(&DesignInputs::default(), Rings::Inner, "Y30");
    let base = compute_all(&mixed);
    let moved = stepped(&mixed, "calibration.alpha_br_per_C", Step::Raise);
    let r = compute_all(&moved);
    assert_eq!(
        r.model.inner_alpha_br_per_C,
        base.model.inner_alpha_br_per_C
    );
    assert_eq!(
        r.model.outer_alpha_br_per_C,
        moved.calibration.alpha_br_per_C
    );
    assert_ne!(
        base.model.inner_alpha_br_per_C,
        base.model.outer_alpha_br_per_C
    );
}

/// `br_temperature`'s grade entry. A grade whose coefficient is nearer zero loses a smaller share
/// when hot, design by design, though not always fewer N·m; the cost named is SmCo's, the
/// nearest: it starts (at 20 °C) weaker than every sintered NdFeB grade. "Nearer zero" alone
/// does not mean weaker: from the Y30 probe design every other grade is nearer zero, and SmCo
/// starts stronger.
#[test]
fn a_grade_nearer_zero_loses_a_smaller_share_when_hot_and_smco_starts_below_sintered_ndfeb() {
    let note = notes::note("br_temperature").expect("the temperature note");
    let entry = note
        .change
        .iter()
        .find(|c| c.input == "coupling.magnets.grade_inner")
        .expect("the grade entry");
    assert!(
        entry
            .text
            .split(';')
            .all(|clause| !(clause.contains("nearer zero") && clause.contains("weaker"))),
        "ties no weakness to nearer zero as such: {:?}",
        entry.text
    );
    assert!(
        entry.text.contains("weaker than sintered NdFeB"),
        "names what SmCo starts weaker than: {:?}",
        entry.text
    );
    assert!(
        entry.text.contains("smaller share"),
        "says the loss is a share, not an amount: {:?}",
        entry.text
    );

    // "SmCo, the nearest" and "starts weaker than sintered NdFeB", over the whole grade table.
    let is_smco = |g: &&Grade| matches!(g.family, GradeFamily::SmCo2_17 | GradeFamily::SmCo1_5);
    let (smco_grades, others): (Vec<&Grade>, Vec<&Grade>) = GRADES.iter().partition(is_smco);
    assert!(!smco_grades.is_empty() && others.iter().any(|g| g.family == GradeFamily::NdFeB));
    for s in &smco_grades {
        for o in &others {
            assert!(
                s.alpha_br_per_C.abs() < o.alpha_br_per_C.abs(),
                "{} nearer zero than {}",
                s.id,
                o.id
            );
            if o.family == GradeFamily::NdFeB {
                assert!(s.br_T < o.br_T, "{} Br below {}", s.id, o.id);
            }
        }
    }

    // Both rings on custom dimensions with one grade.
    let graded =
        |design: &DesignInputs, grade: &str| compute_all(&on_grade(design, Rings::Both, grade));
    // The share of the 20 °C pull-out kept at the operating temperature.
    let kept = |r: &DesignResults| r.model.pullout_Nm / r.model.pullout_20C_Nm;
    let default = DesignInputs::default();
    assert!(default.coupling.op_temp_C > 20.0, "the default runs warm");
    let (ndfeb, smco) = (graded(&default, "N42SH"), graded(&default, "SmCo_2_17_30"));
    assert!(smco.model.inner_alpha_br_per_C.abs() < ndfeb.model.inner_alpha_br_per_C.abs());
    assert!(smco.model.outer_alpha_br_per_C.abs() < ndfeb.model.outer_alpha_br_per_C.abs());
    assert!(
        kept(&smco) > kept(&ndfeb),
        "default: loses a smaller share when hot"
    );
    assert!(
        smco.model.pullout_20C_Nm < ndfeb.model.pullout_20C_Nm,
        "default: starts weaker than sintered NdFeB"
    );

    let ferrite = probe_inputs("Y30 ferrite, custom dimensions");
    assert!(
        ferrite.coupling.op_temp_C > 20.0,
        "the Y30 design runs warm"
    );
    let (y30, smco) = (compute_all(&ferrite), graded(&ferrite, "SmCo_2_17_30"));
    assert!(smco.model.inner_alpha_br_per_C.abs() < y30.model.inner_alpha_br_per_C.abs());
    assert!(smco.model.outer_alpha_br_per_C.abs() < y30.model.outer_alpha_br_per_C.abs());
    assert!(
        kept(&smco) > kept(&y30),
        "Y30 design: loses a smaller share when hot"
    );
    // A smaller share is not fewer N·m: SmCo starts far stronger than Y30.
    let lost = |r: &DesignResults| r.model.pullout_20C_Nm - r.model.pullout_Nm;
    assert!(
        lost(&smco) > lost(&y30),
        "Y30 design: a smaller share, yet more N·m lost"
    );
    assert!(
        smco.model.pullout_20C_Nm > y30.model.pullout_20C_Nm,
        "Y30 design: nearer zero, and stronger at 20 °C"
    );
}

// Slice 2 note claims: the slipping, slip-heating and slip-temperature notes (plan T15). Each
// lever moves by one slider step ([`stepped`]); T5's aim test checks each entry's aim, and the
// entries with none (a measured drag, a material) are checked here, design by design.

/// A change entry without its text: (input, step, aim).
type Lever = (&'static str, Step, Option<(&'static str, Sense)>);

/// The note `id`'s change entries, without their texts.
fn levers_of(id: &str) -> Vec<Lever> {
    notes::note(id)
        .unwrap_or_else(|| panic!("no note {id}"))
        .change
        .iter()
        .map(|c| (c.input, c.step, c.aim))
        .collect()
}

/// `when_slipping` at the five probe designs. Past pull-out the torque swings between plus
/// and minus the pull-out once per pole pair passed (the slip frequency), leaving a drag far
/// below the pull-out; the gearbox input sees that swing through the ratio and the efficiency,
/// and cold-high at the input lies above it (the first watch-out: cold, the swings can reach
/// cold-high).
#[test]
fn the_slip_note_s_claims_hold_at_every_probe_design() {
    assert_eq!(
        levers_of("when_slipping"),
        [
            (
                "temperature.duty.fault_trip_s",
                Step::Lower,
                Some(("temperature.thermal.temp_at_fault_C", Sense::Down))
            ),
            (
                "metal.slip_rpm",
                Step::Lower,
                Some(("temperature.slip_loss.total_W", Sense::Down))
            ),
        ]
    );
    // Sentence 0 answers the gearbox ripple row that links the note (spec 3.1): the swinging
    // torque is the gearbox's (sentence 4's ripple, below).
    let sentence_zero = notes::note("when_slipping").unwrap().sentences[0];
    assert!(
        sentence_zero.contains("gearbox"),
        "sentence 0 names the gearbox: {sentence_zero:?}"
    );
    for (name, design) in probes::probe_designs() {
        let r = compute_all(&design);
        let c = &design.coupling;
        // Sentence 1: one swing per pole pair passed, so the swings come at the pole pairs
        // times the slip revolutions per second.
        let pole_pairs = c.npole as f64 / 2.0;
        assert!(
            close(
                r.metal.slip_freq_Hz,
                pole_pairs * design.metal.slip_rpm / 60.0
            ),
            "{name}: the slip frequency"
        );
        assert_eq!(r.temperature.duty.pole_pairs, pole_pairs, "{name}");
        // Sentences 1 and 2: on average the load gets only the drag, under a tenth of the
        // pull-out (the most is the Y30 design's, its drag from the example design's fields).
        assert!(
            r.temperature.slip_loss.drag_Nm < 0.1 * r.model.pullout_Nm,
            "{name}: {} against {}",
            r.temperature.slip_loss.drag_Nm,
            r.model.pullout_Nm
        );
        // Sentence 4: the ripple at the gearbox input is the swing's amplitude, the pull-out,
        // through the ratio and the efficiency.
        assert!(
            close(
                r.model.gearbox_input_ripple_Nm * c.gear_ratio * c.gear_efficiency,
                r.model.pullout_Nm
            ),
            "{name}: the ripple"
        );
        // Watch-out 1: cold, the swings reach cold-high, which at the gearbox input lies above
        // the ripple at the operating temperature (the GUI test
        // `nothing_checks_a_gearbox_torque_against_its_rating_yet` holds the "nothing checks").
        assert!(
            r.metal.cold_high_input_Nm > r.model.gearbox_input_ripple_Nm,
            "{name}: cold-high at the gearbox input"
        );
        // Entry 2: slower slipping, slower swings.
        let slower = compute_all(&stepped(&design, "metal.slip_rpm", Step::Lower));
        assert!(
            slower.metal.slip_freq_Hz < r.metal.slip_freq_Hz,
            "{name}: slower swings"
        );
    }
}

/// `slip_heating` at the five probe designs: the heat is the drag times the slip speed; the
/// magnetic-steel parts' loss grows as the slip speed to the power 1.5 (skin-limited), the
/// shells', the cap's and the magnets' as its square, and with no magnetic steel the hub's,
/// cup's and web's as its square too; a library sleeve and liner material other than 316L, or
/// a thinner sleeve or liner, heats less, and the thinner one widens the running clearance.
#[test]
fn the_slip_heating_note_s_claims_hold_at_every_probe_design() {
    let total_down = Some(("temperature.slip_loss.total_W", Sense::Down));
    assert_eq!(
        levers_of("slip_heating"),
        [
            ("metal.measured_drag_Nm", Step::Enter, None),
            ("materials.parts.sleeve_liner", Step::Choose, None),
            ("metal.sleeve_mm", Step::Lower, total_down),
            ("metal.liner_mm", Step::Lower, total_down),
        ]
    );
    // Sentences 2 and 4 say magnetic steel: a 304 back iron is steel but not magnetic, and its
    // hub, cup and web take the whole-thickness form (the speed squared, below). Every "steel"
    // in the sentences is " magnetic steel" (the space keeps "non-magnetic steel" out).
    let sentences = notes::note("slip_heating").unwrap().sentences;
    assert!(
        sentences[2].starts_with("In thick magnetic steel "),
        "{:?}",
        sentences[2]
    );
    assert!(
        sentences[4].starts_with("Without magnetic steel back iron "),
        "{:?}",
        sentences[4]
    );
    for s in sentences {
        assert_eq!(
            s.matches("steel").count(),
            s.matches(" magnetic steel").count(),
            "only magnetic steel: {s:?}"
        );
    }
    let code = |id: &str| {
        SLEEVE_LINER_CHOICES
            .iter()
            .find(|&&(_, name)| name == id)
            .map(|&(code, _)| code)
            .unwrap_or_else(|| panic!("no sleeve and liner material {id}"))
    };
    let (mut steel, mut no_steel) = (0, 0);
    for (name, design) in probes::probe_designs() {
        let r = compute_all(&design);
        let loss = &r.temperature.slip_loss;
        // Sentence 1: the heat power is the drag torque times the slip speed.
        assert!(
            close(loss.drag_Nm * r.temperature.duty.slip_rad_s, loss.total_W),
            "{name}: power"
        );
        // Sentences 2 to 4, at twice the slip speed (inside its slider).
        let mut faster = design.clone();
        faster.metal.slip_rpm *= 2.0;
        let fast = compute_all(&faster).temperature.slip_loss;
        let grows = |before: f64, after: f64| (after / before).log2();
        let whole_thickness = [
            ("sleeve", loss.sleeve_W, fast.sleeve_W),
            ("liner", loss.liner_W, fast.liner_W),
            ("cap", loss.cap_W, fast.cap_W),
            ("magnets", loss.magnets_W, fast.magnets_W),
        ];
        for (part, before, after) in whole_thickness {
            assert!(
                (grows(before, after) - 2.0).abs() < 1e-9,
                "{name}: the {part} as the speed squared"
            );
        }
        let body = [
            ("hub", loss.hub_W, fast.hub_W),
            ("cup", loss.cup_W, fast.cup_W),
            ("web", loss.web_W, fast.web_W),
        ];
        let magnetic_steel = r.materials.circuit_backiron == 1;
        let power = if magnetic_steel {
            steel += 1;
            1.5
        } else {
            no_steel += 1;
            2.0
        };
        for (part, before, after) in body {
            assert!(
                (grows(before, after) - power).abs() < 1e-9,
                "{name}: the {part} as the speed to the power {power}"
            );
        }
        // Entry 2: titanium, Inconel and PEEK conduct less than 316L, so the sleeve and the
        // liner heat less (the probe designs use 316L).
        assert_eq!(design.materials.parts.sleeve_liner, code("316L_annealed"));
        for id in ["Ti6Al4V_annealed", "IN625_annealed", "PEEK_unfilled"] {
            let mut other = design.clone();
            other.materials.parts.sleeve_liner = code(id);
            let o = compute_all(&other).temperature.slip_loss;
            assert!(
                o.sleeve_W < loss.sleeve_W && o.liner_W < loss.liner_W,
                "{name}: {id}"
            );
            assert!(o.total_W < loss.total_W, "{name}: {id}, the total");
        }
        // Entries 3 and 4: a thinner sleeve or liner widens the running clearance (the loss it
        // lowers is the aim test's).
        for path in ["metal.sleeve_mm", "metal.liner_mm"] {
            let thinner = compute_all(&stepped(&design, path, Step::Lower));
            assert!(
                thinner.metal.min_running_clearance_mm > r.metal.min_running_clearance_mm,
                "{name}: {path} widens the clearance"
            );
        }
    }
    assert!(
        steel >= 1 && no_steel >= 1,
        "designs with and without magnetic steel among the probes"
    );
}

/// The slip notes' non-magnetic-body sources (`slip_heating` sources[3] and
/// `thermal_time_constant` sources[1]): with a non-magnetic back-iron pick the hub, cup and web
/// take the pick's own conductivity and specific heat (304 stainless at its probe design), so
/// the rules they cite are the non-magnetic body's, not aluminium's alone.
#[test]
fn the_non_magnetic_body_is_the_picked_metal_not_only_aluminium() {
    for (id, i) in [("slip_heating", 3), ("thermal_time_constant", 1)] {
        let sources = notes::note(id).unwrap().sources;
        assert!(
            sources[i].contains("non-magnetic body parts"),
            "{id}: {:?}",
            sources[i]
        );
        assert!(
            sources.iter().all(|s| !s.contains("aluminium")),
            "{id}: {sources:?}"
        );
    }
    let mut not_aluminium = 0;
    for (name, design) in probes::probe_designs() {
        let pick = chosen(&BACK_IRON_CHOICES, design.materials.parts.back_iron)
            .unwrap_or_else(|| panic!("{name}: a back-iron choice"));
        if pick.ferromagnetic {
            continue;
        }
        let m = compute_all(&design).materials;
        assert_ne!(m.circuit_backiron, 1, "{name}: no magnetic steel");
        assert_eq!(
            (m.body_sigma_S_m, m.body_c_J_kgK),
            (pick.engine.sigma_S_m, pick.engine.cp_J_kgK),
            "{name}: the body is the picked metal"
        );
        if pick.id != "6061_T6" {
            not_aluminium += 1;
        }
    }
    assert!(
        not_aluminium >= 1,
        "a non-magnetic body other than aluminium among the probes"
    );
}

/// The two notes' measured-drag entries (`slip_heating` and `thermal_time_constant`, A-2): a
/// drag measured on the bench replaces the estimated loss and its high case in the temperature
/// results (both become the measurement), while the slip-loss rows the notes call "estimated"
/// keep the estimate.
#[test]
fn a_measured_drag_replaces_the_loss_and_its_high_case() {
    for id in ["slip_heating", "thermal_time_constant"] {
        assert!(
            levers_of(id).contains(&("metal.measured_drag_Nm", Step::Enter, None)),
            "{id}: the measured-drag entry"
        );
    }
    for (name, design) in probes::probe_designs() {
        assert_eq!(design.metal.measured_drag_Nm, None, "{name}: none measured");
        let estimate = compute_all(&design).temperature;
        let mut measured = design.clone();
        measured.metal.measured_drag_Nm = Some(0.02);
        let t = compute_all(&measured).temperature;
        let power = 0.02 * t.duty.slip_rad_s;
        assert!(close(t.slip_loss.used_W, power), "{name}: the loss used");
        assert!(close(t.slip_loss.high_W, power), "{name}: its high case");
        assert_eq!(t.thermal.steady_est_C, t.thermal.steady_high_C, "{name}");
        assert_eq!(t.slip_loss.total_W, estimate.slip_loss.total_W, "{name}");
        assert_eq!(t.slip_loss.drag_Nm, estimate.slip_loss.drag_Nm, "{name}");
        // Unmeasured, the high case is the estimate times the allowance the check uses.
        let multiplier = design.temperature.slip_loss.high_multiplier;
        assert!(multiplier > 1.0, "{name}");
        assert!(
            close(
                estimate.slip_loss.high_W,
                estimate.slip_loss.total_W * multiplier
            ),
            "{name}: the high case"
        );
    }
}

/// `thermal_time_constant` at the five probe designs: one lump of heat capacity C and
/// conductance G, starting at the hot-day start; the steady rise P/G, the time constant C/G;
/// one slip event's P·t/C; the life's average rise; the peak the temperature check reads; and
/// "never" exactly when the high-case steady temperature stays at or below the limit.
#[test]
fn the_slip_temperature_note_s_claims_hold_at_every_probe_design() {
    use magcoupling::engine::meta::NumOrText;
    assert_eq!(
        levers_of("thermal_time_constant"),
        [
            (
                "temperature.thermal.conductance_W_K",
                Step::Raise,
                Some(("temperature.thermal.steady_high_C", Sense::Down))
            ),
            ("metal.measured_drag_Nm", Step::Enter, None),
            (
                "temperature.duty.fault_trip_s",
                Step::Lower,
                Some(("temperature.magnet_life.peak_C", Sense::Down))
            ),
        ]
    );
    // Sentence 0 answers the time-constant row Start here opens the note on (spec 3.1): mass
    // changes how fast, which the time constant measures (the heavier design, below). "Only"
    // keeps it clear of the trap "more mass lowers the steady temperature".
    let sentence_zero = notes::note("thermal_time_constant").unwrap().sentences[0];
    assert!(
        sentence_zero.contains("time constant"),
        "sentence 0 names the time constant: {sentence_zero:?}"
    );
    assert!(
        sentence_zero.contains("mass changes only how fast"),
        "sentence 0 keeps the mass trap guard: {sentence_zero:?}"
    );
    let (mut never, mut reached) = (0, 0);
    for (name, design) in probes::probe_designs() {
        let t = compute_all(&design).temperature;
        let (th, life) = (&t.thermal, &t.slip_life);
        let (p, p_high) = (t.slip_loss.used_W, t.slip_loss.high_W);
        let g = design.temperature.thermal.conductance_W_K;
        let c = th.heat_capacity_J_K;
        // Sentence 1: the lump starts at the hot-day start, hot ambient plus driving rise.
        let duty = &design.temperature.duty;
        assert_eq!(
            th.start_C,
            duty.hot_ambient_C + duty.driving_rise_C,
            "{name}"
        );
        assert_eq!(th.start_C, t.summary.hot_day_start_C, "{name}");
        // Sentence 2: toward a rise of P/G above the start, τ_th = C/G.
        assert!(close(th.steady_rise_est_C, p / g), "{name}");
        assert!(close(th.steady_rise_high_C, p_high / g), "{name}");
        assert!(close(th.time_constant_s, c / g), "{name}");
        assert_eq!(
            th.steady_high_C,
            th.start_C + th.steady_rise_high_C,
            "{name}"
        );
        // Sentence 0: more heat capacity changes only how fast, not the steady rise.
        let mut heavier = design.clone();
        heavier.temperature.thermal.c_ndfeb += 50.0;
        let h = compute_all(&heavier).temperature.thermal;
        assert!(h.time_constant_s > th.time_constant_s, "{name}: slower");
        assert_eq!(h.steady_high_C, th.steady_high_C, "{name}: the same rise");
        // Sentence 3: one slip event adds P·t/C; the life adds the slip duty times P/G.
        assert!(
            close(
                life.rise_per_event_high_C,
                p_high * design.metal.slip_event_s / c
            ),
            "{name}: one event"
        );
        assert!(
            close(life.avg_rise_high_C, life.slip_duty * th.steady_rise_high_C),
            "{name}: the average rise"
        );
        // Sentence 4: the peak is the start plus the hotter of a fault slip and one slip event,
        // plus that average rise; the temperature check's limit margins read it.
        let peak = t.magnet_life.peak_C;
        let hotter = th
            .temp_at_fault_C
            .max(th.start_C + life.rise_per_event_high_C);
        assert!(
            close(peak, hotter + life.avg_rise_high_C),
            "{name}: the peak"
        );
        assert_eq!(
            t.magnet_life.margin_limit_C,
            t.summary.magnet_limit_C - peak,
            "{name}"
        );
        assert_eq!(
            t.adhesive_life.margin_C,
            t.summary.adhesive_limit_C - peak,
            "{name}"
        );
        // Entry 1: a better heat path also shortens τ_th.
        let better = compute_all(&stepped(
            &design,
            "temperature.thermal.conductance_W_K",
            Step::Raise,
        ));
        assert!(
            better.temperature.thermal.time_constant_s < th.time_constant_s,
            "{name}: shorter τ_th"
        );
        // Watch-out 1: "never" exactly when the high-case steady temperature stays at or below
        // the limit (every probe design starts below it).
        let limit = t.summary.governing_limit_C;
        assert!(th.start_C < limit, "{name}: starts below the limit");
        let is_never = matches!(th.time_to_limit_high, NumOrText::Text(_));
        assert_eq!(is_never, th.steady_high_C <= limit, "{name}");
        if is_never {
            never += 1;
        } else {
            reached += 1;
        }
    }
    assert!(
        never >= 1 && reached >= 1,
        "'never' and a time to the limit among the probes"
    );
}

/// `thermal_time_constant`'s fault-trip entry: a hot-day start at or above the limit (the
/// hottest day and driving rise the sliders allow) keeps the peak there, however short the
/// unbroken slip.
#[test]
fn a_shorter_fault_slip_cannot_bring_the_peak_under_a_limit_the_start_reaches() {
    for (name, mut design) in probes::probe_designs() {
        let duty = &mut design.temperature.duty;
        duty.hot_ambient_C = 80.0;
        duty.driving_rise_C = 60.0;
        while let Ok(shorter) =
            probes::step_input(&design, "temperature.duty.fault_trip_s", Step::Lower)
        {
            design = shorter;
        }
        let t = compute_all(&design).temperature;
        assert!(
            t.thermal.start_C >= t.summary.governing_limit_C,
            "{name}: the start reaches the limit"
        );
        assert!(
            t.magnet_life.peak_C > t.summary.governing_limit_C,
            "{name}: the peak stays above it at a {} s trip",
            design.temperature.duty.fault_trip_s
        );
    }
}

/// The slip notes' stored-field watch-outs (`when_slipping` and `slip_heating`): the slip
/// losses read fields stored from the example design's 3D model, the same at every probe
/// design, so they do not follow the magnet grade (the Y30 design, the example's geometry with
/// ferrite on both rings, has the example's losses). When the fields follow the design, this
/// fails, and those watch-outs change in the same commit.
#[test]
fn the_slip_losses_read_fields_stored_from_the_example_design() {
    let example = DesignInputs::default();
    let stored = |d: &DesignInputs| {
        let s = &d.temperature.slip_loss;
        [
            s.b_hub_T,
            s.b_cup_T,
            s.b_sleeve_T,
            s.b_liner_T,
            s.cap_integral_T2m4,
            s.web_integral_T2m2,
            s.b_magnet_T,
            s.b_hub_free_T,
            s.b_cup_free_T,
            s.web_integral_free_T2m2,
        ]
    };
    for (name, design) in probes::probe_designs() {
        assert_eq!(stored(&design), stored(&example), "{name}");
    }
    let y30 = compute_all(&probe_inputs("Y30 ferrite, custom dimensions"));
    let r = compute_all(&example);
    assert_ne!(y30.model.pullout_Nm, r.model.pullout_Nm);
    assert_eq!(
        y30.temperature.slip_loss.total_W,
        r.temperature.slip_loss.total_W
    );
}

// Slice 2 note claims: the demagnetization and ferrite notes (plan T16). Their entries choose a
// part and grade or enter a reverse field from an outside 3D calculation, so none has an aim for
// T5's aim test; what they and the sentences say is checked here, design by design.

/// The like-poles reverse field input, the one both notes' Enter entries name.
const LIKE_POLES: &str = "temperature.demag.h_rev_likepole_kA_m";

/// Both notes' change entries: a part and grade on each ring, and the like-poles field entered
/// from an outside 3D calculation.
const DEMAG_LEVERS: [Lever; 3] = [
    ("coupling.magnets.part_inner", Step::Choose, None),
    ("coupling.magnets.part_outer", Step::Choose, None),
    (LIKE_POLES, Step::Enter, None),
];

/// `design`'s four stored reverse fields [kA/m]: rings aligned, at pull-out, like poles facing
/// (slipping), one ring alone.
fn reverse_fields(design: &DesignInputs) -> [f64; 4] {
    let d = &design.temperature.demag;
    [
        d.h_rev_aligned_kA_m,
        d.h_rev_pullout_kA_m,
        d.h_rev_likepole_kA_m,
        d.h_rev_single_ring_kA_m,
    ]
}

/// The number in a result that is a number or a text ("n/a").
fn num(value: NumOrText) -> f64 {
    match value {
        NumOrText::Num(x) => x,
        NumOrText::Text(text) => panic!("a number, not {text:?}"),
    }
}

/// Whether `text` names `word` as a whole word (no letter or digit on either side), so "N50"
/// is not found in "N50M".
fn names_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

#[test]
fn names_word_finds_whole_words_only() {
    assert!(names_word("For N52, N50 and 2:17 SmCo.", "N50"));
    assert!(names_word("For N52, N50 and 2:17 SmCo.", "2:17 SmCo"));
    assert!(!names_word("For N52 and N50M.", "N50"));
    assert!(!names_word("For XN50 only.", "N50"));
    assert!(!names_word("For N52 only.", "N42"));
}

/// `demagnetization` at the five probe designs. Like poles facing (slipping) is the worst of the
/// stored reverse fields, so on NdFeB its onset is the lowest and sets the ring's limit, less the
/// margin; the onsets are shifted so the reference magnet's onset is the magnet's own rating; the
/// lower ring limit governs; the two margins the note owns read the peak against the skipping
/// onset and the limit.
#[test]
fn the_demagnetization_note_s_claims_hold_at_every_probe_design() {
    assert_eq!(levers_of("demagnetization"), DEMAG_LEVERS);
    let [aligned, pullout, like_poles, single] = reverse_fields(&DesignInputs::default());
    assert!(
        like_poles > aligned.max(pullout).max(single),
        "like poles facing is the worst stored field"
    );
    let mut ndfeb = 0;
    for (name, design) in probes::probe_designs() {
        let r = compute_all(&design);
        let (d, life) = (&r.temperature.demag, &r.temperature.magnet_life);
        assert_eq!(
            d.magnet_limit_C,
            d.inner_magnet_limit_C.min(d.outer_magnet_limit_C),
            "{name}: the lower ring limit governs"
        );
        assert_eq!(
            life.margin_onset_C,
            d.onset_skipping_C - life.peak_C,
            "{name}"
        );
        assert_eq!(
            life.margin_limit_C,
            d.magnet_limit_C - life.peak_C,
            "{name}"
        );
        // Heat lowers an NdFeB knee (a negative coefficient): the hot side decides.
        if d.beta_used_per_C < 0.0 {
            ndfeb += 1;
            for other in [d.onset_aligned_C, d.onset_pullout_C, d.onset_single_ring_C] {
                assert!(
                    d.onset_skipping_C < other,
                    "{name}: slipping's onset is the lowest"
                );
            }
            assert_eq!(
                d.magnet_limit_C,
                d.onset_skipping_C - design.temperature.demag.design_margin_C,
                "{name}: the slipping onset less the margin"
            );
            assert!(
                close(d.t_ref_model_C - d.calibration_offset_C, num(d.tmax_lib_C)),
                "{name}: shifted so the reference magnet's onset is the rating"
            );
        }
    }
    assert_eq!(ndfeb, 4, "the four NdFeB probe designs");
}

/// `demagnetization`'s two grade entries on the example design (N42SH on both rings), at the
/// grades rated hotter that the review probes. On both rings, custom dimensions and the grade
/// raise the magnet limit and lower the pull-out, and the adhesive then governs; on one ring
/// alone that ring's limit rises but the other ring's still governs. Every grade rated hotter
/// than the example's is weaker (a lower Br).
#[test]
fn a_hotter_rated_grade_raises_its_ring_s_limit_costs_torque_and_alone_may_not_help() {
    let example = DesignInputs::default();
    let base = compute_all(&example);
    let d0 = &base.temperature.demag;
    assert_eq!(
        (d0.hcj20_used_kA_m, d0.beta_used_per_C),
        (N42SH.hcj20_kA_m, N42SH.beta_hcj_per_C),
        "the example's rings are N42SH"
    );
    let rating = num(d0.tmax_lib_C);
    assert!(
        base.temperature.summary.governing_limit_C < base.temperature.summary.adhesive_limit_C,
        "on the example the magnets govern"
    );
    for g in GRADES.iter().filter(|g| g.tmax_C > rating) {
        assert!(g.br_T < d0.br20_T, "{} is weaker", g.id);
    }
    for id in ["N38UH", "N35EH", "N33AH", "SmCo_2_17_26"] {
        let g = grade(id).unwrap_or_else(|| panic!("no grade {id}"));
        assert!(g.tmax_C > rating, "{id} is rated hotter");
        let r = compute_all(&on_grade(&example, Rings::Both, id));
        let (d, summary) = (&r.temperature.demag, &r.temperature.summary);
        assert!(
            d.magnet_limit_C > d0.magnet_limit_C,
            "{id}: the limit rises"
        );
        assert!(
            r.model.pullout_Nm < base.model.pullout_Nm,
            "{id}: torque falls"
        );
        assert_eq!(
            summary.governing_limit_C, summary.adhesive_limit_C,
            "{id}: the adhesive governs"
        );
        assert!(summary.adhesive_limit_C < d.magnet_limit_C, "{id}");
        for rings in [Rings::Inner, Rings::Outer] {
            let d = compute_all(&on_grade(&example, rings, id))
                .temperature
                .demag;
            let own = match rings {
                Rings::Inner => d.inner_magnet_limit_C,
                Rings::Outer | Rings::Both => d.outer_magnet_limit_C,
            };
            assert!(
                own > d0.magnet_limit_C,
                "{id} {rings:?}: its own limit rises"
            );
            assert_eq!(
                d.magnet_limit_C, d0.magnet_limit_C,
                "{id} {rings:?}: the other ring still governs"
            );
        }
    }
}

/// `demagnetization`'s second watch-out names exactly the grades whose onsets the rating shift
/// raises on custom dimensions: a negative offset, so each onset is the uncalibrated one plus its
/// size; the two 2:17 SmCo grades go by one name. On the example design an onset lies past its
/// grade's tested range. When the grade data or the shift change, this fails and the watch-out
/// changes in the same commit (BL-057).
#[test]
fn the_demagnetization_watch_out_names_the_grades_the_rating_shift_raises() {
    let watch_out = notes::note("demagnetization").expect("the note").watch_out[1];
    let example = DesignInputs::default();
    let knee = example.temperature.demag.knee_fraction;
    let like_poles = example.temperature.demag.h_rev_likepole_kA_m;
    let mut raised = 0;
    for g in &GRADES {
        let d = compute_all(&on_grade(&example, Rings::Both, g.id))
            .temperature
            .demag;
        let shifted_up = d.calibration_offset_C < 0.0;
        if shifted_up {
            raised += 1;
            let uncalibrated = demag_onset_C(
                like_poles,
                d.hcj20_used_kA_m,
                d.beta_used_per_C,
                knee,
                d.alpha_br,
                0.0,
            );
            assert!(
                d.onset_skipping_C > uncalibrated,
                "{}: shifted upward",
                g.id
            );
        }
        let name = if g.family == GradeFamily::SmCo2_17 {
            "2:17 SmCo"
        } else {
            g.id
        };
        assert_eq!(
            names_word(watch_out, name),
            shifted_up,
            "{} (offset {} °C) in {watch_out:?}",
            g.id,
            d.calibration_offset_C
        );
    }
    assert!(raised > 0, "some grade's onsets are raised");
    let d = compute_all(&example).temperature.demag;
    let [_, tested_to] = N42SH.coefficient_range_C.expect("N42SH's tested range");
    assert!(
        d.onset_aligned_C > tested_to,
        "an onset past the tested range"
    );
}

/// The demagnetization notes' stored-field claims: the four reverse fields are the example
/// design's at every probe design, and the onsets do not follow the magnet count, ring radius,
/// gap or block size (each move changes the pull-out, so it took effect). When the fields follow
/// the design, this fails, and the watch-out and both notes' Enter entries change in the same
/// commit.
#[test]
fn the_reverse_fields_are_the_example_design_s_whatever_the_design() {
    let example = DesignInputs::default();
    for (name, design) in probes::probe_designs() {
        assert_eq!(reverse_fields(&design), reverse_fields(&example), "{name}");
    }
    let onsets = |r: &DesignResults| {
        let d = &r.temperature.demag;
        [
            d.onset_aligned_C,
            d.onset_pullout_C,
            d.onset_skipping_C,
            d.onset_single_ring_C,
        ]
    };
    let custom = on_grade(&example, Rings::Both, N42SH.id);
    for (what, from, moved) in [
        ("magnet count", &example, probe_inputs("8 magnets per ring")),
        ("ring radius", &example, probe_inputs("20 mm ring radius")),
        (
            "gap",
            &example,
            stepped(&example, "metal.face_gap_mm", Step::Raise),
        ),
        (
            "block size",
            &custom,
            stepped(
                &custom,
                "coupling.magnets.manual_inner_thickness_mm",
                Step::Raise,
            ),
        ),
    ] {
        let (before, after) = (compute_all(from), compute_all(&moved));
        assert_ne!(
            after.model.pullout_Nm, before.model.pullout_Nm,
            "{what}: the move took effect"
        );
        assert_eq!(onsets(&after), onsets(&before), "{what}");
    }
}

/// `ferrite_cold_demag`'s sentences and watch-outs. Only ferrite's coercivity rises with heat,
/// and its Br falls: cooling lowers its knee while its field grows. At the Y30 probe design no
/// hot onset exists, each ring's hot limit is its rating, and the cold limit is the slipping cold
/// onset (the highest) plus the margin. With the stored fields that onset lies above the rating
/// and the cold limit above the coldest temperature the slider allows, so the check fails. On
/// NdFeB the check reads n/a. Y30's coefficients are measured over a range the coldest
/// temperature lies below.
#[test]
fn the_ferrite_note_s_claims_hold() {
    let note = notes::note("ferrite_cold_demag").expect("the ferrite note");
    for g in &GRADES {
        assert_eq!(
            g.beta_hcj_per_C > 0.0,
            g.family == GradeFamily::Ferrite,
            "{}",
            g.id
        );
    }
    let y30 = grade("Y30").expect("Y30");
    assert!(y30.alpha_br_per_C < 0.0, "its own field grows as it cools");
    let ferrite_design = "Y30 ferrite, custom dimensions";
    let design = probe_inputs(ferrite_design);
    let r = compute_all(&design);
    let d = &r.temperature.demag;
    for onset in [
        d.onset_aligned_C,
        d.onset_pullout_C,
        d.onset_skipping_C,
        d.onset_single_ring_C,
    ] {
        assert_eq!(onset, f64::INFINITY, "no hot onset");
    }
    assert_eq!(
        (d.inner_magnet_limit_C, d.outer_magnet_limit_C),
        (y30.tmax_C, y30.tmax_C),
        "the hot limit is the rating"
    );
    let [aligned, pullout, skipping, single] = [
        d.cold_onset_aligned_C,
        d.cold_onset_pullout_C,
        d.cold_onset_skipping_C,
        d.cold_onset_single_ring_C,
    ]
    .map(num);
    assert!(
        skipping > aligned.max(pullout).max(single),
        "slipping sets the cold onset"
    );
    assert_eq!(
        num(d.cold_limit_C),
        skipping + design.temperature.demag.design_margin_C
    );
    assert!(
        skipping > y30.tmax_C,
        "the like-poles field passes the knee at every temperature up to the rating"
    );
    assert!(
        num(d.cold_limit_C) > slider("metal.min_temp_C").max,
        "the check fails whatever the coldest temperature"
    );
    assert!(
        design.metal.min_temp_C < num(d.cold_limit_C),
        "it fails here"
    );
    for (name, ndfeb) in probes::probe_designs() {
        if name == ferrite_design {
            continue;
        }
        let d = compute_all(&ndfeb).temperature.demag;
        assert_eq!(d.cold_limit_C, NumOrText::Text(NO_COLD_ONSET), "{name}");
        assert!(d.cold_check.starts_with(NO_COLD_ONSET), "{name}");
    }
    let [from, to] = y30.coefficient_range_C.expect("Y30's tested range");
    assert!(
        note.watch_out[1].contains(&format!("from {from} to {to} °C")),
        "{:?}",
        note.watch_out[1]
    );
    assert!(
        design.metal.min_temp_C < from,
        "extrapolated to the coldest"
    );
}

/// `ferrite_cold_demag`'s entries. Y30 on one ring makes the cold check apply on that ring while
/// the library NdFeB ring keeps the hot limit (it trades cold risk for hot), and the check fails
/// though the NdFeB ring has nothing to fail: it passes only if both rings pass. The trade-off
/// the first entry names is far less torque than a same-size library NdFeB part (under half the
/// pull-out). At every NdFeB probe design, Y30 on custom dimensions, on either ring or both,
/// leaves under half the pull-out and turns the hot-minimum check red where the NdFeB design just
/// passes it. At every probe design, Y30 on custom dimensions set to a library part's own size,
/// on either ring, leaves under half that part's pull-out, for every library part. "Same-size"
/// is needed: the manual dimensions keep their own values, so from M5045, a smaller part that
/// fits, Y30 on the example's inner ring does not leave under half the pull-out. An entered
/// like-poles field lower than the stored one lowers the cold limit: half the stored field still
/// fails, the slider's lowest passes.
#[test]
fn the_ferrite_note_s_entries_hold() {
    assert_eq!(levers_of("ferrite_cold_demag"), DEMAG_LEVERS);
    let entry = notes::note("ferrite_cold_demag")
        .expect("the ferrite note")
        .change[0]
        .text;
    assert!(
        entry.contains("far less torque"),
        "the Y30 entry names the torque it costs: {entry:?}"
    );
    assert!(
        entry.contains("same-size"),
        "the Y30 entry compares with a same-size library part: {entry:?}"
    );
    let mut ndfeb = 0;
    for (name, design) in probes::probe_designs() {
        let r = compute_all(&design);
        // Heat lowers an NdFeB knee (a negative coefficient); the Y30 design is skipped.
        if r.temperature.demag.beta_used_per_C > 0.0 {
            continue;
        }
        ndfeb += 1;
        // The requirement at the design's own hot-low, so the NdFeB design just passes.
        let mut just_passing = design.clone();
        just_passing.metal.required_min_Nm = r.metal.torque_hot_low_Nm;
        let required = just_passing.metal.required_min_Nm;
        let before = compute_all(&just_passing);
        assert!(
            before.metal.torque_hot_low_Nm >= required,
            "{name}: NdFeB passes"
        );
        for rings in [Rings::Inner, Rings::Outer, Rings::Both] {
            let after = compute_all(&on_grade(&just_passing, rings, "Y30"));
            assert!(
                after.model.pullout_Nm < before.model.pullout_Nm / 2.0,
                "{name} {rings:?}: far less torque"
            );
            assert!(
                after.metal.torque_hot_low_Nm < required,
                "{name} {rings:?}: hot-low falls below the requirement"
            );
            assert_ne!(
                after.metal.hot_min_check, before.metal.hot_min_check,
                "{name} {rings:?}: the hot-minimum check turns red"
            );
        }
    }
    assert_eq!(ndfeb, 4, "the four NdFeB probe designs");
    // A ring's block as the model uses it: length, width and thickness.
    let size = |m: &ModelResults, rings: Rings| match rings {
        Rings::Inner => [m.inner_length_mm, m.inner_width_mm, m.inner_thickness_mm],
        Rings::Outer => [m.outer_length_mm, m.outer_width_mm, m.outer_thickness_mm],
        Rings::Both => unreachable!("one ring at a time"),
    };
    for (name, design) in probes::probe_designs() {
        for part in &library::MAGNET_LIBRARY {
            for rings in [Rings::Inner, Rings::Outer] {
                let mut on_part = design.clone();
                let mut same_size = on_grade(&design, rings, "Y30");
                let (p, m) = (
                    &mut on_part.coupling.magnets,
                    &mut same_size.coupling.magnets,
                );
                match rings {
                    Rings::Inner => {
                        p.part_inner = part.part.to_owned();
                        m.manual_inner_length_mm = part.length_mm;
                        m.manual_inner_width_mm = part.width_mm;
                        m.manual_inner_thickness_mm = part.thickness_mm;
                    }
                    Rings::Outer => {
                        p.part_outer = part.part.to_owned();
                        m.manual_outer_length_mm = part.length_mm;
                        m.manual_outer_width_mm = part.width_mm;
                        m.manual_outer_thickness_mm = part.thickness_mm;
                    }
                    Rings::Both => unreachable!("one ring at a time"),
                }
                let (with_part, with_y30) =
                    (compute_all(&on_part).model, compute_all(&same_size).model);
                let what = format!("{name}, {} on the {rings:?} ring", part.part);
                assert_eq!(
                    size(&with_y30, rings),
                    size(&with_part, rings),
                    "{what}: same size"
                );
                assert!(
                    with_y30.pullout_Nm < with_part.pullout_Nm / 2.0,
                    "{what}: far less torque: {} -> {}",
                    with_part.pullout_Nm,
                    with_y30.pullout_Nm
                );
            }
        }
    }
    // Without "same-size" the entry is false: a part giving way to the grade leaves the manual
    // dimensions at their own values, so from M5045, a smaller part that fits, Y30 on the
    // example's inner ring does not leave under half the pull-out.
    let example = DesignInputs::default();
    let mut m5045 = example.clone();
    m5045.coupling.magnets.part_inner = "M5045".to_owned();
    let (from, to) = (
        compute_all(&m5045),
        compute_all(&on_grade(&m5045, Rings::Inner, "Y30")),
    );
    assert!(blocks_fit(&m5045.coupling, &from.model), "M5045 fits");
    let (part_mm, manual_mm) = (
        size(&from.model, Rings::Inner),
        size(&to.model, Rings::Inner),
    );
    assert!(
        part_mm.iter().zip(manual_mm).all(|(p, m)| *p < m),
        "the manual block is larger every way: {part_mm:?} against {manual_mm:?}"
    );
    assert!(
        to.model.pullout_Nm >= from.model.pullout_Nm / 2.0,
        "not far less torque: {} -> {}",
        from.model.pullout_Nm,
        to.model.pullout_Nm
    );
    let d0 = compute_all(&example).temperature.demag;
    assert_eq!(d0.cold_limit_C, NumOrText::Text(NO_COLD_ONSET));
    let ferrite = probe_inputs("Y30 ferrite, custom dimensions");
    let entered = |h: f64| {
        let mut d = ferrite.clone();
        d.temperature.demag.h_rev_likepole_kA_m = h;
        compute_all(&d).temperature.demag
    };
    let stored = ferrite.temperature.demag.h_rev_likepole_kA_m;
    let (at_stored, at_half, at_lowest) = (
        entered(stored),
        entered(stored / 2.0),
        entered(slider(LIKE_POLES).min),
    );
    let coldest = ferrite.metal.min_temp_C;
    assert!(num(at_half.cold_limit_C) < num(at_stored.cold_limit_C));
    assert!(num(at_lowest.cold_limit_C) < num(at_half.cold_limit_C));
    assert!(
        coldest < num(at_half.cold_limit_C),
        "half the stored field still fails"
    );
    assert!(coldest >= num(at_lowest.cold_limit_C), "the lowest passes");
    assert_eq!(at_half.cold_check, at_stored.cold_check);
    assert_ne!(at_lowest.cold_check, at_stored.cold_check);
    for (rings, ring) in [(Rings::Inner, RING_INNER), (Rings::Outer, RING_OUTER)] {
        let d = compute_all(&on_grade(&example, rings, "Y30"))
            .temperature
            .demag;
        assert_eq!(d.cold_ring, ring, "{rings:?}");
        assert_eq!(d.cold_limit_C, at_stored.cold_limit_C, "{rings:?}");
        assert_eq!(d.cold_check, at_stored.cold_check, "{rings:?}: fails");
        assert_eq!(
            d.magnet_limit_C, d0.magnet_limit_C,
            "{rings:?}: the NdFeB ring keeps the hot limit"
        );
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
/// temperature off 20 °C; a grade-mode ring with eleven harmonics at six poles; the
/// coercivity from the inputs with a bench drag entered; a hot-day start above the limit
/// (E12); and a dozen full-run cases under nine of the drift guard's augmentations (the
/// back-iron, grade, axial-override, ferrite and part-material branches of every chain).
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
        // The Hcj inputs act for both rings, and a bench drag replaces the loss estimate; a drag
        // of exactly 0 is no heating at all (E13).
        (
            "coercivity from the inputs, drag measured".to_owned(),
            design(&[
                ("temperature.demag.coercivity_source", Value::Int(0)),
                ("metal.measured_drag_Nm", Value::Num(0.02)),
            ]),
        ),
        (
            "drag measured as 0 (E13)".to_owned(),
            design(&[("metal.measured_drag_Nm", Value::Num(0.0))]),
        ),
        (
            "hot-day start above the limit (E12)".to_owned(),
            design(&[("temperature.duty.driving_rise_C", Value::Num(45.0))]),
        ),
    ];
    let augmentations: Vec<_> = augmentations()
        .into_iter()
        .filter(|(l, _)| {
            [
                "as generated",
                "harmonics 11",
                "back iron 1018",
                "back iron 6061",
                "grade mode",
                "axial 8 mm",
                "ferrite inputs",
                "unrated ferrite",
                "sleeve Ti, cap acetal",
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
    // Not vacuous: every input checked moves some explained result at some point.
    for input in paths {
        if !moved_pairs.iter().any(|(i, _)| i == input) {
            failures.push(format!(
                "{input} moves no explained result at any design point"
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
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two),
    // every one of them with explained results downstream now that every chain is explained.
    let r = Registry::build();
    let paths: Vec<String> = ASSUMPTIONS
        .iter()
        .flat_map(|a| a.paths.iter().map(|p| (*p).to_owned()))
        .collect();
    assert_eq!(paths.len(), 15);
    assert!(
        SCOPE.iter().all(|c| c.status == Status::Explained),
        "every chain of decision 31 is explained"
    );
    for p in &paths {
        assert!(
            !r.downstream(p).is_empty(),
            "{p}: no explained result depends on it"
        );
    }
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
fn records_never_panic_on_inputs_that_bypass_set() {
    // The records are proven over validated inputs (the drift guard sets every input through
    // `set`); a design file or share link can hold what `set` refuses (decision D3: the GUI
    // validates at its boundaries). Evaluating every record over such inputs, written straight
    // into the struct, gives a value or an EvalError, never a panic: an invalid harmonic code,
    // selector codes outside their choices, a NaN temperature and a NaN reserve.
    let r = Registry::build();
    let mut inputs = DesignInputs::default();
    inputs.coupling.max_harmonic = 4;
    inputs.temperature.adhesive.selected = 0;
    inputs.clamps.screw_class = 9;
    inputs.materials.parts.back_iron = 99;
    inputs.coupling.op_temp_C = f64::NAN;
    inputs.metal.max_diameter_mm = f64::NAN;
    assert!(
        inputs.validate().is_err(),
        "validate() names every one of them"
    );
    let results = compute_all(&inputs);
    let src = Design {
        inputs: &inputs,
        results: &results,
    };
    let (mut values, mut errors) = (0, 0);
    for eq in r.equations() {
        match r.evaluate(eq, &src, None) {
            Ok(_) => values += 1,
            Err(_) => errors += 1,
        }
    }
    // Every record returned (a value or an EvalError), and most give a value.
    assert_eq!(values + errors, r.equations().len());
    assert!(values > errors, "{values} values, {errors} errors");
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
    // The "used by" list follows the workbook's chain, with no hover dead end: Calculator C35
    // feeds each ring's coefficient (C69, C70), and the governing cold limit reads the cold
    // ring's skipping cold onset, as the magnet limit reads the skipping onset.
    for (path, reader) in [
        ("model.alpha_br_per_C", "model.inner_alpha_br_per_C"),
        ("model.alpha_br_per_C", "model.outer_alpha_br_per_C"),
        (
            "temperature.demag.cold_onset_skipping_C",
            "temperature.demag.cold_limit_C",
        ),
        (
            "temperature.demag.onset_skipping_C",
            "temperature.demag.magnet_limit_C",
        ),
    ] {
        assert!(
            r.used_by(path).iter().any(|u| u == reader),
            "{path} is not used by {reader}: {:?}",
            r.used_by(path)
        );
    }
}

// ---------------------------------------------------------------------------------------
// Guidance slice 2: live values in the notes, at the five probe designs, and the change
// entries' directions (spec 3.2 and section 6, tests 2 and 3)
// ---------------------------------------------------------------------------------------

/// The probe designs (`probes::probe_designs`), each with its name and its results.
fn probe_points() -> Vec<(&'static str, DesignInputs, DesignResults)> {
    probes::probe_designs()
        .into_iter()
        .map(|(name, inputs)| {
            let results = compute_all(&inputs);
            (name, inputs, results)
        })
        .collect()
}

/// A live value as these engine tests put it into note text: a finite number or a whole
/// number as Rust prints it; `None` (read as `note_text::NO_VALUE`) otherwise. The panel's
/// own text (four significant digits and the unit) is tested in `gui::explorer`.
fn engine_value(results: &DesignResults, path: &str) -> Option<String> {
    match results.get(path)? {
        Value::Num(x) if x.is_finite() => Some(x.to_string()),
        Value::Int(n) => Some(n.to_string()),
        Value::Num(_) | Value::Text(_) | Value::None => None,
    }
}

/// The placeholders in `text` followed by their own unit typed again: the value is drawn with
/// its unit, so `{model.pullout_Nm} N·m` would show the unit twice.
fn units_typed_after(text: &str, unit_of: &dyn Fn(&str) -> &'static str) -> Vec<String> {
    let Ok(pieces) = note_text::segments(text) else {
        return Vec::new();
    };
    pieces
        .windows(2)
        .filter_map(|pair| match pair {
            [
                note_text::Segment::Value(path),
                note_text::Segment::Text(after),
            ] => {
                let unit = unit_of(path);
                let typed = !unit.is_empty() && unit != "-" && after.trim_start().starts_with(unit);
                typed.then(|| (*path).to_owned())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn a_unit_typed_after_a_live_value_is_found() {
    let unit_of = |path: &str| {
        if path == "model.pullout_Nm" {
            "N·m"
        } else {
            ""
        }
    };
    assert_eq!(
        units_typed_after("Here {model.pullout_Nm} N·m.", &unit_of),
        ["model.pullout_Nm"]
    );
    assert!(units_typed_after("Here {model.pullout_Nm}.", &unit_of).is_empty());
    assert!(units_typed_after("Here {model.f_end} at the ends.", &unit_of).is_empty());
}

/// Whether `text` holds a live value outside a "here" clause (spec 3.2): a number stands where
/// the text says "here" or "on this design", so it reads as this design's, not as a constant.
/// Both match as whole words (the runs of letters, in any case), so "where", "there" and
/// "upon this design" do not count.
fn live_value_outside_a_here_clause(text: &str) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .collect();
    let here_clause =
        words.contains(&"here") || words.windows(3).any(|w| w == ["on", "this", "design"]);
    text.contains('{') && !here_clause
}

#[test]
fn a_live_value_outside_a_here_clause_is_found() {
    assert!(live_value_outside_a_here_clause(
        "The pull-out is {model.pullout_Nm}."
    ));
    assert!(!live_value_outside_a_here_clause(
        "Here the pull-out is {model.pullout_Nm}."
    ));
    assert!(!live_value_outside_a_here_clause(
        "The end factor is {model.f_end} on this design."
    ));
    assert!(!live_value_outside_a_here_clause(
        "No live value in this one."
    ));
    // "here" counts as a whole word only: where, there, somewhere and adhere hold no "here"
    // clause, and "on this design" counts as those three words.
    for outside in [
        "Where the gap closes, the pull-out is {model.pullout_Nm}.",
        "There the pull-out is {model.pullout_Nm}.",
        "Somewhere the pull-out is {model.pullout_Nm}.",
        "The magnets adhere; the pull-out is {model.pullout_Nm}.",
        "Upon this design the pull-out is {model.pullout_Nm}.",
    ] {
        assert!(live_value_outside_a_here_clause(outside), "{outside}");
    }
    for inside in [
        "The pull-out, here, is {model.pullout_Nm}.",
        "The pull-out (here: {model.pullout_Nm}) falls.",
        "On this\ndesign the pull-out is {model.pullout_Nm}.",
    ] {
        assert!(!live_value_outside_a_here_clause(inside), "{inside}");
    }
}

/// Each result's type and unit, by path.
fn result_kinds(results: &DesignResults) -> BTreeMap<String, (FieldType, &'static str)> {
    result_rows(results)
        .into_iter()
        .map(|r| (r.path, (r.meta.ty, r.meta.unit)))
        .collect()
}

/// What is wrong with `note`'s live values (spec 3.2), each fault named: a brace in a title, a
/// sentence 0 or a source (no live value stands there); a text the parser refuses; a unit typed
/// after a placeholder; a placeholder that names no result or no number result; a placeholder
/// outside a "here" clause; a brace left after rendering at a probe design.
fn placeholder_problems(
    note: &notes::Note,
    kinds: &BTreeMap<String, (FieldType, &'static str)>,
    points: &[(&str, DesignInputs, DesignResults)],
) -> Vec<String> {
    let unit_of = |path: &str| kinds.get(path).map_or("", |(_, unit)| *unit);
    let mut failures = Vec::new();
    let plain = [note.title]
        .into_iter()
        .chain(note.sentences.first().copied())
        .chain(note.sources.iter().copied());
    for text in plain {
        if text.contains(['{', '}']) {
            failures.push(format!("{}: a brace in {text:?}", note.id));
        }
    }
    for text in note_text::rendered_texts(note) {
        if let Err(fault) = note_text::segments(text) {
            failures.push(format!("{}: {fault}", note.id));
        }
        for path in units_typed_after(text, &unit_of) {
            failures.push(format!("{}: the unit typed after {{{path}}}", note.id));
        }
        if live_value_outside_a_here_clause(text) {
            failures.push(format!(
                "{}: a live value outside a 'here' clause: {text:?}",
                note.id
            ));
        }
    }
    for path in note_text::placeholders(note) {
        match kinds.get(path) {
            None => failures.push(format!("{}: {{{path}}} names no result", note.id)),
            Some((FieldType::F64 | FieldType::I64 | FieldType::NumOrText, _)) => {}
            Some((other, _)) => failures.push(format!(
                "{}: {{{path}}} is a {other:?} result, not a number",
                note.id
            )),
        }
    }
    for (name, _, results) in points {
        for text in note_text::rendered_texts(note) {
            let drawn = note_text::render(text, |path| engine_value(results, path));
            if drawn.contains(['{', '}']) {
                failures.push(format!("{} at the {name} design: {drawn:?}", note.id));
            }
        }
    }
    failures
}

#[test]
fn every_note_placeholder_names_a_number_result_and_renders_at_every_probe_design() {
    let points = probe_points();
    let kinds = result_kinds(&points[0].2);
    let failures: Vec<String> = notes::NOTES
        .iter()
        .flat_map(|note| placeholder_problems(note, &kinds, &points))
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn the_placeholder_check_names_each_kind_of_mistake() {
    let points = probe_points();
    let kinds = result_kinds(&points[0].2);
    let good = notes::Note {
        id: "test.placeholders",
        title: "A title",
        equations: &[],
        sentences: &[
            "A plain first sentence.",
            "Here the pull-out is {model.pullout_Nm}.",
            "On this design the end factor is {model.f_end}.",
        ],
        change: &[Change {
            input: "metal.face_gap_mm",
            step: Step::Lower,
            aim: Some(("model.pullout_Nm", Sense::Up)),
            text: "the pull-out here is {model.pullout_Nm}.",
        }],
        watch_out: &[],
        diagram: None,
        sources: &["A source."],
        review: notes::Review::Draft,
    };
    assert_eq!(
        placeholder_problems(&good, &kinds, &points),
        Vec::<String>::new()
    );
    for (bad, fault) in [
        (
            notes::Note {
                title: "A {model.f_end} title",
                ..good
            },
            "a brace in",
        ),
        (
            notes::Note {
                sentences: &["First {model.f_end}.", "Here {model.f_end}.", "Third."],
                ..good
            },
            "a brace in",
        ),
        (
            notes::Note {
                sources: &["A {model.f_end} source."],
                ..good
            },
            "a brace in",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {model.f_end", "Third."],
                ..good
            },
            "with no '}' after it",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {model.f_end", "Third."],
                ..good
            },
            "at the default design",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {= model.a / model.b}.", "Third."],
                ..good
            },
            "is not a result path",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {model.pullout_Nm} N·m.", "Third."],
                ..good
            },
            "the unit typed after {model.pullout_Nm}",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {model.nothing}.", "Third."],
                ..good
            },
            "{model.nothing} names no result",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "Here {materials.cup_wall_check}.", "Third."],
                ..good
            },
            "{materials.cup_wall_check} is a Text result, not a number",
        ),
        (
            notes::Note {
                sentences: &["Plain.", "The pull-out is {model.pullout_Nm}.", "Third."],
                ..good
            },
            "a live value outside a 'here' clause",
        ),
    ] {
        let problems = placeholder_problems(&bad, &kinds, &points);
        assert!(
            problems.iter().any(|p| p.contains(fault)),
            "{fault}: {problems:?}"
        );
    }
}

#[test]
fn every_note_reads_in_plain_words_at_every_probe_design() {
    let mut failures = Vec::new();
    for (name, _, results) in &probe_points() {
        for note in notes::NOTES.iter().filter(|n| notes::shown(n.id)) {
            // The rule's audit-id check is the one that keeps audit ids out of the shipped
            // notes' texts and sources (A-13, RF-5): the guard does not know them.
            let problems = notes::plain_words_problems(note, |path| engine_value(results, path));
            failures.extend(
                problems
                    .into_iter()
                    .map(|problem| format!("at the {name} design, {problem}")),
            );
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// One change entry's aim checked at `points`: at each design where the input moves one
/// slider step its way (`probes::step_input`) and the aimed result is a finite number before
/// and after, the result must move the way the entry says. The failures, and at how many
/// designs the aim was checked.
fn aim_failures(
    id: &str,
    change: &Change,
    points: &[(&str, DesignInputs, DesignResults)],
) -> (Vec<String>, usize) {
    let Some((aim, sense)) = change.aim else {
        return (Vec::new(), 0);
    };
    let mut failures = Vec::new();
    let mut checked = 0;
    for (name, inputs, results) in points {
        let Ok(stepped) = probes::step_input(inputs, change.input, change.step) else {
            continue;
        };
        let (Some(Value::Num(before)), Some(Value::Num(after))) =
            (results.get(aim), compute_all(&stepped).get(aim))
        else {
            continue;
        };
        if !(before.is_finite() && after.is_finite()) {
            continue;
        }
        checked += 1;
        if !sense.holds(before, after) {
            failures.push(format!(
                "{id}: {} {} should move {aim} {sense:?} at the {name} design: {before} -> {after}",
                change.step.word(),
                change.input
            ));
        }
    }
    (failures, checked)
}

#[test]
fn every_change_entry_moves_its_aim_the_way_it_says() {
    let points = probe_points();
    let mut failures = Vec::new();
    for note in notes::NOTES {
        for change in note.change.iter().filter(|c| c.aim.is_some()) {
            let (found, checked) = aim_failures(note.id, change, &points);
            failures.extend(found);
            if checked == 0 {
                failures.push(format!(
                    "{}: {} {} is checked at no probe design (held, at its end or no number at all five)",
                    note.id,
                    change.step.word(),
                    change.input
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn the_aim_check_confirms_a_true_direction_and_catches_a_false_one() {
    let points = probe_points();
    let entry = |input, step, aim| Change {
        input,
        step,
        aim: Some(aim),
        text: "",
    };
    // True at every probe design: a smaller gap, longer magnets (stepped from the length in
    // use, the axial length being blank), and a hotter magnet that weakens.
    for truth in [
        entry(
            "metal.face_gap_mm",
            Step::Lower,
            ("model.pullout_Nm", Sense::Up),
        ),
        entry(
            "coupling.magnets.axial_length_mm",
            Step::Raise,
            ("model.pullout_Nm", Sense::Up),
        ),
        entry(
            "coupling.op_temp_C",
            Step::Raise,
            ("model.pullout_Nm", Sense::Down),
        ),
    ] {
        let (failures, checked) = aim_failures("test", &truth, &points);
        assert_eq!(failures, Vec::<String>::new(), "{truth:?}");
        assert_eq!(checked, points.len(), "{truth:?}");
    }
    let false_claim = entry(
        "metal.face_gap_mm",
        Step::Lower,
        ("model.pullout_Nm", Sense::Down),
    );
    let (failures, checked) = aim_failures("test", &false_claim, &points);
    assert_eq!((failures.len(), checked), (points.len(), points.len()));
    // An input a pick sets at every probe design is checked nowhere: the manual Br (the
    // library part sets it, or on the ferrite rings the grade).
    let held = entry(
        "coupling.magnets.manual_inner_br_T",
        Step::Raise,
        ("model.pullout_Nm", Sense::Up),
    );
    assert_eq!(aim_failures("test", &held, &points), (Vec::new(), 0));
}
