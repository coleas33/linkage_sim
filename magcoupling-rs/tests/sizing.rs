//! Addendum A1: the axial length override (Task 5), inverse sizing (Task 6) and the space
//! claim (Task 7), end to end through `compute_all`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::housing::{INSIDE_THE_SPACE_CLAIM, SPACE_CLAIM_UNKNOWN};
use magcoupling::engine::meta::NumOrText;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
    is_valid, solve,
};

/// The hot-low torque with production variation of a design (what sizing makes meet the target).
fn hot_low(inputs: &DesignInputs) -> f64 {
    compute_all(inputs).metal.torque_hot_low_Nm
}

/// The solved point, or a panic naming the outcome.
fn solved(outcome: Result<SizingOutcome, SizingError>) -> SizingPoint {
    match outcome {
        Ok(SizingOutcome::Solved(point)) => point,
        other => panic!("expected a solution, got {other:?}"),
    }
}

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
    // Decision A2-3: the measured correction keys on the part names, as the workbook's C42
    // does, so a length override keeps it for the prototype's rings with no back iron (no
    // step in torque as a sized length passes the prototype's 12.7 mm). The magnets' mass
    // follows the length; the housing inputs do not (decision 28: no rule sizes them).
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 0;
    let at_part = compute_all(&inputs);
    inputs.coupling.magnets.axial_length_mm = Some(20.0);
    let long = compute_all(&inputs);
    assert_eq!(at_part.model.f_cal, at_part.calibration.f_cal_updated);
    assert_eq!(long.model.f_cal, at_part.model.f_cal);
    assert!((long.mass.magnets_g / at_part.mass.magnets_g - 20.0 / 12.7).abs() < 1e-12);
    assert_eq!(long.metal.axial_stack_mm, at_part.metal.axial_stack_mm);
    assert_eq!(
        long.retainers.retainer_span_mm,
        at_part.retainers.retainer_span_mm
    );
}

#[test]
fn each_free_variable_round_trips() {
    // Spec "Addendum testing": solving for the forward result's torque returns the original
    // value (1e-6). Each base design is one where that value is also the smallest that meets
    // the torque: the hot-low torque grows with the length (~ L - c_end * pole pitch) at any
    // length; T(4), T(6) and T(8) poles are below T(10) and T(8) on the default hub (0.40,
    // 0.79, 1.67 against 2.29 N m); the torque rises with the ring radius from the smallest
    // radius whose flats fit (9.77 mm) to about 14 mm.
    let cases: [(FreeVariable, &[f64]); 3] = [
        (FreeVariable::AxialLength, &[6.0, 12.7, 25.0]),
        (FreeVariable::MagnetsPerRing, &[8.0, 10.0]),
        (FreeVariable::RingRadius, &[10.15, 12.5]),
    ];
    for (variable, values) in cases {
        for &v0 in values {
            let base = variable.apply(&DesignInputs::default(), v0);
            let target = hot_low(&base);
            let p = solved(solve(&base, variable, target));
            assert!(
                (p.value - v0).abs() <= 1e-6,
                "{variable:?} {v0}: {}",
                p.value
            );
            assert!(p.torque_hot_low_Nm >= target, "{variable:?} {v0}");
            // Every other input stays fixed.
            assert_eq!(
                p.inputs,
                variable.apply(&base, p.value),
                "{variable:?} {v0}"
            );
            assert_eq!(p.torque_hot_low_Nm, hot_low(&p.inputs));
        }
    }
}

#[test]
fn a_target_beyond_the_range_is_not_reachable() {
    // Spec: "not reachable" with the best value achieved inside the variable's slider range;
    // it never extrapolates past the range.
    for variable in FreeVariable::ALL {
        let range = variable.range();
        match solve(&DesignInputs::default(), variable, 10.0) {
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => {
                assert!(
                    (range.min..=range.max).contains(&best.value),
                    "{variable:?}: {}",
                    best.value
                );
                assert!(best.torque_hot_low_Nm < 10.0, "{variable:?}");
                assert_eq!(
                    best.inputs,
                    variable.apply(&DesignInputs::default(), best.value)
                );
                // The best of the values the search tried: no valid grid value does better.
                for value in variable.grid() {
                    let design = variable.apply(&DesignInputs::default(), value);
                    let r = compute_all(&design);
                    if is_valid(&design, &r) {
                        assert!(
                            r.metal.torque_hot_low_Nm <= best.torque_hot_low_Nm,
                            "{variable:?} {value}"
                        );
                    }
                }
                // The ring radius's torque peaks between two grid values: its best is the
                // refined peak, not a grid value.
                if variable == FreeVariable::RingRadius {
                    assert!(!variable.grid().contains(&best.value), "{}", best.value);
                }
            }
            other => panic!("{variable:?}: {other:?}"),
        }
    }
    // The length's torque grows with the length: its best is the slider's end, 50.8 mm.
    match solve(&DesignInputs::default(), FreeVariable::AxialLength, 10.0) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 50.8),
        other => panic!("{other:?}"),
    }
}

#[test]
fn poles_stay_even() {
    // Spec: magnets per ring is discrete and the poles stay even, whatever the base design holds
    // (an odd pole count set straight on the struct: set() checks no step grid).
    let grid = FreeVariable::MagnetsPerRing.grid();
    assert_eq!(grid.first(), Some(&4.0));
    assert_eq!(grid.last(), Some(&40.0));
    assert!(grid.iter().all(|n| n % 2.0 == 0.0), "{grid:?}");
    let mut odd = DesignInputs::default();
    odd.coupling.npole = 11;
    for target in [0.3, 1.0, 1.7, 2.2] {
        let p = solved(solve(&odd, FreeVariable::MagnetsPerRing, target));
        assert_eq!(p.value % 2.0, 0.0, "{target}: {}", p.value);
        assert_eq!(p.inputs.coupling.npole % 2, 0, "{target}");
        assert_eq!(p.inputs.coupling.npole as f64, p.value);
    }
}

#[test]
fn the_smaller_of_two_meeting_intervals_is_returned() {
    // The torque is not monotone in the ring radius: on the default design it rises to about
    // 14 mm, falls to about 20 mm and rises again to the slider's end. At 2.5 N m both
    // 11.7 to 15.7 mm and 29.8 to 30 mm meet; the smallest value is the answer.
    let p = solved(solve(
        &DesignInputs::default(),
        FreeVariable::RingRadius,
        2.5,
    ));
    assert!(p.value > 11.0 && p.value < 13.0, "{}", p.value);
    let at =
        |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&DesignInputs::default(), radius));
    assert!(at(p.value) >= 2.5);
    assert!(
        at(p.value - 1e-6) < 2.5,
        "the boundary, to the bisection's tolerance"
    );
    assert!(at(30.0) >= 2.5 && at(20.0) < 2.5, "a second interval meets");
}

#[test]
fn a_meeting_interval_inside_one_cell_is_found() {
    // With no back iron and a 0.5 mm face gap the torque peaks near 12.12 mm of ring radius,
    // between two grid values (11.953125 and 12.28125 mm) that both miss 2.0695 N m: the
    // interval that meets lies inside one cell. The peak search finds it, and the answer is
    // its lower end, not the later interval near the slider's end (29.47 mm).
    let mut base = DesignInputs::default();
    base.coupling.backiron = 0;
    base.metal.face_gap_mm = 0.5;
    let target = 2.0695;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!(p.value > 12.0 && p.value < 12.2, "{}", p.value);
    assert!(at(p.value) >= target);
    assert!(
        at(p.value - 1e-6) < target,
        "the lower end, to the bisection's tolerance"
    );
    // Both grid values around it miss: the case the coarse scan alone cannot see.
    let grid = FreeVariable::RingRadius.grid();
    let cell = grid
        .windows(2)
        .find(|w| w[0] <= p.value && p.value <= w[1])
        .unwrap();
    assert!(at(cell[0]) < target && at(cell[1]) < target, "{cell:?}");
}

#[test]
fn a_target_just_below_the_true_peak_is_solved() {
    // The default design's torque peaks at 2.59511225 N m near 13.591 mm of ring radius; the
    // nearest grid value, 13.59375 mm, reads 2.59511211. A target between the two is met only
    // near the peak, which no grid value reaches.
    let base = DesignInputs::default();
    let target = 2.5951122;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    assert!(
        FreeVariable::RingRadius
            .grid()
            .into_iter()
            .all(|x| at(x) < target)
    );
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!(p.value > 13.5 && p.value < 13.7, "{}", p.value);
    assert!(at(p.value) >= target && at(p.value - 1e-6) < target);
}

#[test]
fn a_target_met_only_where_the_keyway_starts_to_leave_wall_is_found() {
    // A 24 mm bore with a 4 mm keyway leaves hub wall past the key only above 16.05 mm of ring
    // radius (16.05 - 0.05 bond - 12 - 4), where the torque is already falling: 2.46 N m is
    // met just above that edge, missed at the next grid value and met again from about 29 mm.
    // The validity edge is sampled, so the answer is the edge, not the later interval.
    let mut base = DesignInputs::default();
    base.coupling.bore_mm = 24.0;
    base.coupling.keyway_depth_mm = 4.0;
    let target = 2.46;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    let next = FreeVariable::RingRadius
        .grid()
        .into_iter()
        .find(|&x| x > 16.05)
        .unwrap();
    assert!(at(next) < target && at(29.0) >= target, "{next}");
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!((p.value - 16.05).abs() <= 1e-6, "{}", p.value);
    assert!(compute_all(&p.inputs).model.hub_wall_past_key_mm > 0.0);
    assert!(p.torque_hot_low_Nm >= target);
}

#[test]
fn short_lengths_where_f_end_is_not_positive_never_count() {
    // Audit M9: with c_end = 0.5 the end-effect factor is 0 or negative below about 4.4 mm
    // (c_end x pole pitch), and so is the torque. A tiny target is met just above that length,
    // never inside the out-of-range region, and the solved design reads "OK".
    let mut base = DesignInputs::default();
    base.coupling.c_end = 0.5;
    let short = compute_all(&FreeVariable::AxialLength.apply(&base, 2.0));
    assert_eq!(short.model.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    assert!(short.metal.torque_hot_low_Nm < 0.0);
    let p = solved(solve(&base, FreeVariable::AxialLength, 1e-6));
    let r = compute_all(&p.inputs);
    assert_eq!(r.model.end_effect_check, "OK");
    let pitch = r.model.pole_pitch_mm;
    assert!(
        p.value > 0.5 * pitch && p.value < 0.5 * pitch + 0.01,
        "{} vs {}",
        p.value,
        0.5 * pitch
    );
}

#[test]
fn nothing_valid_in_the_range_reports_no_best_value() {
    // 25.4 mm wide manual blocks on the default hub fit no even pole count from 4 up (the inner
    // flat is 2 x 10.15 x tan(pi/N) mm): every value is invalid, so there is no best value.
    let mut base = DesignInputs::default();
    base.coupling.magnets.part_inner = String::new();
    base.coupling.magnets.part_outer = String::new();
    base.coupling.magnets.manual_inner_width_mm = 25.4;
    base.coupling.magnets.manual_outer_width_mm = 25.4;
    assert_eq!(
        solve(&base, FreeVariable::MagnetsPerRing, 0.1),
        Ok(SizingOutcome::NotReachable { best: None })
    );
}

#[test]
fn overlapping_arcs_never_count() {
    // Decision A2-4: arcs fit when their blocks do not overlap at the magnet mid-radius. On the
    // default hub 12 arcs of 6.35 mm share a 6.14 mm pitch there (the fill C66 clamps the
    // share to 1 and prices them as if they fitted): they never count, so 2.3 N m (between
    // 10 poles' 2.285 and 12 poles' 2.541 N m) is not reachable and the best is 10 poles.
    let mut arcs = DesignInputs::default();
    arcs.coupling.faceted = 0;
    let twelve = FreeVariable::MagnetsPerRing.apply(&arcs, 12.0);
    let r = compute_all(&twelve);
    let share = pitch_share(
        r.model.inner_width_mm,
        twelve.coupling.inner_back_apothem_mm,
        r.model.inner_thickness_mm,
        12.0,
    );
    assert!(share > 1.0 && r.model.fill_inner == 1.0, "{share}");
    assert!(!blocks_fit(&twelve.coupling, &r.model));
    assert!(r.metal.torque_hot_low_Nm >= 2.3);
    match solve(&arcs, FreeVariable::MagnetsPerRing, 2.3) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 10.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn arcs_fit_exactly_at_a_pitch_share_of_one() {
    // The equality edge of the arc rule (a share of at most 1): manual inner arcs exactly one
    // mid-radius pitch wide fit; the next wider double does not.
    let mut d = DesignInputs::default();
    d.coupling.faceted = 0;
    d.coupling.magnets.part_inner = String::new();
    let a = d.coupling.inner_back_apothem_mm;
    let t = d.coupling.magnets.manual_inner_thickness_mm;
    let n = d.coupling.npole as f64;
    let pitch = 2.0 * std::f64::consts::PI * (a + t / 2.0) / n;
    d.coupling.magnets.manual_inner_width_mm = pitch;
    let r = compute_all(&d);
    let share = |r: &magcoupling::DesignResults| {
        pitch_share(r.model.inner_width_mm, a, r.model.inner_thickness_mm, n)
    };
    assert_eq!(share(&r), 1.0, "the equality");
    assert!(blocks_fit(&d.coupling, &r.model));
    d.coupling.magnets.manual_inner_width_mm = pitch.next_up();
    let r = compute_all(&d);
    assert!(share(&r) > 1.0);
    assert!(!blocks_fit(&d.coupling, &r.model));
}

#[test]
fn a_hub_the_keyway_breaks_through_never_counts() {
    // Decision A2-4: a keyway that leaves no hub wall (Calculator C53 <= 0) is not a design.
    // With a 16 mm bore and a 2.5 mm keyway the wall past the key is positive only above
    // 10.55 mm of ring radius (10.55 - 0.05 bond - 8 - 2.5), though the flats fit from
    // 9.77 mm: 0.5 N m is met at that edge.
    let mut base = DesignInputs::default();
    base.coupling.bore_mm = 16.0;
    base.coupling.keyway_depth_mm = 2.5;
    let p = solved(solve(&base, FreeVariable::RingRadius, 0.5));
    assert!(compute_all(&p.inputs).model.hub_wall_past_key_mm > 0.0);
    assert!((p.value - 10.55).abs() <= 1e-6, "{}", p.value);
    let below = FreeVariable::RingRadius.apply(&base, p.value - 1e-6);
    assert!(!is_valid(&below, &compute_all(&below)));
}

#[test]
fn the_hub_wall_counts_only_while_positive() {
    // The equality edge of the hub rule (wall past the keyway > 0): a keyway exactly as deep as
    // the hub wall leaves none and does not count; the next shallower double leaves some.
    let mut d = DesignInputs::default();
    let wall = compute_all(&d).model.hub_wall_mm;
    d.coupling.keyway_depth_mm = wall;
    let r = compute_all(&d);
    assert_eq!(r.model.hub_wall_past_key_mm, 0.0, "the equality");
    assert!(!is_valid(&d, &r));
    d.coupling.keyway_depth_mm = wall.next_down();
    let r = compute_all(&d);
    assert!(r.model.hub_wall_past_key_mm > 0.0);
    assert!(is_valid(&d, &r));
}

#[test]
fn invalid_targets_and_inputs_fail_loudly() {
    for target in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for variable in FreeVariable::ALL {
            match solve(&DesignInputs::default(), variable, target) {
                Err(SizingError::InvalidTarget(t)) => {
                    assert!(t == target || (t.is_nan() && target.is_nan()))
                }
                other => panic!("{target} {variable:?}: {other:?}"),
            }
        }
    }
    let mut bad = DesignInputs::default();
    bad.coupling.max_harmonic = 4; // outside its choices, set on the struct
    bad.metal.face_gap_mm = f64::NAN;
    match solve(&bad, FreeVariable::AxialLength, 2.0) {
        Err(SizingError::InvalidInputs(errors)) => {
            let found: Vec<(&str, &SetErrorKind)> =
                errors.iter().map(|e| (e.path.as_str(), &e.kind)).collect();
            assert_eq!(
                found,
                [
                    (
                        "coupling.max_harmonic",
                        &SetErrorKind::NotAChoice { code: 4 }
                    ),
                    ("metal.face_gap_mm", &SetErrorKind::NotFinite),
                ]
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_default_design_sized_to_its_requirement() {
    // The default design's hot low torque, 2.285 N m, misses the 2.5 N m requirement ("Below hot
    // minimum"). Sized by length it meets it a little above 12.7 mm; sized by radius it meets it
    // with a cup wider than the 43 mm claim; sized by poles it cannot: 12 poles no longer fit
    // the default hub, so the best is the design's own 10.
    let base = DesignInputs::default();
    let required = base.metal.required_min_Nm;
    assert!(hot_low(&base) < required);
    let p = solved(solve(&base, FreeVariable::AxialLength, required));
    assert!(p.value > 12.7 && p.value < 20.0, "{}", p.value);
    assert_eq!(
        compute_all(&p.inputs).metal.hot_min_check,
        "Estimate covers hot min"
    );
    let p = solved(solve(&base, FreeVariable::RingRadius, required));
    assert!(
        compute_all(&p.inputs).metal.diameter_reserve_mm < 0.0,
        "{}",
        p.value
    );
    match solve(&base, FreeVariable::MagnetsPerRing, required) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 10.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_search_constants_are_the_documented_ones() {
    assert_eq!((SCAN_CELLS, VALUE_TOLERANCE_MM), (64, 1e-9));
    assert_eq!(
        FreeVariable::ALL[0],
        FreeVariable::AxialLength,
        "the default"
    );
    for variable in FreeVariable::ALL {
        let grid = variable.grid();
        let range = variable.range();
        assert_eq!(
            (grid[0], *grid.last().unwrap()),
            (range.min, range.max),
            "{variable:?}"
        );
        if variable != FreeVariable::MagnetsPerRing {
            assert_eq!(grid.len(), SCAN_CELLS + 1, "{variable:?}");
        }
    }
    assert_eq!(
        FreeVariable::AxialLength.path(),
        "coupling.magnets.axial_length_mm"
    );
    assert_eq!(FreeVariable::MagnetsPerRing.path(), "coupling.npole");
    assert_eq!(
        FreeVariable::RingRadius.path(),
        "coupling.inner_back_apothem_mm"
    );
}

#[test]
fn the_default_design_is_inside_its_space_claim() {
    // Report 6.5: rotating OD 42.8 mm (set by the cap) of 43; stack 31.8 of 35; large-diameter
    // stack 18.8 of the 20 mm bay. The autofit suggests the rule's 2.0 mm cup wall (decision 27).
    let r = compute_all(&DesignInputs::default());
    let h = &r.housing;
    assert_eq!(
        (
            h.diameter_overshoot_mm,
            h.length_overshoot_mm,
            h.bay_overshoot_mm
        ),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(h.space_claim_check, INSIDE_THE_SPACE_CLAIM);
    assert!((r.metal.diameter_reserve_mm - 0.2).abs() < 1e-12);
    assert!((r.metal.axial_reserve_mm - 3.2).abs() < 1e-12);
    assert!((r.metal.large_dia_reserve_mm - 1.2).abs() < 1e-12);
    assert_eq!(r.materials.cup_wall_suggested_mm, NumOrText::Num(2.0));
}

#[test]
fn the_space_claim_is_exceeded_exactly_when_a_dimension_exceeds_it_per_axis() {
    // Spec "Addendum testing": the envelope-exceeded callout triggers exactly when a derived
    // dimension exceeds the space claim, per axis. Each claim is put at its dimension exactly
    // (the comparison at equality, asserted first), then 0.5 mm under and over it.
    type Claim = fn(&mut DesignInputs, f64);
    type Read = fn(&magcoupling::DesignResults) -> (f64, f64); // (dimension, overshoot)
    let axes: [(&str, Claim, Read); 3] = [
        (
            "diameter",
            |i, x| i.metal.max_diameter_mm = x,
            |r| (r.metal.rotating_od_mm, r.housing.diameter_overshoot_mm),
        ),
        (
            "overall length",
            |i, x| i.metal.max_overall_axial_mm = x,
            |r| (r.metal.axial_stack_mm, r.housing.length_overshoot_mm),
        ),
        (
            "large-diameter bay",
            |i, x| i.metal.max_large_dia_axial_mm = x,
            |r| (r.metal.large_dia_stack_mm, r.housing.bay_overshoot_mm),
        ),
    ];
    for (axis, set_claim, read) in axes {
        let (dimension, _) = read(&compute_all(&DesignInputs::default()));
        let at_claim = |claim: f64| {
            let mut inputs = DesignInputs::default();
            set_claim(&mut inputs, claim);
            compute_all(&inputs)
        };
        let r = at_claim(dimension);
        assert_eq!(
            read(&r).0,
            dimension,
            "{axis}: the claim sits at the dimension"
        );
        assert_eq!(read(&r).1, 0.0, "{axis}: at the claim is inside");
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{axis}"
        );
        let r = at_claim(dimension - 0.5);
        assert_eq!(read(&r).1, dimension - (dimension - 0.5), "{axis}");
        assert_eq!(
            r.housing.space_claim_check,
            format!("Exceeds the space claim: {axis} 0.50 mm over"),
            "only this axis"
        );
        let r = at_claim(dimension + 0.5);
        assert_eq!(read(&r).1, 0.0, "{axis}");
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{axis}"
        );
    }
}

#[test]
fn a_sized_design_shows_its_overshoot() {
    // Both modes: the space claim is part of every forward calculation, so a design inverse
    // sizing returns shows it too. Sized by ring radius to the 2.5 N m requirement, the cup
    // grows past the 43 mm claim; the axial stack does not move.
    let base = DesignInputs::default();
    let p = solved(solve(
        &base,
        FreeVariable::RingRadius,
        base.metal.required_min_Nm,
    ));
    let r = compute_all(&p.inputs);
    let over = r.metal.rotating_od_mm - base.metal.max_diameter_mm;
    assert!(over > 0.0, "{over}");
    assert_eq!(r.housing.diameter_overshoot_mm, over);
    assert_eq!(
        (r.housing.length_overshoot_mm, r.housing.bay_overshoot_mm),
        (0.0, 0.0)
    );
    assert!(
        r.housing
            .space_claim_check
            .starts_with("Exceeds the space claim: diameter "),
        "{}",
        r.housing.space_claim_check
    );
}

#[test]
fn several_axes_are_named_in_order_and_nan_is_unknown() {
    let mut inputs = DesignInputs::default();
    inputs.metal.max_diameter_mm = 40.0;
    inputs.metal.max_large_dia_axial_mm = 18.0;
    let h = compute_all(&inputs).housing;
    assert_eq!(
        h.space_claim_check,
        "Exceeds the space claim: diameter 2.80 mm over, large-diameter bay 0.80 mm over"
    );
    // A claim that is not a number (set on the struct: validate() names it) is not "inside",
    // and it hides no known overshoot: the axis reads unknown among the exceeded ones.
    inputs.metal.max_overall_axial_mm = f64::NAN;
    let h = compute_all(&inputs).housing;
    assert!(h.length_overshoot_mm.is_nan());
    assert_eq!(
        h.space_claim_check,
        "Exceeds the space claim: diameter 2.80 mm over, overall length unknown, \
         large-diameter bay 0.80 mm over"
    );
    // With nothing exceeded, a NaN axis makes the whole claim unknown.
    let mut inputs = DesignInputs::default();
    inputs.metal.max_overall_axial_mm = f64::NAN;
    assert_eq!(
        compute_all(&inputs).housing.space_claim_check,
        SPACE_CLAIM_UNKNOWN
    );
}

#[test]
fn a_tiny_overshoot_reads_at_least_a_hundredth() {
    // Any dimension past its claim reads at least "0.01 mm over", never "0.00 mm over": each
    // claim 0.001 mm under its dimension (the overshoot itself stays exact).
    type Claim = fn(&mut DesignInputs, f64);
    let axes: [(&str, Claim, f64); 3] = [
        ("diameter", |i, x| i.metal.max_diameter_mm = x, 42.8),
        (
            "overall length",
            |i, x| i.metal.max_overall_axial_mm = x,
            31.8,
        ),
        (
            "large-diameter bay",
            |i, x| i.metal.max_large_dia_axial_mm = x,
            18.8,
        ),
    ];
    let base = compute_all(&DesignInputs::default());
    let dimensions = [
        base.metal.rotating_od_mm,
        base.metal.axial_stack_mm,
        base.metal.large_dia_stack_mm,
    ];
    for ((axis, set_claim, nominal), dimension) in axes.into_iter().zip(dimensions) {
        assert!((dimension - nominal).abs() < 1e-12, "{axis}: {dimension}");
        let mut inputs = DesignInputs::default();
        set_claim(&mut inputs, dimension - 0.001);
        let h = compute_all(&inputs).housing;
        let over = [
            h.diameter_overshoot_mm,
            h.length_overshoot_mm,
            h.bay_overshoot_mm,
        ];
        assert!(
            over.iter().any(|&o| o > 0.0 && o < 0.005),
            "{axis}: {over:?}"
        );
        assert_eq!(
            h.space_claim_check,
            format!("Exceeds the space claim: {axis} 0.01 mm over"),
            "{axis}"
        );
    }
}

#[test]
fn a_length_sized_design_reads_inside_the_space_claim_at_any_length() {
    // Decision A2-8 as recommended (no engine rule): the axial length moves no dimension the
    // space claim reads (the axial stack C134 and the large-diameter stack C137 are sums of
    // class N inputs), so a length-sized design reads "Inside the space claim" at any length.
    // Sized to its own 2.5 N m requirement the default design's magnets (13.77 mm) outgrow
    // the 13.0 mm hub; sized to 9.9 N m (50.6 mm) they outgrow the 15.5 mm cup and the
    // 14.5 mm retainer span. The override's help says to recheck them; M4 decides the rule.
    let base = DesignInputs::default();
    for (target, longer_than) in [
        (base.metal.required_min_Nm, base.metal.hub_length_mm),
        (9.9, base.metal.cup_depth_mm),
    ] {
        let p = solved(solve(&base, FreeVariable::AxialLength, target));
        assert!(p.value > longer_than, "{target}: {}", p.value);
        let r = compute_all(&p.inputs);
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{target}"
        );
        assert_eq!(
            r.metal.axial_stack_mm,
            compute_all(&base).metal.axial_stack_mm
        );
    }
    let p = solved(solve(&base, FreeVariable::AxialLength, 9.9));
    assert!(p.value > 50.0 && p.value > base.metal.retainer_span_mm);
}
