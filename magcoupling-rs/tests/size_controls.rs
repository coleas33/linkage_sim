//! Manual outside diameter and overall length controls, evaluated through the same engine.

use magcoupling::engine::meta::input_rows;
use magcoupling::engine::model::blocks_fit;
use magcoupling::engine::size_controls::{
    DiameterBreakdown, DiameterFit, FIT_TOLERANCE_MM, diameter_breakdown, dimensions,
    dimensions_with_minimum_diameter, length_breakdown, resize_diameter,
    resize_diameter_with_minimum, resize_length, ring_radius_range, smallest_fitting_diameter,
    with_ring_radius,
};
use magcoupling::engine::sizing::FreeVariable;
use magcoupling::{DesignInputs, compute_all};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

#[test]
fn dimensions_are_the_existing_housing_dimensions_and_ranges() {
    let controls = dimensions(&DesignInputs::default()).unwrap();
    close(controls.diameter_mm, 42.8);
    close(controls.overall_length_mm, 31.8);
    close(controls.diameter_range.min, 40.38163688425198);
    close(controls.diameter_range.max, 80.0);
    close(controls.length_range.min, 21.1);
    close(controls.length_range.max, 69.9);
}

#[test]
fn editing_the_size_already_shown_is_an_exact_noop() {
    for overridden in [false, true] {
        let mut base = DesignInputs::default();
        if overridden {
            base.coupling.magnets.axial_length_mm = Some(15.0);
        }
        let size = dimensions(&base).unwrap();
        assert_eq!(resize_diameter(&base, size.diameter_mm).unwrap(), base);
        assert_eq!(resize_length(&base, size.overall_length_mm).unwrap(), base);
    }
}

#[test]
fn diameter_moves_the_hub_and_cap_with_the_existing_allowances() {
    let base = DesignInputs::default();
    let changed = resize_diameter(&base, 41.0).unwrap();
    let mut expected = base.clone();
    expected.coupling.inner_back_apothem_mm = 9.294049135334363;
    expected.metal.cap_od_mm = 41.0;
    expected.metal.cap_thread_dia_mm = 39.2;
    close(
        changed.coupling.inner_back_apothem_mm,
        expected.coupling.inner_back_apothem_mm,
    );
    close(changed.metal.cap_od_mm, 41.0);
    close(changed.metal.cap_thread_dia_mm, 39.2);
    expected.coupling.inner_back_apothem_mm = changed.coupling.inner_back_apothem_mm;
    expected.metal.cap_od_mm = changed.metal.cap_od_mm;
    expected.metal.cap_thread_dia_mm = changed.metal.cap_thread_dia_mm;
    assert_eq!(
        changed, expected,
        "only the radius, cap OD and cap thread may change"
    );
    let results = compute_all(&changed);
    close(results.model.cup_od_mm, 39.526464605669034);
    close(dimensions(&changed).unwrap().diameter_mm, 41.0);
    // The existing negative thread-to-cup allowance stays visible, rather than being fixed.
    close(
        changed.metal.cap_thread_dia_mm - results.model.cup_od_mm,
        -0.326464605669032,
    );
}

#[test]
fn eight_magnets_can_fit_at_the_smaller_manually_selected_size() {
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    let changed = resize_diameter(&base, 40.5).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 9.087538537612023);
    close(changed.metal.cap_od_mm, 40.5);
    close(compute_all(&changed).model.cup_od_mm, 40.1362321464911);
    assert!(
        compute_all(&changed)
            .model
            .inner_flat_check
            .starts_with("OK")
    );
    assert!(
        compute_all(&changed).metal.torque_hot_low_Nm < compute_all(&base).metal.torque_hot_low_Nm
    );
    assert_eq!(changed.coupling.npole, 8);
}

#[test]
fn an_explicit_minimum_allows_eight_magnets_at_38_mm() {
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    assert!(
        resize_diameter(&base, 38.0).is_err(),
        "the inherited radius floor stays unchanged"
    );
    let changed = resize_diameter_with_minimum(&base, 38.0, 37.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 7.932689121972913);
    close(changed.metal.cap_od_mm, 38.0);
    close(changed.metal.cap_thread_dia_mm, 36.2);
    assert!(
        compute_all(&changed)
            .model
            .inner_flat_check
            .starts_with("OK")
    );
    assert!(
        compute_all(&changed).metal.torque_hot_low_Nm < compute_all(&base).metal.torque_hot_low_Nm
    );
}

#[test]
fn arc_diameter_uses_the_arc_geometry_and_keeps_gap_and_cross_sections() {
    let mut base = DesignInputs::default();
    base.coupling.faceted = 0;
    let changed = resize_diameter(&base, 41.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 9.25);
    close(compute_all(&changed).model.cup_od_mm, 37.68);
    close(dimensions(&changed).unwrap().diameter_mm, 41.0);
    assert_eq!(changed.metal.face_gap_mm, 1.4);
    assert_eq!(changed.coupling.magnets, base.coupling.magnets);
}

#[test]
fn a_cap_smaller_than_the_cup_keeps_its_signed_offset() {
    let mut base = DesignInputs::default();
    base.metal.cap_od_mm = 35.0;
    base.metal.cap_thread_dia_mm = 34.0;
    let changed = resize_diameter(&base, 40.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 9.519228596571793);
    close(changed.metal.cap_od_mm, 33.67353539433097);
    close(changed.metal.cap_thread_dia_mm, 32.67353539433097);
    close(compute_all(&changed).model.cup_od_mm, 40.0);
    close(dimensions(&changed).unwrap().diameter_mm, 40.0);
}

#[test]
fn length_preserves_axial_allowances_and_only_sets_the_override() {
    let base = DesignInputs::default();
    let changed = resize_length(&base, 29.1).unwrap();
    close(changed.coupling.magnets.axial_length_mm.unwrap(), 10.0);
    let mut expected = base.clone();
    expected.coupling.magnets.axial_length_mm = changed.coupling.magnets.axial_length_mm;
    assert_eq!(changed, expected);
    let results = compute_all(&changed);
    close(results.model.inner_length_mm, 10.0);
    close(results.model.outer_length_mm, 10.0);
    close(results.housing.hub_length_mm, 10.3);
    close(results.housing.cup_depth_mm, 12.8);
    close(results.housing.retainer_span_mm, 11.8);
    close(results.metal.axial_stack_mm, 29.1);
    assert!(results.metal.torque_hot_low_Nm < compute_all(&base).metal.torque_hot_low_Nm);
}

#[test]
fn mixed_library_and_manual_rings_keep_their_own_length_allowances() {
    let mut base = DesignInputs::default();
    base.coupling.magnets.part_outer.clear();
    base.coupling.magnets.manual_outer_length_mm = 20.0;
    base.metal.cup_depth_mm = 22.5;
    base.metal.retainer_span_mm = 22.0;
    let changed = resize_length(&base, 33.8).unwrap();
    close(changed.coupling.magnets.axial_length_mm.unwrap(), 15.0);
    let results = compute_all(&changed);
    close(results.housing.hub_length_mm, 15.3);
    close(results.housing.cup_depth_mm, 17.5);
    close(results.housing.retainer_span_mm, 17.0);
    close(results.metal.axial_stack_mm, 33.8);
    assert_eq!(changed.coupling.magnets.part_inner, "B842SH");
    assert_eq!(changed.coupling.magnets.part_outer, "");
    assert_eq!(changed.coupling.magnets.manual_outer_length_mm, 20.0);
}

#[test]
fn negative_axial_allowances_keep_the_existing_ring_length_floor() {
    let mut base = DesignInputs::default();
    base.metal.hub_length_mm = 5.0;
    base.metal.cup_depth_mm = 5.0;
    base.metal.retainer_span_mm = 5.0;
    let original = dimensions(&base).unwrap();
    close(original.overall_length_mm, 21.3);
    close(original.length_range.min, 18.3);
    close(original.length_range.max, 67.1);
    assert_eq!(
        resize_length(&base, original.overall_length_mm).unwrap(),
        base
    );
    let changed = resize_length(&base, 25.0).unwrap();
    close(changed.coupling.magnets.axial_length_mm.unwrap(), 8.7);
    let results = compute_all(&changed);
    close(results.housing.hub_length_mm, 8.7);
    close(results.housing.cup_depth_mm, 8.7);
    close(results.housing.retainer_span_mm, 8.7);
    close(results.metal.axial_stack_mm, 25.0);
    assert_eq!(changed.metal.hub_length_mm, 5.0);
    assert_eq!(changed.metal.cup_depth_mm, 5.0);
    assert_eq!(changed.metal.retainer_span_mm, 5.0);
}

#[test]
fn boss_sets_an_outside_diameter_floor_without_a_noop_resize() {
    let mut base = DesignInputs::default();
    base.metal.boss_od_mm = 50.0;
    let size = dimensions(&base).unwrap();
    close(size.diameter_mm, 50.0);
    close(size.diameter_range.min, 50.0);
    assert_eq!(resize_diameter(&base, 50.0).unwrap(), base);
    assert!(resize_diameter(&base, 49.0).is_err());
    let changed = resize_diameter(&base, 51.0).unwrap();
    close(dimensions(&changed).unwrap().diameter_mm, 51.0);
    assert_eq!(changed.metal.boss_od_mm, 50.0);
}

#[test]
fn metadata_limits_for_radius_cap_and_thread_all_bound_the_diameter() {
    let mut base = DesignInputs::default();
    base.metal.cap_thread_dia_mm = 50.0;
    let size = dimensions(&base).unwrap();
    close(size.diameter_range.max, 72.8);
    let changed = resize_diameter(&base, size.diameter_range.max).unwrap();
    close(changed.metal.cap_thread_dia_mm, 80.0);
    assert!(resize_diameter(&base, 72.81).is_err());
    assert!(resize_diameter(&base, size.diameter_range.min - 0.01).is_err());
    assert!(resize_length(&base, size.length_range.max + 0.01).is_err());
    assert!(resize_length(&base, size.length_range.min - 0.01).is_err());
}

#[test]
fn repeated_round_trips_do_not_accumulate_housing_allowance_drift() {
    let mut current = DesignInputs::default();
    let before = compute_all(&current);
    for _ in 0..20 {
        current = resize_diameter(&current, 41.0).unwrap();
        current = resize_length(&current, 29.1).unwrap();
        current = resize_diameter(&current, 42.8).unwrap();
        current = resize_length(&current, 31.8).unwrap();
    }
    close(current.coupling.inner_back_apothem_mm, 10.15);
    close(current.metal.cap_od_mm, 42.8);
    close(current.metal.cap_thread_dia_mm, 41.0);
    close(current.coupling.magnets.axial_length_mm.unwrap(), 12.7);
    let after = compute_all(&current);
    close(after.housing.hub_length_mm, before.housing.hub_length_mm);
    close(after.housing.cup_depth_mm, before.housing.cup_depth_mm);
    close(
        after.housing.retainer_span_mm,
        before.housing.retainer_span_mm,
    );
    close(
        after.metal.torque_hot_low_Nm,
        before.metal.torque_hot_low_Nm,
    );
}

#[test]
fn malformed_designs_and_nonfinite_requests_are_refused_without_panicking() {
    let base = DesignInputs::default();
    for requested in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(resize_diameter(&base, requested).is_err());
        assert!(resize_length(&base, requested).is_err());
    }
    let mut invalid = base.clone();
    invalid.metal.face_gap_mm = f64::NAN;
    assert!(dimensions(&invalid).is_none());
    assert!(resize_diameter(&invalid, 41.0).is_err());
    assert!(resize_length(&invalid, 29.1).is_err());
    for count in [i64::MIN, -4, 0, 1, 3, i64::MAX] {
        invalid = base.clone();
        invalid.coupling.npole = count;
        assert!(dimensions(&invalid).is_none(), "count {count}");
        assert!(resize_diameter(&invalid, 41.0).is_err());
    }
}

#[test]
fn changing_the_minimum_only_changes_the_range_and_preserves_current_dimensions() {
    let base = DesignInputs::default();
    let original = base.clone();
    for minimum in [0.1, 37.0, 50.0, 200.0] {
        let size = dimensions_with_minimum_diameter(&base, minimum).unwrap();
        close(size.diameter_mm, 42.8);
        close(size.overall_length_mm, 31.8);
        close(size.minimum_diameter_range.min, 32.07508531276967);
        close(size.minimum_diameter_range.max, 80.0);
        close(
            size.diameter_range.min,
            minimum.clamp(32.07508531276967, 80.0),
        );
        assert_eq!(
            resize_diameter_with_minimum(&base, 42.8, minimum).unwrap(),
            original
        );
    }
    assert_eq!(base, original);
    close(
        dimensions(&base).unwrap().diameter_range.min,
        40.38163688425198,
    );
}

#[test]
fn minimum_cannot_remove_the_hub_wall_around_the_shaft() {
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    let size = dimensions_with_minimum_diameter(&base, 20.0).unwrap();
    close(size.minimum_diameter_range.min, 31.759599557017577);
    close(size.diameter_range.min, 31.759599557017577);
    let changed = resize_diameter_with_minimum(&base, size.diameter_range.min, 20.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 5.05);
    assert!(compute_all(&changed).model.hub_wall_mm >= 0.0);
    assert!(compute_all(&changed).model.hub_wall_past_key_mm < 0.0);
    assert!(resize_diameter_with_minimum(&base, 31.75, 20.0).is_err());
}

#[test]
fn explicit_minimum_keeps_block_fit_failures_visible() {
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    let changed = resize_diameter_with_minimum(&base, 37.0, 37.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 7.47074935571727);
    let results = compute_all(&changed);
    assert!(results.model.inner_flat_check.starts_with("TOO NARROW"));
    assert!(results.model.hub_wall_mm > 0.0);
    assert!(results.metal.torque_hot_low_Nm.is_finite());
}

#[test]
fn cap_thread_and_boss_absolute_floors_still_apply() {
    let mut thread_limited = DesignInputs::default();
    thread_limited.metal.cap_thread_dia_mm = 20.0;
    let size = dimensions_with_minimum_diameter(&thread_limited, 25.0).unwrap();
    close(size.diameter_range.min, 42.8);
    assert!(resize_diameter_with_minimum(&thread_limited, 42.7, 25.0).is_err());

    let mut cap_limited = DesignInputs::default();
    cap_limited.metal.cap_od_mm = 20.0;
    let size = dimensions_with_minimum_diameter(&cap_limited, 25.0).unwrap();
    close(size.diameter_range.min, 41.32646460566903);
    assert!(resize_diameter_with_minimum(&cap_limited, 41.3, 25.0).is_err());

    let mut boss_limited = DesignInputs::default();
    boss_limited.coupling.npole = 8;
    boss_limited.metal.boss_od_mm = 35.0;
    let size = dimensions_with_minimum_diameter(&boss_limited, 25.0).unwrap();
    close(size.diameter_range.min, 35.0);
    close(size.minimum_diameter_range.min, 35.0);
    assert!(resize_diameter_with_minimum(&boss_limited, 34.9, 25.0).is_err());
    close(
        dimensions(&resize_diameter_with_minimum(&boss_limited, 35.0, 25.0).unwrap())
            .unwrap()
            .diameter_mm,
        35.0,
    );
}

#[test]
fn explicit_minimum_preserves_upper_bounds_and_works_for_arcs() {
    let mut base = DesignInputs::default();
    base.coupling.faceted = 0;
    let size = dimensions_with_minimum_diameter(&base, 20.0).unwrap();
    close(size.minimum_diameter_range.min, 32.6);
    close(size.diameter_range.max, 80.0);
    let changed = resize_diameter_with_minimum(&base, 37.0, 20.0).unwrap();
    close(changed.coupling.inner_back_apothem_mm, 7.25);
    close(dimensions(&changed).unwrap().diameter_mm, 37.0);
    let largest = resize_diameter_with_minimum(&base, 80.0, 20.0).unwrap();
    close(largest.metal.cap_od_mm, 80.0);
    assert!(resize_diameter_with_minimum(&base, 80.1, 20.0).is_err());
    assert!(resize_diameter_with_minimum(&base, 79.9, 200.0).is_err());
}

#[test]
fn a_loaded_design_below_the_legacy_radius_floor_is_preserved_and_editable() {
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    let mut current = resize_diameter_with_minimum(&base, 38.0, 37.0).unwrap();
    let legacy_size = dimensions(&current).unwrap();
    close(legacy_size.diameter_mm, 38.0);
    assert!(legacy_size.diameter_mm < legacy_size.diameter_range.min);
    assert_eq!(resize_diameter(&current, 38.0).unwrap(), current);
    for _ in 0..20 {
        current = resize_diameter_with_minimum(&current, 37.0, 37.0).unwrap();
        current = resize_diameter_with_minimum(&current, 38.0, 37.0).unwrap();
    }
    close(current.coupling.inner_back_apothem_mm, 7.932689121972913);
    close(current.metal.cap_od_mm, 38.0);
    close(current.metal.cap_thread_dia_mm, 36.2);
    let shortened = resize_length(&current, 29.1).unwrap();
    close(shortened.coupling.inner_back_apothem_mm, 7.932689121972913);
    close(dimensions(&shortened).unwrap().overall_length_mm, 29.1);
}

#[test]
fn invalid_minimum_settings_are_refused_even_for_unchanged_size() {
    let base = DesignInputs::default();
    for minimum in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(dimensions_with_minimum_diameter(&base, minimum).is_none());
        assert!(resize_diameter_with_minimum(&base, 42.8, minimum).is_err());
    }
    for requested in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(resize_diameter_with_minimum(&base, requested, 37.0).is_err());
    }
    let mut invalid = base.clone();
    invalid.metal.face_gap_mm = f64::NAN;
    assert!(dimensions_with_minimum_diameter(&invalid, 37.0).is_none());
    assert!(resize_diameter_with_minimum(&invalid, 38.0, 37.0).is_err());
}

/// The default design with `npole` magnets per ring.
fn with_poles(npole: i64) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = npole;
    inputs
}

/// The axial housing around the outer ring, from the design's own inputs: the cap, the cavity's
/// margin over its ring (the cup depth in effect minus the ring's length), the web and the boss.
fn housing_around_the_outer_ring(inputs: &DesignInputs) -> f64 {
    let results = compute_all(inputs);
    let metal = &inputs.metal;
    metal.cap_axial_mm
        + (results.housing.cup_depth_mm - results.model.outer_length_mm)
        + metal.web_mm
        + metal.boss_length_mm
}

#[test]
fn diameter_breakdown_names_cap_cup_and_boss_and_its_largest_is_the_outside_diameter() {
    let base = DesignInputs::default();
    let parts = diameter_breakdown(&base).unwrap();
    assert_eq!(
        parts,
        DiameterBreakdown {
            cap_mm: base.metal.cap_od_mm,
            cup_mm: compute_all(&base).model.cup_od_mm,
            boss_mm: base.metal.boss_od_mm,
        }
    );
    let largest = |b: DiameterBreakdown| b.cup_mm.max(b.cap_mm).max(b.boss_mm);
    assert_eq!(largest(parts), dimensions(&base).unwrap().diameter_mm);
    assert_eq!(largest(parts), parts.cap_mm, "the cap sets the default OD");

    let mut cup_sets = base.clone();
    cup_sets.metal.cap_od_mm = 35.0;
    cup_sets.metal.cap_thread_dia_mm = 34.0;
    let parts = diameter_breakdown(&cup_sets).unwrap();
    assert_eq!(largest(parts), parts.cup_mm);
    assert_eq!(largest(parts), dimensions(&cup_sets).unwrap().diameter_mm);

    let mut boss_sets = base.clone();
    boss_sets.metal.boss_od_mm = 50.0;
    let parts = diameter_breakdown(&boss_sets).unwrap();
    assert_eq!(largest(parts), parts.boss_mm);
    assert_eq!(largest(parts), dimensions(&boss_sets).unwrap().diameter_mm);
    close(parts.boss_mm, 50.0);
}

#[test]
fn eight_magnets_grow_the_cup_while_the_cap_keeps_the_outside_diameter() {
    // The owner's case (spec 1, T1): 10 -> 8 magnets at the default ring grows the cup, and
    // the outside diameter stays because the cap sets it.
    let ten = diameter_breakdown(&DesignInputs::default()).unwrap();
    let eight = diameter_breakdown(&with_poles(8)).unwrap();
    assert!(
        eight.cup_mm > ten.cup_mm,
        "{} vs {}",
        eight.cup_mm,
        ten.cup_mm
    );
    assert!(
        eight.cup_mm < eight.cap_mm,
        "{} vs {}",
        eight.cup_mm,
        eight.cap_mm
    );
    assert_eq!(eight.cap_mm, ten.cap_mm);
    close(
        dimensions(&with_poles(8)).unwrap().diameter_mm,
        eight.cap_mm,
    );
    close(eight.cap_mm, DesignInputs::default().metal.cap_od_mm);
}

#[test]
fn length_breakdown_is_the_outer_ring_and_the_housing_around_it() {
    let base = DesignInputs::default();
    let parts = length_breakdown(&base).unwrap();
    // The default housing is 19.1 mm: 0.8 cap + (15.5 cavity - 12.7 ring) + 2.5 web + 13.0 boss.
    close(parts.magnets_mm, compute_all(&base).model.outer_length_mm);
    close(parts.housing_mm, housing_around_the_outer_ring(&base));
    close(
        parts.magnets_mm + parts.housing_mm,
        dimensions(&base).unwrap().overall_length_mm,
    );

    // The length slider moves the magnets; the housing keeps its allowances.
    let shorter = resize_length(&base, 29.1).unwrap();
    let after = length_breakdown(&shorter).unwrap();
    close(after.magnets_mm, 10.0);
    close(after.housing_mm, parts.housing_mm);
    close(after.magnets_mm + after.housing_mm, 29.1);

    // The cup cavity holds the outer ring: a 20 mm manual outer ring counts, not the 12.7 mm
    // inner part. Its 22.5 mm cavity leaves 2.5 mm: 0.8 cap + 2.5 + 2.5 web + 13.0 boss = 18.8.
    let mut mixed = base.clone();
    mixed.coupling.magnets.part_outer.clear();
    mixed.coupling.magnets.manual_outer_length_mm = 20.0;
    mixed.metal.cup_depth_mm = 22.5;
    mixed.metal.retainer_span_mm = 22.0;
    let parts = length_breakdown(&mixed).unwrap();
    close(parts.magnets_mm, 20.0);
    close(parts.housing_mm, housing_around_the_outer_ring(&mixed));
    close(
        parts.magnets_mm + parts.housing_mm,
        dimensions(&mixed).unwrap().overall_length_mm,
    );
    let resized = resize_length(&mixed, 33.8).unwrap();
    let after = length_breakdown(&resized).unwrap();
    close(after.magnets_mm, 15.0);
    close(after.housing_mm, housing_around_the_outer_ring(&resized));
    close(after.housing_mm, parts.housing_mm);

    // At the ring-length floor the cavity has no margin: the housing is the cap, web and boss
    // (0.8 + 2.5 + 13.0 = 16.3 mm).
    let mut floored = base.clone();
    floored.metal.hub_length_mm = 5.0;
    floored.metal.cup_depth_mm = 5.0;
    floored.metal.retainer_span_mm = 5.0;
    let at_floor = resize_length(&floored, 25.0).unwrap();
    let parts = length_breakdown(&at_floor).unwrap();
    close(parts.magnets_mm, 8.7);
    close(parts.housing_mm, housing_around_the_outer_ring(&at_floor));
    close(
        parts.housing_mm,
        floored.metal.cap_axial_mm + floored.metal.web_mm + floored.metal.boss_length_mm,
    );
}

#[test]
fn a_housing_shorter_than_its_outer_ring_has_no_length_breakdown() {
    // Blank override, a 40 mm manual outer ring in the default 15.5 mm cavity: the stack
    // (31.8 mm) is shorter than the ring, so no split describes it. The size controls stay.
    let mut short_housing = DesignInputs::default();
    short_housing.coupling.magnets.part_outer.clear();
    short_housing.coupling.magnets.manual_outer_length_mm = 40.0;
    close(dimensions(&short_housing).unwrap().overall_length_mm, 31.8);
    assert_eq!(length_breakdown(&short_housing), None);
    // With the override set the cavity follows the ring (decision A2-8): a split again.
    short_housing.coupling.magnets.axial_length_mm = Some(40.0);
    let parts = length_breakdown(&short_housing).unwrap();
    close(parts.magnets_mm, 40.0);
    close(
        parts.housing_mm,
        housing_around_the_outer_ring(&short_housing),
    );
}

#[test]
fn breakdowns_are_none_exactly_where_the_size_controls_are() {
    let mut invalid = DesignInputs::default();
    invalid.metal.face_gap_mm = f64::NAN;
    assert_eq!(dimensions(&invalid), None);
    assert_eq!(diameter_breakdown(&invalid), None);
    assert_eq!(length_breakdown(&invalid), None);
    for count in [i64::MIN, -4, 0, 1, 2, 3, 9, 41, 42, i64::MAX] {
        let odd = with_poles(count);
        assert_eq!(dimensions(&odd), None, "count {count}");
        assert_eq!(diameter_breakdown(&odd), None, "count {count}");
        assert_eq!(length_breakdown(&odd), None, "count {count}");
    }
}

/// `base` with true arcs instead of flat blocks.
fn as_arcs(mut base: DesignInputs) -> DesignInputs {
    base.coupling.faceted = 0;
    base
}

/// `base` with a manual outer ring of 10 mm wide blocks, so the outer ring sets the fit.
fn with_wide_outer_blocks(mut base: DesignInputs) -> DesignInputs {
    base.coupling.magnets.part_outer.clear();
    base.coupling.magnets.manual_outer_width_mm = 10.0;
    base
}

/// The design after the diameter slider moves to an outside diameter (with the Minimum OD
/// setting when there is one).
fn resized_to(design: &DesignInputs, diameter_mm: f64, minimum_mm: Option<f64>) -> DesignInputs {
    match minimum_mm {
        Some(minimum) => resize_diameter_with_minimum(design, diameter_mm, minimum),
        None => resize_diameter(design, diameter_mm),
    }
    .expect("the diameter lies inside the Size tab's range")
}

/// Whether both rings' blocks fit at an outside diameter, mapped to the inputs by the
/// diameter slider's own resize.
fn fits_at(design: &DesignInputs, diameter_mm: f64, minimum_mm: Option<f64>) -> bool {
    let resized = resized_to(design, diameter_mm, minimum_mm);
    blocks_fit(&resized.coupling, &compute_all(&resized).model)
}

/// The diameter of an `At` fit, or a panic naming the other form.
fn fit_edge(fit: Option<DiameterFit>) -> f64 {
    match fit {
        Some(DiameterFit::At(diameter_mm)) => diameter_mm,
        other => panic!("expected a fit inside the range, got {other:?}"),
    }
}

#[test]
fn smallest_fitting_diameter_fits_at_the_result_and_not_a_thousandth_below() {
    // Flat blocks, 10 magnets, the legacy range: the inner flats bind inside the range.
    let base = DesignInputs::default();
    let edge = fit_edge(smallest_fitting_diameter(&base, 10, None));
    let range = dimensions(&base).unwrap().diameter_range;
    assert!(range.min < edge - 1e-3 && edge < range.max, "{edge}");
    assert!(fits_at(&base, edge, None));
    assert!(!fits_at(&base, edge - 1e-3, None));

    // The owner's 8 magnets with a lowered Minimum OD: between the existing 37 mm (too
    // narrow) and 38 mm (fits) cases.
    let edge = fit_edge(smallest_fitting_diameter(&base, 8, Some(37.0)));
    assert!(37.0 < edge && edge < 38.0, "{edge}");
    let eight = with_poles(8);
    assert!(fits_at(&eight, edge, Some(37.0)));
    assert!(!fits_at(&eight, edge - 1e-3, Some(37.0)));

    // Arcs, 10 magnets, a lowered Minimum OD: the inner arcs' pitch share binds.
    let arcs = as_arcs(base.clone());
    let edge = fit_edge(smallest_fitting_diameter(&arcs, 10, Some(20.0)));
    assert!(fits_at(&arcs, edge, Some(20.0)));
    assert!(!fits_at(&arcs, edge - 1e-3, Some(20.0)));

    // Wide outer blocks: the outer ring binds, for both shapes.
    for wide in [
        with_wide_outer_blocks(base.clone()),
        as_arcs(with_wide_outer_blocks(base.clone())),
    ] {
        let edge = fit_edge(smallest_fitting_diameter(&wide, 10, Some(20.0)));
        assert!(fits_at(&wide, edge, Some(20.0)));
        assert!(!fits_at(&wide, edge - 1e-3, Some(20.0)));
    }
}

#[test]
fn smallest_fitting_diameter_reports_both_range_ends() {
    let base = DesignInputs::default();
    // Four magnets fit at the smallest diameter, legacy or lowered.
    let four = with_poles(4);
    assert_eq!(
        smallest_fitting_diameter(&base, 4, None),
        Some(DiameterFit::AtMinimum(
            dimensions(&four).unwrap().diameter_range.min
        ))
    );
    assert_eq!(
        smallest_fitting_diameter(&base, 4, Some(20.0)),
        Some(DiameterFit::AtMinimum(
            dimensions_with_minimum_diameter(&four, 20.0)
                .unwrap()
                .diameter_range
                .min
        ))
    );
    // Arcs at 10 magnets fit from the legacy 9 mm radius floor up.
    let arcs = as_arcs(base.clone());
    let legacy = dimensions(&arcs).unwrap().diameter_range;
    assert_eq!(
        smallest_fitting_diameter(&arcs, 10, None),
        Some(DiameterFit::AtMinimum(legacy.min))
    );
    assert!(fits_at(&arcs, legacy.min, None));
    // A Minimum OD above the fit's edge is the answer as it is.
    assert_eq!(
        smallest_fitting_diameter(&base, 10, Some(50.0)),
        Some(DiameterFit::AtMinimum(50.0))
    );
    // A Minimum OD at the top of the range leaves one diameter: either form, never a search.
    assert_eq!(
        smallest_fitting_diameter(&base, 10, Some(200.0)),
        Some(DiameterFit::AtMinimum(80.0))
    );
    assert_eq!(
        smallest_fitting_diameter(&base, 40, Some(200.0)),
        Some(DiameterFit::Nowhere(80.0))
    );
    // Forty magnets fit nowhere up to the largest diameter, for both shapes.
    for design in [base.clone(), as_arcs(base.clone())] {
        let mut forty = design.clone();
        forty.coupling.npole = 40;
        let largest = dimensions(&forty).unwrap().diameter_range.max;
        assert_eq!(
            smallest_fitting_diameter(&design, 40, None),
            Some(DiameterFit::Nowhere(largest))
        );
        assert!(!fits_at(&forty, largest, None));
    }
}

#[test]
fn the_magnet_count_argument_replaces_the_designs_own() {
    let ten = DesignInputs::default();
    let eight = with_poles(8);
    assert_eq!(
        smallest_fitting_diameter(&ten, 8, Some(37.0)),
        smallest_fitting_diameter(&eight, 8, Some(37.0))
    );
    assert_eq!(
        smallest_fitting_diameter(&eight, 10, None),
        smallest_fitting_diameter(&ten, 10, None)
    );
    assert_ne!(
        smallest_fitting_diameter(&ten, 8, Some(37.0)),
        smallest_fitting_diameter(&ten, 10, Some(37.0))
    );
}

#[test]
fn blocks_fit_never_stops_as_the_diameter_grows_at_a_fixed_magnet_count() {
    // The basis of the bisection (module docs): over the widest diameter range (the lowest
    // Minimum OD) of every magnet count, flat blocks and arcs, the default blocks and a wide
    // outer ring, a design that fits keeps fitting as the diameter grows. Each sweep also
    // pins smallest_fitting_diameter to the samples on either side of its answer.
    const SAMPLES: usize = 48;
    let lowest = 0.1;
    let base = DesignInputs::default();
    for (blocks_name, blocks) in [
        ("default blocks", base.clone()),
        ("wide outer blocks", with_wide_outer_blocks(base.clone())),
    ] {
        for (shape_name, shape) in [("flat blocks", 1), ("arcs", 0)] {
            // How often each form was the answer: at the minimum, inside the range, nowhere.
            let mut forms = [0; 3];
            for npole in (4..=40).step_by(2) {
                let mut design = blocks.clone();
                design.coupling.faceted = shape;
                design.coupling.npole = npole;
                let case = format!("{blocks_name}, {shape_name}, {npole} magnets");
                let range = dimensions_with_minimum_diameter(&design, lowest)
                    .unwrap()
                    .diameter_range;
                let diameters: Vec<f64> = (0..=SAMPLES)
                    .map(|i| {
                        if i == SAMPLES {
                            range.max
                        } else {
                            range.min + (range.max - range.min) * i as f64 / SAMPLES as f64
                        }
                    })
                    .collect();
                let fits: Vec<bool> = diameters
                    .iter()
                    .map(|&diameter| fits_at(&design, diameter, Some(lowest)))
                    .collect();
                assert!(
                    fits.windows(2).all(|pair| pair[0] <= pair[1]),
                    "{case}: {fits:?}"
                );
                match smallest_fitting_diameter(&design, npole, Some(lowest)).unwrap() {
                    DiameterFit::AtMinimum(diameter) => {
                        forms[0] += 1;
                        assert_eq!(diameter, range.min, "{case}");
                        assert!(fits.iter().all(|&fit| fit), "{case}");
                    }
                    DiameterFit::At(edge) => {
                        forms[1] += 1;
                        for (&diameter, &fit) in diameters.iter().zip(&fits) {
                            if diameter >= edge {
                                assert!(fit, "{case}: {diameter} at or above {edge}");
                            } else if diameter < edge - FIT_TOLERANCE_MM {
                                assert!(!fit, "{case}: {diameter} below {edge}");
                            }
                        }
                    }
                    DiameterFit::Nowhere(diameter) => {
                        forms[2] += 1;
                        assert_eq!(diameter, range.max, "{case}");
                        assert!(fits.iter().all(|&fit| !fit), "{case}");
                    }
                }
            }
            // Not vacuous: every form occurs, so the sweeps cross fit edges.
            assert!(
                forms.iter().all(|&count| count > 0),
                "{blocks_name}, {shape_name}: {forms:?}"
            );
        }
    }
}

#[test]
fn ring_radius_range_spans_the_size_tabs_diameter_range() {
    let base = DesignInputs::default();
    let radius = FreeVariable::RingRadius.range();
    let diameters = dimensions(&base).unwrap().diameter_range;
    let ring = ring_radius_range(&base, None).unwrap();
    close(ring.min, radius.min);
    close(
        ring.min,
        resize_diameter(&base, diameters.min)
            .unwrap()
            .coupling
            .inner_back_apothem_mm,
    );
    assert_eq!(
        ring.max,
        resize_diameter(&base, diameters.max)
            .unwrap()
            .coupling
            .inner_back_apothem_mm
    );
    assert!(
        ring.max < radius.max,
        "the cap's 80 mm limit binds first: {}",
        ring.max
    );
    assert_eq!((ring.step, ring.log), (radius.step, radius.log));
    // A Minimum OD at the top of the range leaves one radius.
    let top = ring_radius_range(&base, Some(200.0)).unwrap();
    assert_eq!(top.min, top.max);
    assert_eq!(top.max, ring.max);

    // A lowered Minimum OD reaches below the radius slider's 9 mm floor (8 magnets at 37 mm):
    // its smallest radius is what the diameter slider gives at the range's smallest diameter.
    let eight = with_poles(8);
    let ring = ring_radius_range(&eight, Some(37.0)).unwrap();
    let smallest = dimensions_with_minimum_diameter(&eight, 37.0)
        .unwrap()
        .diameter_range
        .min;
    close(smallest, 37.0);
    close(
        ring.min,
        resize_diameter_with_minimum(&eight, smallest, 37.0)
            .unwrap()
            .coupling
            .inner_back_apothem_mm,
    );
    assert!(ring.min < radius.min);
}

#[test]
fn the_ring_radius_range_does_not_follow_the_ring_along_the_boss_plateau() {
    // A 50 mm boss sets the outside diameter over a plateau of ring radii, every one of them a
    // 50 mm design. Wherever on that plateau the ring sits, the range's floor is the cup's own
    // and every radius in the range applies.
    let base = DesignInputs::default();
    let radius = FreeVariable::RingRadius.range();
    // The lowest floor a Minimum OD reaches: the hub around the shaft (bore radius + bondline).
    let shaft_mm = base.coupling.bore_mm / 2.0 + base.metal.bond_inner_mm;
    let mut boss = base.clone();
    boss.metal.boss_od_mm = 50.0;
    for current in [radius.min, base.coupling.inner_back_apothem_mm, 12.0, 14.0] {
        let mut design = boss.clone();
        design.coupling.inner_back_apothem_mm = current;
        let size = dimensions(&design).unwrap();
        assert_eq!(size.diameter_mm, boss.metal.boss_od_mm, "ring {current}");
        assert_eq!(
            size.diameter_range.min, boss.metal.boss_od_mm,
            "ring {current}"
        );
        // The same design with the default boss, which sets no part of the diameter range.
        let mut no_boss = design.clone();
        no_boss.metal.boss_od_mm = base.metal.boss_od_mm;
        let plain = ring_radius_range(&no_boss, None).unwrap();

        // Without a Minimum OD the floor is the radius input's; above the boss nothing changes.
        let ring = ring_radius_range(&design, None).unwrap();
        assert_eq!(ring.min, radius.min, "ring {current}");
        assert_eq!(ring.max, plain.max, "ring {current}");
        // A Minimum OD at or below the boss leaves every radius a 50 mm design, so the floor is
        // the shaft's; above the boss it is the same diameter floor as without the boss.
        for minimum in [0.1, 37.0, boss.metal.boss_od_mm] {
            let lowered = ring_radius_range(&design, Some(minimum)).unwrap();
            close(lowered.min, shaft_mm);
            assert_eq!(
                lowered.max, plain.max,
                "ring {current}, Minimum OD {minimum}"
            );
        }
        assert_eq!(
            ring_radius_range(&design, Some(60.0)),
            ring_radius_range(&no_boss, Some(60.0)),
            "ring {current}"
        );

        // Every radius down to the floor, below the current one too, applies and keeps the
        // boss's outside diameter.
        let widest = ring_radius_range(&design, Some(0.1)).unwrap();
        for apothem_mm in [widest.min, radius.min, 10.0, current] {
            let case = format!("ring {current} -> {apothem_mm}");
            let shown = with_ring_radius(&design, apothem_mm).unwrap_or_else(|| panic!("{case}"));
            assert_eq!(shown.coupling.inner_back_apothem_mm, apothem_mm, "{case}");
            assert_eq!(
                dimensions(&shown).unwrap().diameter_mm,
                boss.metal.boss_od_mm,
                "{case}"
            );
        }
    }

    // Moving the ring along the plateau, as a ring-size solve shows it (the cap following the
    // cup), moves neither end.
    let ring = ring_radius_range(&boss, None).unwrap();
    for apothem_mm in [ring.min, 12.0] {
        let moved = with_ring_radius(&boss, apothem_mm).unwrap();
        assert_eq!(
            dimensions(&moved).unwrap().diameter_mm,
            boss.metal.boss_od_mm,
            "{apothem_mm}"
        );
        let again = ring_radius_range(&moved, None).unwrap();
        assert_eq!(again.min, ring.min, "{apothem_mm}");
        close(again.max, ring.max);
    }

    // On the plateau the cap and thread input floors still bound the radius: a small cap's
    // thread, following the cup down, meets its input minimum above the shaft's floor.
    let mut small_cap = boss.clone();
    small_cap.metal.cap_od_mm = 25.0;
    small_cap.metal.cap_thread_dia_mm = 24.0;
    let thread_min = input_rows(&small_cap)
        .iter()
        .find(|row| row.path == "metal.cap_thread_dia_mm")
        .and_then(|row| row.meta.range)
        .unwrap()
        .min;
    let floor = ring_radius_range(&small_cap, Some(0.1)).unwrap().min;
    assert!(floor > shaft_mm, "{floor}");
    let at_floor = with_ring_radius(&small_cap, floor).unwrap();
    close(at_floor.metal.cap_thread_dia_mm, thread_min);
    assert_eq!(
        dimensions(&at_floor).unwrap().diameter_mm,
        boss.metal.boss_od_mm
    );
    assert_eq!(with_ring_radius(&small_cap, floor - 1e-3), None);
}

#[test]
fn a_boss_above_every_reachable_diameter_leaves_the_cups_whole_radius_reach() {
    // A boss larger than the largest diameter the cap can reach (its input maximum) is the only
    // outside diameter left; every ring radius the cup reaches gives it, so the range is the
    // same as without the boss rather than the current radius alone.
    let base = DesignInputs::default();
    let plain_size = dimensions(&base).unwrap();
    let mut largest = base.clone();
    largest.metal.boss_od_mm = plain_size.diameter_range.max + 10.0;
    let size = dimensions(&largest).unwrap();
    assert_eq!(size.diameter_range.min, largest.metal.boss_od_mm);
    assert_eq!(size.diameter_range.max, largest.metal.boss_od_mm);

    for minimum in [None, Some(0.1), Some(37.0)] {
        let plain = ring_radius_range(&base, minimum).unwrap();
        let ring = ring_radius_range(&largest, minimum).unwrap();
        // With a Minimum OD at or below the boss the floor is the shaft's, as at the lowest one.
        let floor = match minimum {
            None => plain.min,
            Some(_) => ring_radius_range(&base, Some(0.1)).unwrap().min,
        };
        close(ring.min, floor);
        close(ring.max, plain.max);
        let current = base.coupling.inner_back_apothem_mm;
        assert!(
            ring.min < current && current < ring.max,
            "{minimum:?}: {ring:?}"
        );
        for apothem_mm in [ring.min, ring.max] {
            let shown = with_ring_radius(&largest, apothem_mm)
                .unwrap_or_else(|| panic!("{minimum:?}: {apothem_mm}"));
            assert_eq!(
                dimensions(&shown).unwrap().diameter_mm,
                largest.metal.boss_od_mm
            );
        }
    }
}

#[test]
fn fit_and_ring_range_are_none_where_the_size_controls_are() {
    let base = DesignInputs::default();
    let mut invalid = base.clone();
    invalid.metal.face_gap_mm = f64::NAN;
    assert_eq!(smallest_fitting_diameter(&invalid, 10, None), None);
    assert_eq!(ring_radius_range(&invalid, None), None);
    for count in [i64::MIN, -4, 0, 1, 2, 3, 9, 41, 42, i64::MAX] {
        assert_eq!(
            ring_radius_range(&with_poles(count), None),
            None,
            "count {count}"
        );
        assert_eq!(
            smallest_fitting_diameter(&base, count, None),
            None,
            "count {count}"
        );
        assert_eq!(
            smallest_fitting_diameter(&base, count, Some(37.0)),
            None,
            "count {count}"
        );
    }
    for minimum in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert_eq!(smallest_fitting_diameter(&base, 10, Some(minimum)), None);
        assert_eq!(ring_radius_range(&base, Some(minimum)), None);
    }
}

#[test]
fn a_ring_radius_applies_through_the_diameter_sliders_own_mapping() {
    let mut small_cap = DesignInputs::default();
    small_cap.metal.cap_od_mm = 35.0;
    small_cap.metal.cap_thread_dia_mm = 34.0;
    // (case, design, a diameter inside its range, the Minimum OD setting)
    let cases = [
        ("default", DesignInputs::default(), 41.0, None),
        ("arcs", as_arcs(DesignInputs::default()), 41.0, None),
        ("cap smaller than the cup", small_cap, 40.0, None),
        ("below the radius floor", with_poles(8), 38.0, Some(37.0)),
    ];
    let cup_of = |design: &DesignInputs| compute_all(design).model.cup_od_mm;
    for (case, base, diameter_mm, minimum_mm) in cases {
        let slid = resized_to(&base, diameter_mm, minimum_mm);
        let apothem_mm = slid.coupling.inner_back_apothem_mm;
        let shown = with_ring_radius(&base, apothem_mm).unwrap_or_else(|| panic!("{case}"));
        assert_eq!(shown.coupling.inner_back_apothem_mm, apothem_mm, "{case}");

        // The outside diameter is the one the slider gave for that radius.
        let outside = dimensions(&shown).unwrap().diameter_mm;
        assert!((outside - diameter_mm).abs() < 1e-9, "{case}: {outside}");
        // The cap and thread followed the cup, as the slider moves them.
        for (name, got, want) in [
            ("cap OD", shown.metal.cap_od_mm, slid.metal.cap_od_mm),
            (
                "cap thread",
                shown.metal.cap_thread_dia_mm,
                slid.metal.cap_thread_dia_mm,
            ),
        ] {
            assert!((got - want).abs() < 1e-9, "{case}: {name} {got} vs {want}");
        }
        // Their signed offsets from the cup are kept.
        let offsets = |design: &DesignInputs| {
            (
                design.metal.cap_od_mm - cup_of(design),
                design.metal.cap_thread_dia_mm - cup_of(design),
            )
        };
        let (before, after) = (offsets(&base), offsets(&shown));
        assert!(
            (before.0 - after.0).abs() < 1e-9,
            "{case}: {before:?} {after:?}"
        );
        assert!(
            (before.1 - after.1).abs() < 1e-9,
            "{case}: {before:?} {after:?}"
        );

        // Nothing else moved.
        let mut rest = shown.clone();
        rest.coupling.inner_back_apothem_mm = base.coupling.inner_back_apothem_mm;
        rest.metal.cap_od_mm = base.metal.cap_od_mm;
        rest.metal.cap_thread_dia_mm = base.metal.cap_thread_dia_mm;
        assert_eq!(
            rest, base,
            "{case}: only the radius, cap OD and thread move"
        );
    }
}

#[test]
fn the_current_ring_radius_is_an_exact_noop() {
    let base = DesignInputs::default();
    assert_eq!(
        with_ring_radius(&base, base.coupling.inner_back_apothem_mm),
        Some(base)
    );
}

#[test]
fn every_radius_the_size_tab_can_reach_is_applicable_and_none_beyond_it() {
    let mut boss = DesignInputs::default();
    boss.metal.boss_od_mm = 50.0;
    let designs = [
        ("default", DesignInputs::default()),
        ("arcs", as_arcs(DesignInputs::default())),
        ("eight magnets", with_poles(8)),
        ("boss sets the diameter", boss),
    ];
    let legacy_floor = FreeVariable::RingRadius.range().min;
    let mut reached_below_the_legacy_floor = false;
    for (case, base) in designs {
        // The widest reach is the lowest Minimum OD; a higher one only narrows it.
        let widest = ring_radius_range(&base, Some(0.1)).unwrap();
        for minimum in [None, Some(0.1), Some(37.0), Some(50.0)] {
            let range = ring_radius_range(&base, minimum).unwrap();
            for end in [range.min, range.max] {
                assert!(
                    with_ring_radius(&base, end).is_some(),
                    "{case}, Minimum OD {minimum:?}: {end}"
                );
            }
        }
        for inside in [widest.min, 0.5 * (widest.min + widest.max), widest.max] {
            assert!(
                with_ring_radius(&base, inside).is_some(),
                "{case}: {inside}"
            );
        }
        assert_eq!(with_ring_radius(&base, widest.min - 1e-3), None, "{case}");
        assert_eq!(with_ring_radius(&base, widest.max + 1e-3), None, "{case}");
        reached_below_the_legacy_floor |= widest.min < legacy_floor;
    }
    assert!(reached_below_the_legacy_floor, "the reach is not vacuous");
}

#[test]
fn a_ring_radius_the_mapping_cannot_define_is_refused() {
    let base = DesignInputs::default();
    for apothem_mm in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert_eq!(with_ring_radius(&base, apothem_mm), None, "{apothem_mm}");
    }
    let mut invalid = base.clone();
    invalid.metal.face_gap_mm = f64::NAN;
    assert_eq!(with_ring_radius(&invalid, 10.0), None);
    for count in [i64::MIN, -4, 0, 1, 2, 3, 9, 41, 42, i64::MAX] {
        assert_eq!(
            with_ring_radius(&with_poles(count), 10.0),
            None,
            "count {count}"
        );
    }
}
