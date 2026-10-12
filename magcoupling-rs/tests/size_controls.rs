//! Manual outside diameter and overall length controls, evaluated through the same engine.

use magcoupling::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter, resize_diameter_with_minimum,
    resize_length,
};
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
