//! BL-063: the daily-cycle bond screen at the entered bondline
//! (`temperature.adhesive_life.daily_entered_screen`, `src/engine/temperature.rs`), end to end
//! through `compute_all`. The workbook's screen (Temperature design!C201 and C202) reads the
//! shear at the recommended bondline (C103) and stays cell for cell; the second screen reads
//! the shear at the bondline the user entered (Metal design!C120). Two Rust-only results. The
//! answer "report both" was first the conservative answer under the owner's ruling of 2026-11-10
//! on open questions; the owner approved it on 2026-11-16.

use std::cmp::Ordering;

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::meta::{Value, result_rows};

const WORKBOOK_SHEAR: &str = "temperature.adhesive_life.daily_peak_shear_MPa";
const WORKBOOK_SCREEN: &str = "temperature.adhesive_life.daily_screen";
const ENTERED_SHEAR: &str = "temperature.adhesive_life.daily_entered_peak_shear_MPa";
const ENTERED_SCREEN: &str = "temperature.adhesive_life.daily_entered_screen";
const BELOW: &str = "Below the fatigue endurance";
const ABOVE: &str = "Above the fatigue endurance: qualify by thermal cycling";

#[test]
fn the_default_design_is_above_the_endurance_at_the_entered_bondline_and_below_it_at_the_recommended_one()
 {
    let inputs = DesignInputs::default();
    let results = compute_all(&inputs);
    let rows = result_rows(&results);
    let at = |path: &str| {
        rows.iter()
            .position(|row| row.path == path)
            .unwrap_or_else(|| panic!("no result {path}"))
    };
    // The fatigue endurance: AA 326's 15 MPa lap shear times the 0.2 share.
    let limit = results.temperature.adhesive.lap_shear_MPa
        * inputs.temperature.adhesive_life.fatigue_endurance;
    assert_eq!(limit, 3.0);
    let (swing, worst) = (
        inputs.temperature.adhesive_life.daily_swing_C,
        results.temperature.mismatch.worst_swing_C,
    );

    // The workbook's screen, at the 0.1 mm recommended bondline: 2.563 MPa, below.
    let (shear, screen) = (&rows[at(WORKBOOK_SHEAR)], &rows[at(WORKBOOK_SCREEN)]);
    assert_eq!(shear.cell.as_deref(), Some("Temperature design!C201"));
    assert_eq!(screen.cell.as_deref(), Some("Temperature design!C202"));
    assert_eq!(
        shear.value,
        Value::Num(results.temperature.mismatch.peak_shear_recommended_MPa * swing / worst)
    );
    let Value::Num(workbook) = shear.value else {
        panic!("{:?}", shear.value)
    };
    assert_eq!(format!("{workbook:.3}"), "2.563");
    assert_eq!(screen.value, Value::Text(BELOW.to_owned()));

    // The second screen, at the 0.05 mm entered bondline: 4.932 MPa, above. Two Rust-only
    // results with no cell, straight after the workbook's two.
    let (shear, screen) = (&rows[at(ENTERED_SHEAR)], &rows[at(ENTERED_SCREEN)]);
    assert_eq!(
        (at(ENTERED_SHEAR), at(ENTERED_SCREEN)),
        (at(WORKBOOK_SCREEN) + 1, at(WORKBOOK_SCREEN) + 2)
    );
    for row in [shear, screen] {
        assert!(row.meta.rust_only, "{}", row.path);
        assert_eq!((row.meta.cell, &row.cell), (None, &None), "{}", row.path);
    }
    assert_eq!(
        (shear.meta.unit, shear.meta.label),
        ("MPa", "Peak end shear per daily cycle, entered bondline")
    );
    assert_eq!(
        (screen.meta.unit, screen.meta.label),
        ("", "Daily-cycle screen, entered bondline")
    );
    assert_eq!(
        shear.value,
        Value::Num(results.temperature.mismatch.peak_shear_current_MPa * swing / worst)
    );
    let Value::Num(entered) = shear.value else {
        panic!("{:?}", shear.value)
    };
    assert_eq!(format!("{entered:.3}"), "4.932");
    assert!(workbook < limit && limit < entered);
    assert_eq!(screen.value, Value::Text(ABOVE.to_owned()));
}

#[test]
fn each_screen_reads_its_own_bondline_over_a_spread_of_bondlines_and_swings() {
    // Every shear here is positive (N42SH on a steel hub), so the size comparison of the
    // negative-shear correction is the signed one.
    let screen = |shear: f64, limit: f64| if shear < limit { BELOW } else { ABOVE };
    let mut verdicts = std::collections::BTreeSet::new();
    for entered in [0.01, 0.05, 0.1, 0.2] {
        for swing in [10.0, 30.0, 60.0] {
            let mut entered_bits = None;
            for recommended in [0.01, 0.05, 0.1, 0.2, 0.5] {
                let what =
                    format!("{entered} mm entered, {recommended} mm recommended, {swing} °C");
                let mut inputs = DesignInputs::default();
                inputs.metal.bond_inner_mm = entered;
                inputs.temperature.mismatch.recommended_bondline_mm = recommended;
                inputs.temperature.adhesive_life.daily_swing_C = swing;
                let results = compute_all(&inputs);
                let (m, life) = (
                    &results.temperature.mismatch,
                    &results.temperature.adhesive_life,
                );
                let limit = results.temperature.adhesive.lap_shear_MPa
                    * inputs.temperature.adhesive_life.fatigue_endurance;
                let (workbook, second) =
                    (life.daily_peak_shear_MPa, life.daily_entered_peak_shear_MPa);

                // The workbook's two cells keep their formula, on the recommended bondline.
                assert_eq!(
                    workbook.to_bits(),
                    (m.peak_shear_recommended_MPa * swing / m.worst_swing_C).to_bits(),
                    "{what}"
                );
                assert_eq!(life.daily_screen, screen(workbook, limit), "{what}");
                // The second screen is the same formula on the entered bondline, and the
                // recommended bondline does not move it.
                assert_eq!(
                    second.to_bits(),
                    (m.peak_shear_current_MPa * swing / m.worst_swing_C).to_bits(),
                    "{what}"
                );
                assert_eq!(life.daily_entered_screen, screen(second, limit), "{what}");
                assert_eq!(
                    *entered_bits.get_or_insert(second.to_bits()),
                    second.to_bits(),
                    "{what}"
                );
                // Equal bondlines give equal shears and screens; the thicker one shears less.
                match entered.partial_cmp(&recommended).unwrap() {
                    Ordering::Equal => {
                        assert_eq!(second.to_bits(), workbook.to_bits(), "{what}");
                        assert_eq!(life.daily_entered_screen, life.daily_screen, "{what}");
                    }
                    Ordering::Greater => assert!(second < workbook, "{what}"),
                    Ordering::Less => assert!(second > workbook, "{what}"),
                }
                verdicts.insert((life.daily_screen.clone(), life.daily_entered_screen.clone()));
            }
        }
    }
    // The spread reaches all four pairs of verdicts.
    assert_eq!(verdicts.len(), 4, "{verdicts:?}");
}
