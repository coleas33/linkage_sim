//! BL-062, correction E26: the magnets' eddy heat (`temperature.slip_loss.magnets_W`,
//! `src/engine/temperature.rs`) reads each ring's conductivity by its grade's family
//! (`GradeFamily::magnet_sigma_S_m`), end to end through `compute_all`. Sintered NdFeB, every
//! library part and a ring with no grade keep the input (Temperature design!C113). The
//! per-family values were first the conservative answer under the owner's ruling of 2026-11-10 on
//! open questions; the owner approved them on 2026-11-16. The switch that turns E26 off is tested in `tests/deviations.rs`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::grades::{GRADES, GradeFamily};
use magcoupling::engine::library::MAGNET_LIBRARY;
use magcoupling::engine::meta::{Value, result_rows};
use magcoupling::engine::temperature::SlipLossResults;

/// The NdFeB conductivity input's default [S/m] (Temperature design!C113).
const INPUT_S_M: f64 = 6.7e5;

const INNER_SIGMA: &str = "temperature.slip_loss.inner_magnet_sigma_S_m";
const OUTER_SIGMA: &str = "temperature.slip_loss.outer_magnet_sigma_S_m";

/// The default design with both rings on their custom (manual) dimensions and these grades
/// ("" for no grade).
fn rings(inner: &str, outer: &str) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.part_inner.clear();
    inputs.coupling.magnets.part_outer.clear();
    inputs.coupling.magnets.grade_inner = inner.to_owned();
    inputs.coupling.magnets.grade_outer = outer.to_owned();
    inputs
}

/// The slip losses of a design.
fn losses(inputs: &DesignInputs) -> SlipLossResults {
    compute_all(inputs).temperature.slip_loss
}

/// The magnets' eddy heat [W].
fn magnets_w(inputs: &DesignInputs) -> f64 {
    losses(inputs).magnets_W
}

/// Equal to within rounding (the heat is the conductivity times one fixed product).
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs()
}

/// Every result row's value, bit for bit (a NaN as its bits).
fn all_values(inputs: &DesignInputs) -> Vec<String> {
    result_rows(&compute_all(inputs))
        .into_iter()
        .map(|row| match row.value {
            Value::Num(x) => format!("{} {:016x}", row.path, x.to_bits()),
            other => format!("{} {other:?}", row.path),
        })
        .collect()
}

#[test]
fn a_ferrite_pair_heats_far_less_than_an_ndfeb_pair_and_a_smco_pair_more() {
    // One design, four grade pairs of the same size. The stored field and the block size do not
    // follow the grade, so the heats stand in the ratio of the conductivities.
    let ndfeb = magnets_w(&rings("N42", "N42"));
    let (ferrite, sm2co17, smco5) = (
        magnets_w(&rings("Y30", "Y30")),
        magnets_w(&rings("SmCo_2_17_26", "SmCo_2_17_26")),
        magnets_w(&rings("SmCo_1_5_20", "SmCo_1_5_20")),
    );
    assert!(close(ferrite / ndfeb, 1.0e2 / INPUT_S_M), "{ferrite}");
    assert!(close(sm2co17 / ndfeb, 1.54e6 / INPUT_S_M), "{sm2co17}");
    assert!(close(smco5 / ndfeb, 2.5e6 / INPUT_S_M), "{smco5}");
    assert!(ferrite < ndfeb / 1000.0 && ndfeb < sm2co17 && sm2co17 < smco5);
}

#[test]
fn every_grade_pair_reads_its_family_s_conductivity() {
    // All 17 grades, each on both rings: the two rows show the family's value, or the input
    // for sintered NdFeB, and the heat follows in proportion.
    let ndfeb = magnets_w(&rings("N42", "N42"));
    assert_eq!(format!("{ndfeb:.4}"), "0.2278");
    for g in &GRADES {
        let want = match g.family {
            GradeFamily::NdFeB => INPUT_S_M,
            GradeFamily::SmCo2_17 => 1.54e6,
            GradeFamily::SmCo1_5 => 2.5e6,
            GradeFamily::Ferrite => 1.0e2,
            GradeFamily::BondedNdFeB => 2.5e4,
        };
        let loss = losses(&rings(g.id, g.id));
        assert_eq!(
            (loss.inner_magnet_sigma_S_m, loss.outer_magnet_sigma_S_m),
            (want, want),
            "{}",
            g.id
        );
        assert!(close(loss.magnets_W / ndfeb, want / INPUT_S_M), "{}", g.id);
        if g.family == GradeFamily::NdFeB {
            assert_eq!(loss.magnets_W, ndfeb, "{}: bit for bit", g.id);
        }
    }
}

#[test]
fn a_mixed_pair_reads_the_mean_of_its_two_rings() {
    // Y30 inside, N42 outside: half the NdFeB pair's heat plus half the ferrite pair's. Each
    // ring's half is taken on the inner block, so the order of the rings does not matter.
    let (ndfeb, ferrite) = (
        magnets_w(&rings("N42", "N42")),
        magnets_w(&rings("Y30", "Y30")),
    );
    let mixed = losses(&rings("Y30", "N42"));
    assert_eq!(
        (mixed.inner_magnet_sigma_S_m, mixed.outer_magnet_sigma_S_m),
        (1.0e2, INPUT_S_M)
    );
    assert!(close(mixed.magnets_W, (ndfeb + ferrite) / 2.0));
    assert!(ferrite < mixed.magnets_W && mixed.magnets_W < ndfeb);
    let swapped = losses(&rings("N42", "Y30"));
    assert_eq!(
        (
            swapped.inner_magnet_sigma_S_m,
            swapped.outer_magnet_sigma_S_m
        ),
        (INPUT_S_M, 1.0e2)
    );
    assert_eq!(swapped.magnets_W, mixed.magnets_W);
    // Two families with values of their own: SmCo5 inside, Sm2Co17 outside.
    let smco = losses(&rings("SmCo_1_5_20", "SmCo_2_17_30"));
    assert!(close(
        smco.magnets_W / ndfeb,
        (2.5e6 + 1.54e6) / 2.0 / INPUT_S_M
    ));
}

#[test]
fn a_library_part_and_a_ring_with_no_grade_read_the_input() {
    // Every library part is sintered NdFeB. A grade typed beside a library part is not read.
    let default = losses(&DesignInputs::default());
    assert_eq!(default.magnets_W, 0.22784639081300484);
    for spec in &MAGNET_LIBRARY {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = spec.part.to_owned();
        inputs.coupling.magnets.part_outer = spec.part.to_owned();
        inputs.coupling.magnets.grade_inner = "Y30".to_owned();
        inputs.coupling.magnets.grade_outer = "Y30".to_owned();
        let loss = losses(&inputs);
        assert_eq!(
            (loss.inner_magnet_sigma_S_m, loss.outer_magnet_sigma_S_m),
            (INPUT_S_M, INPUT_S_M),
            "{}",
            spec.part
        );
    }
    // Manual dimensions with no grade, or with a grade name the table does not hold.
    for grade in ["", "Y31"] {
        let loss = losses(&rings(grade, grade));
        assert_eq!(
            (loss.inner_magnet_sigma_S_m, loss.outer_magnet_sigma_S_m),
            (INPUT_S_M, INPUT_S_M),
            "{grade:?}"
        );
        assert_eq!(loss.magnets_W, default.magnets_W, "{grade:?}");
    }
    // A library part beside a ferrite ring on manual dimensions keeps its own half.
    let mut one = DesignInputs::default();
    one.coupling.magnets.part_outer.clear();
    one.coupling.magnets.grade_outer = "Y30".to_owned();
    let loss = losses(&one);
    assert_eq!(
        (loss.inner_magnet_sigma_S_m, loss.outer_magnet_sigma_S_m),
        (INPUT_S_M, 1.0e2)
    );
}

#[test]
fn the_input_moves_an_ndfeb_pair_s_heat_and_leaves_a_ferrite_pair_s_alone() {
    let raised = |mut inputs: DesignInputs| {
        inputs.temperature.slip_loss.sigma_ndfeb_S_m = 1.0e6;
        inputs
    };
    // Sintered NdFeB on both rings, and no grade: the heat follows the input.
    for grade in ["N42", ""] {
        let (before, after) = (
            magnets_w(&rings(grade, grade)),
            magnets_w(&raised(rings(grade, grade))),
        );
        assert!(close(after / before, 1.0e6 / INPUT_S_M), "{grade:?}");
    }
    // Ferrite on both rings: neither reads the input, and no result moves at all.
    assert_eq!(
        all_values(&rings("Y30", "Y30")),
        all_values(&raised(rings("Y30", "Y30")))
    );
    // A mixed pair: only the NdFeB ring's half follows it.
    let (before, after) = (
        losses(&rings("Y30", "N42")),
        losses(&raised(rings("Y30", "N42"))),
    );
    assert_eq!(
        (after.inner_magnet_sigma_S_m, after.outer_magnet_sigma_S_m),
        (1.0e2, 1.0e6)
    );
    assert!(close(
        after.magnets_W / before.magnets_W,
        (1.0e2 + 1.0e6) / (1.0e2 + INPUT_S_M)
    ));
}

#[test]
fn a_measured_drag_still_replaces_the_estimate() {
    // With a bench drag entered the loss used is the drag times the slip speed, whatever the
    // magnets' estimate says: a SmCo5 pair and an NdFeB pair then use the same loss.
    let measured = |grade: &str| {
        let mut inputs = rings(grade, grade);
        inputs.metal.measured_drag_Nm = Some(0.05);
        compute_all(&inputs).temperature
    };
    let (ndfeb, smco) = (measured("N42"), measured("SmCo_1_5_20"));
    assert!(smco.slip_loss.magnets_W > 3.0 * ndfeb.slip_loss.magnets_W);
    assert!(smco.slip_loss.total_W > ndfeb.slip_loss.total_W);
    assert_eq!(smco.slip_loss.used_W, 0.05 * smco.duty.slip_rad_s);
    assert_eq!(smco.slip_loss.used_W, ndfeb.slip_loss.used_W);
    assert_eq!(smco.slip_loss.high_W, ndfeb.slip_loss.high_W);
    assert_eq!(
        smco.thermal.steady_rise_high_C,
        ndfeb.thermal.steady_rise_high_C
    );
}

#[test]
fn the_two_conductivity_rows_are_rust_only_numbers_with_no_cell() {
    let rows = result_rows(&compute_all(&DesignInputs::default()));
    let at = |path: &str| {
        rows.iter()
            .position(|row| row.path == path)
            .unwrap_or_else(|| panic!("no result {path}"))
    };
    // Beside the other conductivities the losses read, before the skin depth.
    assert_eq!(
        (at(INNER_SIGMA), at(OUTER_SIGMA)),
        (
            at("temperature.slip_loss.cap_sigma_S_m") + 1,
            at("temperature.slip_loss.skin_depth_mm") - 1
        )
    );
    for (path, label, ring) in [
        (
            INNER_SIGMA,
            "Inner ring magnet conductivity used",
            "the inner ring's magnets",
        ),
        (
            OUTER_SIGMA,
            "Outer ring magnet conductivity used",
            "the outer ring's magnets",
        ),
    ] {
        let row = &rows[at(path)];
        assert!(row.meta.rust_only, "{path}");
        assert_eq!((row.meta.cell, &row.cell), (None, &None), "{path}");
        assert_eq!((row.meta.unit, row.meta.label), ("S/m", label));
        assert!(row.meta.help.contains(ring), "{path}: {}", row.meta.help);
        assert_eq!(row.value, Value::Num(INPUT_S_M), "{path}");
        // The help calls no value measured or approved, and says the heat is right for no grade.
        let help = row.meta.help.to_lowercase();
        for word in ["measur", "approv", "correct", "exact"] {
            assert!(!help.contains(word), "{path}: {word}");
        }
    }
}
