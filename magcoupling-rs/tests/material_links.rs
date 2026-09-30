//! Addendum A5 physics links: what a part's material choice feeds into the
//! engine. Picking a steel or a sleeve material equals typing its library values
//! into the inputs it stands for; a non-ferromagnetic back iron selects the
//! free-space circuit and becomes the hub, cup and boss material; the backiron
//! input still overrides a ferromagnetic choice; the default choices change nothing.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{cell_values_for, num};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, SLEEVE_LINER_CHOICES, material,
};
use magcoupling::engine::materials::PartMaterialInputs;
use magcoupling::engine::meta::{InputSet, SetErrorKind, Value, input_rows, result_rows};

/// Cells whose RESULT value differs (input cells are what the user typed).
fn changed_results(
    inputs_a: &DesignInputs,
    inputs_b: &DesignInputs,
    dev: Deviations,
) -> BTreeSet<String> {
    let result_cells = |inputs: &DesignInputs| -> BTreeMap<String, Value> {
        result_rows(&compute_all_with(inputs, dev))
            .into_iter()
            .filter_map(|r| r.cell.map(|c| (c, r.value)))
            .collect()
    };
    let (a, b) = (result_cells(inputs_a), result_cells(inputs_b));
    a.iter()
        .filter(|(cell, v)| !parity_close(v, &b[*cell]))
        .map(|(cell, _)| cell.clone())
        .collect()
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-12 * want.abs()
}

fn with_back_iron(code: i64) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.materials.parts.back_iron = code;
    inputs
}

#[test]
fn each_selector_offers_the_library_choices() {
    let rows = input_rows(&PartMaterialInputs::default());
    for (path, choices) in [
        ("back_iron", &BACK_IRON_CHOICES[..]),
        ("sleeve_liner", &SLEEVE_LINER_CHOICES[..]),
        ("cap_housing", &CAP_HOUSING_CHOICES[..]),
    ] {
        let row = rows.iter().find(|r| r.path == path).expect(path);
        assert!(row.meta.rust_only, "{path}");
        let want: Vec<(i64, &str)> = choices
            .iter()
            .map(|&(code, id)| (code, material(id).expect("a material").label))
            .collect();
        assert_eq!(row.meta.choices, &want[..], "{path}");
        assert_eq!(row.value, Value::Int(1), "{path}: the workbook's material");
    }
}

#[test]
fn the_default_choices_change_nothing() {
    for dev in [Deviations::NONE, Deviations::ALL] {
        let base = DesignInputs::defaults_with(dev);
        let mut explicit = base.clone();
        explicit
            .set("materials.parts.back_iron", Value::Int(1))
            .expect("a choice");
        explicit
            .set("materials.parts.sleeve_liner", Value::Int(1))
            .expect("a choice");
        explicit
            .set("materials.parts.cap_housing", Value::Int(1))
            .expect("a choice");
        assert_eq!(
            compute_all_with(&explicit, dev),
            compute_all_with(&base, dev)
        );
    }
    let res = compute_all(&DesignInputs::default()).materials;
    assert_eq!(
        (
            res.circuit_backiron,
            res.back_iron_material.as_str(),
            res.sleeve_liner_material.as_str(),
            res.cap_material.as_str()
        ),
        (1, "4140 annealed", "316L annealed", "6061-T6 aluminium")
    );
}

#[test]
fn picking_a_steel_equals_typing_its_values() {
    // A ferromagnetic back iron supplies conductivity, density, specific heat, expansion
    // and modulus, and its design flux density where the library has one (1018: 1.7 T);
    // otherwise Materials C13 stays. Incremental permeability stays C15.
    for &(code, id) in &BACK_IRON_CHOICES[1..6] {
        let m = material(id).expect("a material");
        assert!(m.ferromagnetic, "{id}");
        let picked = with_back_iron(code);
        let mut typed = DesignInputs::default();
        let e = m.engine;
        let s = &mut typed.materials.steel;
        s.conductivity_S_m = e.sigma_S_m;
        s.specific_heat_J_kgK = e.cp_J_kgK;
        s.cte_per_C = e.cte_per_C;
        s.modulus_GPa = e.modulus_GPa;
        if let Some(b) = m.design_flux_density_T {
            s.bsat_T = b;
        }
        typed.metal.steel_density_g_mm3 = e.density_g_mm3;
        for dev in [Deviations::NONE, Deviations::ALL] {
            assert!(
                changed_results(&picked, &typed, dev).is_empty(),
                "{id}: {:?}",
                changed_results(&picked, &typed, dev)
            );
        }
        assert_eq!(compute_all(&picked).materials.back_iron_material, m.label);
    }
}

#[test]
fn picking_a_sleeve_equals_typing_its_values() {
    // The sleeve and liner (and the endplates, as Metal design C44 prices them).
    for &(code, id) in &SLEEVE_LINER_CHOICES[1..] {
        let e = material(id).expect("a material").engine;
        let mut picked = DesignInputs::default();
        picked.materials.parts.sleeve_liner = code;
        let mut typed = DesignInputs::default();
        typed.temperature.slip_loss.sigma_316_S_m = e.sigma_S_m;
        typed.temperature.thermal.c_316 = e.cp_J_kgK;
        typed.metal.sleeve_density_g_mm3 = e.density_g_mm3;
        for dev in [Deviations::NONE, Deviations::ALL] {
            assert!(changed_results(&picked, &typed, dev).is_empty(), "{id}");
        }
    }
}

#[test]
fn the_cap_choice_prices_the_cap_only() {
    // The cap's mass, loss and heat; the aluminium adapter (C188) keeps Metal design C42.
    let base = cell_values_for(&DesignInputs::default(), Deviations::ALL);
    for &(code, id) in &CAP_HOUSING_CHOICES[1..] {
        let e = material(id).expect("a material").engine;
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.cap_housing = code;
        let c = cell_values_for(&inputs, Deviations::ALL);
        let ratio = |cell: &str| num(&c[cell]) / num(&base[cell]);
        assert!(
            close(ratio("Metal design!C180"), e.density_g_mm3 / 0.0027),
            "{id} cap mass"
        );
        assert!(
            close(ratio("Temperature design!C128"), e.sigma_S_m / 2.5e7),
            "{id} cap loss"
        );
        assert_eq!(
            c["Temperature design!C112"],
            Value::Num(e.sigma_S_m),
            "{id}"
        );
        assert_eq!(
            c["Metal design!C188"], base["Metal design!C188"],
            "{id} adapter"
        );
        assert_eq!(c["Calculator!C93"], base["Calculator!C93"], "{id} torque");
    }
}

#[test]
fn a_non_ferromagnetic_back_iron_selects_the_free_space_circuit() {
    // Spec, Addendum testing: "a non-ferromagnetic back iron switches the circuit factor".
    // 304 and 6061 give the torque of the no-back-iron circuit (the geometry factor
    // s_free), as the backiron input 0 does, and E9's "No back iron".
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let free = cell_values_for(&no_iron, Deviations::ALL);
    let steel = cell_values_for(&DesignInputs::default(), Deviations::ALL);
    for code in [7, 8] {
        let c = cell_values_for(&with_back_iron(code), Deviations::ALL);
        assert_eq!(c["Calculator!C93"], free["Calculator!C93"], "{code}");
        assert_ne!(c["Calculator!C93"], steel["Calculator!C93"], "{code}");
        assert_eq!(c["Materials!C22"], Value::Text("No back iron".into()));
        assert_eq!(
            compute_all(&with_back_iron(code))
                .materials
                .circuit_backiron,
            0
        );
    }
}

#[test]
fn a_non_ferromagnetic_back_iron_is_the_hub_cup_and_boss_material() {
    // With no back iron the body is the picked material: density (E9, C111-C113),
    // specific heat (E15), conductivity (E17) and expansion and modulus (E18). The
    // default choice with C6 = 0 keeps the workbook's aluminium.
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let aluminium = cell_values_for(&no_iron, Deviations::ALL);
    let c304 = cell_values_for(&with_back_iron(7), Deviations::ALL);
    let e304 = material("304_annealed").expect("304").engine;
    let ratio = |cell: &str| num(&c304[cell]) / num(&aluminium[cell]);
    for cell in ["Calculator!C111", "Calculator!C112", "Calculator!C113"] {
        assert!(close(ratio(cell), e304.density_g_mm3 / 0.0027), "{cell}");
    }
    // E17's closed form is linear in conductivity (same fields and geometry).
    for cell in [
        "Temperature design!C123",
        "Temperature design!C124",
        "Temperature design!C125",
    ] {
        assert!(close(ratio(cell), e304.sigma_S_m / 2.5e7), "{cell}");
    }
    assert_ne!(
        c304["Temperature design!C104"],
        aluminium["Temperature design!C104"]
    );
    // The library's 6061 differs from the workbook aluminium only in the modulus E18 reads
    // (68.3 GPa Kaiser against 68.9 GPa Alliance; decision table of the A-1 plan).
    let c6061 = with_back_iron(8);
    let changed = changed_results(&no_iron, &c6061, Deviations::ALL);
    let want: BTreeSet<String> = ["C104", "C105", "C201"]
        .iter()
        .map(|c| format!("Temperature design!{c}"))
        .collect();
    assert_eq!(changed, want);
}

#[test]
fn the_backiron_input_overrides_a_ferromagnetic_choice() {
    // C6 = 0 with 1018 picked: the free-space circuit and the workbook's aluminium body.
    let mut inputs = with_back_iron(2);
    inputs.coupling.backiron = 0;
    let res = compute_all(&inputs);
    assert_eq!(res.materials.circuit_backiron, 0);
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let aluminium = compute_all(&no_iron);
    assert_eq!(res.mass.cup_g, aluminium.mass.cup_g, "the E9 aluminium cup");
    assert_eq!(res.model.pullout_Nm, aluminium.model.pullout_Nm);
}

#[test]
fn the_design_flux_density_feeds_the_wall_check() {
    // Decision 20: t_bi = B_gap tau_p / (pi B_design). 1018's 1.7 T (the workbook's comment)
    // thins the wall needed by 1.5/1.7 and the 1.8 mm wall passes; 17-4PH has no sourced
    // design value, so Materials C13 (1.5 T) stays.
    let base = cell_values_for(&DesignInputs::default(), Deviations::ALL);
    let c1018 = cell_values_for(&with_back_iron(2), Deviations::ALL);
    let need = |c: &BTreeMap<String, Value>| num(&c["Calculator!C104"]);
    assert!(close(need(&c1018), need(&base) * 1.5 / 1.7));
    assert_eq!(c1018["Calculator!C36"], Value::Num(1.7));
    assert_eq!(
        base["Materials!C22"],
        Value::Text("Too thin: raise Metal design C122 to at least 2.0 mm".into())
    );
    assert_eq!(c1018["Materials!C22"], Value::Text("OK".into()));
    let c174 = cell_values_for(&with_back_iron(5), Deviations::ALL);
    assert_eq!(need(&c174), need(&base));
    assert_eq!(c174["Calculator!C36"], Value::Num(1.5));
}

#[test]
fn a_code_outside_the_choices_gives_nan_not_another_material() {
    // Decision D3 for the new selectors: never another material; validate() names it.
    let mut inputs = DesignInputs::default();
    inputs.materials.parts.back_iron = 99;
    inputs.materials.parts.sleeve_liner = 0;
    inputs.materials.parts.cap_housing = -1;
    let res = compute_all(&inputs);
    assert_eq!(res.materials.back_iron_material, "#N/A");
    assert!(res.mass.cup_g.is_nan() && res.temperature.slip_loss.sleeve_W.is_nan());
    assert!(res.retainers.cap_g.is_nan());
    let errors = inputs.validate().expect_err("three invalid codes");
    let paths: Vec<(&str, &SetErrorKind)> =
        errors.iter().map(|e| (e.path.as_str(), &e.kind)).collect();
    assert_eq!(
        paths,
        [
            (
                "materials.parts.back_iron",
                &SetErrorKind::NotAChoice { code: 99 }
            ),
            (
                "materials.parts.sleeve_liner",
                &SetErrorKind::NotAChoice { code: 0 }
            ),
            (
                "materials.parts.cap_housing",
                &SetErrorKind::NotAChoice { code: -1 }
            ),
        ]
    );
}
