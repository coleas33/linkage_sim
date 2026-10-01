//! The A5 materials library against the approved Addendum A data file
//! (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every value
//! and citation, the engine values (workbook numbers where the workbook has
//! them, decisions 21 to 26), and the default materials equal to the inputs.

mod common;

use common::{ADDENDUM_DATA, read_json, repo_path};
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, MATERIALS, SLEEVE_LINER_CHOICES, Sourced, material,
};
use magcoupling::engine::materials::{AL6061, Steel4140};
use magcoupling::engine::metal_design::MetalDesignInputs;
use magcoupling::engine::temperature::{SlipLossInputs, ThermalInputs};

/// Decision 6 A re-sources the 416 expansion coefficient (the data file keeps a placeholder).
const RA_416: &str = "https://www.rolledalloys.com/wp-content/uploads/416_stainless-steel-data-sheet-rolled-alloys.pdf";

fn data_materials() -> Vec<serde_json::Value> {
    read_json(&repo_path(ADDENDUM_DATA))["materials"]
        .as_array()
        .expect("a materials array")
        .clone()
}

/// The data file's selected value and source of a field, as a [`Sourced`] holds them.
fn assert_sourced(what: &str, rust: Sourced, field: &serde_json::Value) {
    let value = field["value"].as_f64();
    let source = if value.is_some() {
        field["source_url"].as_str()
    } else {
        None
    };
    assert_eq!((rust.value, rust.source), (value, source), "{what}");
}

/// Engine value = the workbook default the data file records, else the sourced value.
fn engine_expected(m: &serde_json::Value, key: &str) -> f64 {
    m["workbook_defaults"][key]["default"]
        .as_f64()
        .or_else(|| m["fields"][key]["value"].as_f64())
        .unwrap_or_else(|| panic!("{} {key}", m["id"]))
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-15 * want.abs()
}

#[test]
fn materials_equal_the_addendum_data_file() {
    let data = data_materials();
    let ids: Vec<&str> = data
        .iter()
        .map(|m| m["id"].as_str().expect("an id"))
        .collect();
    let rust: Vec<&str> = MATERIALS.iter().map(|m| m.id).collect();
    assert_eq!(rust, ids, "ids and order");
    for (m, j) in MATERIALS.iter().zip(&data) {
        let id = m.id;
        let f = &j["fields"];
        assert_eq!(m.name, j["name"].as_str().expect("a name"), "{id}");
        assert_eq!(
            m.ferromagnetic,
            j["ferromagnetic"]["value"].as_bool().expect("a bool"),
            "{id}"
        );
        assert_eq!(
            Some(m.ferromagnetic_source),
            j["ferromagnetic"]["source_url"].as_str(),
            "{id}"
        );
        for (what, rust, key) in [
            ("mu_r", m.mu_r, "mu_r"),
            ("bsat", m.bsat_T, "bsat_T"),
            ("sigma", m.sigma_S_m, "sigma_S_m"),
            ("density", m.density_g_cm3, "density_g_cm3"),
            ("E", m.modulus_GPa, "E_GPa"),
            ("yield", m.yield_MPa, "yield_MPa"),
            ("cp", m.cp_J_kgK, "cp_J_kgK"),
        ] {
            assert_sourced(&format!("{id} {what}"), rust, &f[key]);
        }
        if id == "416_annealed" {
            assert_eq!(
                f["cte_1e-6_per_K"]["value"].as_f64(),
                Some(10.5),
                "the placeholder"
            );
            assert_eq!(
                m.cte_1e6_per_K,
                Sourced {
                    value: Some(10.08),
                    source: Some(RA_416)
                }
            );
        } else {
            assert_sourced(&format!("{id} cte"), m.cte_1e6_per_K, &f["cte_1e-6_per_K"]);
        }
        assert_eq!(
            m.design_flux_density_T,
            j["design_flux_density_T"]["value"].as_f64(),
            "{id} design flux density"
        );
        // Engine values: the workbook's number where one exists, else the sourced value.
        let e = &m.engine;
        assert_eq!(
            e.sigma_S_m,
            engine_expected(j, "sigma_S_m"),
            "{id} engine sigma"
        );
        assert_eq!(e.cp_J_kgK, engine_expected(j, "cp_J_kgK"), "{id} engine cp");
        assert_eq!(e.modulus_GPa, engine_expected(j, "E_GPa"), "{id} engine E");
        assert!(
            close(
                e.density_g_mm3,
                engine_expected(j, "density_g_cm3") / 1000.0
            ),
            "{id} engine density {}",
            e.density_g_mm3
        );
        let cte = if id == "416_annealed" {
            10.08
        } else {
            engine_expected(j, "cte_1e-6_per_K")
        };
        assert!(
            close(e.cte_per_C, cte * 1e-6),
            "{id} engine cte {}",
            e.cte_per_C
        );
    }
}

#[test]
fn the_default_materials_are_the_workbook_inputs() {
    // Picking the default material of a part changes nothing: its engine values are
    // bit-equal to the inputs the engine reads (decisions 21 to 26).
    let steel = Steel4140::default();
    let m4140 = material("4140_annealed").expect("4140").engine;
    assert_eq!(m4140.sigma_S_m, steel.conductivity_S_m);
    assert_eq!(m4140.cp_J_kgK, steel.specific_heat_J_kgK);
    assert_eq!(m4140.cte_per_C, steel.cte_per_C);
    assert_eq!(m4140.modulus_GPa, steel.modulus_GPa);
    let md = MetalDesignInputs::default();
    assert_eq!(m4140.density_g_mm3, md.steel_density_g_mm3);
    assert_eq!(
        material("4140_annealed").and_then(|m| m.design_flux_density_T),
        Some(steel.bsat_T)
    );
    let (sl, th) = (SlipLossInputs::default(), ThermalInputs::default());
    let m316 = material("316L_annealed").expect("316L").engine;
    assert_eq!(m316.sigma_S_m, sl.sigma_316_S_m);
    assert_eq!(m316.cp_J_kgK, th.c_316);
    assert_eq!(m316.density_g_mm3, md.sleeve_density_g_mm3);
    let m6061 = material("6061_T6").expect("6061").engine;
    assert_eq!(m6061.sigma_S_m, AL6061.conductivity_S_m);
    assert_eq!(m6061.cp_J_kgK, th.c_aluminium);
    assert_eq!(m6061.density_g_mm3, md.al_density_g_mm3);
    for (choices, default) in [
        (&BACK_IRON_CHOICES[..], "4140_annealed"),
        (&SLEEVE_LINER_CHOICES[..], "316L_annealed"),
        (&CAP_HOUSING_CHOICES[..], "6061_T6"),
    ] {
        assert_eq!(choices[0], (1, default));
    }
}

#[test]
fn each_part_offers_the_spec_choices_with_their_roles() {
    // Spec A5 choices; the data file's roles agree.
    let data = data_materials();
    let roles = |id: &str| -> String {
        data.iter()
            .find(|m| m["id"] == id)
            .map(|m| m["roles"].to_string())
            .unwrap_or_default()
    };
    for (choices, role) in [
        (&BACK_IRON_CHOICES[..], "back_iron"),
        (&SLEEVE_LINER_CHOICES[..], "sleeve_liner"),
        (&CAP_HOUSING_CHOICES[..], "cap_housing"),
    ] {
        for &(_, id) in choices {
            assert!(roles(id).contains(role), "{id} is not a {role} material");
        }
    }
    let non_magnetic: Vec<&str> = BACK_IRON_CHOICES
        .iter()
        .filter_map(|&(_, id)| material(id))
        .filter(|m| !m.ferromagnetic)
        .map(|m| m.id)
        .collect();
    assert_eq!(
        non_magnetic,
        ["304_annealed", "6061_T6"],
        "the demonstration back irons"
    );
    let sleeves_ferromagnetic = SLEEVE_LINER_CHOICES
        .iter()
        .filter_map(|&(_, id)| material(id))
        .any(|m| m.ferromagnetic);
    assert!(!sleeves_ferromagnetic, "no listed sleeve is ferromagnetic");
}

#[test]
fn plain_and_low_alloy_steels_need_plating() {
    // The data file's warning input: 4140, 1018 and 12L14; stainless and non-ferrous do not.
    let plated: Vec<&str> = MATERIALS
        .iter()
        .filter(|m| m.needs_plating)
        .map(|m| m.id)
        .collect();
    assert_eq!(
        plated,
        ["4140_annealed", "1018_hot_rolled", "12L14_cold_drawn"]
    );
}
