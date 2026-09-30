//! The A6 grade table and the part table against the approved Addendum A data
//! file (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every
//! value and every citation, every part resolving to a grade, and the places
//! where a part's workbook value differs from its grade on purpose.

mod common;

use common::{ADDENDUM_DATA, read_json, repo_path};
use magcoupling::engine::constants::NDFEB_DENSITY_G_MM3;
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library::{MAGNET_LIBRARY, N42SH_BR_CORRECTED_T};
use magcoupling::engine::temperature::DemagInputs;

fn data() -> serde_json::Value {
    read_json(&repo_path(ADDENDUM_DATA))
}

fn f(json: &serde_json::Value) -> f64 {
    json.as_f64()
        .unwrap_or_else(|| panic!("{json} is a number"))
}

fn opt_f(json: &serde_json::Value) -> Option<f64> {
    json.as_f64()
}

fn opt_s(json: &serde_json::Value) -> Option<&str> {
    json.as_str()
}

#[test]
fn grades_equal_the_addendum_data_file() {
    let doc = data();
    let grades = doc["grades"].as_object().expect("a grades object");
    // serde_json keeps object keys sorted: compare the sets (the table itself keeps
    // the data file's order).
    let ids: std::collections::BTreeSet<&str> = grades.keys().map(String::as_str).collect();
    let rust: std::collections::BTreeSet<&str> = GRADES.iter().map(|g| g.id).collect();
    assert_eq!(rust, ids, "grade ids");
    assert_eq!(GRADES.len(), ids.len(), "no duplicate id");
    for g in &GRADES {
        let j = &grades[g.id];
        let fields = &j["fields"];
        let lit = &j["engine_literals"];
        let id = g.id;
        assert_eq!(g.name, j["display_name"].as_str().expect("a name"), "{id}");
        // Engine literals, bit for bit (typed as literals, never converted).
        assert_eq!(g.br_T, f(&lit["br_T"]), "{id} Br");
        assert_eq!(g.hcj20_kA_m, f(&lit["hcj20_kA_m"]), "{id} Hcj");
        assert_eq!(g.alpha_br_per_C, f(&lit["alpha_br_per_C"]), "{id} alpha");
        assert_eq!(
            g.beta_hcj_reference_per_C,
            f(&lit["beta_hcj_per_C"]),
            "{id} beta"
        );
        assert_eq!(g.tmax_C, f(&lit["tmax_C"]), "{id} Tmax");
        assert_eq!(g.density_g_mm3, f(&lit["density_g_mm3"]), "{id} density");
        // The engine's beta keeps a workbook default where the data file records one.
        let engine_beta = opt_f(&j["workbook_default_literals"]["beta_hcj_per_C"])
            .unwrap_or_else(|| f(&lit["beta_hcj_per_C"]));
        assert_eq!(g.beta_hcj_per_C, engine_beta, "{id} engine beta");
        // Display fields, in the data file's own units.
        assert_eq!(g.hcb_kA_m, f(&fields["Hcb_kA_m"]["value"]), "{id} Hcb");
        assert_eq!(
            g.bhmax_kJ_m3,
            f(&fields["BHmax_kJ_m3"]["value"]),
            "{id} BHmax"
        );
        assert_eq!(g.mu_rec, opt_f(&fields["mu_rec"]["value"]), "{id} mu_rec");
        let range = fields["alpha_Br_pct_per_C"]["temp_range_C"]
            .as_array()
            .map(|r| [f(&r[0]), f(&r[1])]);
        assert_eq!(g.coefficient_range_C, range, "{id} range");
        // Every value cites the data file's source.
        let s = &g.sources;
        for (what, rust, key) in [
            ("br", s.br, "Br_T"),
            ("hcj", s.hcj, "Hcj_kA_m"),
            ("hcb", s.hcb, "Hcb_kA_m"),
            ("bhmax", s.bhmax, "BHmax_kJ_m3"),
            ("alpha", s.alpha, "alpha_Br_pct_per_C"),
            ("beta", s.beta, "beta_Hcj_pct_per_C"),
            ("tmax", s.tmax, "Tmax_C"),
            ("density", s.density, "density_g_cm3"),
        ] {
            assert_eq!(
                Some(rust),
                opt_s(&fields[key]["value_source_url"]),
                "{id} {what} source"
            );
        }
        assert_eq!(
            s.mu_rec,
            opt_s(&fields["mu_rec"]["value_source_url"]),
            "{id} mu_rec source"
        );
        assert_eq!(
            s.coefficient_range,
            opt_s(&fields["alpha_Br_pct_per_C"]["temp_range_source_url"]),
            "{id} range source"
        );
    }
}

#[test]
fn every_library_part_resolves_to_a_grade() {
    // Spec, Addendum testing: "every library part resolves to a grade" (the workbook's
    // grade text, N50M included).
    for spec in &MAGNET_LIBRARY {
        assert!(
            grade(spec.grade).is_some(),
            "{} names {}",
            spec.part,
            spec.grade
        );
    }
}

#[test]
fn library_parts_agree_with_their_grade_except_where_registered() {
    // Grade data lives once: a part's workbook Br and Tmax equal its grade's, except
    // the N42SH rows' 1.29 T (E3 corrects it to the grade's 1.30 T) and the SuperMagnetMan
    // arcs' 1.42 T (decision 2 A keeps the workbook Br; K&J's N50 minimum is 1.41 T).
    for spec in &MAGNET_LIBRARY {
        let g: &Grade = grade(spec.grade).expect("resolves");
        assert_eq!(spec.tmax_C, g.tmax_C, "{} Tmax", spec.part);
        let expected_br = match (spec.grade, spec.vendor) {
            ("N42SH", _) => 1.29,
            (_, "SuperMagnetMan") => 1.42,
            _ => g.br_T,
        };
        assert_eq!(spec.br_T, expected_br, "{} Br", spec.part);
    }
}

#[test]
fn e3_corrects_to_the_n42sh_grade_minimum() {
    assert_eq!(N42SH_BR_CORRECTED_T, N42SH.br_T);
    assert_eq!(N42SH.br_T, 1.30);
}

#[test]
fn n42sh_keeps_the_workbook_coercivity() {
    // Decisions 17 A and 18 A: the default part's grade is default-neutral for E20.
    let demag = DemagInputs::default();
    assert_eq!(N42SH.hcj20_kA_m, demag.hcj20_kA_m);
    assert_eq!(N42SH.beta_hcj_per_C, demag.beta_hcj_per_C);
    assert_eq!(N42SH.beta_hcj_reference_per_C, -0.0055);
}

#[test]
fn sintered_ndfeb_grades_are_bit_equal_to_the_engine_constants() {
    // Report 2.5: alpha(Br) and density of every sintered NdFeB grade equal the engine's
    // -0.0012 /°C (Calibration!C22) and 0.0075 g/mm³, as literals.
    let alpha = magcoupling::engine::calibration::CalibrationInputs::default().alpha_br_per_C;
    for g in GRADES.iter().filter(|g| g.family == GradeFamily::NdFeB) {
        assert_eq!(g.alpha_br_per_C, alpha, "{}", g.id);
        assert_eq!(g.density_g_mm3, NDFEB_DENSITY_G_MM3, "{}", g.id);
    }
}

#[test]
fn only_ferrite_has_a_positive_beta() {
    for g in &GRADES {
        let positive = g.beta_hcj_per_C > 0.0;
        assert_eq!(positive, g.family == GradeFamily::Ferrite, "{}", g.id);
        assert_eq!(
            g.beta_hcj_per_C.signum(),
            g.beta_hcj_reference_per_C.signum(),
            "{}",
            g.id
        );
    }
    assert_eq!(grade("Y30").map(|g| g.beta_hcj_per_C), Some(0.0035)); // decision 3 A
}

#[test]
fn every_part_cites_its_vendor_page_for_coating_and_magnetization() {
    for spec in &MAGNET_LIBRARY {
        let page = spec.page;
        match spec.vendor {
            "K&J" => {
                assert_eq!(
                    page,
                    format!(
                        "https://www.kjmagnetics.com/proddetail.asp?prod={}",
                        spec.part
                    )
                );
                assert_eq!(
                    spec.coating, "Nickel-Copper-Nickel (Ni-Cu-Ni)",
                    "{}",
                    spec.part
                );
                assert_eq!(
                    spec.magnetization, "Magnetized Through Thickness",
                    "{}",
                    spec.part
                );
            }
            "SuperMagnetMan" => {
                assert_eq!(
                    page,
                    format!(
                        "https://supermagnetman.com/products/{}",
                        spec.part.to_lowercase()
                    )
                );
                assert_eq!(spec.coating, "Nickel", "{}", spec.part);
                assert_eq!(spec.magnetization, "Radially magnetized", "{}", spec.part);
            }
            other => panic!("{}: unexpected vendor {other}", spec.part),
        }
    }
}
