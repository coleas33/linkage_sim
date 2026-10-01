//! The A6 grade table and the part table against the approved Addendum A data
//! file (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every
//! value and every citation, every part resolving to a grade, and the places
//! where a part's workbook value differs from its grade on purpose.

mod common;

use common::{ADDENDUM_DATA, read_json, repo_path};
use magcoupling::engine::constants::NDFEB_DENSITY_G_MM3;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library::{MAGNET_LIBRARY, N42SH_BR_CORRECTED_T, grade_id};
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
fn each_grade_lists_the_parts_that_resolve_to_it() {
    // The data file's `library_parts` is decision 2 A (E19 on: M5045 is N50);
    // `library_parts_under_decision_2_B_or_C` is the workbook's own mapping (E19 off).
    let doc = data();
    for g in &GRADES {
        for (dev, key) in [
            (Deviations::ALL, "library_parts"),
            (Deviations::NONE, "library_parts_under_decision_2_B_or_C"),
        ] {
            let listed = &doc["grades"][g.id][key];
            let listed = if listed.is_null() {
                &doc["grades"][g.id]["library_parts"]
            } else {
                listed
            };
            let want: Vec<&str> = listed
                .as_array()
                .expect("a part list")
                .iter()
                .map(|p| p.as_str().expect("a part"))
                .collect();
            let got: Vec<&str> = MAGNET_LIBRARY
                .iter()
                .filter(|spec| grade_id(spec, dev) == g.id)
                .map(|spec| spec.part)
                .collect();
            assert_eq!(got, want, "{} {key}", g.id);
        }
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

#[test]
fn a_grade_ring_scales_with_its_own_alpha_and_weighs_at_its_density() {
    // Decision A2-7 end to end: hard ferrite Y30 picked for manual dimensions on both rings,
    // with the calculator's alpha (Calibration C22) left at the NdFeB -0.0012: the rings take
    // Y30's -0.20 %/C and 5.0 g/cm3. Setting C22 to Y30's value by hand (the A-1 way) gives
    // the same Calculator torques and temperature limits; only C35 and the parts C22 alone
    // drives (the Calibration prototype, C151) differ.
    use magcoupling::{DesignInputs, compute_all};
    let mut graded = DesignInputs::default();
    let m = &mut graded.coupling.magnets;
    m.part_inner = String::new();
    m.part_outer = String::new();
    m.grade_inner = "Y30".to_owned();
    m.grade_outer = "Y30".to_owned();
    let r = compute_all(&graded);
    assert_eq!(
        (r.model.inner_alpha_br_per_C, r.model.outer_alpha_br_per_C),
        (-0.002, -0.002)
    );
    let volume = r.model.inner_length_mm * r.model.inner_width_mm * r.model.inner_thickness_mm;
    let volume_o = r.model.outer_length_mm * r.model.outer_width_mm * r.model.outer_thickness_mm;
    assert_eq!(r.mass.magnets_g, 10.0 * (volume + volume_o) * 0.005);
    let mut by_hand = graded.clone();
    by_hand.calibration.alpha_br_per_C = -0.002;
    let h = compute_all(&by_hand);
    assert_eq!(r.model.pullout_Nm, h.model.pullout_Nm);
    assert_eq!(r.metal.torque_cold_high_Nm, h.metal.torque_cold_high_Nm);
    assert_eq!(
        r.temperature.summary.governing_limit_C,
        h.temperature.summary.governing_limit_C
    );
    assert_eq!(r.temperature.demag.alpha_br, -0.002);
    assert_eq!(r.model.alpha_br_per_C, -0.0012);
}

#[test]
fn mixed_rings_each_take_their_own_alpha_and_density_either_way_round() {
    // Decision A2-7 end to end with two different coefficients, where a swap of the rings'
    // wiring shows (with identical rings every product commutes): the library NdFeB part
    // (B842SH: C22's -0.12 %/C and the NdFeB density) beside a Y30 ring in the grade mode
    // (-0.20 %/C, 5.0 g/cm3), each way round. Under E20 with the coercivity source at 1,
    // Y30's positive beta puts that ring on the cold side, so the NdFeB ring governs the hot
    // limit, with the onsets it has in the default design (two NdFeB parts), and the Y30 ring
    // the cold one.
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    use magcoupling::{DesignInputs, compute_all};
    let th = |alpha: f64, t: f64| 1.0 + alpha * (t - 20.0); // model::br_factor, t in °C
    let ndfeb = (-0.0012, NDFEB_DENSITY_G_MM3);
    let y30 = (-0.002, 0.005);
    let default_demag = compute_all(&DesignInputs::default()).temperature.demag;
    for (y30_inner, ((alpha_i, rho_i), (alpha_o, rho_o)), (hot_ring, cold_ring)) in [
        (false, (ndfeb, y30), (RING_INNER, RING_OUTER)),
        (true, (y30, ndfeb), (RING_OUTER, RING_INNER)),
    ] {
        let label = if y30_inner { "Y30 inner" } else { "Y30 outer" };
        let mut d = DesignInputs::default();
        let m = &mut d.coupling.magnets;
        if y30_inner {
            m.part_inner = String::new();
            m.grade_inner = "Y30".to_owned();
        } else {
            m.part_outer = String::new();
            m.grade_outer = "Y30".to_owned();
        }
        let r = compute_all(&d);
        assert_eq!(
            (r.model.inner_alpha_br_per_C, r.model.outer_alpha_br_per_C),
            (alpha_i, alpha_o),
            "{label}"
        );
        assert_eq!(
            (
                r.model.inner_magnet_density_g_mm3,
                r.model.outer_magnet_density_g_mm3
            ),
            (rho_i, rho_o),
            "{label}"
        );
        // Torques at another temperature: both rings' factors, in the engine's operand order.
        let (t_min, t_op) = (d.metal.min_temp_C, r.metal.op_temp_C);
        assert_eq!(
            r.metal.torque_cold_Nm,
            r.metal.torque_20C_Nm * (th(alpha_i, t_min) * th(alpha_o, t_min)),
            "{label}"
        );
        assert_eq!(
            r.metal.cold_for_hot_min_Nm,
            d.metal.required_min_Nm
                * ((th(alpha_i, t_min) / th(alpha_i, t_op))
                    * (th(alpha_o, t_min) / th(alpha_o, t_op))),
            "{label}"
        );
        // Each ring's E20 check with its own coefficient: the block (C43) shows the governing
        // NdFeB ring's, and that ring's onsets are the default design's bit for bit.
        let demag = &r.temperature.demag;
        assert_eq!(
            (demag.demag_ring.as_str(), demag.cold_ring.as_str()),
            (hot_ring, cold_ring),
            "{label}"
        );
        assert_eq!(demag.alpha_br, ndfeb.0, "{label}");
        assert_eq!(
            (
                demag.onset_aligned_C,
                demag.onset_pullout_C,
                demag.onset_skipping_C,
                demag.onset_single_ring_C,
                demag.magnet_limit_C
            ),
            (
                default_demag.onset_aligned_C,
                default_demag.onset_pullout_C,
                default_demag.onset_skipping_C,
                default_demag.onset_single_ring_C,
                default_demag.magnet_limit_C
            ),
            "{label}"
        );
        // Each ring weighs at its own density; the bond block is the inner ring's.
        let volume_i =
            r.model.inner_length_mm * r.model.inner_width_mm * r.model.inner_thickness_mm;
        let volume_o =
            r.model.outer_length_mm * r.model.outer_width_mm * r.model.outer_thickness_mm;
        let n = d.coupling.npole as f64;
        assert_eq!(
            r.mass.magnets_g,
            n * (volume_i * rho_i + volume_o * rho_o),
            "{label}"
        );
        assert_eq!(
            r.temperature.adhesive.block_mass_g,
            volume_i * rho_i,
            "{label}"
        );
    }
}
