//! Materials sheet ('Materials').
//!
//! Port of `reference/magcoupling-py/magcoupling/materials.py`. This module
//! holds the input structs ([`Steel4140`], [`ElectrolessNickel`],
//! [`ScrewClasses`] and their group [`MaterialsInputs`]), the aluminium alloys
//! ([`AL7075`], [`AL6061`]: the Python `aluminium` member has no metadata, so
//! they are static data, not inputs), [`ScrewClasses::proof`] and the sheet's
//! results ([`MaterialsResults`], [`compute`], 7 result cells).
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied are E9 (Materials!C22
//! reads "No back iron" when Calculator!C6 = 0) and E10 (Materials!C20 to C22
//! follow the corrected gap flux density through Calculator!C104; no code
//! here).

use super::compat::{ceiling, fmt_fixed};
use super::deviations::{DeviationId, Deviations};
use super::material_library::PartProperties;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, param_rust_only, results};

inputs! {
    /// 4140 steel properties (Materials!C13:C19).
    pub struct Steel4140 {
        fields {
            bsat_T: f64 = 1.5 => param("T", "Design flux density for the back-iron check",
                "Annealed 4140. 1018 would be about 1.7 T; use about 1.4 T for pre-hardened stock.", "Materials!C13")
                .range(0.5, 2.2, 0.01)
                .assumption(),
            conductivity_S_m: f64 = 4.5e6 => param("S/m", "Electrical conductivity",
                "Resistivity about 0.22 µΩ·m.", "Materials!C14")
                .range(1e6, 1e7, 1e4)
                .log(),
            mu_r_incremental: f64 = 200.0 => param("-", "Incremental relative permeability (with the magnet bias)",
                "", "Materials!C15")
                .range(1.0, 2000.0, 1.0)
                .log(),
            specific_heat_J_kgK: f64 = 473.0 => param("J/(kg·K)", "Specific heat",
                "", "Materials!C16")
                .range(300.0, 1000.0, 1.0),
            cte_per_C: f64 = 12.3e-6 => param("1/°C", "Expansion coefficient", "", "Materials!C17")
                .range(5e-6, 25e-6, 1e-7),
            modulus_GPa: f64 = 205.0 => param("GPa", "Elastic modulus", "", "Materials!C18")
                .range(50.0, 250.0, 1.0),
            density_g_cm3: f64 = 7.85 => param("g/cm³", "Density",
                "Same as 1018; the mass model uses Metal design C132.", "Materials!C19")
                .range(7.0, 8.2, 0.01),
        }
    }
}

inputs! {
    /// Electroless nickel plating (Materials!C26).
    pub struct ElectrolessNickel {
        fields {
            thickness_mm: f64 = 0.015 => param("mm", "Plating thickness per surface",
                "High-phosphorus EN (10–12 % P): non-magnetic as plated. Typical 0.013–0.025 mm.", "Materials!C26")
                .range(0.0, 0.05, 0.001),
        }
    }
}

/// An aluminium alloy's properties (Materials!C34:C43). Plain data in Python
/// (no metadata), so static consts here, not inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct AluminiumAlloy {
    pub name: &'static str,
    pub yield_MPa: f64,
    pub shear_MPa: f64,
    pub head_pressure_limit_MPa: f64,
    pub key_bearing_allow_MPa: f64,
    pub conductivity_S_m: f64,
}

/// 7075-T6 (Materials!C34:C38): clamp collars and adapters.
pub const AL7075: AluminiumAlloy = AluminiumAlloy {
    name: "7075-T6",
    yield_MPa: 503.0,
    shear_MPa: 331.0,
    head_pressure_limit_MPa: 400.0,
    key_bearing_allow_MPa: 100.0,
    conductivity_S_m: 1.9e7,
};

/// 6061-T6 (Materials!C39:C43): cap, housing, brackets.
pub const AL6061: AluminiumAlloy = AluminiumAlloy {
    name: "6061-T6",
    yield_MPa: 276.0,
    shear_MPa: 207.0,
    head_pressure_limit_MPa: 250.0,
    key_bearing_allow_MPa: 60.0,
    conductivity_S_m: 2.5e7,
};

inputs! {
    /// Screw class strengths (Materials!C48:C50).
    pub struct ScrewClasses {
        fields {
            proof_12_9_MPa: f64 = 970.0 => param("MPa", "Class 12.9 proof stress",
                "ISO 898-1.", "Materials!C48")
                .range(500.0, 1200.0, 5.0),
            proof_10_9_MPa: f64 = 830.0 => param("MPa", "Class 10.9 proof stress",
                "", "Materials!C49")
                .range(400.0, 1100.0, 5.0),
            yield_A4_70_MPa: f64 = 450.0 => param("MPa", "Stainless A4-70 yield stress",
                "", "Materials!C50")
                .range(200.0, 800.0, 5.0),
        }
    }
}

impl ScrewClasses {
    /// Proof stress by class code: 1 = 12.9, 2 = 10.9, 3 = A4-70 (workbook CHOOSE order).
    /// Python raises KeyError for any other code; here it is NaN (decision D3): no
    /// panic, and `DesignInputs::validate` reports the code.
    pub fn proof(&self, code: i64) -> f64 {
        match code {
            1 => self.proof_12_9_MPa,
            2 => self.proof_10_9_MPa,
            3 => self.yield_A4_70_MPa,
            _ => f64::NAN,
        }
    }
}

inputs! {
    /// The material of each part (Addendum A5, Rust-only selectors). Code 1 of each
    /// is the workbook's material, whose values are the inputs; the choices are the
    /// library records of `material_library` (`BACK_IRON_CHOICES` and the others,
    /// tested equal to these texts).
    pub struct PartMaterialInputs {
        fields {
            back_iron: i64 = 1 => param_rust_only("-", "Back iron material (hub, cup and boss)",
                "Addendum A5. 1 = the workbook's 4140: the steel inputs above. Another choice supplies its library conductivity, density, specific heat, expansion and modulus, and its design flux density where the library has one (else C13 stays). A non-ferromagnetic choice selects the free-space circuit and becomes the hub, cup and boss material; with a ferromagnetic one, Calculator C6 = 0 still selects the free-space circuit (the override).")
                .choices(&[
                    (1, "4140 annealed"),
                    (2, "1018 hot rolled"),
                    (3, "12L14 cold drawn"),
                    (4, "416 stainless, annealed"),
                    (5, "17-4PH H1150"),
                    (6, "17-4PH H900"),
                    (7, "304 stainless (non-magnetic)"),
                    (8, "6061-T6 aluminium"),
                ]),
            sleeve_liner: i64 = 1 => param_rust_only("-", "Sleeve and liner material",
                "Addendum A5. 1 = the workbook's 316L (Temperature design C111 and C139, Metal design C44). Another choice supplies its conductivity, density and specific heat; the endplates follow it, as C44 prices them.")
                .choices(&[
                    (1, "316L annealed"),
                    (2, "Ti-6Al-4V grade 5"),
                    (3, "Inconel 625"),
                    (4, "PEEK"),
                ]),
            cap_housing: i64 = 1 => param_rust_only("-", "Cap and housing material",
                "Addendum A5. 1 = the workbook's 6061-T6 (Materials C43, Temperature design C140, Metal design C42). Another choice supplies the cap's conductivity, density and specific heat; the aluminium adapter and the clamp alloy keep their own inputs.")
                .choices(&[
                    (1, "6061-T6 aluminium"),
                    (2, "7075-T6 aluminium"),
                    (3, "Acetal (POM-H)"),
                ]),
        }
    }
}

inputs! {
    /// Every Materials input, grouped as the Python `MaterialsInputs` (plus the
    /// Rust-only part selectors).
    pub struct MaterialsInputs {
        fields {}
        groups {
            steel: Steel4140,
            nickel: ElectrolessNickel,
            screws: ScrewClasses,
            parts: PartMaterialInputs,
        }
    }
}

results! {
    /// Materials sheet results (Materials!C20:C22, C27:C30).
    pub struct MaterialsResults {
        fields {
            backiron_thickness_needed_mm: f64 => out("mm", "Back-iron thickness needed at this flux density", "", "Materials!C20"),
            cup_wall_corner_mm: f64 => out("mm", "Cup wall at the pocket corners", "", "Materials!C21"),
            cup_wall_check: String => out("", "Cup wall check",
                "A thicker corner wall grows the cup OD by twice the change; recheck the cap thread and envelope.",
                "Materials!C22"),
            cup_wall_suggested_mm: NumOrText => out_rust_only("mm", "Suggested cup wall at the pocket corners (autofit)",
                "Addendum A1 autofit, decision 27: the wall check's rule, the back-iron thickness needed rounded up to 0.1 mm (the number the check's advice quotes). The wall stays an input (Metal design C122). 'n/a' when the check reads 'No back iron'."),
            hub_flats_under_mm: f64 => out("mm", "Machine the hub flats under by", "On the apothem.", "Materials!C27"),
            cup_pockets_over_mm: f64 => out("mm", "Machine the cup pockets over by", "On the apothem.", "Materials!C28"),
            bores_over_dia_mm: f64 => out("mm", "Machine bores over (on diameter)", "", "Materials!C29"),
            ods_under_dia_mm: f64 => out("mm", "Machine outside diameters under (on diameter)", "", "Materials!C30"),
            circuit_backiron: i64 => out_rust_only("-", "Back-iron circuit in effect",
                "1 = steel circuit, 0 = free space: Calculator C6, or 0 for a non-ferromagnetic back iron (Addendum A5)."),
            back_iron_material: String => out_rust_only("", "Back iron material", ""),
            sleeve_liner_material: String => out_rust_only("", "Sleeve and liner material", ""),
            cap_material: String => out_rust_only("", "Cap and housing material", ""),
            steel_density_g_mm3: f64 => out_rust_only("g/mm³", "Back-iron steel density in effect",
                "Plan A-3 (a term of the equation explorer): Metal design C132, or a ferromagnetic back-iron pick's library density (Addendum A5); the mass model prices the steel parts with it."),
            body_sigma_S_m: f64 => out_rust_only("S/m", "Hub, cup and boss conductivity without back iron",
                "The workbook's 6061-T6 (Materials C43), or a non-ferromagnetic back-iron pick's (Addendum A5); correction E17 prices the aluminium parts' slip losses with it."),
            body_density_g_mm3: f64 => out_rust_only("g/mm³", "Hub, cup and boss density without back iron",
                "Metal design C42, or a non-ferromagnetic back-iron pick's (Addendum A5): the aluminium hub, cup and boss (E9)."),
            body_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Hub, cup and boss specific heat without back iron",
                "Temperature design C140, or a non-ferromagnetic back-iron pick's (Addendum A5): correction E15's heat capacity."),
            sleeve_sigma_S_m: f64 => out_rust_only("S/m", "Sleeve and liner conductivity in effect",
                "Temperature design C111, or the sleeve and liner pick's (Addendum A5)."),
            sleeve_density_g_mm3: f64 => out_rust_only("g/mm³", "Sleeve, liner and endplate density in effect",
                "Metal design C44, or the sleeve and liner pick's (Addendum A5)."),
            sleeve_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Sleeve, liner and endplate specific heat in effect",
                "Temperature design C139, or the sleeve and liner pick's (Addendum A5)."),
            cap_density_g_mm3: f64 => out_rust_only("g/mm³", "Cap density in effect",
                "Metal design C42, or the cap pick's (Addendum A5)."),
            cap_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Cap specific heat in effect",
                "Temperature design C140, or the cap pick's (Addendum A5)."),
        }
    }
}

/// Text of `materials.cup_wall_suggested_mm` when the wall check reads "No back iron":
/// no magnetic rule sizes an aluminium cup's wall.
pub const NO_WALL_RULE: &str = "n/a";

/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
/// `backiron` is the Calculator selector C6 in effect (1 steel, 0 none); only E9
/// reads it. `parts` names the materials in effect (Rust-only results).
pub fn compute(
    mat: &MaterialsInputs,
    t_bi_req_mm: f64,
    wall_corner_mm: f64,
    backiron: i64,
    parts: &PartProperties,
    dev: Deviations,
) -> MaterialsResults {
    // Addendum A1 autofit (decision 27): the rule's wall, the number the advice quotes.
    let suggested = ceiling(t_bi_req_mm, 0.1);
    let no_back_iron = dev.is_on(DeviationId::E9) && backiron == 0;
    let check = if no_back_iron {
        "No back iron".to_owned() // as the Calculator's C105/C106 read
    } else if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!(
            "Too thin: raise Metal design C122 to at least {} mm",
            fmt_fixed(suggested, 1)
        )
    };
    let t = mat.nickel.thickness_mm;
    MaterialsResults {
        backiron_thickness_needed_mm: t_bi_req_mm,
        cup_wall_corner_mm: wall_corner_mm,
        cup_wall_check: check,
        cup_wall_suggested_mm: if no_back_iron {
            NumOrText::Text(NO_WALL_RULE)
        } else {
            NumOrText::Num(suggested)
        },
        hub_flats_under_mm: t,
        cup_pockets_over_mm: t,
        bores_over_dia_mm: 2.0 * t,
        ods_under_dia_mm: 2.0 * t,
        circuit_backiron: backiron,
        back_iron_material: parts.back_iron.label().to_owned(),
        sleeve_liner_material: parts.sleeve_liner.label().to_owned(),
        cap_material: parts.cap.label().to_owned(),
        steel_density_g_mm3: parts.steel.density_g_mm3,
        body_sigma_S_m: parts.body.sigma_S_m,
        body_density_g_mm3: parts.body.density_g_mm3,
        body_c_J_kgK: parts.body.cp_J_kgK,
        sleeve_sigma_S_m: parts.sleeve_liner.props.sigma_S_m,
        sleeve_density_g_mm3: parts.sleeve_liner.props.density_g_mm3,
        sleeve_c_J_kgK: parts.sleeve_liner.props.cp_J_kgK,
        cap_density_g_mm3: parts.cap.props.density_g_mm3,
        cap_c_J_kgK: parts.cap.props.cp_J_kgK,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::material_library::resolve;
    use crate::engine::metal_design::MetalDesignInputs;
    use crate::engine::temperature::{SlipLossInputs, ThermalInputs};

    /// The default parts (every selector at code 1).
    fn parts() -> PartProperties {
        let mat = MaterialsInputs::default();
        resolve(
            &mat.parts,
            &mat.steel,
            1,
            &MetalDesignInputs::default(),
            &SlipLossInputs::default(),
            &ThermalInputs::default(),
        )
    }

    #[test]
    fn wall_check_passes_at_equality() {
        let mat = MaterialsInputs::default();
        // Architecture section 7 step 8: the comparison sits exactly at equality.
        let (needed, corner) = (1.8, 1.8);
        assert_eq!(needed, corner);
        assert_eq!(
            compute(&mat, needed, corner, 1, &parts(), Deviations::NONE).cup_wall_check,
            "OK"
        );
        assert_eq!(
            compute(&mat, 1.90415278222222, 1.8, 1, &parts(), Deviations::NONE).cup_wall_check,
            "Too thin: raise Metal design C122 to at least 2.0 mm" // Materials!C22
        );
    }

    #[test]
    fn the_wall_suggestion_is_the_number_the_advice_quotes() {
        // Addendum A1 autofit, decision 27: the wall stays an input and the rule's wall is a
        // suggestion, the back-iron need rounded up to 0.1 mm (the advice's own number).
        let mat = MaterialsInputs::default();
        let r = compute(&mat, 1.90415278222222, 1.8, 1, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(2.0));
        assert_eq!(
            r.cup_wall_check,
            "Too thin: raise Metal design C122 to at least 2.0 mm"
        );
        // The suggestion does not depend on the wall, and a need of exactly 1.9 mm stays 1.9 mm
        // (the 1e-12 guard of `ceiling`): 19 steps of 0.1, as Excel's CEILING gives it.
        let r = compute(&mat, 1.90415278222222, 2.5, 1, &parts(), Deviations::NONE);
        assert_eq!(
            (r.cup_wall_suggested_mm, r.cup_wall_check.as_str()),
            (NumOrText::Num(2.0), "OK")
        );
        let r = compute(&mat, 1.9, 1.8, 1, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(19.0 * 0.1));
        assert_eq!(
            r.cup_wall_check,
            "Too thin: raise Metal design C122 to at least 1.9 mm"
        );
        // No back iron (E9): no magnetic rule, as the check reads "No back iron"; without E9 the
        // workbook still advises a wall, and so does the suggestion.
        let e9 = Deviations::only(DeviationId::E9);
        let r = compute(&mat, 1.90415278222222, 1.8, 0, &parts(), e9);
        assert_eq!(
            (r.cup_wall_suggested_mm, r.cup_wall_check.as_str()),
            (NumOrText::Text(NO_WALL_RULE), "No back iron")
        );
        let r = compute(&mat, 1.90415278222222, 1.8, 0, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(2.0));
    }

    #[test]
    fn e9_no_back_iron_replaces_the_wall_advice_only_at_code_0() {
        let mat = MaterialsInputs::default();
        let too_thin = "Too thin: raise Metal design C122 to at least 2.0 mm";
        let check = |backiron, dev| {
            compute(&mat, 1.90415278222222, 1.8, backiron, &parts(), dev).cup_wall_check
        };
        let e9 = Deviations::only(DeviationId::E9);
        assert_eq!(check(0, e9), "No back iron");
        assert_eq!(check(0, Deviations::NONE), too_thin); // the workbook ignores C6 here
        assert_eq!(check(1, e9), too_thin);
        // A code outside {0, 1} keeps the wall advice, as the Calculator's C105/C106
        // test `== 0`; `DesignInputs::validate` reports the code.
        assert_eq!(check(2, e9), too_thin);
        // A wall that meets the need still reads "No back iron" with no back iron.
        assert_eq!(
            compute(&mat, 1.8, 1.8, 0, &parts(), e9).cup_wall_check,
            "No back iron"
        );
    }

    #[test]
    fn the_materials_in_effect_are_the_picks_values() {
        // Plan A-3: the Rust-only materials in effect read what `resolve` picked. At the
        // defaults they are the inputs; a non-ferromagnetic back iron (6061) becomes the hub,
        // cup and boss material and leaves the steel at the inputs; the sleeve and cap picks
        // supply their library values.
        use crate::engine::material_library::material;
        let md = MetalDesignInputs::default();
        let with = |back_iron: i64, sleeve_liner: i64, cap_housing: i64| {
            let mut mat = MaterialsInputs::default();
            mat.parts.back_iron = back_iron;
            mat.parts.sleeve_liner = sleeve_liner;
            mat.parts.cap_housing = cap_housing;
            let p = resolve(
                &mat.parts,
                &mat.steel,
                1,
                &md,
                &SlipLossInputs::default(),
                &ThermalInputs::default(),
            );
            compute(&mat, 1.9, 1.8, p.backiron, &p, Deviations::NONE)
        };
        let r = with(1, 1, 1);
        assert_eq!(
            (
                r.steel_density_g_mm3,
                r.body_sigma_S_m,
                r.body_density_g_mm3,
                r.sleeve_density_g_mm3,
                r.cap_density_g_mm3
            ),
            (
                md.steel_density_g_mm3,
                AL6061.conductivity_S_m,
                md.al_density_g_mm3,
                md.sleeve_density_g_mm3,
                md.al_density_g_mm3
            )
        );
        let al = material("6061_T6").unwrap().engine;
        let r = with(8, 1, 1);
        assert_eq!(
            (r.body_sigma_S_m, r.body_density_g_mm3, r.body_c_J_kgK),
            (al.sigma_S_m, al.density_g_mm3, al.cp_J_kgK)
        );
        assert_eq!(
            r.steel_density_g_mm3, md.steel_density_g_mm3,
            "a non-ferromagnetic pick leaves the steel at the inputs"
        );
        let (ti, pom) = (
            material("Ti6Al4V_annealed").unwrap().engine,
            material("POM_H_acetal").unwrap().engine,
        );
        let r = with(1, 2, 3);
        assert_eq!(
            (r.sleeve_sigma_S_m, r.sleeve_density_g_mm3, r.sleeve_c_J_kgK),
            (ti.sigma_S_m, ti.density_g_mm3, ti.cp_J_kgK)
        );
        assert_eq!(
            (r.cap_density_g_mm3, r.cap_c_J_kgK),
            (pom.density_g_mm3, pom.cp_J_kgK)
        );
    }

    #[test]
    fn proof_follows_the_class_code_and_is_nan_outside_it() {
        let s = ScrewClasses::default();
        assert_eq!([s.proof(1), s.proof(2), s.proof(3)], [970.0, 830.0, 450.0]);
        for bad in [0, 4, -1, i64::MIN, i64::MAX] {
            assert!(s.proof(bad).is_nan(), "{bad}");
        }
    }
}
