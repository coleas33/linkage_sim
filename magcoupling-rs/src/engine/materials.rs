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
use super::meta::{inputs, out, out_rust_only, param, param_rust_only, results};

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
            hub_flats_under_mm: f64 => out("mm", "Machine the hub flats under by", "On the apothem.", "Materials!C27"),
            cup_pockets_over_mm: f64 => out("mm", "Machine the cup pockets over by", "On the apothem.", "Materials!C28"),
            bores_over_dia_mm: f64 => out("mm", "Machine bores over (on diameter)", "", "Materials!C29"),
            ods_under_dia_mm: f64 => out("mm", "Machine outside diameters under (on diameter)", "", "Materials!C30"),
            circuit_backiron: i64 => out_rust_only("-", "Back-iron circuit in effect",
                "1 = steel circuit, 0 = free space: Calculator C6, or 0 for a non-ferromagnetic back iron (Addendum A5)."),
            back_iron_material: String => out_rust_only("", "Back iron material", ""),
            sleeve_liner_material: String => out_rust_only("", "Sleeve and liner material", ""),
            cap_material: String => out_rust_only("", "Cap and housing material", ""),
        }
    }
}

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
    let check = if dev.is_on(DeviationId::E9) && backiron == 0 {
        "No back iron".to_owned() // as the Calculator's C105/C106 read
    } else if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!(
            "Too thin: raise Metal design C122 to at least {} mm",
            fmt_fixed(ceiling(t_bi_req_mm, 0.1), 1)
        )
    };
    let t = mat.nickel.thickness_mm;
    MaterialsResults {
        backiron_thickness_needed_mm: t_bi_req_mm,
        cup_wall_corner_mm: wall_corner_mm,
        cup_wall_check: check,
        hub_flats_under_mm: t,
        cup_pockets_over_mm: t,
        bores_over_dia_mm: 2.0 * t,
        ods_under_dia_mm: 2.0 * t,
        circuit_backiron: backiron,
        back_iron_material: parts.back_iron.label().to_owned(),
        sleeve_liner_material: parts.sleeve_liner.label().to_owned(),
        cap_material: parts.cap.label().to_owned(),
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
    fn proof_follows_the_class_code_and_is_nan_outside_it() {
        let s = ScrewClasses::default();
        assert_eq!([s.proof(1), s.proof(2), s.proof(3)], [970.0, 830.0, 450.0]);
        for bad in [0, 4, -1, i64::MIN, i64::MAX] {
            assert!(s.proof(bad).is_nan(), "{bad}");
        }
    }
}
