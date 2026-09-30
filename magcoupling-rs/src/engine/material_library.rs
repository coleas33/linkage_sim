//! Materials per part (spec Addendum A5): the library, the choices each part
//! offers, and (with the selectors) what a choice feeds into the engine.
//!
//! Data: `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`, key
//! `"materials"`, approved with the Addendum A verification (decisions 5 to 7
//! and 20 to 26, option A; resolutions M1 to M24 in its section 4.2). Each
//! property keeps the data file's selected value and its source in a
//! [`Sourced`]; `tests/material_library.rs` compares every value and citation
//! with the data file. One value differs from the data file on purpose: the 416
//! expansion coefficient, re-sourced from a standard-416 sheet as decision 6 A
//! asks (Rolled Alloys, 5.6e-6 /F from 70 to 212 F = 10.08e-6 /K; the data file
//! kept Zapp's 10.5 as a placeholder).
//!
//! [`Material::engine`] holds what the engine uses when a material is picked:
//! the workbook's number where the workbook has one (4140, 316L, 6061, 7075:
//! decisions 21 to 26 keep them as the defaults, the sourced value is the
//! reference), else the sourced value, as engine-unit literals. The default
//! choice of each part is the workbook's material, whose values ARE the
//! existing inputs (Materials C13 to C18, Temperature design C111, C139, C140,
//! Metal design C42, C44, C132): picking it changes nothing, so parity and the
//! differential tests hold (decision table of the A-1 plan).

/// A property as the data file selects it: the value in the data file's unit and
/// its source, or `None` where no source was found or the property does not apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sourced {
    pub value: Option<f64>,
    pub source: Option<&'static str>,
}

/// A sourced property.
const fn sourced(value: f64, source: &'static str) -> Sourced {
    Sourced {
        value: Some(value),
        source: Some(source),
    }
}

/// A property with no source (a null in the data file, with its reason there).
const NOT_SOURCED: Sourced = Sourced {
    value: None,
    source: None,
};

/// What the engine uses when a material is picked (engine units).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct EngineProps {
    /// Electrical conductivity [S/m]: slip losses.
    pub sigma_S_m: f64,
    /// Density [g/mm³]: mass and heat capacity.
    pub density_g_mm3: f64,
    /// Specific heat [J/(kg·K)]: heat capacity.
    pub cp_J_kgK: f64,
    /// Expansion coefficient [1/°C]: the bond-stress screen.
    pub cte_per_C: f64,
    /// Elastic modulus [GPa]: the bond-stress screen.
    pub modulus_GPa: f64,
}

/// One library material.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct Material {
    /// The data file's id.
    pub id: &'static str,
    /// The selector's text.
    pub label: &'static str,
    pub name: &'static str,
    pub condition: &'static str,
    /// Selects the steel circuit (as back iron) and short-circuits the gap (as sleeve).
    pub ferromagnetic: bool,
    pub ferromagnetic_source: &'static str,
    /// Relative permeability (reference only: secant, maximum or chart-read values;
    /// no source gives the incremental value at the magnet bias, so the engine keeps
    /// Materials C15 for every steel).
    pub mu_r: Sourced,
    /// Saturation flux density [T]: informational; drives the low-saturation warning.
    pub bsat_T: Sourced,
    pub sigma_S_m: Sourced,
    pub density_g_cm3: Sourced,
    /// Expansion coefficient [1e-6/K].
    pub cte_1e6_per_K: Sourced,
    pub modulus_GPa: Sourced,
    /// Yield strength [MPa] (published minimum where one exists; not read by the engine).
    pub yield_MPa: Sourced,
    pub cp_J_kgK: Sourced,
    /// The wall check's design flux density [T] (decision 20): workbook values only,
    /// 4140 1.5 T (Materials C13) and 1018 about 1.7 T (that cell's comment).
    pub design_flux_density_T: Option<f64>,
    pub engine: EngineProps,
    /// Plain or low-alloy steel: needs plating against corrosion.
    pub needs_plating: bool,
    /// The report's resolutions (M#) and flags.
    pub notes: &'static str,
}

/// Every library material, in the data file's order.
pub const MATERIALS: [Material; 14] = [
    Material {
        id: "4140_annealed",
        label: "4140 annealed",
        name: "AISI 4140 (EN 42CrMo4), annealed",
        condition: "Annealed (815-870 C), ~197 HB. Workbook plan: annealed 4140 with electroless nickel.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        mu_r: sourced(
            363.0,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            4330000.0,
            "https://www.lucefin.com/wp-content/files_mf/152353604042CrMo4.pdf",
        ),
        density_g_cm3: sourced(7.85, "https://www.azom.com/article.aspx?ArticleID=6769"),
        cte_1e6_per_K: sourced(12.2, "https://www.azom.com/article.aspx?ArticleID=6769"),
        modulus_GPa: sourced(
            205.0,
            "https://otaisteel.com/aisi-4140-material-data-sheet/",
        ),
        yield_MPa: sourced(
            415.0,
            "https://otaisteel.com/aisi-4140-material-data-sheet/",
        ),
        cp_J_kgK: sourced(
            461.0,
            "https://www.lucefin.com/wp-content/files_mf/152353604042CrMo4.pdf",
        ),
        design_flux_density_T: Some(1.5),
        engine: EngineProps {
            sigma_S_m: 4500000.0,
            density_g_mm3: 0.00785,
            cp_J_kgK: 473.0,
            cte_per_C: 12.3e-6,
            modulus_GPa: 205.0,
        },
        needs_plating: true,
        notes: "Default back iron (the Materials sheet's inputs). mu_r 363 is the secant value at 1.5 T, not the incremental value the skin depth needs (decision 22). M13.",
    },
    Material {
        id: "1018_hot_rolled",
        label: "1018 hot rolled",
        name: "AISI 1018 (UNS G10180)",
        condition: "Hot rolled bar for yield; annealed for resistivity, CTE and specific heat (as the sources state). Cold drawn yield is 370 MPa.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            5848000.0,
            "https://www.azom.com/article.aspx?ArticleID=6115",
        ),
        density_g_cm3: sourced(7.87, "https://www.azom.com/article.aspx?ArticleID=6115"),
        cte_1e6_per_K: sourced(
            12.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        modulus_GPa: sourced(205.0, "https://www.azom.com/article.aspx?ArticleID=6115"),
        yield_MPa: sourced(
            220.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        cp_J_kgK: sourced(
            486.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        design_flux_density_T: Some(1.7),
        engine: EngineProps {
            sigma_S_m: 5848000.0,
            density_g_mm3: 0.00787,
            cp_J_kgK: 486.0,
            cte_per_C: 12.0e-6,
            modulus_GPa: 205.0,
        },
        needs_plating: true,
        notes: "Aggregator sources only; sigma derived at 20 C, flagged +-30 % (decision 7 A; 7 % IACS alternate 4.06e6). Design flux 1.7 T: the workbook's comment on Materials!C13.",
    },
    Material {
        id: "12L14_cold_drawn",
        label: "12L14 cold drawn",
        name: "AISI 12L14 (UNS G12144) resulfurized, leaded free-machining steel",
        condition: "Cold drawn bar (yield). Other properties: condition not stated by the sources.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.azom.com/article.aspx?ArticleID=6604",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(5747000.0, "https://www.theworldmaterial.com/12l14-steel/"),
        density_g_cm3: sourced(7.87, "https://www.azom.com/article.aspx?ArticleID=6604"),
        cte_1e6_per_K: sourced(11.5, "https://www.azom.com/article.aspx?ArticleID=6604"),
        modulus_GPa: sourced(200.0, "https://www.theworldmaterial.com/12l14-steel/"),
        yield_MPa: sourced(415.0, "https://www.azom.com/article.aspx?ArticleID=6604"),
        cp_J_kgK: sourced(472.0, "https://www.theworldmaterial.com/12l14-steel/"),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 5747000.0,
            density_g_mm3: 0.00787,
            cp_J_kgK: 472.0,
            cte_per_C: 11.5e-6,
            modulus_GPa: 200.0,
        },
        needs_plating: true,
        notes: "Aggregator sources only; sigma flagged +-30 % (decision 7 A; 7.1 % IACS alternate 4.12e6). Contains lead (RoHS/REACH). No sourced design flux density.",
    },
    Material {
        id: "416_annealed",
        label: "416 stainless, annealed",
        name: "Type 416 (UNS S41600) free-machining martensitic stainless",
        condition: "Annealed (Carpenter: anneal 650-760 C cool in air, ~187 HB). Hardened values listed as alternates.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.carpentertechnology.com/blog/magnetic-properties-of-stainless-steels",
        mu_r: sourced(
            110.0,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        bsat_T: sourced(
            1.6,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        sigma_S_m: sourced(
            1754000.0,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        density_g_cm3: sourced(
            7.64,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        cte_1e6_per_K: sourced(
            10.08,
            "https://www.rolledalloys.com/wp-content/uploads/416_stainless-steel-data-sheet-rolled-alloys.pdf",
        ),
        modulus_GPa: sourced(
            200.0,
            "https://www.smithmetal.com/pdf/stainless/416-stainless.pdf",
        ),
        yield_MPa: sourced(
            276.0,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        cp_J_kgK: sourced(
            460.5,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1754000.0,
            density_g_mm3: 0.00764,
            cp_J_kgK: 460.5,
            cte_per_C: 10.08e-6,
            modulus_GPa: 200.0,
        },
        needs_plating: false,
        notes: "Standard martensitic 416 (decision 6 A): Bsat 1.60 T is a proxy (SMAG Table 4, 410 at 0.15 % C); CTE 10.08e-6 re-sourced from Rolled Alloys (the sheet prints 'Coefficient of Thermal Expansion* 5.6 in/in F x 10-6' in its 212 F column, footnote '* 70F to indicated temperature': 5.6e-6 /F over 70 to 212 F = 10.08e-6 /K over 21 to 100 C); yield 276 is typical (no minimum published). mu_r 110 secant at 1.5 T.",
    },
    Material {
        id: "17-4PH_H1150",
        label: "17-4PH H1150",
        name: "17-4 PH (UNS S17400) precipitation-hardened stainless, condition H1150",
        condition: "H1150 (aged 4 h at 1150 F / 621 C, air cool). Chosen as the default because it is the stable, tough, lower-distortion condition; H900 is listed separately.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        mu_r: sourced(
            76.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1250000.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        density_g_cm3: sourced(
            7.82,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cte_1e6_per_K: sourced(
            11.9,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        modulus_GPa: sourced(
            199.9,
            "https://www.rolledalloys.com/wp-content/uploads/17-4_Data-sheet-rolled-alloys.pdf",
        ),
        yield_MPa: sourced(
            725.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cp_J_kgK: sourced(
            460.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1250000.0,
            density_g_mm3: 0.00782,
            cp_J_kgK: 460.0,
            cte_per_C: 11.9e-6,
            modulus_GPa: 199.9,
        },
        needs_plating: false,
        notes: "Low induction: ARMCO Fig. 3 gives about 1.07 T at 140 Oe (a lower bound on Bsat, not Bsat). sigma is ARMCO's H900 figure as a proxy (M24). mu_r chart-read.",
    },
    Material {
        id: "17-4PH_H900",
        label: "17-4PH H900",
        name: "17-4 PH (UNS S17400), condition H900",
        condition: "H900 (aged 1 h at 900 F / 482 C, air cool). Maximum strength, lowest toughness.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        mu_r: sourced(
            96.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1250000.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        density_g_cm3: sourced(
            7.8,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cte_1e6_per_K: sourced(
            10.8,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        modulus_GPa: sourced(
            199.9,
            "https://www.rolledalloys.com/wp-content/uploads/17-4_Data-sheet-rolled-alloys.pdf",
        ),
        yield_MPa: sourced(
            1170.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cp_J_kgK: sourced(
            460.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1250000.0,
            density_g_mm3: 0.0078,
            cp_J_kgK: 460.0,
            cte_per_C: 10.8e-6,
            modulus_GPa: 199.9,
        },
        needs_plating: false,
        notes: "ARMCO Fig. 3 gives about 1.35 T at 140 Oe (a lower bound on Bsat). mu_r chart-read.",
    },
    Material {
        id: "304_annealed",
        label: "304 stainless (non-magnetic)",
        name: "AISI 304 (UNS S30400) austenitic stainless, annealed",
        condition: "Annealed sheet/strip (AK Steel data sheet).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.carpentertechnology.com/blog/magnetic-properties-of-stainless-steels",
        mu_r: sourced(
            1.02,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1389000.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        density_g_cm3: sourced(
            8.03,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        cte_1e6_per_K: sourced(
            16.9,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        modulus_GPa: sourced(
            193.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        yield_MPa: sourced(
            205.0,
            "https://www.sandmeyersteel.com/wp-content/uploads/Alloy304-304L-APR2013.pdf",
        ),
        cp_J_kgK: sourced(
            500.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1389000.0,
            density_g_mm3: 0.00803,
            cp_J_kgK: 500.0,
            cte_per_C: 16.9e-6,
            modulus_GPa: 193.0,
        },
        needs_plating: false,
        notes: "Non-magnetic demonstration back iron: an open magnetic circuit. mu_r 1.02 is AK Steel's upper limit at 200 Oe. M3 (yield minimum), M20.",
    },
    Material {
        id: "6061_T6",
        label: "6061-T6 aluminium",
        name: "Aluminium 6061-T6 (T651)",
        condition: "T6 temper (solution heat treated and artificially aged).",
        ferromagnetic: false,
        ferromagnetic_source: "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        mu_r: sourced(
            1.000022,
            "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            24940000.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        density_g_cm3: sourced(
            2.7,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        cte_1e6_per_K: sourced(
            23.6,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        modulus_GPa: sourced(
            68.3,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        yield_MPa: sourced(
            241.0,
            "https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25011-T6-Sheet_200",
        ),
        cp_J_kgK: sourced(
            896.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 25000000.0,
            density_g_mm3: 0.0027,
            cp_J_kgK: 900.0,
            cte_per_C: 23.6e-6,
            modulus_GPa: 68.3,
        },
        needs_plating: false,
        notes: "Default cap and housing (the workbook's C42, C43 and C140). As a back iron: non-magnetic demonstration. mu_r: pure-aluminium proxy. M5.",
    },
    Material {
        id: "7075_T6",
        label: "7075-T6 aluminium",
        name: "Aluminium 7075-T6 (T651)",
        condition: "T6 temper (solution heat treated and artificially aged).",
        ferromagnetic: false,
        ferromagnetic_source: "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        mu_r: sourced(
            1.000022,
            "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            19140000.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        density_g_cm3: sourced(
            2.8,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        cte_1e6_per_K: sourced(
            23.4,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        modulus_GPa: sourced(
            71.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        yield_MPa: sourced(
            462.0,
            "https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25012-T6-Sheet_203",
        ),
        cp_J_kgK: sourced(
            960.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 19000000.0,
            density_g_mm3: 0.0028,
            cp_J_kgK: 960.0,
            cte_per_C: 23.4e-6,
            modulus_GPa: 71.0,
        },
        needs_plating: false,
        notes: "The workbook uses 7075-T6 for clamp collars and adapters (clamps.alloy), which this choice does not change. M6.",
    },
    Material {
        id: "316L_annealed",
        label: "316L annealed",
        name: "316L (UNS S31603) austenitic stainless, annealed",
        condition: "Annealed sheet/strip/plate (AK Steel and Sandmeyer data sheets).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.sandmeyersteel.com/wp-content/uploads/316-316l-317l-spec-sheet.pdf",
        mu_r: sourced(
            1.02,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1351000.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        density_g_cm3: sourced(
            7.99,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        cte_1e6_per_K: sourced(
            16.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        modulus_GPa: sourced(
            193.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        yield_MPa: sourced(
            172.0,
            "https://www.sandmeyersteel.com/wp-content/uploads/316-316l-317l-spec-sheet.pdf",
        ),
        cp_J_kgK: sourced(
            500.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1350000.0,
            density_g_mm3: 0.008,
            cp_J_kgK: 500.0,
            cte_per_C: 16.0e-6,
            modulus_GPa: 193.0,
        },
        needs_plating: false,
        notes: "Default sleeve and liner (the workbook's C111, C139 and Metal design C44, which also sets the endplates). M4, M21.",
    },
    Material {
        id: "Ti6Al4V_annealed",
        label: "Ti-6Al-4V grade 5",
        name: "Titanium grade 5 (Ti-6Al-4V, UNS R56400), annealed",
        condition: "Annealed (700-785 C per ASM sheet).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        mu_r: sourced(
            1.00005,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            595200.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        density_g_cm3: sourced(
            4.42,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        cte_1e6_per_K: sourced(
            9.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        modulus_GPa: sourced(
            113.8,
            "https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
        ),
        yield_MPa: sourced(
            828.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        cp_J_kgK: sourced(
            580.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 595200.0,
            density_g_mm3: 0.00442,
            cp_J_kgK: 580.0,
            cte_per_C: 9.0e-6,
            modulus_GPa: 113.8,
        },
        needs_plating: false,
        notes: "About 0.44 times the conductivity of 316L. sigma at 0 C (4-6 % high at room temperature). M7, M22, M23.",
    },
    Material {
        id: "IN625_annealed",
        label: "Inconel 625",
        name: "INCONEL alloy 625 (UNS N06625), annealed",
        condition: "Annealed (resistivity: annealed 2100 F / 1 h; yield: annealed rod, bar, plate nominal range).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        mu_r: sourced(
            1.0006,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            775200.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        density_g_cm3: sourced(
            8.44,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        cte_1e6_per_K: sourced(
            12.8,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        modulus_GPa: sourced(
            207.5,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        yield_MPa: sourced(
            414.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        cp_J_kgK: sourced(
            410.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 775200.0,
            density_g_mm3: 0.00844,
            cp_J_kgK: 410.0,
            cte_per_C: 12.8e-6,
            modulus_GPa: 207.5,
        },
        needs_plating: false,
        notes: "About 0.57 times the conductivity of 316L. Yield: lower bound of the composite range.",
    },
    Material {
        id: "PEEK_unfilled",
        label: "PEEK",
        name: "PEEK, unfilled (Ensinger TECAPEEK natural stock shapes; Victrex 450G as cross-check)",
        condition: "Unfilled, natural/beige; stock-shape values at 23 C. Not a conductor.",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1e-13,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        density_g_cm3: sourced(
            1.31,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        cte_1e6_per_K: sourced(
            50.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        modulus_GPa: sourced(
            4.0,
            "https://www.victrex.com/-/media/downloads/datasheets/victrex_tds_450g.pdf",
        ),
        yield_MPa: sourced(
            98.0,
            "https://www.victrex.com/-/media/downloads/datasheets/victrex_tds_450g.pdf",
        ),
        cp_J_kgK: sourced(
            1100.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1e-13,
            density_g_mm3: 0.00131,
            cp_J_kgK: 1100.0,
            cte_per_C: 50.0e-6,
            modulus_GPa: 4.0,
        },
        needs_plating: false,
        notes: "Conductivity is an upper bound from the volume resistivity: zero for slip loss. CTE three times 316L. M14 to M17.",
    },
    Material {
        id: "POM_H_acetal",
        label: "Acetal (POM-H)",
        name: "Acetal, POM homopolymer (Delrin 150 series / TECAFORM AD natural)",
        condition: "HOMOPOLYMER (not copolymer), natural stock shape, 73 F (23 C).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1e-13,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        density_g_cm3: sourced(
            1.41,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        cte_1e6_per_K: sourced(
            122.4,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        modulus_GPa: sourced(3.1, "https://cdn.thomasnet.com/ccp/00072207/74788.pdf"),
        yield_MPa: sourced(
            75.84,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        cp_J_kgK: sourced(
            1465.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1e-13,
            density_g_mm3: 0.00141,
            cp_J_kgK: 1465.0,
            cte_per_C: 122.4e-6,
            modulus_GPa: 3.1,
        },
        needs_plating: false,
        notes: "Homopolymer (Delrin 150). Conductivity is an upper bound: zero for slip loss. CTE about ten times aluminium. M9, M10.",
    },
];

/// The material with this data-file id (exact text).
pub fn material(id: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|m| m.id == id)
}

/// Back iron (hub, cup and boss) choices: selector code and material id. Code 1,
/// the workbook's 4140, is the default: the Materials sheet inputs.
pub const BACK_IRON_CHOICES: [(i64, &str); 8] = [
    (1, "4140_annealed"),
    (2, "1018_hot_rolled"),
    (3, "12L14_cold_drawn"),
    (4, "416_annealed"),
    (5, "17-4PH_H1150"),
    (6, "17-4PH_H900"),
    (7, "304_annealed"),
    (8, "6061_T6"),
];

/// Sleeve and liner choices (the endplates follow them, as Metal design C44 does).
/// Code 1, the workbook's 316L, is the default.
pub const SLEEVE_LINER_CHOICES: [(i64, &str); 4] = [
    (1, "316L_annealed"),
    (2, "Ti6Al4V_annealed"),
    (3, "IN625_annealed"),
    (4, "PEEK_unfilled"),
];

/// Cap and housing choices (the engine models the cap; the aluminium adapter and the
/// clamp alloy keep their own inputs). Code 1, the workbook's 6061-T6, is the default.
pub const CAP_HOUSING_CHOICES: [(i64, &str); 3] =
    [(1, "6061_T6"), (2, "7075_T6"), (3, "POM_H_acetal")];

/// The material a selector code picks among `choices`; `None` for a code outside them.
pub fn chosen(choices: &[(i64, &'static str)], code: i64) -> Option<&'static Material> {
    choices
        .iter()
        .find(|&&(c, _)| c == code)
        .and_then(|&(_, id)| material(id))
}

/// What a code outside the choices supplies (decision D3: never another material):
/// NaN properties, so the results say the input is invalid; `validate()` names it.
const NO_PROPS: EngineProps = EngineProps {
    sigma_S_m: f64::NAN,
    density_g_mm3: f64::NAN,
    cp_J_kgK: f64::NAN,
    cte_per_C: f64::NAN,
    modulus_GPa: f64::NAN,
};

/// The name results show for a code outside the choices.
pub const NO_MATERIAL: &str = "#N/A";

/// One part's material in effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartMaterial {
    /// The library record picked; `None` for a code outside the choices.
    pub material: Option<&'static Material>,
    /// Whether the pick is the part's default (the workbook's material: the inputs).
    pub is_default: bool,
    /// The values the engine reads for this part.
    pub props: EngineProps,
}

impl PartMaterial {
    /// The selector text of the pick, or [`NO_MATERIAL`].
    pub fn label(&self) -> &'static str {
        self.material.map_or(NO_MATERIAL, |m| m.label)
    }
}

/// What the parts' materials feed into the engine. `api::compute` builds it from
/// the selectors and the inputs; at the default choices every value is the input
/// it stands for, bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct PartProperties {
    /// The circuit Calculator C6 selects in effect: 0 (free space) for a
    /// non-ferromagnetic back iron, else the C6 input, which thus overrides a
    /// ferromagnetic choice toward "no back iron".
    pub backiron: i64,
    /// The design flux density of the wall check (Materials C13 in effect).
    pub design_flux_T: f64,
    /// The back-iron pick (its name and properties, for display and the warnings).
    pub back_iron: PartMaterial,
    /// The steel circuit's values: the Materials inputs, or a ferromagnetic pick's.
    pub steel: EngineProps,
    /// The hub, cup and boss when there is no back iron: the workbook's aluminium
    /// (Metal design C42, Temperature design C140, Materials C43, and E18's 6061
    /// expansion and modulus), or a non-ferromagnetic pick's own values.
    pub body: EngineProps,
    /// Sleeve, liner and endplates (expansion and modulus not read).
    pub sleeve_liner: PartMaterial,
    /// The cap (expansion and modulus not read).
    pub cap: PartMaterial,
}

/// The workbook values of a part, as [`EngineProps`].
#[allow(non_snake_case)]
const fn props(
    sigma_S_m: f64,
    density_g_mm3: f64,
    cp_J_kgK: f64,
    cte_per_C: f64,
    modulus_GPa: f64,
) -> EngineProps {
    EngineProps {
        sigma_S_m,
        density_g_mm3,
        cp_J_kgK,
        cte_per_C,
        modulus_GPa,
    }
}

/// A part's material: the default pick stands for `workbook` (the inputs); another
/// pick supplies its engine values; a code outside the choices gives [`NO_PROPS`].
fn part(choices: &[(i64, &'static str)], code: i64, workbook: EngineProps) -> PartMaterial {
    let material = chosen(choices, code);
    let is_default = code == choices[0].0;
    let props = match material {
        Some(_) if is_default => workbook,
        Some(m) => m.engine,
        None => NO_PROPS,
    };
    PartMaterial {
        material,
        is_default,
        props,
    }
}

/// Resolves the three part selectors against the inputs (Addendum A5 physics links):
/// a ferromagnetic back iron keeps the steel circuit and supplies the steel values and
/// its design flux density where the library has one (else Materials C13 stays); a
/// non-ferromagnetic one selects the free-space circuit and becomes the hub, cup and
/// boss material; the sleeve and cap picks supply their conductivity, density and
/// specific heat. Incremental permeability always stays Materials C15 (no source
/// gives it at the magnet bias).
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn resolve(
    choice: &super::materials::PartMaterialInputs,
    steel: &super::materials::Steel4140,
    backiron: i64,
    md: &super::metal_design::MetalDesignInputs,
    slip: &super::temperature::SlipLossInputs,
    thermal: &super::temperature::ThermalInputs,
) -> PartProperties {
    use super::materials::AL6061;
    use super::temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA};
    let workbook_steel = props(
        steel.conductivity_S_m,
        md.steel_density_g_mm3,
        steel.specific_heat_J_kgK,
        steel.cte_per_C,
        steel.modulus_GPa,
    );
    let back_iron = part(&BACK_IRON_CHOICES, choice.back_iron, workbook_steel);
    let non_magnetic = back_iron.material.is_some_and(|m| !m.ferromagnetic);
    let design_flux_T = match back_iron.material {
        Some(m) if !back_iron.is_default => m.design_flux_density_T.unwrap_or(steel.bsat_T),
        Some(_) => steel.bsat_T,
        None => f64::NAN,
    };
    let aluminium = props(
        AL6061.conductivity_S_m,
        md.al_density_g_mm3,
        thermal.c_aluminium,
        AL_HUB_CTE_PER_C,
        AL_HUB_MODULUS_GPA,
    );
    PartProperties {
        backiron: if non_magnetic { 0 } else { backiron },
        design_flux_T,
        back_iron,
        // A non-magnetic pick leaves the steel values at the inputs: only cells the
        // workbook still prices as steel with no back iron (E9 off) read them.
        steel: if non_magnetic {
            workbook_steel
        } else {
            back_iron.props
        },
        body: if non_magnetic {
            back_iron.props
        } else {
            aluminium
        },
        sleeve_liner: part(
            &SLEEVE_LINER_CHOICES,
            choice.sleeve_liner,
            props(
                slip.sigma_316_S_m,
                md.sleeve_density_g_mm3,
                thermal.c_316,
                f64::NAN,
                f64::NAN,
            ),
        ),
        cap: part(
            &CAP_HOUSING_CHOICES,
            choice.cap_housing,
            props(
                AL6061.conductivity_S_m,
                md.al_density_g_mm3,
                thermal.c_aluminium,
                f64::NAN,
                f64::NAN,
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_choice_names_a_library_material_and_codes_start_at_one() {
        for choices in [
            &BACK_IRON_CHOICES[..],
            &SLEEVE_LINER_CHOICES[..],
            &CAP_HOUSING_CHOICES[..],
        ] {
            for (i, &(code, id)) in choices.iter().enumerate() {
                assert_eq!(code, i as i64 + 1, "{id}");
                assert!(material(id).is_some(), "{id}");
            }
            assert_eq!(chosen(choices, 0), None);
            assert_eq!(chosen(choices, choices.len() as i64 + 1), None);
        }
        assert_eq!(
            chosen(&BACK_IRON_CHOICES, 7).map(|m| m.ferromagnetic),
            Some(false)
        );
    }
}
