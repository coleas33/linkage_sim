//! Magnet grades (spec Addendum A6): one record per grade, so grade data lives
//! once. Library parts name their grade ([`crate::engine::library::MagnetSpec::grade`]);
//! a grade can also be picked for manual dimensions
//! (`coupling.magnets.grade_inner`, `grade_outer`).
//!
//! Data: `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`, key
//! `"grades"`, approved with the Addendum A verification (decisions 1, 3, 4, 17
//! and 18, option A; resolutions R1 to R17 in its section 2.4). Every value
//! cites its source in [`GradeSources`]; `tests/grades.rs` compares every value
//! and citation with the data file. Values are the data file's engine-unit
//! literals (`engine_literals`), typed as literals and never converted at load
//! time (-0.55 / 100 is -0.0055000000000000005, not -0.0055).
//!
//! What the engine reads: Br and Tmax of a grade picked for manual dimensions
//! (`model::resolve_magnets`), E3's corrected N42SH Br ([`N42SH`]), and with
//! E20 the Hcj and beta of each ring's grade (the demagnetization block checks
//! both rings; the weaker governs, A-1 plan decision A13).
//! Hcb, (BH)max, mu_rec, the coefficient ranges and the reference beta are for
//! display. alpha(Br) and density are read for a ring in the grade mode (manual
//! dimensions with a grade: Addendum A-2 decision A2-7); a library part (every one
//! sintered NdFeB, whose grade values equal them) keeps the calculator's single alpha
//! (Calibration!C22) and the NdFeB density.

/// K&J Magnetics, Neodymium Magnet Specifications & Tolerances (the NdFeB basis, decision D1/1).
pub const KJ_SPECS: &str = "https://www.kjmagnetics.com/neodymium-magnet-specifications.asp";
/// Arnold Recoma sintered SmCo, combined datasheet 160301 (p4 Recoma 20, p8 Recoma 26, p12 Recoma 30).
pub const ARNOLD_RECOMA: &str =
    "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf";
/// Arnold (Constantinides), APEEM 2006, slide 23: recoil permeability "about 1.05".
pub const ARNOLD_APEEM_2006: &str = "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Manufacturing-and-performance-comparison-between-bonded-and-sintered-permanent-magnets-Constantinides-APEEM-2006-psn-hi-res.pdf";
/// Arnold TECHNotes TN 0303 (family coefficients, averages over about 20 to 120 °C).
pub const ARNOLD_TN_0303: &str =
    "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/TN_0303_rev_150715.pdf";
/// Eclipse Magnetics, Ferrite/Ceramic Magnets Datasheet.
pub const ECLIPSE_FERRITE: &str =
    "https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf";
/// Alliance LLC, Ferrite C-5 (the +0.35 %/°C beta, decision 3 A).
pub const ALLIANCE_C5: &str =
    "https://allianceorg.com/magnetic-materials/ceramic-magnets/ferrite-c-5/";
/// Alliance LLC, Compression Bonded Neo (BCN-19).
pub const ALLIANCE_BONDED_NEO: &str =
    "https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/";

/// The material family of a grade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradeFamily {
    /// Sintered NdFeB.
    NdFeB,
    /// Sintered Sm2Co17.
    SmCo2_17,
    /// Sintered SmCo5.
    SmCo1_5,
    /// Sintered hard (strontium) ferrite: its beta(Hcj) is positive.
    Ferrite,
    /// Isotropic compression-bonded NdFeB.
    BondedNdFeB,
}

/// Where each value of a grade comes from (a URL of the data file's `sources`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GradeSources {
    pub br: &'static str,
    pub hcj: &'static str,
    pub hcb: &'static str,
    pub bhmax: &'static str,
    pub alpha: &'static str,
    pub beta: &'static str,
    /// The measured range of alpha and beta; `None` where none is printed.
    pub coefficient_range: Option<&'static str>,
    /// `None` where mu_rec is not sourced.
    pub mu_rec: Option<&'static str>,
    pub tmax: &'static str,
    pub density: &'static str,
}

/// One magnet grade (the selected value of each field: the published minimum
/// where one exists, report section 2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct Grade {
    /// The data file's key; parts and the grade inputs name a grade by it.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    pub family: GradeFamily,
    /// Remanence at 20 °C [T].
    pub br_T: f64,
    /// Intrinsic coercivity at 20 °C [kA/m].
    pub hcj20_kA_m: f64,
    /// Normal coercivity [kA/m] (display).
    pub hcb_kA_m: f64,
    /// Maximum energy product [kJ/m³] (display).
    pub bhmax_kJ_m3: f64,
    /// Reversible temperature coefficient of Br [1/°C] (read in the grade mode, decision A2-7).
    pub alpha_br_per_C: f64,
    /// beta(Hcj) the engine uses with E20 [1/°C]: the reference value, except
    /// N42SH, which keeps the workbook's -0.005 (decision 18 A).
    pub beta_hcj_per_C: f64,
    /// The sourced beta(Hcj) [1/°C]. Positive for hard ferrite: its coercivity
    /// falls as it cools, so its demagnetization risk is at cold.
    pub beta_hcj_reference_per_C: f64,
    /// The temperature range [°C] alpha and beta were measured over.
    pub coefficient_range_C: Option<[f64; 2]>,
    /// Recoil permeability (display; class-level value).
    pub mu_rec: Option<f64>,
    /// Maximum operating temperature [°C] (the calibration rating of the demag block).
    pub tmax_C: f64,
    /// Density [g/mm³] (read in the grade mode, decision A2-7).
    pub density_g_mm3: f64,
    pub sources: GradeSources,
    /// The report's resolutions (R#) and flags for this grade.
    pub notes: &'static str,
}

/// Every grade, in the data file's order.
pub const GRADES: [Grade; 17] = [
    Grade {
        id: "N35",
        name: "N35",
        family: GradeFamily::NdFeB,
        br_T: 1.17,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 262.6,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17 (K&J low end, not stated as guaranteed).",
    },
    Grade {
        id: "N42",
        name: "N42",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N48",
        name: "N48",
        family: GradeFamily::NdFeB,
        br_T: 1.38,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 358.1,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17 (Hcj: K&J 954.9, Arnold 875).",
    },
    Grade {
        id: "N52",
        name: "N52",
        family: GradeFamily::NdFeB,
        br_T: 1.45,
        hcj20_kA_m: 875.4,
        hcb_kA_m: 836.0,
        bhmax_kJ_m3: 393.9,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 60.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R3 (Hcb 836 Arnold: K&J's 891.3 exceeds Hcj), R4 (Tmax 80 K&J; Arnold catalog 60, Eclipse 70), R16, R17.",
    },
    Grade {
        id: "N50",
        name: "N50",
        family: GradeFamily::NdFeB,
        br_T: 1.41,
        hcj20_kA_m: 875.4,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 382.0,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Library grade (M5044, M5026; M5045 under E19), not in the spec's A6 list (decision 30). R5, R16, R17.",
    },
    Grade {
        id: "N42M",
        name: "N42M",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1114.1,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.006,
        beta_hcj_reference_per_C: -0.006,
        coefficient_range_C: Some([20.0, 100.0]),
        mu_rec: Some(1.05),
        tmax_C: 100.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N50M",
        name: "N50M",
        family: GradeFamily::NdFeB,
        br_T: 1.41,
        hcj20_kA_m: 1114.1,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 382.0,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.00675,
        beta_hcj_reference_per_C: -0.00675,
        coefficient_range_C: Some([20.0, 100.0]),
        mu_rec: Some(1.05),
        tmax_C: 100.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Library grade (M5045 as stored in the workbook), not in the spec's A6 list. R1 (Hcb), R6 (Tmax), R16, R17.",
    },
    Grade {
        id: "N42H",
        name: "N42H",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1352.8,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0057,
        beta_hcj_reference_per_C: -0.0057,
        coefficient_range_C: Some([20.0, 120.0]),
        mu_rec: Some(1.05),
        tmax_C: 120.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N42SH",
        name: "N42SH",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1592.0,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.005,
        beta_hcj_reference_per_C: -0.0055,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 150.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Default grade (B842SH on both rings). Hcj 1592 Arnold (decision 17 A; K&J 1591.5). The engine keeps the workbook beta -0.005 /C; Arnold's -0.0055 is the reference (decision 18 A). R16, R17.",
    },
    Grade {
        id: "N38UH",
        name: "N38UH",
        family: GradeFamily::NdFeB,
        br_T: 1.22,
        hcj20_kA_m: 1989.4,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 286.5,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0051,
        beta_hcj_reference_per_C: -0.0051,
        coefficient_range_C: Some([20.0, 180.0]),
        mu_rec: Some(1.05),
        tmax_C: 180.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N35EH",
        name: "N35EH",
        family: GradeFamily::NdFeB,
        br_T: 1.17,
        hcj20_kA_m: 2387.3,
        hcb_kA_m: 859.4,
        bhmax_kJ_m3: 262.6,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0047,
        beta_hcj_reference_per_C: -0.0047,
        coefficient_range_C: Some([20.0, 200.0]),
        mu_rec: Some(1.05),
        tmax_C: 200.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N33AH",
        name: "N33AH",
        family: GradeFamily::NdFeB,
        br_T: 1.14,
        hcj20_kA_m: 2705.6,
        hcb_kA_m: 811.7,
        bhmax_kJ_m3: 246.7,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.00375,
        beta_hcj_reference_per_C: -0.00375,
        coefficient_range_C: Some([20.0, 220.0]),
        mu_rec: Some(1.05),
        tmax_C: 220.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R2 ((BH)max K&J 246.7; Arnold 215 min), R16, R17.",
    },
    Grade {
        id: "SmCo_2_17_26",
        name: "Recoma 26 (Sm2Co17 grade 26)",
        family: GradeFamily::SmCo2_17,
        br_T: 1.0,
        hcj20_kA_m: 1200.0,
        hcb_kA_m: 680.0,
        bhmax_kJ_m3: 185.0,
        alpha_br_per_C: -0.00035,
        beta_hcj_per_C: -0.00247,
        beta_hcj_reference_per_C: -0.00247,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 350.0,
        density_g_mm3: 0.0083,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 26, the low-Hcj sub-grade (decision 4 A; typ Hcj 2000). R9, R10, R11 (Tmax 350: may be considerably lower at a low load line), R14 (mu_rec class value).",
    },
    Grade {
        id: "SmCo_2_17_30",
        name: "Recoma 30 (Sm2Co17 grade 30)",
        family: GradeFamily::SmCo2_17,
        br_T: 1.09,
        hcj20_kA_m: 1040.0,
        hcb_kA_m: 700.0,
        bhmax_kJ_m3: 215.0,
        alpha_br_per_C: -0.00035,
        beta_hcj_per_C: -0.0025,
        beta_hcj_reference_per_C: -0.0025,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.0083,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 30, the low-Hcj sub-grade (decision 4 A; 30HE/30S reach 1500/1750). R9, R10, R14.",
    },
    Grade {
        id: "SmCo_1_5_20",
        name: "Recoma 20 (SmCo5 grade 20)",
        family: GradeFamily::SmCo1_5,
        br_T: 0.85,
        hcj20_kA_m: 2000.0,
        hcb_kA_m: 640.0,
        bhmax_kJ_m3: 140.0,
        alpha_br_per_C: -0.00045,
        beta_hcj_per_C: -0.0019,
        beta_hcj_reference_per_C: -0.0019,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.0084,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 20 (decision 4 A). R8 (beta -0.19 is 37-52 % smaller than the SmCo5 family values: the non-conservative side), R10, R14.",
    },
    Grade {
        id: "Y30",
        name: "Ferrite Y30 (= C5)",
        family: GradeFamily::Ferrite,
        br_T: 0.37,
        hcj20_kA_m: 180.0,
        hcb_kA_m: 175.0,
        bhmax_kJ_m3: 26.0,
        alpha_br_per_C: -0.002,
        beta_hcj_per_C: 0.0035,
        beta_hcj_reference_per_C: 0.0035,
        coefficient_range_C: Some([20.0, 120.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.005,
        sources: GradeSources {
            br: ECLIPSE_FERRITE,
            hcj: ECLIPSE_FERRITE,
            hcb: ECLIPSE_FERRITE,
            bhmax: ECLIPSE_FERRITE,
            alpha: ECLIPSE_FERRITE,
            beta: ALLIANCE_C5,
            coefficient_range: Some(ARNOLD_TN_0303),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ECLIPSE_FERRITE,
            density: ECLIPSE_FERRITE,
        },
        notes: "Positive beta: coercivity falls as the magnet cools (decision 3 A: +0.35 %/C, Alliance C-5; Eclipse and TN 0303 give +0.27). Coefficient range borrowed from TN 0303's Ferrite 8 row (R15). Density: range midpoint (R12). R14.",
    },
    Grade {
        id: "Bonded_NdFeB_BCN19",
        name: "Bonded NdFeB (Alliance BCN-19)",
        family: GradeFamily::BondedNdFeB,
        br_T: 0.65,
        hcj20_kA_m: 880.0,
        hcb_kA_m: 416.0,
        bhmax_kJ_m3: 72.0,
        alpha_br_per_C: -0.0014,
        beta_hcj_per_C: -0.0036,
        beta_hcj_reference_per_C: -0.0036,
        coefficient_range_C: None,
        mu_rec: None,
        tmax_C: 140.0,
        density_g_mm3: 0.0058,
        sources: GradeSources {
            br: ALLIANCE_BONDED_NEO,
            hcj: ALLIANCE_BONDED_NEO,
            hcb: ALLIANCE_BONDED_NEO,
            bhmax: ALLIANCE_BONDED_NEO,
            alpha: ALLIANCE_BONDED_NEO,
            beta: ALLIANCE_BONDED_NEO,
            coefficient_range: None,
            mu_rec: None,
            tmax: ALLIANCE_BONDED_NEO,
            density: ALLIANCE_BONDED_NEO,
        },
        notes: "Alliance BCN-19 (about 0.65 T). alpha and beta are Alliance's operating-point coefficients (Bd, Hd), no range printed (R13). mu_rec not sourced (Arnold gives 1.1-1.7 for isotropic bonded Neo). Density: range midpoint (R12).",
    },
];

/// The default part's grade (B842SH on both rings): E3's corrected Br is its
/// published minimum (decision D1).
pub const N42SH: &Grade = &GRADES[8];

/// The grade named by exact text (like the part lookup); `None` for an empty or
/// unknown name.
pub fn grade(id: &str) -> Option<&'static Grade> {
    if id.is_empty() {
        return None;
    }
    GRADES.iter().find(|g| g.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n42sh_is_the_named_grade() {
        assert_eq!(N42SH.id, "N42SH");
        assert_eq!(grade("N42SH"), Some(N42SH));
    }

    #[test]
    fn lookup_is_exact_text() {
        assert_eq!(grade("Y30").map(|g| g.family), Some(GradeFamily::Ferrite));
        for miss in ["", "n42sh", "N42SH ", "N 42", "Recoma 26"] {
            assert_eq!(grade(miss), None, "{miss:?}");
        }
        let ids: std::collections::BTreeSet<&str> = GRADES.iter().map(|g| g.id).collect();
        assert_eq!(ids.len(), GRADES.len(), "grade ids are unique");
    }
}
