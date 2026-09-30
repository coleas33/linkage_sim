//! Stock magnet library ('Magnet library' sheet).
//!
//! Port of `reference/magcoupling-py/magcoupling/library.py`. The calculator
//! looks magnets up by exact part text, like the workbook's INDEX/MATCH.
//! Dimensions in mm, Br is the 20 °C remanence in tesla, tmax the supplier
//! rating in °C. The rows hold the WORKBOOK values (1.29 T for the N42SH rows),
//! so the static-data test still compares them with the Python engine. The
//! approved correction E3 (N42SH remanence, decision D1) is applied by
//! [`br_T`], which the Calculator calls where it resolves a part
//! (`model::resolve_magnets`). The 3D field run (`fields3d`, M3) must be rerun
//! with the corrected Br.

use super::deviations::{DeviationId, Deviations};

/// E3: the N42SH rows' remanence with the approved correction: the vendor's
/// grade minimum (K&J 13.0 kG), like the library's other K&J rows (decision D1).
pub const N42SH_BR_CORRECTED_T: f64 = 1.30;

/// The 20 °C remanence the Calculator uses for a library part: the row's
/// workbook value, or with E3 on the corrected N42SH value.
#[allow(non_snake_case)]
pub fn br_T(spec: &MagnetSpec, dev: Deviations) -> f64 {
    if dev.is_on(DeviationId::E3) && spec.grade == "N42SH" {
        N42SH_BR_CORRECTED_T
    } else {
        spec.br_T
    }
}

/// One library row (Python `MagnetSpec`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct MagnetSpec {
    pub part: &'static str,
    pub vendor: &'static str,
    pub shape: &'static str,
    /// Axial length [mm].
    pub length_mm: f64,
    /// Tangential width [mm].
    pub width_mm: f64,
    /// Radial thickness, the magnetized direction [mm].
    pub thickness_mm: f64,
    pub grade: &'static str,
    /// Remanence at 20 °C [T].
    pub br_T: f64,
    /// Supplier maximum operating temperature [°C].
    pub tmax_C: f64,
    pub notes: &'static str,
}

#[allow(clippy::too_many_arguments, non_snake_case)]
const fn row(
    part: &'static str,
    vendor: &'static str,
    shape: &'static str,
    dims: [f64; 3],
    grade: &'static str,
    br_T: f64,
    tmax_C: f64,
    notes: &'static str,
) -> MagnetSpec {
    MagnetSpec {
        part,
        vendor,
        shape,
        length_mm: dims[0],
        width_mm: dims[1],
        thickness_mm: dims[2],
        grade,
        br_T,
        tmax_C,
        notes,
    }
}

/// The library, in the Python row order (`_ROWS`).
pub const MAGNET_LIBRARY: [MagnetSpec; 15] = [
    row(
        "B842SH",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1/2 x 1/4 x 1/8 in, magnetized through 1/8 in",
    ),
    row(
        "B842",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "",
    ),
    row(
        "B842-N52",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    ),
    row(
        "B822",
        "K&J",
        "block",
        [12.7, 3.17, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/8 x 1/8 in",
    ),
    row(
        "B862",
        "K&J",
        "block",
        [12.7, 9.5, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/8 in",
    ),
    row(
        "B882",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/8 in",
    ),
    row(
        "B882-N52",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    ),
    row(
        "B861",
        "K&J",
        "block",
        [12.7, 9.5, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/16 in",
    ),
    row(
        "B881",
        "K&J",
        "block",
        [12.7, 12.7, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/16 in (check stock)",
    ),
    row(
        "B442",
        "K&J",
        "block",
        [6.35, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "1/4 x 1/4 x 1/8 in",
    ),
    row(
        "BX042SH",
        "K&J",
        "block",
        [25.4, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/4 x 1/8 in",
    ),
    row(
        "BX082SH",
        "K&J",
        "block",
        [25.4, 12.7, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/2 x 1/8 in",
    ),
    row(
        "M5044",
        "SuperMagnetMan",
        "arc",
        [13.0, 5.5, 1.31],
        "N50",
        1.42,
        80.0,
        "22.60 OD x 19.97 ID x 13, 29.8 deg, 12 pcs; width = mean arc length",
    ),
    row(
        "M5045",
        "SuperMagnetMan",
        "arc",
        [6.56, 5.6, 1.12],
        "N50M",
        1.42,
        100.0,
        "22.70 OD x 20.47 ID x 6.56, 12 pcs",
    ),
    row(
        "M5026",
        "SuperMagnetMan",
        "arc",
        [15.0, 6.4, 1.67],
        "N50",
        1.42,
        80.0,
        "26.60 OD x 23.26 ID x 15, 12 pcs",
    ),
];

/// Exact-text lookup (Python `lookup`): `None` for an empty or unknown part,
/// and the model then uses the manual dimensions.
pub fn lookup(part: &str) -> Option<&'static MagnetSpec> {
    if part.is_empty() {
        return None;
    }
    MAGNET_LIBRARY.iter().find(|m| m.part == part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e3_corrects_exactly_the_three_n42sh_rows() {
        // The report names B842SH, BX042SH and BX082SH (all at 1.29 T in the
        // workbook); every other row keeps its workbook remanence.
        let named = ["B842SH", "BX042SH", "BX082SH"];
        for spec in &MAGNET_LIBRARY {
            let corrected = named.contains(&spec.part);
            assert_eq!(spec.grade == "N42SH", corrected, "{}", spec.part);
            if corrected {
                assert_eq!(spec.br_T, 1.29, "{}: the workbook row", spec.part);
            }
            let want = if corrected {
                N42SH_BR_CORRECTED_T
            } else {
                spec.br_T
            };
            assert_eq!(br_T(spec, Deviations::NONE), spec.br_T, "{}", spec.part);
            assert_eq!(
                br_T(spec, Deviations::only(DeviationId::E3)),
                want,
                "{}",
                spec.part
            );
            assert_eq!(br_T(spec, Deviations::ALL), want, "{}", spec.part);
        }
    }
}
