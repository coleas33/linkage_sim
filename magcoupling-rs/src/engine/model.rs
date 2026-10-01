//! Calculator sheet ('Calculator'): the core coupling model.
//!
//! Port of `reference/magcoupling-py/magcoupling/model.py`: the magnet and
//! coupling inputs, the 73 result cells ([`ModelResults`]), and the helpers
//! (`resolve_magnets`, `select_calibration_factor`, `geometry_factor`,
//! `harmonic_amplitude`, `shear_stress`, shared with the sweeps). Two formulas
//! several sheets share live here once, so they stay bit-identical everywhere:
//! `corner_radius` (√(r_face² + (w/2)²)) and `br_factor` (1 + α (T − 20 °C)).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`. A ring in the
//! grade mode (manual dimensions with a grade) takes its grade's alpha(Br) and density
//! (Addendum A-2 decision A2-7, [`ResolvedMagnet::alpha_br`], [`ResolvedMagnet::density_g_mm3`]).
//!
//! The harmonic set is the Rust-only assumption `coupling.max_harmonic` (Addendum A3):
//! the odd harmonics 1, 3, ... up to 11 ([`ODD_HARMONICS`], [`harmonic_count`]), the
//! workbook's 1, 3, 5 ([`HARMONICS`]) by default. The Calculator, the sweeps and the
//! Calibration prototype sum the same set.
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied are E3 (the N42SH
//! remanence, via `resolve_magnets`, and the manual Br defaults C17 and C27),
//! E6 (Calculator!C63, the ring wall at the flats without the outer
//! bondline), E7 (pull-out at the maximum over angle,
//! [`peak_off_half_pitch`], one search for any odd harmonic set up to 11, decision 29 A;
//! `peak_angle`, the one E7 gate, and `tau_at`, shared
//! with the sweeps and Calibration through `at_pull_out` or directly), E8
//! (arc mode: Calculator!C9 from the corner radius C55, the round pocket in
//! C111), E9 (no back iron: aluminium cup and boss in C111 and C113, as the
//! hub C112 already is) and E10 (Calculator!C103, the gap flux density from
//! each magnet's MMF, Br·t, instead of the mean Br).

use std::f64::consts::PI;

use super::compat::{fmt_fixed, py_max, py_min};
use super::constants::{MU0, NDFEB_DENSITY_G_MM3};
use super::deviations::{DeviationId, Deviations};
use super::grades::{self, Grade};
use super::library::{self, lookup};
use super::meta::{NumOrText, inputs, out, out_rust_only, param, param_rust_only, results};

/// The odd space harmonics the model sums (the workbook's set).
pub const HARMONICS: [u32; 3] = [1, 3, 5];

/// The odd space harmonics the E7 peak search handles, 1 to 11 (Addendum A3: "selectable
/// up to 11"); amplitude `a[i]` of [`peak_off_half_pitch`] belongs to `ODD_HARMONICS[i]`.
pub const ODD_HARMONICS: [u32; 6] = [1, 3, 5, 7, 9, 11];

/// The workbook's highest harmonic: the default of `coupling.max_harmonic`, the set [`HARMONICS`].
pub const WORKBOOK_MAX_HARMONIC: i64 = 5;

/// The choices of `coupling.max_harmonic` (Addendum A3): the highest odd harmonic summed.
pub const MAX_HARMONIC_CHOICES: [(i64, &str); 6] = [
    (1, "1"),
    (3, "1, 3"),
    (5, "1, 3, 5 (workbook)"),
    (7, "1, 3, 5, 7"),
    (9, "1, 3, 5, 7, 9"),
    (11, "1, 3, 5, 7, 9, 11"),
];

/// How many harmonics of [`ODD_HARMONICS`] the model sums for a `max_harmonic` code (3 for
/// the workbook's 5); `None` for a code outside [`MAX_HARMONIC_CHOICES`] (decision D3: every
/// harmonic sum is then NaN, never another set).
pub fn harmonic_count(max_harmonic: i64) -> Option<usize> {
    MAX_HARMONIC_CHOICES
        .iter()
        .position(|&(code, _)| code == max_harmonic)
        .map(|i| i + 1)
}

/// Σ `terms` of the harmonic set, a left fold from 0 as Python's `sum()`; NaN when the set
/// is invalid (`count` is `None`, [`harmonic_count`]).
pub(crate) fn harmonic_sum(count: Option<usize>, terms: impl Iterator<Item = f64>) -> f64 {
    match count {
        Some(_) => terms.fold(0.0, |acc, t| acc + t),
        None => f64::NAN,
    }
}

/// What the per-harmonic cell of `ODD_HARMONICS[i]` shows: `terms[i]` when the set sums
/// that harmonic, 0 when the set leaves it out, NaN when the set is invalid.
pub(crate) fn harmonic_slot(count: Option<usize>, terms: &[f64], i: usize) -> f64 {
    match count {
        Some(_) => terms.get(i).copied().unwrap_or(0.0),
        None => f64::NAN,
    }
}

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";

/// Text of the end-effect checks (`model.end_effect_check`, `calibration.end_effect_check`)
/// when f_end = 1 − c_end · pole pitch / L is 0 or below. Audit M9: the empirical form turns
/// negative for short magnets, and the pull-out with it; the user's decision (2026-09-30) is
/// a flag, no physics change (the GUI greys the numbers computed from the pull-out).
pub const END_EFFECT_OUT_OF_RANGE: &str = "End-effect model out of range";

/// Whether an end-effect factor is inside the model's range (positive; NaN is not). The
/// sweep rows' `f_end` column and inverse sizing use it too.
pub fn end_effect_in_range(f_end: f64) -> bool {
    f_end > 0.0
}

/// "OK" or [`END_EFFECT_OUT_OF_RANGE`], by [`end_effect_in_range`].
pub fn end_effect_check(f_end: f64) -> &'static str {
    if end_effect_in_range(f_end) {
        "OK"
    } else {
        END_EFFECT_OUT_OF_RANGE
    }
}

inputs! {
    /// Magnet parts and the manual fallbacks (Calculator!C11:C27).
    pub struct MagnetInputs {
        fields {
            part_inner: String = "B842SH" => param("-", "Inner magnet part",
                "Looked up in the library by exact text; blank = manual.", "Calculator!C11"),
            part_outer: String = "B842SH" => param("-", "Outer magnet part", "", "Calculator!C12"),
            manual_inner_length_mm: f64 = 12.7 => param("mm", "Manual inner length (axial)",
                "Used only if the part is not in the library.", "Calculator!C14")
                .range(2.0, 50.8, 0.01),
            manual_inner_width_mm: f64 = 6.35 => param("mm", "Manual inner width (tangential)",
                "", "Calculator!C15")
                .range(1.0, 25.4, 0.01),
            manual_inner_thickness_mm: f64 = 3.17 => param("mm", "Manual inner thickness (radial)",
                "", "Calculator!C16")
                .range(0.5, 10.0, 0.01),
            manual_inner_br_T: f64 = 1.30 => param("T", "Manual inner Br at 20 °C",
                "", "Calculator!C17")
                .range(0.2, 1.5, 0.001),
            manual_outer_length_mm: f64 = 12.7 => param("mm", "Manual outer length (axial)",
                "", "Calculator!C24")
                .range(2.0, 50.8, 0.01),
            manual_outer_width_mm: f64 = 6.35 => param("mm", "Manual outer width (tangential)",
                "", "Calculator!C25")
                .range(1.0, 25.4, 0.01),
            manual_outer_thickness_mm: f64 = 3.17 => param("mm", "Manual outer thickness (radial)",
                "", "Calculator!C26")
                .range(0.5, 10.0, 0.01),
            manual_outer_br_T: f64 = 1.30 => param("T", "Manual outer Br at 20 °C",
                "", "Calculator!C27")
                .range(0.2, 1.5, 0.001),
            grade_inner: String = "" => param_rust_only("-", "Inner magnet grade (manual dimensions)",
                "Addendum A6: a grade of the grade table, by exact name (e.g. N42SH, Y30). Used only when the inner part is not in the library: the manual dimensions with the grade's Br at 20 °C and maximum temperature. Blank = the manual Br and no rating."),
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
            axial_length_mm: Option<f64> = None => param_rust_only("mm", "Axial magnet length, both rings",
                "Addendum A1. Blank = each ring's part or manual length. A value sets both rings' axial length and keeps everything else each ring has (part or manual cross-section, grade, Br, rating): blocks cut or stacked to length. The calibration factor still follows the part names (Calculator C42). The hub length, cup cavity depth and retainer span follow the length change (Metal design C123 with the inner ring, C124 with the outer ring, C172 with the longer ring; each at least its ring's length: decision A2-8), so both axial stacks and the space claim follow; housing.* shows them. Inverse sizing's default free variable.")
                .range(2.0, 50.8, 0.01),
        }
    }
}

inputs! {
    /// Calculator inputs (Calculator!C5:C49) and the magnet group.
    pub struct CouplingInputs {
        fields {
            npole: i64 = 10 => param("-", "Number of poles per ring",
                "Even; one block per pole.", "Calculator!C5")
                .range(4.0, 40.0, 2.0),
            backiron: i64 = 1 => param("-", "Back iron",
                "1 = steel circuit, 0 = no intentional back iron.", "Calculator!C6")
                .choices(&[(1, "steel circuit"), (0, "no back iron")]),
            faceted: i64 = 1 => param("-", "Geometry type",
                "1 = flat blocks on polygons, 0 = true arcs.", "Calculator!C7")
                .choices(&[(1, "flat blocks"), (0, "arcs")]),
            inner_back_apothem_mm: f64 = 10.15 => param("mm", "Inner magnet back apothem, including bondline",
                "Machined hub apothem plus the inner bondline.", "Calculator!C8")
                .range(9.0, 30.0, 0.01),
            op_temp_C: f64 = 50.0 => param("°C", "Operating magnet temperature",
                "Br falls with temperature; torque ~ Br².", "Calculator!C10")
                .range(-40.0, 150.0, 0.5),
            bore_mm: f64 = 10.0 => param("mm", "Keyed bore diameter", "", "Calculator!C39")
                .range(4.0, 16.0, 0.1),
            keyway_depth_mm: f64 = 1.7 => param("mm", "Keyway depth in the hub",
                "", "Calculator!C40")
                .range(0.0, 4.0, 0.05),
            c_end: f64 = 0.15 => param("-", "End-effect coefficient",
                "f_end = 1 − c_end · pole pitch / L.", "Calculator!C41")
                .range(0.0, 0.5, 0.005)
                .assumption(),
            max_harmonic: i64 = WORKBOOK_MAX_HARMONIC => param_rust_only("-", "Highest odd harmonic summed",
                "Addendum A3 assumption. The torque model sums the odd space harmonics 1, 3, ... up to this one: the pull-out, the two circuit sums (C95, C96), every sweep row and the Calibration prototype, so the measured correction compares like with like. Workbook: 1, 3, 5. The cells of harmonics 1, 3 and 5 keep their terms; a harmonic left out reads 0 shear stress, and harmonics 7 to 11 add the Rust-only tau7_Pa, tau9_Pa and tau11_Pa.")
                .choices(&MAX_HARMONIC_CHOICES)
                .assumption(),
            mu0: f64 = MU0 => param("T·m/A", "Vacuum permeability", "", "Calculator!C43")
                .range(1.2566e-6, 1.2567e-6, 1e-11),
            gear_ratio: f64 = 5.0 => param("-", "Gearbox ratio", "", "Calculator!C45")
                .range(1.0, 20.0, 0.1),
            gear_efficiency: f64 = 0.95 => param("-", "Gearbox efficiency", "", "Calculator!C46")
                .range(0.5, 1.0, 0.01),
            gearbox_input_rating_Nm: f64 = 0.6 => param("N·m", "Gearbox input torque rating",
                "GAM 5:1 peak input rating.", "Calculator!C47")
                .range(0.05, 5.0, 0.01),
            drive_torque_Nm: f64 = 0.7 => param("N·m", "Torque the coupling must carry for driving (at the wheel)",
                "", "Calculator!C48")
                .range(0.0, 5.0, 0.01),
            drive_safety_factor: f64 = 1.3 => param("-", "Safety factor wanted on the driving torque",
                "", "Calculator!C49")
                .range(1.0, 3.0, 0.05),
        }
        groups {
            magnets: MagnetInputs,
        }
    }
}

results! {
    /// Calculator sheet results (Calculator!C9:C108).
    pub struct ModelResults {
        fields {
            inner_length_mm: f64 => out("mm", "Inner length (axial) used", "", "Calculator!C18"),
            inner_width_mm: f64 => out("mm", "Inner width (tangential) used", "", "Calculator!C19"),
            inner_thickness_mm: f64 => out("mm", "Inner thickness (radial) used",
                "", "Calculator!C20"),
            inner_br_T: f64 => out("T", "Inner Br at 20 °C used", "", "Calculator!C21"),
            inner_tmax_C: NumOrText => out("°C", "Inner max operating temperature (library)",
                "", "Calculator!C22"),
            outer_length_mm: f64 => out("mm", "Outer length (axial) used", "", "Calculator!C28"),
            outer_width_mm: f64 => out("mm", "Outer width (tangential) used", "", "Calculator!C29"),
            outer_thickness_mm: f64 => out("mm", "Outer thickness (radial) used",
                "", "Calculator!C30"),
            outer_br_T: f64 => out("T", "Outer Br at 20 °C used", "", "Calculator!C31"),
            outer_tmax_C: NumOrText => out("°C", "Outer max operating temperature (library)",
                "", "Calculator!C32"),
            active_length_mm: f64 => out("mm", "Active (overlapping) length", "", "Calculator!C33"),
            corner_gap_mm: f64 => out("mm", "Corner gap: inner block corner to outer block face",
                "Derived from the flat-face gap.", "Calculator!C9"),
            alpha_br_per_C: f64 => out("1/°C", "Br temperature coefficient", "", "Calculator!C35"),
            bsat_T: f64 => out("T", "Back-iron saturation flux density", "", "Calculator!C36"),
            cup_wall_corner_mm: f64 => out("mm", "Cup wall thickness at the pocket corners",
                "", "Calculator!C37"),
            hub_wall_mm: f64 => out("mm", "Hub wall under flats", "", "Calculator!C38"),
            f_cal: f64 => out("-", "Calibration factor",
                "0.95 for the steel-backed candidate.", "Calculator!C42"),
            inner_flat_width_mm: f64 => out("mm", "Inner flat width available",
                "", "Calculator!C51"),
            inner_flat_check: String => out("", "Inner flat check", "", "Calculator!C52"),
            hub_wall_past_key_mm: f64 => out("mm", "Hub wall past the keyway",
                "Keep ≥ ~2.5 mm.", "Calculator!C53"),
            inner_face_radius_mm: f64 => out("mm", "Inner block face radius", "", "Calculator!C54"),
            inner_corner_radius_mm: f64 => out("mm", "Inner block corner radius",
                "", "Calculator!C55"),
            outer_face_apothem_mm: f64 => out("mm", "Outer block face apothem",
                "", "Calculator!C56"),
            face_gap_mm: f64 => out("mm", "Gap at the flat centres (effective gap)",
                "", "Calculator!C57"),
            outer_flat_width_mm: f64 => out("mm", "Outer flat width at the faces",
                "", "Calculator!C58"),
            outer_flat_check: String => out("", "Outer flat check", "", "Calculator!C59"),
            outer_back_apothem_mm: f64 => out("mm", "Outer block back apothem",
                "", "Calculator!C60"),
            pocket_corner_radius_mm: f64 => out("mm", "Pocket corner radius", "", "Calculator!C61"),
            cup_od_mm: f64 => out("mm", "Cup outer diameter", "", "Calculator!C62"),
            cup_wall_flat_mm: f64 => out("mm", "Ring wall at the flats", "", "Calculator!C63"),
            gap_radius_mm: f64 => out("mm", "Gap mean radius", "", "Calculator!C64"),
            pole_pitch_mm: f64 => out("mm", "Pole pitch at the gap radius", "", "Calculator!C65"),
            fill_inner: f64 => out("-", "Inner fill factor", "", "Calculator!C66"),
            fill_outer: f64 => out("-", "Outer fill factor", "", "Calculator!C67"),
            br_inner_T_op: f64 => out("T", "Br inner at operating temperature",
                "", "Calculator!C69"),
            br_outer_T_op: f64 => out("T", "Br outer at operating temperature",
                "", "Calculator!C70"),
            k1: f64 => out("1/m", "Harmonic 1 wave number", "", "Calculator!C71"),
            b_i1: f64 => out("T", "Harmonic 1 inner amplitude", "", "Calculator!C72"),
            b_o1: f64 => out("T", "Harmonic 1 outer amplitude", "", "Calculator!C73"),
            s1_iron: f64 => out("-", "Harmonic 1 geometry factor with back iron",
                "", "Calculator!C74"),
            s1_free: f64 => out("-", "Harmonic 1 geometry factor without back iron",
                "", "Calculator!C75"),
            tau1_Pa: f64 => out("Pa", "Harmonic 1 shear stress", "", "Calculator!C76"),
            k3: f64 => out("1/m", "Harmonic 3 wave number", "", "Calculator!C77"),
            b_i3: f64 => out("T", "Harmonic 3 inner amplitude", "", "Calculator!C78"),
            b_o3: f64 => out("T", "Harmonic 3 outer amplitude", "", "Calculator!C79"),
            s3_iron: f64 => out("-", "Harmonic 3 geometry factor with back iron",
                "", "Calculator!C80"),
            s3_free: f64 => out("-", "Harmonic 3 geometry factor without back iron",
                "", "Calculator!C81"),
            tau3_Pa: f64 => out("Pa", "Harmonic 3 shear stress", "", "Calculator!C82"),
            k5: f64 => out("1/m", "Harmonic 5 wave number", "", "Calculator!C83"),
            b_i5: f64 => out("T", "Harmonic 5 inner amplitude", "", "Calculator!C84"),
            b_o5: f64 => out("T", "Harmonic 5 outer amplitude", "", "Calculator!C85"),
            s5_iron: f64 => out("-", "Harmonic 5 geometry factor with back iron",
                "", "Calculator!C86"),
            s5_free: f64 => out("-", "Harmonic 5 geometry factor without back iron",
                "", "Calculator!C87"),
            tau5_Pa: f64 => out("Pa", "Harmonic 5 shear stress", "", "Calculator!C88"),
            tau7_Pa: f64 => out_rust_only("Pa", "Harmonic 7 shear stress",
                "Addendum A3: summed when the highest harmonic (coupling.max_harmonic) is 7 or more; 0 otherwise."),
            tau9_Pa: f64 => out_rust_only("Pa", "Harmonic 9 shear stress",
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 shear stress",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            k7: f64 => out_rust_only("1/m", "Harmonic 7 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i7: f64 => out_rust_only("T", "Harmonic 7 inner amplitude", "As b_i1 to b_i5, for harmonic 7."),
            b_o7: f64 => out_rust_only("T", "Harmonic 7 outer amplitude", "As b_o1 to b_o5, for harmonic 7."),
            s7_iron: f64 => out_rust_only("-", "Harmonic 7 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 7."),
            s7_free: f64 => out_rust_only("-", "Harmonic 7 geometry factor without back iron", "As s1_free to s5_free, for harmonic 7."),
            k9: f64 => out_rust_only("1/m", "Harmonic 9 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i9: f64 => out_rust_only("T", "Harmonic 9 inner amplitude", "As b_i1 to b_i5, for harmonic 9."),
            b_o9: f64 => out_rust_only("T", "Harmonic 9 outer amplitude", "As b_o1 to b_o5, for harmonic 9."),
            s9_iron: f64 => out_rust_only("-", "Harmonic 9 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 9."),
            s9_free: f64 => out_rust_only("-", "Harmonic 9 geometry factor without back iron", "As s1_free to s5_free, for harmonic 9."),
            k11: f64 => out_rust_only("1/m", "Harmonic 11 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i11: f64 => out_rust_only("T", "Harmonic 11 inner amplitude", "As b_i1 to b_i5, for harmonic 11."),
            b_o11: f64 => out_rust_only("T", "Harmonic 11 outer amplitude", "As b_o1 to b_o5, for harmonic 11."),
            s11_iron: f64 => out_rust_only("-", "Harmonic 11 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 11."),
            s11_free: f64 => out_rust_only("-", "Harmonic 11 geometry factor without back iron", "As s1_free to s5_free, for harmonic 11."),
            amp1_Pa: f64 => out_rust_only("Pa", "Harmonic 1 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,1 B_o,1 S_1 / (2 μ0) in the circuit in effect, so harmonic 1's shear stress at electrical angle φ is this times sin(1φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp3_Pa: f64 => out_rust_only("Pa", "Harmonic 3 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,3 B_o,3 S_3 / (2 μ0) in the circuit in effect, so harmonic 3's shear stress at electrical angle φ is this times sin(3φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp5_Pa: f64 => out_rust_only("Pa", "Harmonic 5 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,5 B_o,5 S_5 / (2 μ0) in the circuit in effect, so harmonic 5's shear stress at electrical angle φ is this times sin(5φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp7_Pa: f64 => out_rust_only("Pa", "Harmonic 7 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,7 B_o,7 S_7 / (2 μ0) in the circuit in effect, so harmonic 7's shear stress at electrical angle φ is this times sin(7φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp9_Pa: f64 => out_rust_only("Pa", "Harmonic 9 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,9 B_o,9 S_9 / (2 μ0) in the circuit in effect, so harmonic 9's shear stress at electrical angle φ is this times sin(9φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,11 B_o,11 S_11 / (2 μ0) in the circuit in effect, so harmonic 11's shear stress at electrical angle φ is this times sin(11φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
                "", "Calculator!C90"),
            torque_2d_Nm: f64 => out("N·m", "2D pull-out torque (infinite length)",
                "", "Calculator!C91"),
            f_end: f64 => out("-", "End-effect factor", "", "Calculator!C92"),
            end_effect_check: String => out_rust_only("", "End-effect model check",
                "Audit M9 (the user's decision: a flag, no physics change): 'End-effect model out of range' when f_end is 0 or below, which makes the pull-out and every number computed from it meaningless; 'OK' otherwise."),
            pullout_Nm: f64 => out("N·m", "Pull-out torque at operating temperature",
                "Analytical estimate, not a guaranteed minimum.", "Calculator!C93"),
            pullout_20C_Nm: f64 => out("N·m", "Pull-out torque at 20 °C", "", "Calculator!C94"),
            pullout_iron_Nm: f64 => out("N·m", "Same layout with steel back iron (factor 0.95)",
                "", "Calculator!C95"),
            pullout_noiron_Nm: f64 => out("N·m", "Raw no-back-iron prediction, same layout",
                "", "Calculator!C96"),
            pullout_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle",
                "E7 (plan A-3: a term of the equation explorer): the electrical angle θ at which the torque-angle curve Σ τ_n sin(nθ) of the harmonics summed peaks; π/2 (half a pole pitch, the workbook's assumption) when that is the maximum. Every τ_n is taken at it."),
            iron_circuit_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle, steel circuit (C95)",
                "E7: as pullout_angle_rad, for the steel-backed circuit sum of C95."),
            free_circuit_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle, free-space circuit (C96)",
                "E7: as pullout_angle_rad, for the no-back-iron circuit sum of C96."),
            ripple_freq_Hz: f64 => out("Hz", "Torque ripple frequency at the design slip speed",
                "", "Calculator!C97"),
            gearbox_input_ripple_Nm: f64 => out("N·m", "Estimated gearbox input torque ripple amplitude",
                "", "Calculator!C99"),
            gearbox_reference_Nm: f64 => out("N·m", "Legacy gearbox torque reference (informational)",
                "", "Calculator!C100"),
            required_floor_Nm: f64 => out("N·m", "Required floor: larger of traction need and service minimum",
                "", "Calculator!C101"),
            verdict: String => out("", "Verdict",
                "Only the hot minimum is evaluated.", "Calculator!C102"),
            gap_flux_density_T: f64 => out("T", "Estimated gap flux density (flat circuit)",
                "", "Calculator!C103"),
            backiron_needed_mm: f64 => out("mm", "Back-iron thickness needed (sinusoidal flux, B_sat)",
                "", "Calculator!C104"),
            cup_ring_check: String => out("", "Cup ring check", "", "Calculator!C105"),
            hub_check: String => out("", "Hub check", "", "Calculator!C106"),
            inner_temp_check: String => out("", "Inner magnet temperature check",
                "Library rating only.", "Calculator!C107"),
            outer_temp_check: String => out("", "Outer magnet temperature check",
                "Library rating only.", "Calculator!C108"),
            inner_grade: String => out_rust_only("", "Inner magnet grade used",
                "The library part's grade, or the grade picked for manual dimensions; blank for a manual magnet without a grade."),
            outer_grade: String => out_rust_only("", "Outer magnet grade used",
                "As the inner grade, for the outer ring."),
            inner_alpha_br_per_C: f64 => out_rust_only("1/°C", "Inner Br temperature coefficient used",
                "Decision A2-7: the grade's for manual dimensions with a grade picked; else the calculator's alpha (Calibration C22, shown in C35)."),
            outer_alpha_br_per_C: f64 => out_rust_only("1/°C", "Outer Br temperature coefficient used",
                "As the inner coefficient, for the outer ring."),
            inner_magnet_density_g_mm3: f64 => out_rust_only("g/mm³", "Inner magnet density used",
                "Decision A2-7: the grade's for manual dimensions with a grade picked; else NdFeB, 7.5 g/cm³ (C110)."),
            outer_magnet_density_g_mm3: f64 => out_rust_only("g/mm³", "Outer magnet density used",
                "As the inner density, for the outer ring."),
        }
    }
}

results! {
    /// Calculator sheet mass results (Calculator!C110:C115).
    pub struct MassResults {
        fields {
            magnets_g: f64 => out("g", "Magnets (both rings)", "7.5 g/cm³.", "Calculator!C110"),
            cup_g: f64 => out("g", "Steel cup wall and integral rear web", "", "Calculator!C111"),
            hub_g: f64 => out("g", "Steel keyed inner hub", "", "Calculator!C112"),
            boss_g: f64 => out("g", "Integral steel shaft boss", "", "Calculator!C113"),
            total_g: f64 => out("g", "Preliminary rotating mass, including retainers",
                "Gross geometry plus hardware; holes, threads and slots not subtracted.",
                "Calculator!C114"),
            added_inertia_kgm2: f64 => out("kg·m²", "Added inertia at 0.25 m from the swing axis",
                "Point-mass estimate.", "Calculator!C115"),
        }
    }
}

/// One magnet ring as the Calculator uses it (Python `ResolvedMagnet`, plus the grade).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct ResolvedMagnet {
    pub length_mm: f64,
    pub width_mm: f64,
    pub thickness_mm: f64,
    pub br_T: f64,
    /// Library or grade rating, or `NOT_IN_LIBRARY` for manual magnets without a grade.
    pub tmax_C: NumOrText,
    /// The part's grade, or the grade picked for manual dimensions (Addendum A6).
    pub grade: Option<&'static Grade>,
    /// Whether the ring is in the grade mode: manual dimensions with a grade picked (the
    /// part is not in the library). Only then does the grade supply alpha(Br) and density.
    pub from_grade: bool,
}

impl ResolvedMagnet {
    /// The ring's reversible Br coefficient [1/°C]: its grade's in the grade mode (decision
    /// A2-7), else `calculator_alpha`, the calculator's single alpha (Calibration C22), which
    /// equals every library part's sintered NdFeB grade at its default.
    pub fn alpha_br(&self, calculator_alpha: f64) -> f64 {
        match self.grade {
            Some(g) if self.from_grade => g.alpha_br_per_C,
            _ => calculator_alpha,
        }
    }

    /// The ring's magnet density [g/mm³]: its grade's in the grade mode (decision A2-7), else
    /// the NdFeB density the workbook uses for every magnet.
    pub fn density_g_mm3(&self) -> f64 {
        match self.grade {
            Some(g) if self.from_grade => g.density_g_mm3,
            _ => NDFEB_DENSITY_G_MM3,
        }
    }
}

/// Library values when the part is found (workbook IFERROR/INDEX/MATCH); else the
/// manual dimensions, with the grade's Br and rating when a grade is picked
/// (Addendum A6, a Rust-only mode) and the manual Br and no rating otherwise.
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence),
/// its rating and grade from [`library::tmax_C`] and [`library::grade_id`], which apply E19.
/// The Rust-only `axial_length_mm` (Addendum A1), when set, replaces both rings' length.
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve =
        |part: &str, grade: &str, length_mm: f64, width_mm: f64, thickness_mm: f64, br_T: f64| {
            match lookup(part) {
                Some(spec) => ResolvedMagnet {
                    length_mm: spec.length_mm,
                    width_mm: spec.width_mm,
                    thickness_mm: spec.thickness_mm,
                    br_T: library::br_T(spec, dev),
                    tmax_C: NumOrText::Num(library::tmax_C(spec, dev)),
                    grade: grades::grade(library::grade_id(spec, dev)),
                    from_grade: false,
                },
                None => match grades::grade(grade) {
                    Some(g) => ResolvedMagnet {
                        length_mm,
                        width_mm,
                        thickness_mm,
                        br_T: g.br_T,
                        tmax_C: NumOrText::Num(g.tmax_C),
                        grade: Some(g),
                        from_grade: true,
                    },
                    None => ResolvedMagnet {
                        length_mm,
                        width_mm,
                        thickness_mm,
                        br_T,
                        tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
                        grade: None,
                        from_grade: false,
                    },
                },
            }
        };
    let with_length = |ring: ResolvedMagnet| match m.axial_length_mm {
        Some(length_mm) => ResolvedMagnet { length_mm, ..ring },
        None => ring,
    };
    (
        with_length(resolve(
            &m.part_inner,
            &m.grade_inner,
            m.manual_inner_length_mm,
            m.manual_inner_width_mm,
            m.manual_inner_thickness_mm,
            m.manual_inner_br_T,
        )),
        with_length(resolve(
            &m.part_outer,
            &m.grade_outer,
            m.manual_outer_length_mm,
            m.manual_outer_width_mm,
            m.manual_outer_thickness_mm,
            m.manual_outer_br_T,
        )),
    )
}

/// Whether a block fits its polygon flat: the comparison of the flat checks C52 and C59.
fn flat_fits(flat_mm: f64, width_mm: f64) -> bool {
    flat_mm >= width_mm
}

/// A block's share of its pole pitch at the magnet mid-radius: the width over
/// 2π (back apothem + thickness / 2) / N, the fill C66 (inner ring) and C67 (outer ring, from
/// the outer face apothem) before their min(1, ...). Above 1 the blocks overlap there.
pub fn pitch_share(width_mm: f64, back_apothem_mm: f64, thickness_mm: f64, npole: f64) -> f64 {
    width_mm / (2.0 * PI * (back_apothem_mm + thickness_mm / 2.0) / npole)
}

/// Whether both rings' blocks fit (Addendum A decision A2-4). Faceted blocks (`faceted` 1)
/// fit their polygon flats: the Calculator's C52 and C59 checks pass (their comparison,
/// [`flat_fits`]). Arcs (any other code; the flat checks read "n/a (arcs)") fit when neither
/// ring's blocks overlap at the magnet mid-radius: a [`pitch_share`] of at most 1, where C66
/// and C67 would otherwise clamp the fill and price overlapping arcs as if they fitted.
/// Inverse sizing counts only layouts that fit.
pub fn blocks_fit(ci: &CouplingInputs, r: &ModelResults) -> bool {
    if ci.faceted == 1 {
        flat_fits(r.inner_flat_width_mm, r.inner_width_mm)
            && flat_fits(r.outer_flat_width_mm, r.outer_width_mm)
    } else {
        let n = ci.npole as f64;
        let inner = pitch_share(
            r.inner_width_mm,
            ci.inner_back_apothem_mm,
            r.inner_thickness_mm,
            n,
        );
        let outer = pitch_share(
            r.outer_width_mm,
            r.outer_face_apothem_mm,
            r.outer_thickness_mm,
            n,
        );
        inner <= 1.0 && outer <= 1.0
    }
}

/// Measured correction only for the prototype's circuit (no iron, same poles, B842SH both rings).
pub fn select_calibration_factor(
    backiron: i64,
    npole: i64,
    part_i: &str,
    part_o: &str,
    prototype_poles: f64,
    f_cal_updated: f64,
    f_cal_original: f64,
) -> f64 {
    // Python compares the int npole with the float poles_per_ring (10 == 10.0).
    if backiron == 0 && npole as f64 == prototype_poles && part_i == "B842SH" && part_o == "B842SH"
    {
        f_cal_updated
    } else {
        f_cal_original
    }
}

/// One harmonic's terms (an entry of the Python `shear_stress` dict).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harmonic {
    pub n: u32,
    pub k: f64,
    pub bi: f64,
    pub bo: f64,
    pub s_iron: f64,
    pub s_free: f64,
    pub tau: f64,
}

impl Harmonic {
    /// The geometry factor of the circuit `backiron` selects: `s_iron` for 1 (steel), else `s_free`.
    pub fn s(&self, backiron: i64) -> f64 {
        if backiron == 1 {
            self.s_iron
        } else {
            self.s_free
        }
    }

    /// The torque-angle amplitude [Pa] in the circuit `backiron` selects, B_in B_on S_n / (2 μ0):
    /// the harmonic's shear stress at electrical angle x is this times sin(n x) (E7). The one
    /// expression for `shear_stress`, `at_pull_out` and the `amp*_Pa` results.
    pub fn amplitude(&self, backiron: i64, mu0: f64) -> f64 {
        self.bi * self.bo / (2.0 * mu0) * self.s(backiron)
    }
}

/// S_n: sinh ratio for a steel-backed circuit, exponential form for free-space rings.
pub fn geometry_factor(k: f64, t_i_mm: f64, t_o_mm: f64, g_mm: f64, backiron: i64) -> f64 {
    if backiron == 1 {
        (k * t_i_mm / 1000.0).sinh() * (k * t_o_mm / 1000.0).sinh()
            / (k * (t_i_mm + t_o_mm + g_mm) / 1000.0).sinh()
    } else {
        (1.0 - (-k * t_i_mm / 1000.0).exp())
            * (1.0 - (-k * t_o_mm / 1000.0).exp())
            * (-k * g_mm / 1000.0).exp()
            / 2.0
    }
}

/// Odd-harmonic amplitude of a square-wave magnetization with the given fill factor.
#[allow(non_snake_case)]
pub fn harmonic_amplitude(br_T: f64, n: u32, fill: f64) -> f64 {
    let n = f64::from(n);
    br_T * (4.0 / (n * PI)) * (n * fill * PI / 2.0).sin()
}

/// Corner radius of a flat block on a polygon, √(r_face² + (w/2)²): `r_face_mm`
/// the block's face apothem (back apothem plus thickness), `width_mm` its
/// tangential width. The one formula for Calculator C55, the E8 workbook
/// branches, the sweeps, Calibration and the Metal design sleeve.
pub(crate) fn corner_radius(r_face_mm: f64, width_mm: f64) -> f64 {
    (r_face_mm.powi(2) + (width_mm / 2.0).powi(2)).sqrt()
}

/// Reversible remanence factor at `temp_C`, 1 + α (T − 20 °C): Br(T) = Br20 · factor.
/// The one formula for the Calculator, Calibration, Metal design and Temperature design.
#[allow(non_snake_case)] // unit suffix, as the Python names
pub(crate) fn br_factor(alpha_br: f64, temp_C: f64) -> f64 {
    1.0 + alpha_br * (temp_C - 20.0)
}

/// Torque at magnet temperature `temp_C` over torque at 20 °C, Br_i(T) Br_o(T) / (Br_i Br_o):
/// each ring with its own coefficient (Addendum A-2 decision A2-7). With one coefficient it is
/// the workbook's (1 + α (T − 20 °C))², bit for bit (`x.powi(2)` is `x * x`). The one formula
/// for the Metal design and Temperature design torques at another temperature.
#[allow(non_snake_case)] // unit suffix, as the Python names
pub(crate) fn ring_pair_factor(alpha_inner: f64, alpha_outer: f64, temp_C: f64) -> f64 {
    br_factor(alpha_inner, temp_C) * br_factor(alpha_outer, temp_C)
}

/// Per-harmonic pull-out shear stress [Pa] and its parts, for every harmonic of
/// [`ODD_HARMONICS`] (Python: for `HARMONICS`); the model sums the first
/// [`harmonic_count`] of them (Addendum A3).
#[allow(clippy::too_many_arguments)] // Python signature
pub fn shear_stress(
    br_i: f64,
    br_o: f64,
    fill_i: f64,
    fill_o: f64,
    npole: i64,
    r_g_mm: f64,
    t_i_mm: f64,
    t_o_mm: f64,
    g_mm: f64,
    backiron: i64,
    mu0: f64,
) -> [Harmonic; 6] {
    ODD_HARMONICS.map(|n| {
        let nf = f64::from(n);
        let k = nf * (npole as f64 / 2.0) / (r_g_mm / 1000.0);
        let bi = harmonic_amplitude(br_i, n, fill_i);
        let bo = harmonic_amplitude(br_o, n, fill_o);
        let s_iron = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 1);
        let s_free = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 0);
        let mut hn = Harmonic {
            n,
            k,
            bi,
            bo,
            s_iron,
            s_free,
            tau: 0.0,
        };
        hn.tau = hn.amplitude(backiron, mu0) * (nf * PI / 2.0).sin();
        hn
    })
}

/// E7: the electrical angle of the true pull-out, when half a pole pitch is not it.
///
/// Torque against electrical angle x is T(x) = Σ a_i sin((2i + 1) x): `a[i]` is the
/// amplitude of the odd harmonic 2i + 1 ([`ODD_HARMONICS`], at most six). Half a pitch
/// (x = π/2) is always a stationary point, and the workbook evaluates every harmonic there.
/// The other stationary points solve dT/dx = Σ n a_n cos(n x) = cos x · Q(u) = 0 with
/// u = cos² x, where Q is a polynomial of degree `a.len() − 1` ([`stationary_polynomial`]).
/// T is symmetric about π/2 (odd harmonics), so x in [0, π/2] (u in [0, 1]) suffices.
/// [`roots_in_unit_interval`] finds every root of Q there with no closed form, so a vanishing
/// top amplitude (a ring at a fill of exactly 0.4 zeroes the fifth) cannot cancel digits, and
/// with no scan grid, so no pair of roots can hide inside a cell (decision 29 A). Returns
/// `None` when half a pitch is the maximum, so callers keep the workbook expression and
/// default outputs stay bit-identical; otherwise the angle whose torque exceeds the half-pitch
/// torque by more than 1e-12 relative.
pub fn peak_off_half_pitch(a: &[f64]) -> Option<f64> {
    let torque = |x: f64| {
        a.iter()
            .zip(ODD_HARMONICS)
            .fold(0.0, |acc, (&an, n)| acc + an * (f64::from(n) * x).sin())
    };
    // sin(nπ/2) = 1, −1, 1, ... for n = 1, 3, 5, ...: a1 − a3 + a5 − ...
    let half_pitch = a.iter().enumerate().fold(
        0.0,
        |acc, (i, &an)| if i % 2 == 0 { acc + an } else { acc - an },
    );
    roots_in_unit_interval(&stationary_polynomial(a))
        .into_iter()
        .map(|u| u.sqrt().acos())
        .map(|x| (x, torque(x)))
        .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
        .max_by(|p, q| p.1.total_cmp(&q.1))
        .map(|(x, _)| x)
}

/// Q(u), the coefficients (Q = q[0] + q[1] u + ...) of dT/dx / cos x for
/// T(x) = Σ a_i sin((2i + 1) x): Q = Σ n a_n P_n(u), where cos(n x) = cos x · P_n(cos² x)
/// and P_{n+2} = 2 (2u − 1) P_n − P_{n−2}, P_1 = P_{−1} = 1 (so P_3 = 4u − 3,
/// P_5 = 16u² − 20u + 5). For 1, 3, 5 that is (a1 − 9 a3 + 25 a5) + (12 a3 − 100 a5) u + 80 a5 u².
fn stationary_polynomial(a: &[f64]) -> Vec<f64> {
    let mut q = vec![0.0; a.len()];
    let (mut p_before, mut p) = (vec![1.0], vec![1.0]); // P_{n−2} and P_n, from n = 1
    for (&an, n) in a.iter().zip(ODD_HARMONICS) {
        for (qk, &pk) in q.iter_mut().zip(&p) {
            *qk += f64::from(n) * an * pk;
        }
        let mut next = vec![0.0; p.len() + 1]; // P_{n+2} = 4u P_n − 2 P_n − P_{n−2}
        for (k, &pk) in p.iter().enumerate() {
            next[k + 1] += 4.0 * pk;
            next[k] -= 2.0 * pk;
        }
        for (k, &pk) in p_before.iter().enumerate() {
            next[k] -= pk;
        }
        p_before = std::mem::replace(&mut p, next);
    }
    q
}

/// Every real root in [0, 1] of the polynomial c[0] + c[1] u + c[2] u² + ..., ascending.
///
/// A linear polynomial's root is −c[0] / c[1]. Above degree 1, the roots of the derivative
/// (found the same way) split [0, 1] into intervals on which the polynomial is monotone, so
/// each interval holds at most one root and bisection finds it to the last bit: no root pair
/// can hide between samples, the failure mode of a sampled scan (decision 29 A's Q' guard,
/// applied at every level).
/// Trailing zero coefficients are dropped; a constant (zero included) has no isolated root.
/// A double root is found only where the polynomial is exactly 0 there; that is an inflection
/// of the torque curve, never its peak.
fn roots_in_unit_interval(c: &[f64]) -> Vec<f64> {
    let degree = match c.iter().rposition(|&ck| ck != 0.0) {
        Some(d) if d > 0 => d,
        _ => return Vec::new(),
    };
    let c = &c[..=degree];
    if degree == 1 {
        let root = -c[0] / c[1];
        return if (0.0..=1.0).contains(&root) {
            vec![root]
        } else {
            Vec::new()
        };
    }
    let value = |u: f64| c.iter().rev().fold(0.0, |acc, &ck| acc * u + ck);
    let derivative: Vec<f64> = c
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &ck)| k as f64 * ck)
        .collect();
    let mut breaks = vec![0.0];
    breaks.extend(
        roots_in_unit_interval(&derivative)
            .into_iter()
            .filter(|&u| u > 0.0 && u < 1.0),
    );
    breaks.push(1.0);
    let mut roots = Vec::new();
    for w in breaks.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (f_lo, f_hi) = (value(lo), value(hi));
        if f_lo == 0.0 {
            roots.push(lo);
            continue;
        }
        // A root at `hi` is found as the next interval's `lo` (or at 1 below).
        if f_hi == 0.0 || (f_lo < 0.0) == (f_hi < 0.0) {
            continue;
        }
        let lo_negative = f_lo < 0.0;
        let mut root = None;
        for _ in 0..128 {
            let mid = lo + (hi - lo) / 2.0;
            if mid <= lo || mid >= hi {
                break; // lo and hi are adjacent doubles
            }
            let f_mid = value(mid);
            if f_mid == 0.0 {
                root = Some(mid);
                break;
            }
            if (f_mid < 0.0) == lo_negative {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        roots.push(root.unwrap_or(if value(lo).abs() <= value(hi).abs() {
            lo
        } else {
            hi
        }));
    }
    if value(1.0) == 0.0 {
        roots.push(1.0);
    }
    roots
}

/// The E7 gate every harmonic sum shares (the pull-out, the two circuit sums
/// C95 and C96, the sweep rows, Calibration C40-C42): the angle of the true
/// pull-out when E7 is on and half a pitch is not the maximum of the curve with
/// amplitudes `a` ([`peak_off_half_pitch`]). `None` means: keep the workbook's
/// half-pitch expression, bit for bit.
pub(crate) fn peak_angle(a: &[f64], dev: Deviations) -> Option<f64> {
    if dev.is_on(DeviationId::E7) {
        peak_off_half_pitch(a)
    } else {
        None
    }
}

/// One harmonic's term at electrical angle `x` (E7): amplitude `a` times sin(n x).
pub(crate) fn tau_at(a: f64, n: u32, x: f64) -> f64 {
    a * (f64::from(n) * x).sin()
}

/// E7 for one circuit: every harmonic's `tau` at the true pull-out angle when
/// half a pitch is not the maximum ([`peak_angle`] on the amplitudes
/// B_in,n·B_on,n/(2μ0)·S_n of the circuit `backiron` selects). Returns `h`
/// unchanged, bit for bit, when E7 is off or half a pitch is the maximum, with the angle
/// [`peak_angle`] found (`None`: half a pitch). `h` is the harmonic set summed (Addendum A3).
/// Shared by [`compute`] and the sweep rows.
pub(crate) fn at_pull_out(
    h: &[Harmonic],
    backiron: i64,
    mu0: f64,
    dev: Deviations,
) -> (Vec<Harmonic>, Option<f64>) {
    let amplitude = |x: &Harmonic| x.amplitude(backiron, mu0);
    let mut h_pull = h.to_vec();
    let amplitudes: Vec<f64> = h.iter().map(amplitude).collect();
    let peak = peak_angle(&amplitudes, dev);
    if let Some(x) = peak {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
    }
    (h_pull, peak)
}

/// Half a pole pitch as an electrical angle, π/2: where the workbook evaluates every
/// harmonic, and what the angle results (`pullout_angle_rad`, the circuit angles and the
/// Calibration's) read when [`peak_angle`] keeps it (returns `None`).
pub const HALF_PITCH_RAD: f64 = PI / 2.0;

/// Calculator sheet. Linked values come from Metal design, Calibration and Materials (see `api`).
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn compute(
    ci: &CouplingInputs,
    face_gap_mm: f64,
    bond_inner_mm: f64,
    bond_outer_mm: f64,
    cup_wall_corner_mm: f64,
    alpha_br: f64,
    bsat_T: f64,
    f_cal: f64,
    f_cal_original: f64,
    slip_rpm: f64,
    required_min_Nm: f64,
    dev: Deviations,
) -> ModelResults {
    let (mi, mo) = resolve_magnets(&ci.magnets, dev);
    let (N, a_i) = (ci.npole as f64, ci.inner_back_apothem_mm);
    let L = py_min(mi.length_mm, mo.length_mm);
    let r_face_i = a_i + mi.thickness_mm;
    let r_corner_i = if ci.faceted == 1 {
        corner_radius(r_face_i, mi.width_mm)
    } else {
        r_face_i
    };
    // Corner gap from the flat-face gap. Workbook: always the flat-block corner
    // geometry. E8: the inner corner radius C55, which is the face radius for arcs.
    let corner_gap = if dev.is_on(DeviationId::E8) {
        face_gap_mm - (r_corner_i - r_face_i)
    } else {
        face_gap_mm - (corner_radius(r_face_i, mi.width_mm) - r_face_i)
    };
    let hub_wall = a_i - bond_inner_mm - ci.bore_mm / 2.0;

    let flat_i = 2.0 * a_i * (PI / N).tan();
    let chk_i = if ci.faceted == 1 {
        if flat_fits(flat_i, mi.width_mm) {
            format!("OK, {} mm slack", fmt_fixed(flat_i - mi.width_mm, 2))
        } else {
            "TOO NARROW: increase apothem or reduce poles".to_owned()
        }
    } else {
        "n/a (arcs)".to_owned()
    };
    let A_o = r_corner_i + corner_gap;
    let g_m = A_o - r_face_i;
    let flat_o = 2.0 * A_o * (PI / N).tan();
    let chk_o = if ci.faceted == 1 {
        if flat_fits(flat_o, mo.width_mm) {
            format!(
                "OK, blocks {} mm apart at the faces",
                fmt_fixed(flat_o - mo.width_mm, 2)
            )
        } else {
            "TOO NARROW: increase gap/apothem or reduce poles".to_owned()
        }
    } else {
        "n/a (arcs)".to_owned()
    };
    let A_back = A_o + mo.thickness_mm;
    let r_pocket = if ci.faceted == 1 {
        (A_back + bond_outer_mm) / (PI / N).cos()
    } else {
        A_back + bond_outer_mm
    };
    let OD = 2.0 * (r_pocket + cup_wall_corner_mm);
    // E6: the pocket flat sits at the block back plus the outer bondline (as C61, C62 place it).
    let wall_f = if dev.is_on(DeviationId::E6) {
        OD / 2.0 - (A_back + bond_outer_mm)
    } else {
        OD / 2.0 - A_back
    };
    let R_g = r_face_i + g_m / 2.0;
    let tau_p = 2.0 * PI * R_g / N;
    let al_i = py_min(1.0, pitch_share(mi.width_mm, a_i, mi.thickness_mm, N));
    let al_o = py_min(1.0, pitch_share(mo.width_mm, A_o, mo.thickness_mm, N));

    // Decision A2-7: each ring with its own coefficient (a grade-mode ring's grade, else C22).
    let (alpha_i, alpha_o) = (mi.alpha_br(alpha_br), mo.alpha_br(alpha_br));
    let bri = mi.br_T * br_factor(alpha_i, ci.op_temp_C);
    let bro = mo.br_T * br_factor(alpha_o, ci.op_temp_C);
    let h = shear_stress(
        bri,
        bro,
        al_i,
        al_o,
        ci.npole,
        R_g,
        mi.thickness_mm,
        mo.thickness_mm,
        g_m,
        ci.backiron,
        ci.mu0,
    );
    // Addendum A3: the harmonics summed, 1, 3, ... up to coupling.max_harmonic.
    let count = harmonic_count(ci.max_harmonic);
    let used = &h[..count.unwrap_or(0)];
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `used` (the half-pitch terms) stays for the per-circuit sums below.
    let (h_pull, pull_peak) = at_pull_out(used, ci.backiron, ci.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let tau = harmonic_sum(count, taus.iter().copied()); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
    let T2D = tau * AL;
    let f_end = 1.0 - ci.c_end * tau_p / L;
    let T_pull = T2D * f_end * f_cal;
    let T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro);
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    // Each gives the sum and the E7 angle it is taken at (`None`: half a pitch).
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients: Vec<f64> = used.iter().map(|x| x.bi * x.bo * s(x)).collect();
        let peak = peak_angle(&coefficients, dev);
        let sum = match peak {
            Some(x) => harmonic_sum(
                count,
                used.iter()
                    .zip(&coefficients)
                    .map(|(hn, &c)| tau_at(c, hn.n, x)),
            ),
            None => harmonic_sum(
                count,
                used.iter()
                    .map(|x| x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()),
            ),
        };
        (sum, peak)
    };
    let (iron_sum, iron_peak) = circuit(|x| x.s_iron);
    let (free_sum, free_peak) = circuit(|x| x.s_free);
    let T_iron = iron_sum / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
    let T_noiron = free_sum / (2.0 * ci.mu0) * AL * f_end * f_cal;

    let floor_ = py_max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm);
    // E10: in the series circuit each magnet contributes its own MMF (Br·t), not the mean Br.
    let B_gap = if dev.is_on(DeviationId::E10) {
        (bri * mi.thickness_mm + bro * mo.thickness_mm) / (mi.thickness_mm + mo.thickness_mm + g_m)
    } else {
        (bri + bro) / 2.0 * (mi.thickness_mm + mo.thickness_mm)
            / (mi.thickness_mm + mo.thickness_mm + g_m)
    };
    let t_bi = B_gap * tau_p / (PI * bsat_T);

    // (A block may not start with `if ... {}.to_owned()`: bind the &str first.)
    let thick_check = |wall: f64| -> String {
        let text = if ci.backiron == 0 {
            "No back iron"
        } else if wall >= t_bi {
            "Thickness OK"
        } else {
            "Too thin"
        };
        text.to_owned()
    };
    let temp_check = |tmax: NumOrText| -> String {
        let text = match tmax {
            NumOrText::Text(_) => "unknown",
            NumOrText::Num(t) if ci.op_temp_C <= t => "OK",
            NumOrText::Num(_) => "OVER the magnet rating",
        };
        text.to_owned()
    };
    let [h1, h3, h5, h7, h9, h11] = h;
    let tau_n = |i: usize| harmonic_slot(count, &taus, i);

    ModelResults {
        inner_length_mm: mi.length_mm,
        inner_width_mm: mi.width_mm,
        inner_thickness_mm: mi.thickness_mm,
        inner_br_T: mi.br_T,
        inner_tmax_C: mi.tmax_C,
        outer_length_mm: mo.length_mm,
        outer_width_mm: mo.width_mm,
        outer_thickness_mm: mo.thickness_mm,
        outer_br_T: mo.br_T,
        outer_tmax_C: mo.tmax_C,
        active_length_mm: L,
        corner_gap_mm: corner_gap,
        alpha_br_per_C: alpha_br,
        bsat_T,
        cup_wall_corner_mm,
        hub_wall_mm: hub_wall,
        f_cal,
        inner_flat_width_mm: flat_i,
        inner_flat_check: chk_i,
        hub_wall_past_key_mm: hub_wall - ci.keyway_depth_mm,
        inner_face_radius_mm: r_face_i,
        inner_corner_radius_mm: r_corner_i,
        outer_face_apothem_mm: A_o,
        face_gap_mm: g_m,
        outer_flat_width_mm: flat_o,
        outer_flat_check: chk_o,
        outer_back_apothem_mm: A_back,
        pocket_corner_radius_mm: r_pocket,
        cup_od_mm: OD,
        cup_wall_flat_mm: wall_f,
        gap_radius_mm: R_g,
        pole_pitch_mm: tau_p,
        fill_inner: al_i,
        fill_outer: al_o,
        br_inner_T_op: bri,
        br_outer_T_op: bro,
        k1: h1.k,
        b_i1: h1.bi,
        b_o1: h1.bo,
        s1_iron: h1.s_iron,
        s1_free: h1.s_free,
        tau1_Pa: tau_n(0),
        k3: h3.k,
        b_i3: h3.bi,
        b_o3: h3.bo,
        s3_iron: h3.s_iron,
        s3_free: h3.s_free,
        tau3_Pa: tau_n(1),
        k5: h5.k,
        b_i5: h5.bi,
        b_o5: h5.bo,
        s5_iron: h5.s_iron,
        s5_free: h5.s_free,
        tau5_Pa: tau_n(2),
        tau7_Pa: tau_n(3),
        tau9_Pa: tau_n(4),
        tau11_Pa: tau_n(5),
        k7: h7.k,
        b_i7: h7.bi,
        b_o7: h7.bo,
        s7_iron: h7.s_iron,
        s7_free: h7.s_free,
        k9: h9.k,
        b_i9: h9.bi,
        b_o9: h9.bo,
        s9_iron: h9.s_iron,
        s9_free: h9.s_free,
        k11: h11.k,
        b_i11: h11.bi,
        b_o11: h11.bo,
        s11_iron: h11.s_iron,
        s11_free: h11.s_free,
        amp1_Pa: h[0].amplitude(ci.backiron, ci.mu0),
        amp3_Pa: h[1].amplitude(ci.backiron, ci.mu0),
        amp5_Pa: h[2].amplitude(ci.backiron, ci.mu0),
        amp7_Pa: h[3].amplitude(ci.backiron, ci.mu0),
        amp9_Pa: h[4].amplitude(ci.backiron, ci.mu0),
        amp11_Pa: h[5].amplitude(ci.backiron, ci.mu0),
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
        f_end,
        end_effect_check: end_effect_check(f_end).to_owned(),
        pullout_Nm: T_pull,
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
        pullout_noiron_Nm: T_noiron,
        pullout_angle_rad: pull_peak.unwrap_or(HALF_PITCH_RAD),
        iron_circuit_angle_rad: iron_peak.unwrap_or(HALF_PITCH_RAD),
        free_circuit_angle_rad: free_peak.unwrap_or(HALF_PITCH_RAD),
        ripple_freq_Hz: N / 2.0 * slip_rpm / 60.0,
        gearbox_input_ripple_Nm: T_pull / (ci.gear_ratio * ci.gear_efficiency),
        gearbox_reference_Nm: ci.gearbox_input_rating_Nm * ci.gear_ratio * ci.gear_efficiency,
        required_floor_Nm: floor_,
        verdict: if T_pull < floor_ {
            "Below hot minimum"
        } else {
            "Nominal only: hot test"
        }
        .to_owned(),
        gap_flux_density_T: B_gap,
        backiron_needed_mm: t_bi,
        cup_ring_check: thick_check(cup_wall_corner_mm),
        hub_check: thick_check(hub_wall),
        inner_temp_check: temp_check(mi.tmax_C),
        outer_temp_check: temp_check(mo.tmax_C),
        inner_grade: mi.grade.map_or("", |g| g.id).to_owned(),
        outer_grade: mo.grade.map_or("", |g| g.id).to_owned(),
        inner_alpha_br_per_C: alpha_i,
        outer_alpha_br_per_C: alpha_o,
        inner_magnet_density_g_mm3: mi.density_g_mm3(),
        outer_magnet_density_g_mm3: mo.density_g_mm3(),
    }
}

/// Whether the cup wall, rear web and boss are the body material
/// (`PartProperties::body`: the workbook's aluminium, or a non-ferromagnetic
/// back-iron pick), not the back-iron steel: E9, with no intentional back iron
/// (`backiron` is Calculator!C6 in effect). The one gate the mass model (C111,
/// C113), the heat capacity (E15) and the removed-disc mass (E16) read, so they
/// cannot disagree.
pub(crate) fn cup_is_body_material(backiron: i64, dev: Deviations) -> bool {
    dev.is_on(DeviationId::E9) && backiron != 1
}

/// The density of the cup wall, rear web and boss [g/mm³]: the body material's
/// (C42 by default) when [`cup_is_body_material`], else steel (C132). The mass
/// model's C111 and C113 use it, and E16 prices the disc bored out of the web with
/// it (one source of truth).
pub(crate) fn cup_boss_density(
    backiron: i64,
    steel_density_g_mm3: f64,
    body_density_g_mm3: f64,
    dev: Deviations,
) -> f64 {
    if cup_is_body_material(backiron, dev) {
        body_density_g_mm3
    } else {
        steel_density_g_mm3
    }
}

/// Whether the keyed hub is the body material, not the back-iron steel: the
/// workbook prices it as aluminium whenever C6 is not 1 (C112), with or without
/// E9. Read by the mass model, E15 and E18.
pub(crate) fn hub_is_body_material(backiron: i64) -> bool {
    backiron != 1
}

/// Calculator rows 110-115. Gross solids: no holes, slots or threads subtracted.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn mass_estimate(
    ci: &CouplingInputs,
    r: &ModelResults,
    bond_inner_mm: f64,
    bond_outer_mm: f64,
    cup_depth_mm: f64,
    web_mm: f64,
    hub_length_mm: f64,
    boss_length_mm: f64,
    boss_od_mm: f64,
    steel_density_g_mm3: f64,
    al_density_g_mm3: f64,
    retainers_g: f64,
    hardware_g: f64,
    cap_g: f64,
    endplates_g: f64,
    dev: Deviations,
) -> MassResults {
    let N = ci.npole as f64;
    let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
    let volume_o = r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm;
    let (rho_i, rho_o) = (r.inner_magnet_density_g_mm3, r.outer_magnet_density_g_mm3);
    // Decision A2-7: each ring at its own density; rings of one density keep the workbook's
    // single product (bit for bit: N (V_i + V_o) rho).
    let m_mag = if rho_i == rho_o {
        N * (volume_i + volume_o) * rho_i
    } else {
        N * (volume_i * rho_i + volume_o * rho_o)
    };
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density =
        cup_boss_density(ci.backiron, steel_density_g_mm3, al_density_g_mm3, dev);
    let pocket = r.outer_back_apothem_mm + bond_outer_mm;
    // E8: arcs sit in a round pocket, not a polygon.
    let cavity = if dev.is_on(DeviationId::E8) && ci.faceted != 1 {
        PI * pocket.powi(2)
    } else {
        N * pocket.powi(2) * (PI / N).tan()
    };
    let m_ring = ((PI * (r.cup_od_mm / 2.0).powi(2) - cavity) * cup_depth_mm
        + PI * ((r.cup_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2)) * web_mm)
        * cup_boss_density;
    let hub_area = if ci.faceted == 1 {
        N * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2) * (PI / N).tan()
    } else {
        PI * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2)
    };
    let m_hub = (hub_area - PI * (ci.bore_mm / 2.0).powi(2))
        * hub_length_mm
        * (if hub_is_body_material(ci.backiron) {
            al_density_g_mm3
        } else {
            steel_density_g_mm3
        });
    let m_boss = PI
        * ((boss_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2))
        * boss_length_mm
        * cup_boss_density;
    let total = m_mag + m_ring + m_hub + m_boss + retainers_g + hardware_g + cap_g + endplates_g;
    MassResults {
        magnets_g: m_mag,
        cup_g: m_ring,
        hub_g: m_hub,
        boss_g: m_boss,
        total_g: total,
        added_inertia_kgm2: total / 1000.0 * 0.25_f64.powi(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Calculator at the workbook defaults, links as `api::compute` passes them.
    fn at(ci: &CouplingInputs) -> ModelResults {
        at_with(ci, 0.05, Deviations::NONE)
    }

    /// [`at`] with another outer bondline (Metal design!C121) and deviation set.
    fn at_with(ci: &CouplingInputs, bond_outer_mm: f64, dev: Deviations) -> ModelResults {
        compute(
            ci,
            1.4,
            0.05,
            bond_outer_mm,
            1.8,
            -0.0012,
            1.5,
            0.95,
            0.95,
            2000.0,
            2.5,
            dev,
        )
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() <= 1e-9 * want.abs()
    }

    #[test]
    fn default_design_matches_the_workbook() {
        let r = at(&CouplingInputs::default());
        assert!(close(r.pullout_Nm, 2.6472742027215)); // Calculator!C93
        assert!(close(r.corner_gap_mm, 1.02682560543393)); // C9
        assert_eq!(r.inner_flat_check, "OK, 0.25 mm slack"); // C52
        assert_eq!(r.outer_flat_check, "OK, blocks 3.22 mm apart at the faces"); // C59
        assert_eq!(r.verdict, "Nominal only: hot test"); // C102
        assert_eq!(
            (r.cup_ring_check.as_str(), r.hub_check.as_str()),
            ("Too thin", "Thickness OK")
        ); // C105, C106
        assert_eq!(r.inner_tmax_C, NumOrText::Num(150.0)); // C22
    }

    #[test]
    fn a_grade_gives_manual_dimensions_its_br_and_rating() {
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        ci.magnets.manual_inner_br_T = 1.2; // ignored: the grade supplies Br
        let r = at(&ci);
        assert_eq!(r.inner_br_T, 0.37);
        assert_eq!(r.inner_tmax_C, NumOrText::Num(250.0));
        assert_eq!(r.inner_temp_check, "OK");
        assert_eq!(r.inner_grade, "Y30");
        assert_eq!(
            (r.inner_length_mm, r.inner_thickness_mm),
            (
                ci.magnets.manual_inner_length_mm,
                ci.magnets.manual_inner_thickness_mm
            )
        );
        // The outer ring keeps its library part.
        assert_eq!((r.outer_br_T, r.outer_grade.as_str()), (1.29, "N42SH"));

        // Mirror: each ring reads its own grade field. Different grades on the two rings, so a
        // swapped or copy-pasted field cannot pass (Y30 on both rings would).
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.part_outer = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        ci.magnets.grade_outer = "N42".to_owned();
        ci.magnets.manual_inner_br_T = 1.2; // both ignored: the grades supply Br
        ci.magnets.manual_outer_br_T = 1.1;
        ci.op_temp_C = 100.0; // over N42's 80 C rating, under Y30's 250 C
        let r = at(&ci);
        assert_eq!(
            (r.outer_br_T, r.outer_tmax_C, r.outer_grade.as_str()),
            (1.30, NumOrText::Num(80.0), "N42")
        );
        assert_eq!(r.outer_temp_check, "OVER the magnet rating");
        assert_eq!(
            (r.outer_length_mm, r.outer_thickness_mm),
            (
                ci.magnets.manual_outer_length_mm,
                ci.magnets.manual_outer_thickness_mm
            )
        );
        // The inner ring is unchanged by the outer grade.
        assert_eq!(
            (r.inner_br_T, r.inner_tmax_C, r.inner_grade.as_str()),
            (0.37, NumOrText::Num(250.0), "Y30")
        );
        assert_eq!(r.inner_temp_check, "OK");

        // Each grade field alone leaves the other, manual ring manual.
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_outer = "N42".to_owned(); // part_outer is still B842SH: no effect
        let r = at(&ci);
        assert_eq!(r.inner_br_T, ci.magnets.manual_inner_br_T);
        assert_eq!(r.inner_tmax_C, NumOrText::Text(NOT_IN_LIBRARY));
        assert_eq!(r.inner_grade, "");
        assert_eq!(r.inner_temp_check, "unknown");
        assert_eq!((r.outer_br_T, r.outer_grade.as_str()), (1.29, "N42SH"));
        let mut ci = CouplingInputs::default();
        ci.magnets.part_outer = String::new();
        ci.magnets.grade_inner = "Y30".to_owned(); // part_inner is still B842SH: no effect
        let r = at(&ci);
        assert_eq!(r.outer_br_T, ci.magnets.manual_outer_br_T);
        assert_eq!(r.outer_tmax_C, NumOrText::Text(NOT_IN_LIBRARY));
        assert_eq!(r.outer_grade, "");
        assert_eq!(r.outer_temp_check, "unknown");
        assert_eq!((r.inner_br_T, r.inner_grade.as_str()), (1.29, "N42SH"));
    }

    #[test]
    fn a_grade_ring_takes_its_grade_alpha_and_density() {
        // Decision A2-7: a ring in the grade mode (manual dimensions with a grade picked) takes
        // its grade's Br temperature coefficient and density. A library part (every one
        // sintered NdFeB) and a manual magnet without a grade keep the calculator's single
        // alpha (Calibration C22, here the `alpha_br` argument) and the NdFeB density.
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        let r = at(&ci);
        assert_eq!(
            (r.inner_alpha_br_per_C, r.outer_alpha_br_per_C),
            (-0.002, -0.0012)
        );
        assert_eq!(
            (r.inner_magnet_density_g_mm3, r.outer_magnet_density_g_mm3),
            (0.005, NDFEB_DENSITY_G_MM3)
        );
        assert_eq!(r.br_inner_T_op, 0.37 * br_factor(-0.002, 50.0));
        assert_eq!(r.br_outer_T_op, r.outer_br_T * br_factor(-0.0012, 50.0));
        assert_eq!(
            r.alpha_br_per_C, -0.0012,
            "C35 still shows the calculator's alpha"
        );
        // A gradeless manual magnet and a library part follow the argument, whatever it is.
        let mut manual = CouplingInputs::default();
        manual.magnets.part_inner = String::new();
        let r = compute(
            &manual,
            1.4,
            0.05,
            0.05,
            1.8,
            -0.001,
            1.5,
            0.95,
            0.95,
            2000.0,
            2.5,
            Deviations::NONE,
        );
        assert_eq!(
            (r.inner_alpha_br_per_C, r.outer_alpha_br_per_C),
            (-0.001, -0.001)
        );
        assert_eq!(r.inner_magnet_density_g_mm3, NDFEB_DENSITY_G_MM3);
        // An NdFeB grade in the grade mode has the calculator's default values: nothing moves.
        let mut n42 = CouplingInputs::default();
        n42.magnets.part_inner = String::new();
        n42.magnets.grade_inner = "N42".to_owned();
        let r = at(&n42);
        assert_eq!(
            (r.inner_alpha_br_per_C, r.inner_magnet_density_g_mm3),
            (-0.0012, NDFEB_DENSITY_G_MM3)
        );
    }

    #[test]
    fn a_library_part_wins_over_a_grade_and_an_unknown_grade_is_manual() {
        let mut ci = CouplingInputs::default();
        ci.magnets.grade_inner = "Y30".to_owned(); // the part B842SH is in the library
        ci.magnets.grade_outer = "N42".to_owned(); // likewise for the outer ring
        let r = at(&ci);
        assert_eq!(r, at(&CouplingInputs::default()));
        assert_eq!(
            (r.inner_grade.as_str(), r.outer_grade.as_str()),
            ("N42SH", "N42SH")
        );
        for text in ["", "y30", "Y30 ", "N 42", "N42"] {
            let mut ci = CouplingInputs::default();
            ci.magnets.part_inner = String::new();
            ci.magnets.grade_inner = text.to_owned();
            let r = at(&ci);
            if text == "N42" {
                assert_eq!((r.inner_br_T, r.inner_grade.as_str()), (1.30, "N42"));
                continue;
            }
            assert_eq!(r.inner_br_T, ci.magnets.manual_inner_br_T, "{text:?}");
            assert_eq!(r.inner_tmax_C, NumOrText::Text(NOT_IN_LIBRARY), "{text:?}");
            assert_eq!(r.inner_grade, "", "{text:?}");
        }
    }

    #[test]
    fn part_lookup_is_exact_text() {
        // Review Focus 2: near misses fall back to the manual magnet, like the workbook.
        for part in ["b842sh", "B842SH ", ""] {
            let mut ci = CouplingInputs::default();
            ci.magnets.part_inner = part.to_owned();
            let r = at(&ci);
            assert_eq!(r.inner_tmax_C, NumOrText::Text(NOT_IN_LIBRARY), "{part:?}");
            assert_eq!(r.inner_temp_check, "unknown", "{part:?}");
            assert_eq!(
                select_calibration_factor(0, 10, part, "B842SH", 10.0, 1.0658, 0.95),
                0.95,
                "{part:?}"
            );
        }
        assert_eq!(
            select_calibration_factor(0, 10, "B842SH", "B842SH", 10.0, 1.0658, 0.95),
            1.0658
        );
    }

    #[test]
    fn the_axial_length_override_sets_both_rings_and_keeps_the_rest() {
        // Addendum A1: blocks of the parts' cross-section, grade, Br and rating, cut or stacked
        // to one axial length. Torque ~ L * f_end = L - c_end * pole pitch (the pitch does not
        // depend on L).
        let base = at(&CouplingInputs::default());
        let mut ci = CouplingInputs::default();
        ci.magnets.axial_length_mm = Some(20.0);
        let r = at(&ci);
        assert_eq!(
            (r.inner_length_mm, r.outer_length_mm, r.active_length_mm),
            (20.0, 20.0, 20.0)
        );
        assert_eq!(
            (r.inner_width_mm, r.inner_thickness_mm, r.outer_width_mm),
            (
                base.inner_width_mm,
                base.inner_thickness_mm,
                base.outer_width_mm
            )
        );
        assert_eq!(
            (r.inner_br_T, r.inner_tmax_C),
            (base.inner_br_T, base.inner_tmax_C)
        );
        assert_eq!((r.inner_grade.as_str(), r.f_cal), ("N42SH", base.f_cal));
        assert_eq!(r.pole_pitch_mm, base.pole_pitch_mm);
        let excess = |l: f64| l - CouplingInputs::default().c_end * base.pole_pitch_mm;
        assert!(close(
            r.pullout_Nm / base.pullout_Nm,
            excess(20.0) / excess(12.7)
        ));
        // It overrides manual lengths too, and a blank override changes nothing.
        let mut manual = CouplingInputs::default();
        manual.magnets.part_inner = String::new();
        manual.magnets.manual_inner_length_mm = 30.0;
        manual.magnets.axial_length_mm = Some(15.0);
        let r = at(&manual);
        assert_eq!((r.inner_length_mm, r.outer_length_mm), (15.0, 15.0));
        let mut blank = CouplingInputs::default();
        blank.magnets.axial_length_mm = None;
        assert_eq!(at(&blank), base);
    }

    #[test]
    fn e3_reaches_library_parts_but_never_a_typed_manual_br() {
        let mut m = MagnetInputs {
            part_inner: "BX082SH".to_owned(),
            part_outer: String::new(),
            ..MagnetInputs::default()
        };
        m.manual_outer_br_T = 1.25;
        let br = |dev| {
            let (i, o) = resolve_magnets(&m, dev);
            (i.br_T, o.br_T)
        };
        assert_eq!(br(Deviations::NONE), (1.29, 1.25));
        assert_eq!(br(Deviations::only(DeviationId::E3)), (1.30, 1.25));
        assert_eq!(br(Deviations::ALL), (1.30, 1.25));
    }

    #[test]
    fn e6_round_pocket_wall_at_the_flats_is_the_corner_wall() {
        // Audit report row E6: a round (arc-mode) pocket has a uniform wall, the
        // corner wall (1.8 mm here); the workbook adds the outer bondline to it.
        let ci = CouplingInputs {
            faceted: 0,
            ..CouplingInputs::default()
        };
        for bond in [0.0, 0.05, 0.2] {
            let wall = |dev| at_with(&ci, bond, dev).cup_wall_flat_mm;
            assert!(close(wall(Deviations::NONE), 1.8 + bond), "{bond}");
            for dev in [Deviations::only(DeviationId::E6), Deviations::ALL] {
                assert!(close(wall(dev), 1.8), "{bond} {dev:?}");
            }
        }
    }

    #[test]
    fn half_pitch_stays_the_peak_when_the_third_harmonic_is_small() {
        assert_eq!(peak_off_half_pitch(&[1.0, -0.011, 0.001]), None); // default design: A3/A1 = -0.011
        assert_eq!(peak_off_half_pitch(&[1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn a_large_third_harmonic_moves_the_peak_and_raises_it() {
        let a = [1.0, 0.2, 0.0]; // A1 < 9 A3: half a pitch is a local minimum
        let x = peak_off_half_pitch(&a).expect("the peak moves");
        let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
        assert!(x > 0.0 && x < std::f64::consts::FRAC_PI_2);
        assert!(t(x) > t(std::f64::consts::FRAC_PI_2));
        // stationary: dT/dx = 0 at the returned angle
        let slope = a[0] * x.cos() + 3.0 * a[1] * (3.0 * x).cos() + 5.0 * a[2] * (5.0 * x).cos();
        assert!(slope.abs() < 1e-9, "{slope}");
    }

    /// T(x) = Σ a_i sin((2i + 1) x), the curve [`peak_off_half_pitch`] maximizes.
    fn torque_at(a: &[f64], x: f64) -> f64 {
        a.iter()
            .zip(ODD_HARMONICS)
            .fold(0.0, |acc, (&an, n)| acc + an * (f64::from(n) * x).sin())
    }

    /// Brute-force maximum of `torque_at` on [0, π/2]: a grid, then a ternary search
    /// around the best grid point. Never above the true maximum.
    fn grid_maximum(a: &[f64], points: usize) -> f64 {
        let step = std::f64::consts::FRAC_PI_2 / (points - 1) as f64;
        let (best_i, best) = (0..points)
            .map(|i| (i, torque_at(a, i as f64 * step)))
            .max_by(|p, q| p.1.total_cmp(&q.1))
            .expect("points > 0");
        let (mut lo, mut hi) = (
            (best_i as f64 - 1.0).max(0.0) * step,
            ((best_i + 1) as f64 * step).min(std::f64::consts::FRAC_PI_2),
        );
        for _ in 0..100 {
            let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if torque_at(a, m1) < torque_at(a, m2) {
                lo = m1;
            } else {
                hi = m2;
            }
        }
        best.max(torque_at(a, (lo + hi) / 2.0))
    }

    /// A uniform source in [0, 1) (splitmix64), the property tests' generator.
    fn uniform_source(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random amplitude triple (a1, a3, a5), case `i` of three bands: the general case,
    /// a fifth harmonic in the band |a5/a1| <= 1.5e-13 where the textbook quadratic cancels
    /// (a ring at a fill of exactly 0.4 or 0.8 gives sin(5 fill pi/2) ~ 1e-16), and a5 = 0.
    /// a1 in [0.5, 1.5], a3 in +-a1/2 (A1 < 9 A3 moves the peak), so half a pitch stays
    /// positive and the maximum is a stationary point in (0, pi/2].
    fn random_triple(uniform: &mut impl FnMut() -> f64, i: usize) -> [f64; 3] {
        let a1 = 0.5 + uniform();
        let a3 = (uniform() - 0.5) * a1;
        let sign = if uniform() < 0.5 { -1.0 } else { 1.0 };
        let a5 = match i % 3 {
            0 => (uniform() - 0.5) * 0.6 * a1,
            1 => sign * a1 * 10f64.powf(-20.0 + 7.0 * uniform()) * 1.5,
            _ => 0.0,
        };
        [a1, a3, a5]
    }

    #[test]
    fn peak_angle_is_the_e7_gate() {
        // The one gate every harmonic sum shares: E7 off keeps half a pitch even where it
        // is a local minimum; E7 on is exactly peak_off_half_pitch, which agrees with the
        // workbook set's closed form within 1e-12 rad (decision 29 A).
        let a = [1.0, 0.2, 0.0];
        assert_eq!(peak_angle(&a, Deviations::NONE), None);
        let on = peak_angle(&a, Deviations::only(DeviationId::E7));
        assert!(on.is_some());
        assert_eq!(on, peak_off_half_pitch(&a));
        let closed = closed_form_peak(a).expect("the closed form moves the peak too");
        assert!((on.unwrap() - closed).abs() <= 1e-12, "{on:?} vs {closed}");
        assert_eq!(peak_angle(&a, Deviations::ALL), on);
        assert_eq!(tau_at(2.0, 3, on.unwrap()), 2.0 * (3.0 * on.unwrap()).sin());
    }

    #[test]
    fn a_vanishing_fifth_harmonic_does_not_lose_the_peak() {
        // Final review, E7: with a5 -> 0 the quadratic's small root must tend to the linear
        // root -qc/qb (the a5 = 0 answer), not cancel to a wrong angle or to None.
        let x0 = peak_off_half_pitch(&[1.0, 0.2, 0.0]).expect("the peak moves");
        for a5 in [1e-20, -1e-20, 1e-16, -1e-16, 1e-14, 1.5e-13, -1.5e-13] {
            let x = peak_off_half_pitch(&[1.0, 0.2, a5]).unwrap_or_else(|| panic!("{a5}: None"));
            assert!((x - x0).abs() < 1e-9, "{a5}: {x} vs {x0}");
        }
    }

    #[test]
    fn peak_off_half_pitch_finds_the_brute_force_maximum() {
        // Property test over random amplitude triples (`random_triple`: the general case, a
        // vanishing fifth harmonic, and a5 = 0).
        let mut uniform = uniform_source(0x9E37_79B9_7F4A_7C15);
        let mut misses = Vec::new();
        for i in 0..3000 {
            let a = random_triple(&mut uniform, i);
            let got = match peak_off_half_pitch(&a) {
                Some(x) => {
                    assert!(
                        (0.0..=std::f64::consts::FRAC_PI_2).contains(&x),
                        "{a:?}: {x}"
                    );
                    torque_at(&a, x)
                }
                None => a[0] - a[1] + a[2],
            };
            let want = grid_maximum(&a, 1001);
            if got < want - 4e-12 * (a[0].abs() + a[1].abs() + a[2].abs()) {
                misses.push(format!("{a:?}: {got} < {want}"));
            }
        }
        assert!(
            misses.is_empty(),
            "{} misses, e.g. {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
    }

    /// The closed form E7 used for the workbook set 1, 3, 5 before the harmonic set became a
    /// parameter (kept as the reference decision 29 A compares with): the real roots in
    /// u = cos² x of 80 a5 u² + (12 a3 − 100 a5) u + (a1 − 9 a3 + 25 a5), taken
    /// cancellation-free, filtered as `peak_off_half_pitch` filters.
    fn closed_form_peak(a: [f64; 3]) -> Option<f64> {
        let [a1, a3, a5] = a;
        let torque = |x: f64| a1 * x.sin() + a3 * (3.0 * x).sin() + a5 * (5.0 * x).sin();
        let half_pitch = a1 - a3 + a5;
        let (qa, qb, qc) = (80.0 * a5, 12.0 * a3 - 100.0 * a5, a1 - 9.0 * a3 + 25.0 * a5);
        let roots: Vec<f64> = if qa != 0.0 {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                Vec::new()
            } else {
                let q = -0.5 * (qb + disc.sqrt().copysign(qb));
                if q == 0.0 {
                    vec![0.0]
                } else {
                    vec![q / qa, qc / q]
                }
            }
        } else if qb != 0.0 {
            vec![-qc / qb]
        } else {
            Vec::new()
        };
        roots
            .into_iter()
            .filter(|u| (0.0..=1.0).contains(u))
            .map(|u| u.sqrt().acos())
            .map(|x| (x, torque(x)))
            .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
            .max_by(|p, q| p.1.total_cmp(&q.1))
            .map(|(x, _)| x)
    }

    #[test]
    fn the_general_search_agrees_with_the_closed_form_on_the_workbook_set() {
        // Decision 29 A: one general search replaces the closed form for 1, 3, 5. On 3,000
        // random triples of the three bands both give None in the same cases and otherwise
        // the same angle within 1e-12 rad.
        let mut uniform = uniform_source(0x2545_F491_4F6C_DD1D);
        let mut disagreements = Vec::new();
        for i in 0..3000 {
            let a = random_triple(&mut uniform, i);
            let (general, closed) = (peak_off_half_pitch(&a), closed_form_peak(a));
            let agree = match (general, closed) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-12,
                (None, None) => true,
                _ => false,
            };
            if !agree {
                disagreements.push(format!("{a:?}: {general:?} vs {closed:?}"));
            }
        }
        assert!(
            disagreements.is_empty(),
            "{} disagreements, e.g. {:?}",
            disagreements.len(),
            &disagreements[..disagreements.len().min(5)]
        );
    }

    #[test]
    fn the_general_search_finds_the_brute_force_maximum_up_to_harmonic_11() {
        // Decision 29 A: 3,000 random spectra of 1 to 6 odd harmonics (up to 11). a1 in
        // [0.5, 1.5]; harmonic n in +-a1/n, so half a pitch is often a local minimum and the
        // curve can have several peaks; every fourth spectrum of two or more harmonics has a
        // vanishing top harmonic (|a/a1| <= 1.5e-13). The search is never below the brute
        // force by more than 4e-12 of Σ|a|.
        let mut uniform = uniform_source(0xD1B5_4A32_D192_ED03);
        let mut misses = Vec::new();
        for i in 0..3000 {
            let count = 1 + i % 6;
            let a1 = 0.5 + uniform();
            let mut a = vec![a1];
            for &n in &ODD_HARMONICS[1..count] {
                a.push((uniform() - 0.5) * 2.0 * a1 / f64::from(n));
            }
            if count > 1 && (i / 6) % 4 == 0 {
                let sign = if uniform() < 0.5 { -1.0 } else { 1.0 };
                a[count - 1] = sign * a1 * 10f64.powf(-20.0 + 7.0 * uniform()) * 1.5;
            }
            let got = match peak_off_half_pitch(&a) {
                Some(x) => {
                    assert!(
                        (0.0..=std::f64::consts::FRAC_PI_2).contains(&x),
                        "{a:?}: {x}"
                    );
                    torque_at(&a, x)
                }
                None => torque_at(&a, std::f64::consts::FRAC_PI_2),
            };
            let want = grid_maximum(&a, 2001);
            let scale: f64 = a.iter().map(|x| x.abs()).sum();
            if got < want - 4e-12 * scale {
                misses.push(format!("{a:?}: {got} < {want}"));
            }
        }
        assert!(
            misses.is_empty(),
            "{} misses, e.g. {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
    }

    #[test]
    fn roots_in_unit_interval_finds_every_root_of_a_quintic() {
        // cos(11x) / cos(x) = P_11(u) with u = cos² x has five roots in (0, 1), at
        // x = (2k + 1) π / 22 for k = 0 .. 4.
        let p11 = [-11.0, 220.0, -1232.0, 2816.0, -2816.0, 1024.0];
        let roots = roots_in_unit_interval(&p11);
        let mut want: Vec<f64> = (0..5)
            .map(|k| (f64::from(2 * k + 1) * PI / 22.0).cos().powi(2))
            .collect();
        want.sort_by(f64::total_cmp);
        assert_eq!(roots.len(), 5, "{roots:?}");
        for (got, want) in roots.iter().zip(&want) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn roots_in_unit_interval_finds_a_close_pair_a_scan_would_miss() {
        // Decision 29's failure mode of a sampled scan: two roots inside one cell of a
        // 1,024-cell scan, with the same sign at both cell ends. The roots of the derivative
        // split the interval first, so each piece holds one root.
        let q = [0.5002 * 0.5006, -(0.5002 + 0.5006), 1.0]; // (u - 0.5002)(u - 0.5006)
        let value = |u: f64| q[0] + q[1] * u + q[2] * u * u;
        let (cell_lo, cell_hi) = (512.0 / 1024.0, 513.0 / 1024.0);
        assert!(
            value(cell_lo) > 0.0 && value(cell_hi) > 0.0,
            "same sign at the cell ends"
        );
        let roots = roots_in_unit_interval(&q);
        assert_eq!(roots.len(), 2, "{roots:?}");
        assert!((roots[0] - 0.5002).abs() < 1e-12, "{}", roots[0]);
        assert!((roots[1] - 0.5006).abs() < 1e-12, "{}", roots[1]);
    }

    #[test]
    fn roots_in_unit_interval_handles_the_edges() {
        let none = Vec::<f64>::new();
        assert_eq!(roots_in_unit_interval(&[]), none);
        assert_eq!(roots_in_unit_interval(&[1.0]), none, "a constant");
        assert_eq!(
            roots_in_unit_interval(&[0.0, 0.0]),
            none,
            "zero: no isolated root"
        );
        assert_eq!(roots_in_unit_interval(&[-0.25, 1.0]), [0.25]);
        assert_eq!(roots_in_unit_interval(&[0.0, 1.0]), [0.0], "a root at 0");
        assert_eq!(roots_in_unit_interval(&[-1.0, 1.0]), [1.0], "a root at 1");
        assert_eq!(
            roots_in_unit_interval(&[2.0, -1.0]),
            none,
            "root at 2: outside"
        );
        assert_eq!(
            roots_in_unit_interval(&[1.0, 0.0, 1.0]),
            none,
            "no real root"
        );
        assert_eq!(
            roots_in_unit_interval(&[0.25, -1.0, 1.0]),
            [0.5],
            "double root"
        );
        assert_eq!(
            roots_in_unit_interval(&[-0.25, 1.0, 0.0, 0.0]),
            [0.25],
            "trailing zeros dropped"
        );
    }

    #[test]
    fn e7_circuit_sums_find_the_same_peak_as_the_pull_out() {
        // 6 poles on the default hub (audit row E7): the peak leaves half a pitch in both
        // circuits. Each circuit sum (C95 steel, C96 no iron) finds its own peak and must
        // agree with the pull-out of the circuit `backiron` selects, as at half a pitch.
        let e7 = Deviations::only(DeviationId::E7);
        for backiron in [1, 0] {
            let ci = CouplingInputs {
                npole: 6,
                backiron,
                ..CouplingInputs::default()
            };
            let (off, on) = (at(&ci), at_with(&ci, 0.05, e7));
            let own = if backiron == 1 {
                on.pullout_iron_Nm
            } else {
                on.pullout_noiron_Nm
            };
            assert!(
                close(own, on.pullout_Nm),
                "{backiron}: {own} vs {}",
                on.pullout_Nm
            );
            assert!(on.pullout_Nm > off.pullout_Nm, "{backiron}");
            assert!(on.pullout_iron_Nm > off.pullout_iron_Nm, "{backiron}");
            assert!(on.pullout_noiron_Nm > off.pullout_noiron_Nm, "{backiron}");
            assert_eq!(on.tau_Pa, 0.0 + on.tau1_Pa + on.tau3_Pa + on.tau5_Pa);
        }
        // Default design (10 poles, steel): half a pitch is the peak; E7 changes no bit.
        let ci = CouplingInputs::default();
        assert_eq!(at_with(&ci, 0.05, e7), at(&ci));
    }

    #[test]
    fn harmonic_count_maps_each_choice_and_nothing_else() {
        for (i, &(code, _)) in MAX_HARMONIC_CHOICES.iter().enumerate() {
            assert_eq!(code, i64::from(ODD_HARMONICS[i]));
            assert_eq!(harmonic_count(code), Some(i + 1), "{code}");
        }
        assert_eq!(harmonic_count(WORKBOOK_MAX_HARMONIC), Some(HARMONICS.len()));
        assert_eq!(
            CouplingInputs::default().max_harmonic,
            WORKBOOK_MAX_HARMONIC
        );
        for code in [0, 2, 4, 6, 12, 13, -1, i64::MIN, i64::MAX] {
            assert_eq!(harmonic_count(code), None, "{code}");
        }
    }

    #[test]
    fn the_harmonic_set_adds_or_drops_terms() {
        // Addendum A3: the model sums 1, 3, ... up to the chosen harmonic. The terms of 1, 3
        // and 5 (wave number, amplitudes, geometry factors) do not depend on the set; a left-out
        // harmonic's shear stress reads 0, so the total is always the sum of the six cells.
        let with = |max_harmonic| {
            at(&CouplingInputs {
                max_harmonic,
                ..CouplingInputs::default()
            })
        };
        let (one, workbook, eleven) = (with(1), with(5), with(11));
        assert_eq!(workbook, at(&CouplingInputs::default()));
        assert_eq!((one.tau3_Pa, one.tau5_Pa), (0.0, 0.0));
        assert_eq!(one.tau_Pa, one.tau1_Pa);
        assert_eq!(
            (workbook.tau7_Pa, workbook.tau9_Pa, workbook.tau11_Pa),
            (0.0, 0.0, 0.0)
        );
        for r in [&one, &eleven] {
            assert_eq!(
                (r.k3, r.b_i5, r.s5_iron, r.s1_free),
                (
                    workbook.k3,
                    workbook.b_i5,
                    workbook.s5_iron,
                    workbook.s1_free
                )
            );
        }
        assert!(eleven.tau7_Pa != 0.0 && eleven.tau9_Pa != 0.0 && eleven.tau11_Pa != 0.0);
        let cells = [
            eleven.tau1_Pa,
            eleven.tau3_Pa,
            eleven.tau5_Pa,
            eleven.tau7_Pa,
            eleven.tau9_Pa,
            eleven.tau11_Pa,
        ];
        assert_eq!(eleven.tau_Pa, cells.iter().fold(0.0, |acc, t| acc + t));
        assert!(close(
            eleven.pullout_Nm / workbook.pullout_Nm,
            eleven.tau_Pa / workbook.tau_Pa
        ));
        // The steel circuit sum (C95) is the pull-out's circuit here (both factors 0.95).
        for r in [&one, &workbook, &eleven] {
            assert!(close(r.pullout_iron_Nm, r.pullout_Nm), "{}", r.tau_Pa);
        }
    }

    #[test]
    fn an_invalid_harmonic_set_gives_nan_not_another_set() {
        // Decision D3: a code outside the choices, set on the struct, never selects another set.
        let r = at(&CouplingInputs {
            max_harmonic: 4,
            ..CouplingInputs::default()
        });
        for (what, x) in [
            ("tau", r.tau_Pa),
            ("tau1", r.tau1_Pa),
            ("tau5", r.tau5_Pa),
            ("tau11", r.tau11_Pa),
            ("pull-out", r.pullout_Nm),
            ("steel circuit", r.pullout_iron_Nm),
            ("free-space circuit", r.pullout_noiron_Nm),
        ] {
            assert!(x.is_nan(), "{what}: {x}");
        }
        assert!(r.k1.is_finite() && r.b_i1.is_finite() && r.s1_iron.is_finite());
    }

    #[test]
    fn e7_finds_the_peak_of_an_eleven_harmonic_curve() {
        // Decision 29 A through the model: 6 poles in free space (the layout of E7's probe),
        // every harmonic up to 11. E7's pull-out is the maximum of the model's own curve built
        // from the half-pitch terms (tau_n = a_n sin(n pi/2), so a_n = +-tau_n), and the
        // free-space circuit sum (C96) finds the same peak.
        let ci = CouplingInputs {
            npole: 6,
            backiron: 0,
            max_harmonic: 11,
            ..CouplingInputs::default()
        };
        let off = at(&ci);
        let on = at_with(&ci, 0.05, Deviations::only(DeviationId::E7));
        let half_pitch = [
            off.tau1_Pa,
            off.tau3_Pa,
            off.tau5_Pa,
            off.tau7_Pa,
            off.tau9_Pa,
            off.tau11_Pa,
        ];
        let a: Vec<f64> = half_pitch
            .iter()
            .enumerate()
            .map(|(i, &t)| if i % 2 == 0 { t } else { -t })
            .collect();
        let peak = grid_maximum(&a, 20001);
        assert!(on.tau_Pa > off.tau_Pa, "{} vs {}", on.tau_Pa, off.tau_Pa);
        assert!(close(on.tau_Pa, peak), "{} vs {peak}", on.tau_Pa);
        assert!(close(on.pullout_noiron_Nm, on.pullout_Nm));
    }

    #[test]
    fn calibration_factor_needs_every_prototype_condition() {
        let f = |backiron, npole, poles| {
            select_calibration_factor(backiron, npole, "B842SH", "B842SH", poles, 1.0658, 0.95)
        };
        assert_eq!(f(0, 10, 10.0), 1.0658);
        assert_eq!(f(1, 10, 10.0), 0.95, "steel circuit");
        assert_eq!(f(0, 12, 10.0), 0.95, "other pole count");
        assert_eq!(
            f(0, 12, 12.0),
            1.0658,
            "a 24-magnet prototype matches 12 poles"
        );
    }

    #[test]
    fn the_end_effect_check_is_strict_at_zero() {
        // Audit M9, the user's decision: a flag, no physics change. Architecture section 7
        // step 8: the comparison `f_end > 0` at exact equality, then either side of it.
        assert_eq!(end_effect_check(0.0), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(-0.0), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(f64::MIN_POSITIVE), "OK");
        assert_eq!(end_effect_check(-1e-300), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(f64::NAN), END_EFFECT_OUT_OF_RANGE);
        assert!(end_effect_in_range(1.0) && !end_effect_in_range(f64::NEG_INFINITY));
    }

    #[test]
    fn short_magnets_flag_the_end_effect_model() {
        // Audit M9: f_end = 1 - c_end * pole pitch / L turns negative below L = c_end * pole
        // pitch (1.32 mm at the defaults), and the pull-out with it. 2 mm manual blocks with
        // c_end = 0.5, both inside their sliders, reach it.
        let r = at(&CouplingInputs::default());
        assert_eq!((r.f_end > 0.0, r.end_effect_check.as_str()), (true, "OK"));
        let mut ci = CouplingInputs {
            c_end: 0.5,
            ..CouplingInputs::default()
        };
        ci.magnets.part_inner = String::new();
        ci.magnets.part_outer = String::new();
        ci.magnets.manual_inner_length_mm = 2.0;
        ci.magnets.manual_outer_length_mm = 2.0;
        let r = at(&ci);
        assert!(
            r.f_end < 0.0 && r.pullout_Nm < 0.0,
            "{} {}",
            r.f_end,
            r.pullout_Nm
        );
        assert_eq!(r.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    }

    #[test]
    fn temperature_check_is_inclusive_at_the_rating() {
        let mut ci = CouplingInputs {
            op_temp_C: 150.0,
            ..CouplingInputs::default()
        }; // B842SH rating
        assert_eq!(at(&ci).inner_temp_check, "OK");
        ci.op_temp_C = 150.5;
        assert_eq!(at(&ci).inner_temp_check, "OVER the magnet rating");
    }

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Each right-hand side is a value its left-hand side
        // does not read, so one run supplies the left side and a second run puts the
        // comparison at exact equality (no tolerance involved).
        let base = at(&CouplingInputs::default());

        // Verdict: `T_pull < floor_` is strict, so a pull-out equal to the floor is not "Below".
        let ci = CouplingInputs::default();
        let r = compute(
            &ci,
            1.4,
            0.05,
            0.05,
            1.8,
            -0.0012,
            1.5,
            0.95,
            0.95,
            2000.0,
            base.pullout_Nm,
            Deviations::NONE,
        );
        assert_eq!(r.required_floor_Nm, r.pullout_Nm);
        assert_eq!(r.verdict, "Nominal only: hot test");

        // Thickness check `wall >= t_bi` (one closure serves the cup ring and the hub).
        let r = compute(
            &ci,
            1.4,
            0.05,
            0.05,
            base.backiron_needed_mm,
            -0.0012,
            1.5,
            0.95,
            0.95,
            2000.0,
            2.5,
            Deviations::NONE,
        );
        assert_eq!(r.backiron_needed_mm, r.cup_wall_corner_mm);
        assert_eq!(r.cup_ring_check, "Thickness OK");

        // Flat checks `flat >= width`. Manual magnets make the widths inputs; the inner width
        // moves the outer apothem, so the outer width is copied after the inner one is set.
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.part_outer = String::new();
        ci.magnets.manual_inner_width_mm = base.inner_flat_width_mm;
        ci.magnets.manual_outer_width_mm = at(&ci).outer_flat_width_mm;
        let r = at(&ci);
        assert_eq!(
            (r.inner_width_mm, r.outer_width_mm),
            (r.inner_flat_width_mm, r.outer_flat_width_mm)
        );
        assert_eq!(r.inner_flat_check, "OK, 0.00 mm slack");
        assert_eq!(r.outer_flat_check, "OK, blocks 0.00 mm apart at the faces");
    }

    #[test]
    fn a_grade_ring_weighs_at_its_grade_density() {
        // Decision A2-7: the magnets' mass prices each ring at its own density; rings of one
        // density keep the workbook's single product, bit for bit.
        let md = super::super::metal_design::MetalDesignInputs::default();
        let mass = |ci: &CouplingInputs| {
            let r = at(ci);
            let m = mass_estimate(
                ci,
                &r,
                md.bond_inner_mm,
                md.bond_outer_mm,
                md.cup_depth_mm,
                md.web_mm,
                md.hub_length_mm,
                md.boss_length_mm,
                md.boss_od_mm,
                md.steel_density_g_mm3,
                md.al_density_g_mm3,
                0.0,
                0.0,
                0.0,
                0.0,
                Deviations::NONE,
            );
            (m.magnets_g, r)
        };
        let (workbook, r) = mass(&CouplingInputs::default());
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        let volume_o = r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm;
        assert_eq!(workbook, 10.0 * (volume_i + volume_o) * NDFEB_DENSITY_G_MM3);
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        let (ferrite, r) = mass(&ci);
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        assert_eq!(
            ferrite,
            10.0 * (volume_i * 0.005 + volume_o * NDFEB_DENSITY_G_MM3)
        );
    }

    #[test]
    fn e9_cup_and_boss_follow_the_hub_density() {
        // The Metal design defaults api::compute passes; retainers, hardware, cap and
        // end plates are left at 0 because only the densities matter here.
        let md = super::super::metal_design::MetalDesignInputs::default();
        let mass = |backiron: i64, dev: Deviations| {
            let ci = CouplingInputs {
                backiron,
                ..CouplingInputs::default()
            };
            let r = at(&ci);
            mass_estimate(
                &ci,
                &r,
                md.bond_inner_mm,
                md.bond_outer_mm,
                md.cup_depth_mm,
                md.web_mm,
                md.hub_length_mm,
                md.boss_length_mm,
                md.boss_od_mm,
                md.steel_density_g_mm3,
                md.al_density_g_mm3,
                0.0,
                0.0,
                0.0,
                0.0,
                dev,
            )
        };
        let e9 = Deviations::only(DeviationId::E9);
        // Steel circuit: the workbook's masses, bit for bit.
        assert_eq!(mass(1, e9), mass(1, Deviations::NONE));
        // No back iron (and, like the hub, any code but 1): aluminium cup and boss.
        let ratio = md.al_density_g_mm3 / md.steel_density_g_mm3;
        for backiron in [0, 2] {
            let (off, on) = (mass(backiron, Deviations::NONE), mass(backiron, e9));
            assert!(close(on.cup_g, off.cup_g * ratio), "{backiron}");
            assert!(close(on.boss_g, off.boss_g * ratio), "{backiron}");
            assert_eq!(
                (on.magnets_g, on.hub_g),
                (off.magnets_g, off.hub_g),
                "{backiron}"
            );
            let total = on.magnets_g + on.cup_g + on.hub_g + on.boss_g;
            assert!(close(on.total_g, total), "{backiron}");
        }
    }

    #[test]
    fn e10_gap_flux_density_sums_each_magnets_mmf() {
        // Audit report row E10. Manual magnets make Br and thickness inputs; the
        // outer thickness also moves the geometry, so each case reads g, t and the
        // operating Br back from the results.
        let e10 = Deviations::only(DeviationId::E10);
        let case = |br_i: f64, t_i: f64, br_o: f64, t_o: f64| {
            let mut ci = CouplingInputs::default();
            ci.magnets.part_inner = String::new();
            ci.magnets.part_outer = String::new();
            ci.magnets.manual_inner_br_T = br_i;
            ci.magnets.manual_inner_thickness_mm = t_i;
            ci.magnets.manual_outer_br_T = br_o;
            ci.magnets.manual_outer_thickness_mm = t_o;
            (at(&ci), at_with(&ci, 0.05, e10))
        };
        for (br_i, t_i, br_o, t_o) in [
            (1.44, 3.17, 1.32, 1.59), // higher Br on the thicker ring: workbook understates
            (1.44, 1.59, 1.32, 3.17), // higher Br on the thinner ring: workbook overstates
            (1.30, 3.17, 1.30, 1.59), // equal Br
            (1.44, 3.17, 1.32, 3.17), // equal thickness
            (1.30, 3.17, 1.30, 3.17), // identical rings
        ] {
            let (off, on) = case(br_i, t_i, br_o, t_o);
            let label = format!("{br_i} T x {t_i} mm inside, {br_o} T x {t_o} mm outside");
            // Everything upstream of C103 is untouched.
            assert_eq!(
                (on.br_inner_T_op, on.br_outer_T_op, on.face_gap_mm),
                (off.br_inner_T_op, off.br_outer_T_op, off.face_gap_mm),
                "{label}"
            );
            assert_eq!(on.pullout_Nm, off.pullout_Nm, "{label}");
            let (bi, bo, g) = (on.br_inner_T_op, on.br_outer_T_op, on.face_gap_mm);
            let d = t_i + t_o + g;
            assert!(
                close(on.gap_flux_density_T, (bi * t_i + bo * t_o) / d),
                "{label}"
            );
            // The report's error term: workbook - corrected = -(Br_i - Br_o)(t_i - t_o) / (2 (t_i + t_o + g)).
            let error = off.gap_flux_density_T - on.gap_flux_density_T;
            let want = -(bi - bo) * (t_i - t_o) / (2.0 * d);
            assert!((error - want).abs() <= 1e-12, "{label}: {error} vs {want}");
            // C104 follows C103 with the same pole pitch and saturation flux density.
            assert_eq!(on.pole_pitch_mm, off.pole_pitch_mm, "{label}");
            assert!(
                close(
                    on.backiron_needed_mm / off.backiron_needed_mm,
                    on.gap_flux_density_T / off.gap_flux_density_T
                ),
                "{label}"
            );
        }
        // Identical rings: the same double, not just the parity rule.
        let (off, on) = case(1.30, 3.17, 1.30, 3.17);
        assert_eq!(on.gap_flux_density_T, off.gap_flux_density_T);
        assert_eq!(on, off);
        // Sign: the workbook understates B when the higher-Br ring is also the thicker one.
        let (off, on) = case(1.44, 3.17, 1.32, 1.59);
        assert!(on.gap_flux_density_T > off.gap_flux_density_T);
        let (off, on) = case(1.44, 1.59, 1.32, 3.17);
        assert!(on.gap_flux_density_T < off.gap_flux_density_T);
    }

    #[test]
    fn the_explorer_terms_are_the_engine_s_own_values() {
        // Plan A-3: the Rust-only terms of the equation explorer are reads of what the engine
        // computes. Every harmonic up to 11 has its parts whether or not it is summed; each
        // amplitude is B_i B_o S / (2 mu0) in the circuit in effect (steel here); each summed
        // shear stress is its amplitude at the pull-out angle; the angles are pi/2 or inside
        // [0, pi/2]; the prototype's terms obey the same law at its own angle.
        use crate::engine::api::{DesignInputs, compute_all};
        use crate::engine::meta::{InputSet, Value};
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        for code in [5, 11] {
            let mut inputs = DesignInputs::default();
            inputs
                .set("coupling.max_harmonic", Value::Int(code))
                .unwrap();
            let res = compute_all(&inputs);
            let r = &res.model;
            let parts = [
                (1, r.b_i1, r.b_o1, r.s1_iron, r.amp1_Pa, r.tau1_Pa),
                (3, r.b_i3, r.b_o3, r.s3_iron, r.amp3_Pa, r.tau3_Pa),
                (5, r.b_i5, r.b_o5, r.s5_iron, r.amp5_Pa, r.tau5_Pa),
                (7, r.b_i7, r.b_o7, r.s7_iron, r.amp7_Pa, r.tau7_Pa),
                (9, r.b_i9, r.b_o9, r.s9_iron, r.amp9_Pa, r.tau9_Pa),
                (11, r.b_i11, r.b_o11, r.s11_iron, r.amp11_Pa, r.tau11_Pa),
            ];
            for (n, bi, bo, s, amp, tau) in parts {
                assert!(
                    bi != 0.0 && s != 0.0,
                    "harmonic {n}'s parts exist (code {code})"
                );
                assert!(
                    close(amp, bi * bo * s / (2.0 * MU0)),
                    "harmonic {n}: amplitude (code {code})"
                );
                let want = if i64::from(n) <= code {
                    amp * (f64::from(n) * r.pullout_angle_rad).sin()
                } else {
                    0.0
                };
                assert!(close(tau, want), "harmonic {n}: shear stress (code {code})");
            }
            for angle in [
                r.pullout_angle_rad,
                r.iron_circuit_angle_rad,
                r.free_circuit_angle_rad,
            ] {
                assert!((0.0..=HALF_PITCH_RAD).contains(&angle), "{angle}");
            }
            let c = &res.calibration;
            for (n, amp, tau) in [
                (1, c.amp1_Pa, c.tau1_Pa),
                (3, c.amp3_Pa, c.tau3_Pa),
                (11, c.amp11_Pa, c.tau11_Pa),
            ] {
                let want = if i64::from(n) <= code {
                    amp * (f64::from(n) * c.pullout_angle_rad).sin()
                } else {
                    0.0
                };
                assert!(close(tau, want), "prototype harmonic {n} (code {code})");
            }
        }
    }
}
