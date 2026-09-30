//! Calculator sheet ('Calculator'): the core coupling model.
//!
//! Port of `reference/magcoupling-py/magcoupling/model.py`: the magnet and
//! coupling inputs, the 73 result cells ([`ModelResults`]), and the helpers
//! (`resolve_magnets`, `select_calibration_factor`, `geometry_factor`,
//! `harmonic_amplitude`, `shear_stress`, shared with the sweeps).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`.
//!
//! The harmonic set is the workbook's fixed 1, 3, 5 ([`HARMONICS`]); selecting
//! more harmonics belongs to the Addendum A engine plan.
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied are E3 (the N42SH
//! remanence, via `resolve_magnets`, and the manual Br defaults C17 and C27),
//! E6 (Calculator!C63, the ring wall at the flats without the outer
//! bondline), E7 (pull-out at the maximum over angle,
//! [`peak_off_half_pitch`] and `at_pull_out`, shared with the sweeps) and E8
//! (arc mode: Calculator!C9 from the corner radius C55, the round pocket in
//! C111); planned are E9 (masses), E10 (Calculator!C103).

use std::f64::consts::PI;

use super::compat::{fmt_fixed, py_max, py_min};
use super::constants::{MU0, NDFEB_DENSITY_G_MM3};
use super::deviations::{DeviationId, Deviations};
use super::library::{self, lookup};
use super::meta::{NumOrText, inputs, out, param, results};

/// The odd space harmonics the model sums (the workbook's set).
pub const HARMONICS: [u32; 3] = [1, 3, 5];

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";

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
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
                "", "Calculator!C90"),
            torque_2d_Nm: f64 => out("N·m", "2D pull-out torque (infinite length)",
                "", "Calculator!C91"),
            f_end: f64 => out("-", "End-effect factor", "", "Calculator!C92"),
            pullout_Nm: f64 => out("N·m", "Pull-out torque at operating temperature",
                "Analytical estimate, not a guaranteed minimum.", "Calculator!C93"),
            pullout_20C_Nm: f64 => out("N·m", "Pull-out torque at 20 °C", "", "Calculator!C94"),
            pullout_iron_Nm: f64 => out("N·m", "Same layout with steel back iron (factor 0.95)",
                "", "Calculator!C95"),
            pullout_noiron_Nm: f64 => out("N·m", "Raw no-back-iron prediction, same layout",
                "", "Calculator!C96"),
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

/// One magnet ring as the Calculator uses it (Python `ResolvedMagnet`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct ResolvedMagnet {
    pub length_mm: f64,
    pub width_mm: f64,
    pub thickness_mm: f64,
    pub br_T: f64,
    /// Library rating, or `NOT_IN_LIBRARY` for manual magnets.
    pub tmax_C: NumOrText,
}

/// Library values when the part is found, else the manual values (workbook IFERROR/INDEX/MATCH).
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence).
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve = |part: &str, length_mm: f64, width_mm: f64, thickness_mm: f64, br_T: f64| {
        match lookup(part) {
            Some(spec) => ResolvedMagnet {
                length_mm: spec.length_mm,
                width_mm: spec.width_mm,
                thickness_mm: spec.thickness_mm,
                br_T: library::br_T(spec, dev),
                tmax_C: NumOrText::Num(spec.tmax_C),
            },
            None => ResolvedMagnet {
                length_mm,
                width_mm,
                thickness_mm,
                br_T,
                tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
            },
        }
    };
    (
        resolve(
            &m.part_inner,
            m.manual_inner_length_mm,
            m.manual_inner_width_mm,
            m.manual_inner_thickness_mm,
            m.manual_inner_br_T,
        ),
        resolve(
            &m.part_outer,
            m.manual_outer_length_mm,
            m.manual_outer_width_mm,
            m.manual_outer_thickness_mm,
            m.manual_outer_br_T,
        ),
    )
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

/// Per-harmonic pull-out shear stress [Pa] and its parts, for HARMONICS.
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
) -> [Harmonic; 3] {
    HARMONICS.map(|n| {
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
        hn.tau = bi * bo / (2.0 * mu0) * hn.s(backiron) * (nf * PI / 2.0).sin();
        hn
    })
}

/// E7: the electrical angle of the true pull-out, when half a pole pitch is not it.
///
/// Torque against electrical angle x is T(x) = a1 sin x + a3 sin 3x + a5 sin 5x,
/// `a` the per-harmonic amplitudes. Half a pitch (x = π/2) is always a stationary
/// point, and the workbook evaluates every harmonic there. The other stationary
/// points solve dT/dx = 0; with c = cos x,
/// dT/dx = c·[(a1 − 9 a3 + 25 a5) + (12 a3 − 100 a5) c² + 80 a5 c⁴],
/// a quadratic in u = c². T is symmetric about π/2 (odd harmonics), so
/// x in [0, π/2] suffices. Returns `None` when half a pitch is the maximum, so
/// callers keep the workbook expression and default outputs stay bit-identical;
/// otherwise the angle whose torque exceeds the half-pitch torque by more than
/// 1e-12 relative. Valid for the harmonic set [1, 3, 5] only (`HARMONICS`); the
/// Addendum A plan generalizes it with the harmonic parameter (decision D7).
pub fn peak_off_half_pitch(a: [f64; 3]) -> Option<f64> {
    let [a1, a3, a5] = a;
    let torque = |x: f64| a1 * x.sin() + a3 * (3.0 * x).sin() + a5 * (5.0 * x).sin();
    let half_pitch = a1 - a3 + a5; // sin(π/2) = 1, sin(3π/2) = −1, sin(5π/2) = 1
    let (qa, qb, qc) = (80.0 * a5, 12.0 * a3 - 100.0 * a5, a1 - 9.0 * a3 + 25.0 * a5);
    let roots: Vec<f64> = if qa != 0.0 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            Vec::new()
        } else {
            vec![
                (-qb + disc.sqrt()) / (2.0 * qa),
                (-qb - disc.sqrt()) / (2.0 * qa),
            ]
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

/// E7 for one circuit: every harmonic's `tau` at the true pull-out angle when
/// half a pitch is not the maximum ([`peak_off_half_pitch`] on the amplitudes
/// B_in,n·B_on,n/(2μ0)·S_n of the circuit `backiron` selects). Returns `h`
/// unchanged, bit for bit, when E7 is off or half a pitch is the maximum.
/// Shared by [`compute`] and the sweep rows.
pub(crate) fn at_pull_out(
    h: [Harmonic; 3],
    backiron: i64,
    mu0: f64,
    dev: Deviations,
) -> [Harmonic; 3] {
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * mu0) * x.s(backiron);
    let mut h_pull = h;
    if dev.is_on(DeviationId::E7)
        && let Some(x) = peak_off_half_pitch(h.map(|hn| amplitude(&hn)))
    {
        for hn in h_pull.iter_mut() {
            hn.tau = amplitude(hn) * (f64::from(hn.n) * x).sin();
        }
    }
    h_pull
}

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
        (r_face_i.powi(2) + (mi.width_mm / 2.0).powi(2)).sqrt()
    } else {
        r_face_i
    };
    // Corner gap from the flat-face gap. Workbook: always the flat-block corner
    // geometry. E8: the inner corner radius C55, which is the face radius for arcs.
    let corner_gap = if dev.is_on(DeviationId::E8) {
        face_gap_mm - (r_corner_i - r_face_i)
    } else {
        face_gap_mm
            - (((a_i + mi.thickness_mm).powi(2) + (mi.width_mm / 2.0).powi(2)).sqrt()
                - (a_i + mi.thickness_mm))
    };
    let hub_wall = a_i - bond_inner_mm - ci.bore_mm / 2.0;

    let flat_i = 2.0 * a_i * (PI / N).tan();
    let chk_i = if ci.faceted == 1 {
        if flat_i >= mi.width_mm {
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
        if flat_o >= mo.width_mm {
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
    let al_i = py_min(
        1.0,
        mi.width_mm / (2.0 * PI * (a_i + mi.thickness_mm / 2.0) / N),
    );
    let al_o = py_min(
        1.0,
        mo.width_mm / (2.0 * PI * (A_o + mo.thickness_mm / 2.0) / N),
    );

    let bri = mi.br_T * (1.0 + alpha_br * (ci.op_temp_C - 20.0));
    let bro = mo.br_T * (1.0 + alpha_br * (ci.op_temp_C - 20.0));
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
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `h` (the half-pitch terms) stays for the per-circuit sums below.
    let h_pull = at_pull_out(h, ci.backiron, ci.mu0, dev);
    let tau = h_pull.iter().fold(0.0, |acc, x| acc + x.tau); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
    let T2D = tau * AL;
    let f_end = 1.0 - ci.c_end * tau_p / L;
    let T_pull = T2D * f_end * f_cal;
    let T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro);
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients = h.map(|x| x.bi * x.bo * s(&x));
        let peak = if dev.is_on(DeviationId::E7) {
            peak_off_half_pitch(coefficients)
        } else {
            None
        };
        match peak {
            Some(x) => h
                .iter()
                .zip(coefficients)
                .fold(0.0, |acc, (hn, c)| acc + c * (f64::from(hn.n) * x).sin()),
            None => h.iter().fold(0.0, |acc, x| {
                acc + x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()
            }),
        }
    };
    let T_iron = circuit(|x| x.s_iron) / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
    let T_noiron = circuit(|x| x.s_free) / (2.0 * ci.mu0) * AL * f_end * f_cal;

    let floor_ = py_max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm);
    let B_gap = (bri + bro) / 2.0 * (mi.thickness_mm + mo.thickness_mm)
        / (mi.thickness_mm + mo.thickness_mm + g_m);
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
    let [h1, h3, h5] = h_pull;

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
        tau1_Pa: h1.tau,
        k3: h3.k,
        b_i3: h3.bi,
        b_o3: h3.bo,
        s3_iron: h3.s_iron,
        s3_free: h3.s_free,
        tau3_Pa: h3.tau,
        k5: h5.k,
        b_i5: h5.bi,
        b_o5: h5.bo,
        s5_iron: h5.s_iron,
        s5_free: h5.s_free,
        tau5_Pa: h5.tau,
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
        f_end,
        pullout_Nm: T_pull,
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
        pullout_noiron_Nm: T_noiron,
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
    }
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
    let m_mag = N
        * (r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm
            + r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm)
        * NDFEB_DENSITY_G_MM3;
    let pocket = r.outer_back_apothem_mm + bond_outer_mm;
    // E8: arcs sit in a round pocket, not a polygon.
    let cavity = if dev.is_on(DeviationId::E8) && ci.faceted != 1 {
        PI * pocket.powi(2)
    } else {
        N * pocket.powi(2) * (PI / N).tan()
    };
    let m_ring = ((PI * (r.cup_od_mm / 2.0).powi(2) - cavity) * cup_depth_mm
        + PI * ((r.cup_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2)) * web_mm)
        * steel_density_g_mm3;
    let hub_area = if ci.faceted == 1 {
        N * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2) * (PI / N).tan()
    } else {
        PI * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2)
    };
    let m_hub = (hub_area - PI * (ci.bore_mm / 2.0).powi(2))
        * hub_length_mm
        * (if ci.backiron == 1 {
            steel_density_g_mm3
        } else {
            al_density_g_mm3
        });
    let m_boss = PI
        * ((boss_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2))
        * boss_length_mm
        * steel_density_g_mm3;
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
        assert_eq!(peak_off_half_pitch([1.0, -0.011, 0.001]), None); // default design: A3/A1 = -0.011
        assert_eq!(peak_off_half_pitch([1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn a_large_third_harmonic_moves_the_peak_and_raises_it() {
        let a = [1.0, 0.2, 0.0]; // A1 < 9 A3: half a pitch is a local minimum
        let x = peak_off_half_pitch(a).expect("the peak moves");
        let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
        assert!(x > 0.0 && x < std::f64::consts::FRAC_PI_2);
        assert!(t(x) > t(std::f64::consts::FRAC_PI_2));
        // stationary: dT/dx = 0 at the returned angle
        let slope = a[0] * x.cos() + 3.0 * a[1] * (3.0 * x).cos() + 5.0 * a[2] * (5.0 * x).cos();
        assert!(slope.abs() < 1e-9, "{slope}");
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
}
