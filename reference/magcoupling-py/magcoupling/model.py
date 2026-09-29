"""Core coupling model ('Calculator' sheet).

Coaxial synchronous permanent-magnet coupling: an inner ring of flat blocks on
a polygonal hub and an outer ring of flat blocks in a polygonal cup, both with
alternating radial magnetization.

Approach (the workbook's 2D slab model):
  1. Geometry: polygon apothems from the block size, the flat-face gap and the
     bondlines; corner radius of the inner blocks sets the minimum clearance.
  2. Field: square-wave magnetization of each ring expanded in odd space
     harmonics n = 1, 3, 5 with fill factor (block width / pole pitch).
  3. Shear stress at pull-out: tau_n = B_in·B_on / (2 µ0) · S_n · sin(nπ/2), with
     S_n = sinh(k t_i)·sinh(k t_o) / sinh(k (t_i + t_o + g)) for a steel-backed
     circuit, or (1 − e^-k t_i)(1 − e^-k t_o)·e^-k g / 2 for free-space rings.
  4. Torque = tau · 2π R_g² L, times an end-effect factor (1 − c_end · pole pitch / L)
     and a calibration factor.
  5. Temperature: Br scales linearly (alpha_Br); torque scales with Br².

This is an analytical estimate, not a guaranteed minimum: finite permeability,
saturation, the solid cup web and eddy losses are not solved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ._fields import out, param
from .constants import MU0, NDFEB_DENSITY_G_MM3
from .library import lookup

HARMONICS = (1, 3, 5)


# --------------------------------------------------------------------------- inputs
@dataclass
class MagnetInputs:
    part_inner: str = param("B842SH", "-", "Inner magnet part", "Looked up in the library by exact text; blank = manual.", "Calculator!C11")
    part_outer: str = param("B842SH", "-", "Outer magnet part", "", "Calculator!C12")
    manual_inner_length_mm: float = param(12.7, "mm", "Manual inner length (axial)", "Used only if the part is not in the library.", "Calculator!C14")
    manual_inner_width_mm: float = param(6.35, "mm", "Manual inner width (tangential)", "", "Calculator!C15")
    manual_inner_thickness_mm: float = param(3.17, "mm", "Manual inner thickness (radial)", "", "Calculator!C16")
    manual_inner_br_T: float = param(1.29, "T", "Manual inner Br at 20 °C", "", "Calculator!C17")
    manual_outer_length_mm: float = param(12.7, "mm", "Manual outer length (axial)", "", "Calculator!C24")
    manual_outer_width_mm: float = param(6.35, "mm", "Manual outer width (tangential)", "", "Calculator!C25")
    manual_outer_thickness_mm: float = param(3.17, "mm", "Manual outer thickness (radial)", "", "Calculator!C26")
    manual_outer_br_T: float = param(1.29, "T", "Manual outer Br at 20 °C", "", "Calculator!C27")


@dataclass
class CouplingInputs:
    npole: int = param(10, "-", "Number of poles per ring", "Even; one block per pole.", "Calculator!C5")
    backiron: int = param(1, "-", "Back iron", "1 = steel circuit, 0 = no intentional back iron.", "Calculator!C6",
                          {1: "steel circuit", 0: "no back iron"})
    faceted: int = param(1, "-", "Geometry type", "1 = flat blocks on polygons, 0 = true arcs.", "Calculator!C7",
                         {1: "flat blocks", 0: "arcs"})
    inner_back_apothem_mm: float = param(10.15, "mm", "Inner magnet back apothem, including bondline",
                                         "Machined hub apothem plus the inner bondline.", "Calculator!C8")
    op_temp_C: float = param(50, "°C", "Operating magnet temperature", "Br falls with temperature; torque ~ Br².", "Calculator!C10")
    bore_mm: float = param(10, "mm", "Keyed bore diameter", "", "Calculator!C39")
    keyway_depth_mm: float = param(1.7, "mm", "Keyway depth in the hub", "", "Calculator!C40")
    c_end: float = param(0.15, "-", "End-effect coefficient", "f_end = 1 − c_end · pole pitch / L.", "Calculator!C41")
    mu0: float = param(MU0, "T·m/A", "Vacuum permeability", "", "Calculator!C43")
    gear_ratio: float = param(5, "-", "Gearbox ratio", "", "Calculator!C45")
    gear_efficiency: float = param(0.95, "-", "Gearbox efficiency", "", "Calculator!C46")
    gearbox_input_rating_Nm: float = param(0.6, "N·m", "Gearbox input torque rating", "GAM 5:1 peak input rating.", "Calculator!C47")
    drive_torque_Nm: float = param(0.7, "N·m", "Torque the coupling must carry for driving (at the wheel)", "", "Calculator!C48")
    drive_safety_factor: float = param(1.3, "-", "Safety factor wanted on the driving torque", "", "Calculator!C49")
    magnets: MagnetInputs = None

    def __post_init__(self):
        self.magnets = self.magnets or MagnetInputs()


# --------------------------------------------------------------------------- results
@dataclass
class ModelResults:
    # magnets used
    inner_length_mm: float = out("mm", "Inner length (axial) used", cell="Calculator!C18")
    inner_width_mm: float = out("mm", "Inner width (tangential) used", cell="Calculator!C19")
    inner_thickness_mm: float = out("mm", "Inner thickness (radial) used", cell="Calculator!C20")
    inner_br_T: float = out("T", "Inner Br at 20 °C used", cell="Calculator!C21")
    inner_tmax_C: object = out("°C", "Inner max operating temperature (library)", cell="Calculator!C22")
    outer_length_mm: float = out("mm", "Outer length (axial) used", cell="Calculator!C28")
    outer_width_mm: float = out("mm", "Outer width (tangential) used", cell="Calculator!C29")
    outer_thickness_mm: float = out("mm", "Outer thickness (radial) used", cell="Calculator!C30")
    outer_br_T: float = out("T", "Outer Br at 20 °C used", cell="Calculator!C31")
    outer_tmax_C: object = out("°C", "Outer max operating temperature (library)", cell="Calculator!C32")
    active_length_mm: float = out("mm", "Active (overlapping) length", cell="Calculator!C33")
    # linked constants
    corner_gap_mm: float = out("mm", "Corner gap: inner block corner to outer block face", "Derived from the flat-face gap.", "Calculator!C9")
    alpha_br_per_C: float = out("1/°C", "Br temperature coefficient", cell="Calculator!C35")
    bsat_T: float = out("T", "Back-iron saturation flux density", cell="Calculator!C36")
    cup_wall_corner_mm: float = out("mm", "Cup wall thickness at the pocket corners", cell="Calculator!C37")
    hub_wall_mm: float = out("mm", "Hub wall under flats", cell="Calculator!C38")
    f_cal: float = out("-", "Calibration factor", "0.95 for the steel-backed candidate.", "Calculator!C42")
    # geometry
    inner_flat_width_mm: float = out("mm", "Inner flat width available", cell="Calculator!C51")
    inner_flat_check: str = out("", "Inner flat check", cell="Calculator!C52")
    hub_wall_past_key_mm: float = out("mm", "Hub wall past the keyway", "Keep ≥ ~2.5 mm.", "Calculator!C53")
    inner_face_radius_mm: float = out("mm", "Inner block face radius", cell="Calculator!C54")
    inner_corner_radius_mm: float = out("mm", "Inner block corner radius", cell="Calculator!C55")
    outer_face_apothem_mm: float = out("mm", "Outer block face apothem", cell="Calculator!C56")
    face_gap_mm: float = out("mm", "Gap at the flat centres (effective gap)", cell="Calculator!C57")
    outer_flat_width_mm: float = out("mm", "Outer flat width at the faces", cell="Calculator!C58")
    outer_flat_check: str = out("", "Outer flat check", cell="Calculator!C59")
    outer_back_apothem_mm: float = out("mm", "Outer block back apothem", cell="Calculator!C60")
    pocket_corner_radius_mm: float = out("mm", "Pocket corner radius", cell="Calculator!C61")
    cup_od_mm: float = out("mm", "Cup outer diameter", cell="Calculator!C62")
    cup_wall_flat_mm: float = out("mm", "Ring wall at the flats", cell="Calculator!C63")
    gap_radius_mm: float = out("mm", "Gap mean radius", cell="Calculator!C64")
    pole_pitch_mm: float = out("mm", "Pole pitch at the gap radius", cell="Calculator!C65")
    fill_inner: float = out("-", "Inner fill factor", cell="Calculator!C66")
    fill_outer: float = out("-", "Outer fill factor", cell="Calculator!C67")
    # field and torque
    br_inner_T_op: float = out("T", "Br inner at operating temperature", cell="Calculator!C69")
    br_outer_T_op: float = out("T", "Br outer at operating temperature", cell="Calculator!C70")
    k1: float = out("1/m", "Harmonic 1 wave number", cell="Calculator!C71")
    b_i1: float = out("T", "Harmonic 1 inner amplitude", cell="Calculator!C72")
    b_o1: float = out("T", "Harmonic 1 outer amplitude", cell="Calculator!C73")
    s1_iron: float = out("-", "Harmonic 1 geometry factor with back iron", cell="Calculator!C74")
    s1_free: float = out("-", "Harmonic 1 geometry factor without back iron", cell="Calculator!C75")
    tau1_Pa: float = out("Pa", "Harmonic 1 shear stress", cell="Calculator!C76")
    k3: float = out("1/m", "Harmonic 3 wave number", cell="Calculator!C77")
    b_i3: float = out("T", "Harmonic 3 inner amplitude", cell="Calculator!C78")
    b_o3: float = out("T", "Harmonic 3 outer amplitude", cell="Calculator!C79")
    s3_iron: float = out("-", "Harmonic 3 geometry factor with back iron", cell="Calculator!C80")
    s3_free: float = out("-", "Harmonic 3 geometry factor without back iron", cell="Calculator!C81")
    tau3_Pa: float = out("Pa", "Harmonic 3 shear stress", cell="Calculator!C82")
    k5: float = out("1/m", "Harmonic 5 wave number", cell="Calculator!C83")
    b_i5: float = out("T", "Harmonic 5 inner amplitude", cell="Calculator!C84")
    b_o5: float = out("T", "Harmonic 5 outer amplitude", cell="Calculator!C85")
    s5_iron: float = out("-", "Harmonic 5 geometry factor with back iron", cell="Calculator!C86")
    s5_free: float = out("-", "Harmonic 5 geometry factor without back iron", cell="Calculator!C87")
    tau5_Pa: float = out("Pa", "Harmonic 5 shear stress", cell="Calculator!C88")
    tau_Pa: float = out("Pa", "Total magnetic shear stress at pull-out", "PM-PM couplings typically 100–250 kPa.", "Calculator!C89")
    area_lever_m3: float = out("m³", "Gap area × lever arm (2π R_g² L)", cell="Calculator!C90")
    torque_2d_Nm: float = out("N·m", "2D pull-out torque (infinite length)", cell="Calculator!C91")
    f_end: float = out("-", "End-effect factor", cell="Calculator!C92")
    pullout_Nm: float = out("N·m", "Pull-out torque at operating temperature",
                            "Analytical estimate, not a guaranteed minimum.", "Calculator!C93")
    pullout_20C_Nm: float = out("N·m", "Pull-out torque at 20 °C", cell="Calculator!C94")
    pullout_iron_Nm: float = out("N·m", "Same layout with steel back iron (factor 0.95)", cell="Calculator!C95")
    pullout_noiron_Nm: float = out("N·m", "Raw no-back-iron prediction, same layout", cell="Calculator!C96")
    ripple_freq_Hz: float = out("Hz", "Torque ripple frequency at the design slip speed", cell="Calculator!C97")
    # checks
    gearbox_input_ripple_Nm: float = out("N·m", "Estimated gearbox input torque ripple amplitude", cell="Calculator!C99")
    gearbox_reference_Nm: float = out("N·m", "Legacy gearbox torque reference (informational)", cell="Calculator!C100")
    required_floor_Nm: float = out("N·m", "Required floor: larger of traction need and service minimum", cell="Calculator!C101")
    verdict: str = out("", "Verdict", "Only the hot minimum is evaluated.", "Calculator!C102")
    gap_flux_density_T: float = out("T", "Estimated gap flux density (flat circuit)", cell="Calculator!C103")
    backiron_needed_mm: float = out("mm", "Back-iron thickness needed (sinusoidal flux, B_sat)", cell="Calculator!C104")
    cup_ring_check: str = out("", "Cup ring check", cell="Calculator!C105")
    hub_check: str = out("", "Hub check", cell="Calculator!C106")
    inner_temp_check: str = out("", "Inner magnet temperature check", "Library rating only.", "Calculator!C107")
    outer_temp_check: str = out("", "Outer magnet temperature check", "Library rating only.", "Calculator!C108")


@dataclass
class MassResults:
    magnets_g: float = out("g", "Magnets (both rings)", "7.5 g/cm³.", "Calculator!C110")
    cup_g: float = out("g", "Steel cup wall and integral rear web", cell="Calculator!C111")
    hub_g: float = out("g", "Steel keyed inner hub", cell="Calculator!C112")
    boss_g: float = out("g", "Integral steel shaft boss", cell="Calculator!C113")
    total_g: float = out("g", "Preliminary rotating mass, including retainers",
                         "Gross geometry plus hardware; holes, threads and slots not subtracted.", "Calculator!C114")
    added_inertia_kgm2: float = out("kg·m²", "Added inertia at 0.25 m from the swing axis", "Point-mass estimate.", "Calculator!C115")


# --------------------------------------------------------------------------- helpers
@dataclass
class ResolvedMagnet:
    length_mm: float
    width_mm: float
    thickness_mm: float
    br_T: float
    tmax_C: object  # float, or "n/a" when not in the library


def resolve_magnets(m: MagnetInputs) -> tuple[ResolvedMagnet, ResolvedMagnet]:
    """Library values when the part is found, otherwise the manual values (workbook IFERROR/INDEX/MATCH)."""
    out_ = []
    for part, L, w, t, br in ((m.part_inner, m.manual_inner_length_mm, m.manual_inner_width_mm, m.manual_inner_thickness_mm, m.manual_inner_br_T),
                              (m.part_outer, m.manual_outer_length_mm, m.manual_outer_width_mm, m.manual_outer_thickness_mm, m.manual_outer_br_T)):
        spec = lookup(part)
        if spec:
            out_.append(ResolvedMagnet(spec.length_mm, spec.width_mm, spec.thickness_mm, spec.br_T, spec.tmax_C))
        else:
            out_.append(ResolvedMagnet(L, w, t, br, "n/a"))
    return out_[0], out_[1]


def select_calibration_factor(backiron: int, npole: int, part_i: str, part_o: str,
                              prototype_poles: float, f_cal_updated: float, f_cal_original: float) -> float:
    """Measured correction only for the prototype's circuit (no iron, same poles, B842SH both rings)."""
    if backiron == 0 and npole == prototype_poles and part_i == "B842SH" and part_o == "B842SH":
        return f_cal_updated
    return f_cal_original


def geometry_factor(k: float, t_i_mm: float, t_o_mm: float, g_mm: float, backiron: int) -> float:
    """S_n: sinh ratio for a steel-backed circuit, exponential form for free-space rings."""
    if backiron == 1:
        return (math.sinh(k * t_i_mm / 1000) * math.sinh(k * t_o_mm / 1000)
                / math.sinh(k * (t_i_mm + t_o_mm + g_mm) / 1000))
    return ((1 - math.exp(-k * t_i_mm / 1000)) * (1 - math.exp(-k * t_o_mm / 1000))
            * math.exp(-k * g_mm / 1000) / 2)


def harmonic_amplitude(br_T: float, n: int, fill: float) -> float:
    """Odd-harmonic amplitude of a square-wave magnetization with the given fill factor."""
    return br_T * (4 / (n * math.pi)) * math.sin(n * fill * math.pi / 2)


def shear_stress(br_i: float, br_o: float, fill_i: float, fill_o: float, npole: int, r_g_mm: float,
                 t_i_mm: float, t_o_mm: float, g_mm: float, backiron: int, mu0: float = MU0) -> dict:
    """Per-harmonic pull-out shear stress [Pa] and its parts, for harmonics 1, 3, 5."""
    res = {}
    for n in HARMONICS:
        k = n * (npole / 2) / (r_g_mm / 1000)
        bi = harmonic_amplitude(br_i, n, fill_i)
        bo = harmonic_amplitude(br_o, n, fill_o)
        s_iron = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 1)
        s_free = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 0)
        tau = bi * bo / (2 * mu0) * (s_iron if backiron == 1 else s_free) * math.sin(n * math.pi / 2)
        res[n] = {"k": k, "bi": bi, "bo": bo, "s_iron": s_iron, "s_free": s_free, "tau": tau}
    return res


# --------------------------------------------------------------------------- main model
def compute(ci: CouplingInputs, face_gap_mm: float, bond_inner_mm: float, bond_outer_mm: float,
            cup_wall_corner_mm: float, alpha_br: float, bsat_T: float, f_cal: float, f_cal_original: float,
            slip_rpm: float, required_min_Nm: float) -> ModelResults:
    """Calculator sheet. Linked values come from Metal design, Calibration and Materials (see api.compute_all)."""
    mi, mo = resolve_magnets(ci.magnets)
    N, a_i = ci.npole, ci.inner_back_apothem_mm
    L = min(mi.length_mm, mo.length_mm)
    # corner gap from the flat-face gap (formula always uses the corner geometry)
    corner_gap = face_gap_mm - (math.sqrt((a_i + mi.thickness_mm) ** 2 + (mi.width_mm / 2) ** 2) - (a_i + mi.thickness_mm))
    hub_wall = a_i - bond_inner_mm - ci.bore_mm / 2

    flat_i = 2 * a_i * math.tan(math.pi / N)
    if ci.faceted == 1:
        chk_i = (f"OK, {flat_i - mi.width_mm:.2f} mm slack" if flat_i >= mi.width_mm
                 else "TOO NARROW: increase apothem or reduce poles")
    else:
        chk_i = "n/a (arcs)"
    r_face_i = a_i + mi.thickness_mm
    r_corner_i = math.sqrt(r_face_i ** 2 + (mi.width_mm / 2) ** 2) if ci.faceted == 1 else r_face_i
    A_o = r_corner_i + corner_gap
    g_m = A_o - r_face_i
    flat_o = 2 * A_o * math.tan(math.pi / N)
    if ci.faceted == 1:
        chk_o = (f"OK, blocks {flat_o - mo.width_mm:.2f} mm apart at the faces" if flat_o >= mo.width_mm
                 else "TOO NARROW: increase gap/apothem or reduce poles")
    else:
        chk_o = "n/a (arcs)"
    A_back = A_o + mo.thickness_mm
    r_pocket = (A_back + bond_outer_mm) / math.cos(math.pi / N) if ci.faceted == 1 else A_back + bond_outer_mm
    OD = 2 * (r_pocket + cup_wall_corner_mm)
    wall_f = OD / 2 - A_back
    R_g = r_face_i + g_m / 2
    tau_p = 2 * math.pi * R_g / N
    al_i = min(1.0, mi.width_mm / (2 * math.pi * (a_i + mi.thickness_mm / 2) / N))
    al_o = min(1.0, mo.width_mm / (2 * math.pi * (A_o + mo.thickness_mm / 2) / N))

    bri = mi.br_T * (1 + alpha_br * (ci.op_temp_C - 20))
    bro = mo.br_T * (1 + alpha_br * (ci.op_temp_C - 20))
    h = shear_stress(bri, bro, al_i, al_o, N, R_g, mi.thickness_mm, mo.thickness_mm, g_m, ci.backiron, ci.mu0)
    tau = sum(h[n]["tau"] for n in HARMONICS)
    AL = 2 * math.pi * (R_g / 1000) ** 2 * (L / 1000)
    T2D = tau * AL
    f_end = 1 - ci.c_end * tau_p / L
    T_pull = T2D * f_end * f_cal
    T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro)
    T_iron = sum(h[n]["bi"] * h[n]["bo"] * h[n]["s_iron"] * math.sin(n * math.pi / 2) for n in HARMONICS) / (2 * ci.mu0) * AL * f_end * f_cal_original
    T_noiron = sum(h[n]["bi"] * h[n]["bo"] * h[n]["s_free"] * math.sin(n * math.pi / 2) for n in HARMONICS) / (2 * ci.mu0) * AL * f_end * f_cal

    floor_ = max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm)
    B_gap = (bri + bro) / 2 * (mi.thickness_mm + mo.thickness_mm) / (mi.thickness_mm + mo.thickness_mm + g_m)
    t_bi = B_gap * tau_p / (math.pi * bsat_T)

    def thick_check(wall):
        if ci.backiron == 0:
            return "No back iron"
        return "Thickness OK" if wall >= t_bi else "Too thin"

    def temp_check(tmax):
        if not isinstance(tmax, (int, float)):
            return "unknown"
        return "OK" if ci.op_temp_C <= tmax else "OVER the magnet rating"

    return ModelResults(
        inner_length_mm=mi.length_mm, inner_width_mm=mi.width_mm, inner_thickness_mm=mi.thickness_mm, inner_br_T=mi.br_T,
        inner_tmax_C=mi.tmax_C, outer_length_mm=mo.length_mm, outer_width_mm=mo.width_mm, outer_thickness_mm=mo.thickness_mm,
        outer_br_T=mo.br_T, outer_tmax_C=mo.tmax_C, active_length_mm=L, corner_gap_mm=corner_gap, alpha_br_per_C=alpha_br,
        bsat_T=bsat_T, cup_wall_corner_mm=cup_wall_corner_mm, hub_wall_mm=hub_wall, f_cal=f_cal,
        inner_flat_width_mm=flat_i, inner_flat_check=chk_i, hub_wall_past_key_mm=hub_wall - ci.keyway_depth_mm,
        inner_face_radius_mm=r_face_i, inner_corner_radius_mm=r_corner_i, outer_face_apothem_mm=A_o, face_gap_mm=g_m,
        outer_flat_width_mm=flat_o, outer_flat_check=chk_o, outer_back_apothem_mm=A_back, pocket_corner_radius_mm=r_pocket,
        cup_od_mm=OD, cup_wall_flat_mm=wall_f, gap_radius_mm=R_g, pole_pitch_mm=tau_p, fill_inner=al_i, fill_outer=al_o,
        br_inner_T_op=bri, br_outer_T_op=bro,
        k1=h[1]["k"], b_i1=h[1]["bi"], b_o1=h[1]["bo"], s1_iron=h[1]["s_iron"], s1_free=h[1]["s_free"], tau1_Pa=h[1]["tau"],
        k3=h[3]["k"], b_i3=h[3]["bi"], b_o3=h[3]["bo"], s3_iron=h[3]["s_iron"], s3_free=h[3]["s_free"], tau3_Pa=h[3]["tau"],
        k5=h[5]["k"], b_i5=h[5]["bi"], b_o5=h[5]["bo"], s5_iron=h[5]["s_iron"], s5_free=h[5]["s_free"], tau5_Pa=h[5]["tau"],
        tau_Pa=tau, area_lever_m3=AL, torque_2d_Nm=T2D, f_end=f_end, pullout_Nm=T_pull, pullout_20C_Nm=T_pull20,
        pullout_iron_Nm=T_iron, pullout_noiron_Nm=T_noiron, ripple_freq_Hz=N / 2 * slip_rpm / 60,
        gearbox_input_ripple_Nm=T_pull / (ci.gear_ratio * ci.gear_efficiency),
        gearbox_reference_Nm=ci.gearbox_input_rating_Nm * ci.gear_ratio * ci.gear_efficiency,
        required_floor_Nm=floor_, verdict="Below hot minimum" if T_pull < floor_ else "Nominal only: hot test",
        gap_flux_density_T=B_gap, backiron_needed_mm=t_bi, cup_ring_check=thick_check(cup_wall_corner_mm),
        hub_check=thick_check(hub_wall), inner_temp_check=temp_check(mi.tmax_C), outer_temp_check=temp_check(mo.tmax_C),
    )


def mass_estimate(ci: CouplingInputs, r: ModelResults, bond_inner_mm: float, bond_outer_mm: float, cup_depth_mm: float,
                  web_mm: float, hub_length_mm: float, boss_length_mm: float, boss_od_mm: float, steel_density_g_mm3: float,
                  al_density_g_mm3: float, retainers_g: float, hardware_g: float, cap_g: float, endplates_g: float) -> MassResults:
    """Calculator rows 110–115. Gross solids: no holes, slots or threads subtracted."""
    N = ci.npole
    m_mag = N * (r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm
                 + r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm) * NDFEB_DENSITY_G_MM3
    m_ring = ((math.pi * (r.cup_od_mm / 2) ** 2 - N * (r.outer_back_apothem_mm + bond_outer_mm) ** 2 * math.tan(math.pi / N)) * cup_depth_mm
              + math.pi * ((r.cup_od_mm / 2) ** 2 - (ci.bore_mm / 2) ** 2) * web_mm) * steel_density_g_mm3
    hub_area = (N * (ci.inner_back_apothem_mm - bond_inner_mm) ** 2 * math.tan(math.pi / N) if ci.faceted == 1
                else math.pi * (ci.inner_back_apothem_mm - bond_inner_mm) ** 2)
    m_hub = (hub_area - math.pi * (ci.bore_mm / 2) ** 2) * hub_length_mm * (steel_density_g_mm3 if ci.backiron == 1 else al_density_g_mm3)
    m_boss = math.pi * ((boss_od_mm / 2) ** 2 - (ci.bore_mm / 2) ** 2) * boss_length_mm * steel_density_g_mm3
    total = m_mag + m_ring + m_hub + m_boss + retainers_g + hardware_g + cap_g + endplates_g
    return MassResults(magnets_g=m_mag, cup_g=m_ring, hub_g=m_hub, boss_g=m_boss, total_g=total,
                       added_inertia_kgm2=total / 1000 * 0.25 ** 2)
