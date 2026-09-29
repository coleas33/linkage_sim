"""Independent single-point evaluation of one sweep row (engine: sweeps.py; sheets 'Gap sweep' and 'Pole sweep').

Nothing here imports the engine. The row is assembled from the earlier tasks' references, so each formula lives in
one place in the audit:
- planar model (Task 3, audit.references.planar): Br(T), harmonic amplitudes B_n = Br·4/(n·pi)·sin(n·pi·fill/2),
  wave number k_n = n·pi/pole pitch (= n·(poles/2)/R_gap, the README form), the steel-backed and free-space
  geometry factors S_n, the harmonic shear stress B_in·B_on/(2·mu0)·S_n·sin(k·delta), the torque of a uniform shear
  on a cylinder and the empirical end factor 1 - c_end·pitch/L;
- geometry (Task 4, block_geometry.coupling_geometry): face radius, inner corner radius (block rectangle, or the
  arc radius), outer face apothem, flat-face gap, pocket polygon vertex radius and cup OD, gap radius, pole pitch,
  fill = block width / pole pitch at the block's mid-thickness radius, flat widths;
- gearbox input torque (Task 4, metal_stack.gearbox_input_torque).

What this module adds: the row assembly (README 'Torque model' steps 1-4 at one corner gap and pole count, with
the pull-out evaluated at half a pole pitch, where sin(k·delta) = sin(n·pi/2)), the status priority and the pole
sweep's apothem rule as stated.

A row is parameterized by the corner gap (inner block corner to outer block face). coupling_geometry takes the
flat-face gap, which is the corner gap plus the inner block's corner overhang (corner radius - face radius; zero
for arc magnets), so geometry_at_corner_gap builds it in two explicit steps.

This reproduces the model as the README states it. Whether that model is right (for example the sin(n·pi/2)
evaluation of every harmonic at the fundamental's pull-out angle) is audited by the torque family, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from audit.references.block_geometry import CouplingGeometry, coupling_geometry
from audit.references.metal_stack import gearbox_input_torque
from audit.references.planar import (br_at, end_factor, free_space_factor, harmonic_shear_stress, iron_backed_factor,
                                     square_wave_harmonic, torque_on_cylinder, wave_number)

HARMONICS = (1, 3, 5)

STATUS_INNER_NARROW = "inner flat too narrow"
STATUS_OUTER_NARROW = "outer flat too narrow"
STATUS_OD = "outside OD envelope"
STATUS_BELOW_MIN = "below hot minimum"
STATUS_NOMINAL = "nominal: test needed"

#: Pole-sweep apothem rule as the sheet note states it (engine sweeps.py docstrings, ported from the workbook):
#: "the smallest inner apothem that fits the block width (+0.05 mm) and the keyed bore wall (2.5 mm)".
FLAT_MARGIN_MM = 0.05
KEYED_WALL_MM = 2.5


@dataclass(frozen=True)
class SweepSetup:
    """The values a sweep row needs (workbook units: mm, T, °C)."""
    faceted: int
    backiron: int
    t_i_mm: float
    w_i_mm: float
    t_o_mm: float
    w_o_mm: float
    length_mm: float
    br_i20_T: float
    br_o20_T: float
    alpha_per_C: float
    op_temp_C: float
    bond_inner_mm: float
    bond_outer_mm: float
    cup_wall_corner_mm: float
    bore_mm: float
    keyway_mm: float
    c_end: float
    mu0: float
    gear_ratio: float
    gear_eff: float
    required_floor_Nm: float
    max_diameter_mm: float


def required_floor(drive_torque_Nm: float, drive_safety_factor: float, required_min_Nm: float) -> float:
    """Larger of the traction need (drive torque × safety factor) and the service minimum."""
    return max(drive_torque_Nm * drive_safety_factor, required_min_Nm)


def geometry_at_corner_gap(s: SweepSetup, npole: int, a_i_mm: float, corner_gap_mm: float) -> CouplingGeometry:
    """Task 4's coupling geometry for this row: first the inner block's corner overhang, then the geometry at the
    flat-face gap corner_gap + overhang."""
    kw = dict(npole=npole, inner_back_apothem=a_i_mm, t_i=s.t_i_mm, w_i=s.w_i_mm, t_o=s.t_o_mm, w_o=s.w_o_mm,
              bond_inner=s.bond_inner_mm, bond_outer=s.bond_outer_mm, cup_wall_corner=s.cup_wall_corner_mm,
              bore=s.bore_mm, keyway_depth=s.keyway_mm, faceted=s.faceted == 1)
    probe = coupling_geometry(face_gap=0.0, **kw)
    overhang = probe.inner_corner_radius - probe.inner_face_radius
    return coupling_geometry(face_gap=corner_gap_mm + overhang, **kw)


def sweep_status(s: SweepSetup, geo: CouplingGeometry, pullout_op_Nm: float) -> str:
    """First failing rule in the workbook's priority order (sweeps.py docstring): the hub flat under the inner block
    is narrower than the block, the outer face flat is narrower than the outer block, the cup OD exceeds the
    envelope, the nominal pull-out is below the required floor."""
    if geo.inner_flat_width < s.w_i_mm:
        return STATUS_INNER_NARROW
    if geo.outer_flat_width < s.w_o_mm:
        return STATUS_OUTER_NARROW
    if geo.cup_od > s.max_diameter_mm:
        return STATUS_OD
    if pullout_op_Nm < s.required_floor_Nm:
        return STATUS_BELOW_MIN
    return STATUS_NOMINAL


def sweep_row(s: SweepSetup, npole: int, a_i_mm: float, corner_gap_mm: float, f_cal: float) -> dict:
    """Independent evaluation of one sweep row; keys are the engine's SweepRow field names (minus 'variable')."""
    geo = geometry_at_corner_gap(s, npole, a_i_mm, corner_gap_mm)
    br_i = br_at(s.br_i20_T, s.alpha_per_C, s.op_temp_C)
    br_o = br_at(s.br_o20_T, s.alpha_per_C, s.op_temp_C)
    r_gap_m = geo.gap_radius / 1000.0
    pitch_m = geo.pole_pitch / 1000.0
    t_i_m, t_o_m, g_m = s.t_i_mm / 1000.0, s.t_o_mm / 1000.0, geo.face_gap / 1000.0
    out = {"inner_apothem_mm": a_i_mm, "corner_gap_mm": geo.corner_gap, "centre_gap_mm": geo.face_gap,
           "outer_face_apothem_mm": geo.outer_face_apothem, "cup_od_mm": geo.cup_od, "gap_radius_mm": geo.gap_radius,
           "pole_pitch_mm": geo.pole_pitch, "fill_inner": geo.fill_inner, "fill_outer": geo.fill_outer}
    tau = 0.0
    for n in HARMONICS:
        k = wave_number(n, pitch_m)
        s_n = iron_backed_factor(k, t_i_m, t_o_m, g_m) if s.backiron == 1 else free_space_factor(k, t_i_m, t_o_m, g_m)
        tau_n = harmonic_shear_stress(square_wave_harmonic(br_i, n, geo.fill_inner),
                                      square_wave_harmonic(br_o, n, geo.fill_outer), s_n, k, pitch_m / 2.0, s.mu0)
        out.update({f"k{n}": k, f"s{n}": s_n, f"tau{n}_Pa": tau_n})
        tau += tau_n
    torque_2d = torque_on_cylinder(tau, r_gap_m, s.length_mm / 1000.0)
    f_end = end_factor(s.c_end, geo.pole_pitch, s.length_mm)
    pull_op = torque_2d * f_end * f_cal
    out.update({"tau_Pa": tau, "torque_2d_Nm": torque_2d, "f_end": f_end, "pullout_op_Nm": pull_op,
                # torque is bilinear in the two rings' remanence (README step 4)
                "pullout_20C_Nm": pull_op * (s.br_i20_T * s.br_o20_T) / (br_i * br_o),
                "gearbox_input_Nm": gearbox_input_torque(pull_op, s.gear_ratio, s.gear_eff),
                "status": sweep_status(s, geo, pull_op)})
    return out


def stated_min_inner_apothem(npole: int, w_i_mm: float, bore_mm: float, keyway_mm: float) -> float:
    """The pole sweep's apothem rule as stated: the hub flat holds the block, 2·a·tan(pi/N) >= w_i, plus 0.05 mm on
    the apothem; and a 2.5 mm keyed-bore wall measured, as the note leaves it, from the block-back apothem:
    a >= bore/2 + keyway + 2.5."""
    return max(w_i_mm / (2.0 * math.tan(math.pi / npole)) + FLAT_MARGIN_MM, bore_mm / 2.0 + keyway_mm + KEYED_WALL_MM)
