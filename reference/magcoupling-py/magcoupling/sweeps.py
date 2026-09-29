"""Design-space sweeps ('Gap sweep' and 'Pole sweep' sheets).

Both sweeps rerun the Calculator's 2D harmonic model with one variable changed:
  * gap sweep: the inner-block-corner to outer-face gap (current hub apothem);
    uses the Calculator's calibration factor.
  * pole sweep: the number of poles per ring, with the smallest inner apothem
    that fits the block width (+0.05 mm) and the keyed bore wall; uses the
    original 0.95 factor (the measured correction does not transfer).

Each row also gets a fit/status flag in the workbook's priority order:
inner flat too narrow → outer flat too narrow → outside OD envelope →
below hot minimum → nominal: test needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .model import HARMONICS, shear_stress

GAP_SWEEP_CORNER_GAPS_MM = (0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4)
POLE_SWEEP_POLES = (6, 8, 10, 12, 14, 16)


@dataclass
class SweepRow:
    variable: float            # corner gap [mm] for the gap sweep, poles for the pole sweep
    inner_apothem_mm: float    # C
    corner_gap_mm: float       # D
    centre_gap_mm: float       # E
    outer_face_apothem_mm: float  # F
    cup_od_mm: float           # G
    gap_radius_mm: float       # H
    pole_pitch_mm: float       # I
    fill_inner: float          # J
    fill_outer: float          # K
    k1: float                  # L
    s1: float                  # M
    tau1_Pa: float             # N
    k3: float                  # O
    s3: float                  # P
    tau3_Pa: float             # Q
    k5: float                  # R
    s5: float                  # S
    tau5_Pa: float             # T
    tau_Pa: float              # U
    torque_2d_Nm: float        # V
    f_end: float               # W
    pullout_op_Nm: float       # X
    pullout_20C_Nm: float      # Y
    gearbox_input_Nm: float    # Z
    status: str                # AA


# workbook column letter for each SweepRow field (row numbers start at 6)
SWEEP_COLUMNS = dict(zip(
    ["variable", "inner_apothem_mm", "corner_gap_mm", "centre_gap_mm", "outer_face_apothem_mm", "cup_od_mm", "gap_radius_mm",
     "pole_pitch_mm", "fill_inner", "fill_outer", "k1", "s1", "tau1_Pa", "k3", "s3", "tau3_Pa", "k5", "s5", "tau5_Pa", "tau_Pa",
     "torque_2d_Nm", "f_end", "pullout_op_Nm", "pullout_20C_Nm", "gearbox_input_Nm", "status"],
    ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA"]))


@dataclass
class SweepContext:
    """Values the sweeps borrow from the Calculator / Metal design."""
    faceted: int
    backiron: int
    t_i: float
    w_i: float
    t_o: float
    w_o: float
    L: float
    br_i20: float
    br_o20: float
    br_i_op: float
    br_o_op: float
    bond_outer: float
    cup_wall_corner: float
    c_end: float
    mu0: float
    gear_ratio: float
    gear_eff: float
    required_floor_Nm: float
    max_diameter_mm: float


def _row(ctx: SweepContext, variable: float, npole: int, a_i: float, corner_gap: float, factor: float) -> SweepRow:
    r_face = a_i + ctx.t_i
    F = (math.sqrt(r_face ** 2 + (ctx.w_i / 2) ** 2) if ctx.faceted == 1 else r_face) + corner_gap
    E = F - r_face
    back = F + ctx.t_o + ctx.bond_outer
    G = 2 * ((back / math.cos(math.pi / npole) if ctx.faceted == 1 else back) + ctx.cup_wall_corner)
    H = r_face + E / 2
    I = 2 * math.pi * H / npole
    J = min(1.0, ctx.w_i / (2 * math.pi * (a_i + ctx.t_i / 2) / npole))
    K = min(1.0, ctx.w_o / (2 * math.pi * (F + ctx.t_o / 2) / npole))
    h = shear_stress(ctx.br_i_op, ctx.br_o_op, J, K, npole, H, ctx.t_i, ctx.t_o, E, ctx.backiron, ctx.mu0)
    s = {n: (h[n]["s_iron"] if ctx.backiron == 1 else h[n]["s_free"]) for n in HARMONICS}
    U = sum(h[n]["tau"] for n in HARMONICS)
    V = U * 2 * math.pi * (H / 1000) ** 2 * (ctx.L / 1000)
    W = 1 - ctx.c_end * I / ctx.L
    X = V * W * factor
    Y = X * (ctx.br_i20 * ctx.br_o20) / (ctx.br_i_op * ctx.br_o_op)
    Z = X / (ctx.gear_ratio * ctx.gear_eff)
    if 2 * a_i * math.tan(math.pi / npole) < ctx.w_i:
        status = "inner flat too narrow"
    elif 2 * F * math.tan(math.pi / npole) < ctx.w_o:
        status = "outer flat too narrow"
    elif G > ctx.max_diameter_mm:
        status = "outside OD envelope"
    elif X < ctx.required_floor_Nm:
        status = "below hot minimum"
    else:
        status = "nominal: test needed"
    return SweepRow(variable, a_i, corner_gap, E, F, G, H, I, J, K,
                    h[1]["k"], s[1], h[1]["tau"], h[3]["k"], s[3], h[3]["tau"], h[5]["k"], s[5], h[5]["tau"],
                    U, V, W, X, Y, Z, status)


def gap_sweep(ctx: SweepContext, npole: int, a_i: float, f_cal: float,
              corner_gaps_mm=GAP_SWEEP_CORNER_GAPS_MM) -> list[SweepRow]:
    """Pull-out vs corner gap for the current layout (Calculator calibration factor)."""
    return [_row(ctx, g, npole, a_i, g, f_cal) for g in corner_gaps_mm]


def pole_sweep(ctx: SweepContext, corner_gap_mm: float, bore_mm: float, keyway_depth_mm: float, f_cal_original: float,
               poles=POLE_SWEEP_POLES) -> list[SweepRow]:
    """Pull-out vs poles; smallest apothem that fits the block (+0.05 mm) and the keyed-bore wall (2.5 mm)."""
    rows = []
    for n in poles:
        a_i = max(ctx.w_i / (2 * math.tan(math.pi / n)) + 0.05, bore_mm / 2 + keyway_depth_mm + 2.5)
        rows.append(_row(ctx, n, n, a_i, corner_gap_mm, f_cal_original))
    return rows
