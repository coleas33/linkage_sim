"""Independent reference for the shaft-clamp sizing (engine: clamps.py; sheets 'Shaft clamps', 'Clamp screw sizes').

Written from standards and first principles. Nothing here imports the engine.

Sources
- ISO 68-1 basic metric profile: fundamental triangle height H = (sqrt(3)/2)·P; pitch diameter d2 = d - (3/4)·H;
  external minor diameter d3 = d - (17/12)·H; internal minor diameter D1 = d - (5/4)·H. The internal thread has a
  crest flat of P/4 at D1 and a root gap of P/8 at D.
- ISO 898-1: tensile stress area As = (pi/4)·((d2 + d3)/2)^2, tabulated to 3 significant figures (proof loads are
  built from the tabulated As); proof stress 970 MPa (class 12.9), 830 MPa (class 10.9).
- ISO 3506-1: A4-70 stress at 0.2 % permanent strain, 450 MPa (the usual "proof" stress for stainless screws).
- ISO 965-2 limits of size, medium tolerance class 6g (screw) / 6H (tapped hole), as printed in Bossard, "Metric
  ISO threads" (technical section, 01-2025), p. 97, "Limits for metric (standard) coarse threads according to
  ISO 965".
- ISO 261 coarse pitches, ISO 273 medium clearance holes, ISO 4762 head diameter d_k max, head height k max and hex
  socket size s, DIN 336 / ISO 2306 tap-drill sizes.
- Internal-thread stripping area, FED-STD-H28/2B (also Machinery's Handbook, "Strength of screw threads"):
  A_n = pi·n·Le·Ds,min·[1/(2n) + 0.57735·(Ds,min - En,max)], with n = 1/P, Ds,min the minimum major diameter of
  the external thread and En,max the maximum pitch diameter of the internal thread. It is the internal tooth
  width at the screw's major diameter (P/2 at En,max, widening by tan30° per unit of diameter), times pi·Ds,min·Le/P.
  With the 6g/6H minimum-material limits it is the smallest area an in-tolerance thread pair can have; at basic
  size (Ds = d, En = D2) the tooth width is 7P/8 and A_n = 0.875·pi·d·Le.
- Ultimate shear strength (MMPDS / ASM): 7075-T6 331 MPa, 6061-T6 207 MPa.
- Preload of 75 % of proof load for reusable joints: Shigley, Mechanical Engineering Design, Eq. 8-31.
- Tightening torque T = K·F·d: Shigley Eq. 8-27.
- Clamp friction torque (derived in friction_torque_coefficient): T = C·mu·F·d with
  C = ∫p dθ / ∫p·cosθ dθ over one jaw. C >= 1 for any non-negative bore pressure (cosθ <= 1), with C = 1 for line
  contact, 4/pi for a cosine distribution and pi/2 for uniform pressure (uniform pressure reproduces Shigley's
  press-fit torque (pi/2)·f·p·l·d^2).
- Parallel-key bearing pressure p = 2T/(d·h·L): the force at the shaft surface T/(d/2) over the contact area h·L
  (Shigley, keys and pins).
- Bolted-flange friction torque T = mu·n·F·(D_bc/2).

Clamp geometry (cross-section normal to the shaft axis). The slit plane contains the shaft axis; each tangential
screw axis is normal to the slit plane at offset e from the shaft axis, i.e. a chord of the boss circle (radius R).
Along the screw axis, x = 0 is the middle of the slit, the slit faces are at x = ±slit/2 and the boss OD is at
x = ±sqrt(R^2 - e^2). The flat head seat is a disc of diameter d_k centred on the screw axis in the plane x = x_seat;
the disc lies inside the boss cylinder x^2 + y^2 <= R^2 exactly when x_seat^2 + (e + d_k/2)^2 <= R^2. A screw of
under-head length L seated at x_seat crosses the head-side jaw (grip = x_seat - slit/2) and the open slit, so it
engages L - grip - slit of thread; it stays inside the far jaw while L <= grip + slit + thread available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

SQRT3 = math.sqrt(3.0)
TAN30 = 1.0 / SQRT3
COS30 = SQRT3 / 2.0

#: Float slack for "fits"/"rounds up" decisions on quantities that come out of mm arithmetic.
LENGTH_TOL_MM = 1e-9


@dataclass(frozen=True)
class IsoScrew:
    """ISO metric cap screw data (all mm)."""
    name: str
    d: float           # nominal diameter
    pitch: float       # ISO 261 coarse pitch
    hole: float        # ISO 273 medium clearance hole
    head_dk: float     # ISO 4762 head diameter, max
    head_k: float      # ISO 4762 head height, max (= d)
    hex_s: float       # ISO 4762 hex socket size
    tap_drill: float   # DIN 336 / ISO 2306 tap drill


ISO_SCREWS = (
    IsoScrew("M2.5", 2.5, 0.45, 2.9, 4.5, 2.5, 2.0, 2.05),
    IsoScrew("M3", 3.0, 0.5, 3.4, 5.5, 3.0, 2.5, 2.5),
    IsoScrew("M4", 4.0, 0.7, 4.5, 7.0, 4.0, 3.0, 3.3),
    IsoScrew("M5", 5.0, 0.8, 5.5, 8.5, 5.0, 4.0, 4.2),
    IsoScrew("M6", 6.0, 1.0, 6.6, 10.0, 6.0, 5.0, 5.0),
)


@dataclass(frozen=True)
class ThreadLimits:
    """ISO 965-2 limits of size for a 6g screw and a 6H tapped hole (mm), as printed."""
    d_max: float    # 6g major diameter
    d_min: float
    d2_max: float   # 6g pitch diameter
    d2_min: float
    D2_max: float   # 6H pitch diameter
    D2_min: float
    D1_max: float   # 6H minor diameter
    D1_min: float


#: Bossard "Metric ISO threads" p. 97 (ISO 965-2). M10 is included for the sanity test against ISO 724.
THREAD_LIMITS_6G_6H = {
    "M2.5": ThreadLimits(2.480, 2.380, 2.188, 2.117, 2.303, 2.208, 2.138, 2.013),
    "M3": ThreadLimits(2.980, 2.874, 2.655, 2.580, 2.775, 2.675, 2.599, 2.459),
    "M4": ThreadLimits(3.978, 3.838, 3.523, 3.433, 3.663, 3.545, 3.422, 3.242),
    "M5": ThreadLimits(4.976, 4.826, 4.456, 4.361, 4.605, 4.480, 4.334, 4.134),
    "M6": ThreadLimits(5.974, 5.794, 5.324, 5.212, 5.500, 5.350, 5.153, 4.917),
    "M10": ThreadLimits(9.968, 9.732, 8.994, 8.862, 9.206, 9.026, 8.676, 8.376),
}

#: ISO 898-1 proof stress (12.9, 10.9) and ISO 3506-1 Rp0.2 (A4-70), MPa.
PROOF_STRESS_MPA = {"12.9": 970.0, "10.9": 830.0, "A4-70": 450.0}
#: MMPDS / ASM ultimate shear strength, MPa.
AL_SHEAR_MPA = {"7075-T6": 331.0, "6061-T6": 207.0}


# --------------------------------------------------------------------------- ISO 68-1 / ISO 898-1 thread geometry
def triangle_height(pitch: float) -> float:
    """ISO 68-1 fundamental triangle height H = (sqrt(3)/2)·P."""
    return SQRT3 / 2.0 * pitch


def pitch_diameter(d: float, pitch: float) -> float:
    """ISO 68-1 basic pitch diameter d2 = D2 = d - (3/4)·H."""
    return d - 0.75 * triangle_height(pitch)


def minor_diameter_external(d: float, pitch: float) -> float:
    """ISO 68-1 external-thread minor diameter d3 = d - (17/12)·H (root radius H/6 below D1)."""
    return d - 17.0 / 12.0 * triangle_height(pitch)


def minor_diameter_internal(d: float, pitch: float) -> float:
    """ISO 68-1 internal-thread minor diameter D1 = d - (5/4)·H."""
    return d - 1.25 * triangle_height(pitch)


def stress_area(d: float, pitch: float) -> float:
    """ISO 898-1 tensile stress area As = (pi/4)·((d2 + d3)/2)^2 [mm^2], unrounded."""
    return math.pi / 4.0 * ((pitch_diameter(d, pitch) + minor_diameter_external(d, pitch)) / 2.0) ** 2


def round_sig(x: float, digits: int) -> float:
    """Round to `digits` significant figures."""
    if x == 0.0:
        return 0.0
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def iso_stress_area(d: float, pitch: float) -> float:
    """As as ISO 898-1 tabulates it (3 significant figures) [mm^2]."""
    return round_sig(stress_area(d, pitch), 3)


def round_up(x: float, step: float) -> float:
    """Smallest multiple of `step` that is >= x (with LENGTH_TOL_MM slack for float noise)."""
    return math.ceil((x - LENGTH_TOL_MM) / step) * step


# --------------------------------------------------------------------------- thread stripping (FED-STD-H28/2B)
def internal_tooth_width(x: float, en_max: float, pitch: float) -> float:
    """Axial width of the internal-thread tooth at diameter x: P/2 at the internal pitch diameter en_max, widening
    by tan30° per unit of diameter (each 60° flank moves (x - en_max)/2·tan30°)."""
    return pitch / 2.0 + (x - en_max) * TAN30


def internal_thread_shear_area(ds_min: float, en_max: float, pitch: float, engagement: float) -> float:
    """FED-STD-H28/2B internal-thread shear area, sheared on the cylinder at the screw major diameter ds_min:
    pi·ds_min·Le·w(ds_min)/P = pi·n·Le·Ds,min·[1/(2n) + tan30·(Ds,min - En,max)] [mm^2]."""
    return math.pi * ds_min * engagement * internal_tooth_width(ds_min, en_max, pitch) / pitch


def min_material_shear_area(screw: IsoScrew, engagement: float) -> float:
    """Internal-thread shear area at the ISO 965-2 6g/6H minimum-material limits (smallest screw major diameter,
    largest tapped pitch diameter) [mm^2]."""
    lim = THREAD_LIMITS_6G_6H[screw.name]
    return internal_thread_shear_area(lim.d_min, lim.D2_max, screw.pitch, engagement)


def stripping_capacity(shear_MPa: float, area_mm2: float, safety_factor: float) -> float:
    """Allowable screw force before the tapped thread strips, tau·A_n/SF [N]."""
    return shear_MPa * area_mm2 / safety_factor


# --------------------------------------------------------------------------- strength, friction, fasteners
def preload_proof_share(fraction: float, proof_MPa: float, area_mm2: float) -> float:
    """Preload as a share of proof load, F = fraction·Sp·As [N] (MPa·mm^2 = N)."""
    return fraction * proof_MPa * area_mm2


def friction_torque_coefficient(pressure_shape: Callable[[float], float], n: int = 20000) -> float:
    """C = T/(mu·F·d) for a two-jaw clamp on a shaft of diameter d = 2r and length l.

    theta is measured from the screw-force direction; each jaw presses the bore over |theta| <= pi/2 with pressure
    p(theta) = p0·shape(theta). Force balance on one jaw: F = ∫ p cos(theta) r l dtheta. Friction torque from both
    jaws: T = 2·mu·∫ p r^2 l dtheta. Hence C = T/(mu·F·2r) = ∫p dtheta / ∫p cos(theta) dtheta
    (midpoint rule, n panels).
    """
    h = math.pi / n
    num = den = 0.0
    for i in range(n):
        theta = -math.pi / 2.0 + (i + 0.5) * h
        p = pressure_shape(theta)
        num += p * h
        den += p * math.cos(theta) * h
    return num / den


#: Line contact on the screw axis (the conservative lower bound of friction_torque_coefficient).
LINE_CONTACT_COEFFICIENT = 1.0


def clamp_torque_per_screw(mu: float, preload_N: float, shaft_mm: float, clamp_factor: float) -> float:
    """Friction torque one screw's preload holds: C_line·mu·F·d·(clamp factor) [N·m]."""
    return LINE_CONTACT_COEFFICIENT * mu * preload_N * (shaft_mm / 1000.0) * clamp_factor


def tightening_torque(nut_factor: float, preload_N: float, d_mm: float) -> float:
    """Shigley Eq. 8-27, T = K·F·d [N·m]."""
    return nut_factor * preload_N * d_mm / 1000.0


def head_bearing_pressure(preload_N: float, head_dk: float, hole: float) -> float:
    """Pressure under the head on the annulus between the head diameter and the clearance hole [MPa]."""
    return preload_N / (math.pi / 4.0 * (head_dk ** 2 - hole ** 2))


def screws_needed(required_Nm: float, per_screw_Nm: float) -> int:
    """Smallest n >= 1 with n·T_per >= T_req (relative slack 1e-12 for float noise)."""
    if per_screw_Nm <= 0.0:
        raise ValueError("per-screw torque must be positive")
    n = 1
    while n * per_screw_Nm < required_Nm * (1.0 - 1e-12):
        n += 1
    return n


def screws_fit(clamp_length: float, axial_margin: float, cbore_dia: float, spacing: float) -> int:
    """Count screws placed one by one from the free end: every counterbore keeps `axial_margin` to both clamp ends
    and neighbouring screws sit `spacing` apart (greedy placement)."""
    if spacing <= 0.0:
        raise ValueError("screw spacing must be positive")
    first = axial_margin + cbore_dia / 2.0
    last_allowed = clamp_length - axial_margin - cbore_dia / 2.0
    count = 0
    centre = first
    while centre <= last_allowed + LENGTH_TOL_MM:
        count += 1
        centre = first + count * spacing
    return count


def hex_key_passes_port(hex_s: float, port_d: float = 6.0, port_pitch: float = 1.0) -> bool:
    """A hex key of size s (across flats) passes a tapped port when its across-corners size s/cos30° is below the
    port's internal minor diameter D1 (M6 x 1: D1 = 4.917 mm)."""
    return hex_s / COS30 < minor_diameter_internal(port_d, port_pitch)


def key_bearing_pressure_MPa(torque_Nm: float, shaft_mm: float, contact_mm: float, length_mm: float) -> float:
    """p = 2T/(d·h·L), everything in SI, returned in MPa."""
    return 2.0 * torque_Nm / ((shaft_mm / 1000.0) * (contact_mm / 1000.0) * (length_mm / 1000.0)) / 1e6


def flange_friction_torque_Nm(mu: float, n_screws: int, preload_N: float, bolt_circle_mm: float) -> float:
    """T = mu·n·F·(D_bc/2) [N·m]."""
    return mu * n_screws * preload_N * (bolt_circle_mm / 2.0) / 1000.0


# --------------------------------------------------------------------------- clamp sizing
@dataclass(frozen=True)
class ClampSetup:
    """Everything the sizing needs, in the units of the workbook inputs (mm, N·m, MPa)."""
    shaft_mm: float
    boss_od_mm: float
    slit_mm: float
    ligament_mm: float
    wall_min_mm: float
    grip_min_mm: float
    axial_margin_mm: float
    clamp_length_mm: float
    engagement_x_d: float
    preload_fraction: float
    strip_sf: float
    friction: float
    clamp_factor: float
    nut_factor: float
    screw_class: str          # "12.9", "10.9" or "A4-70"
    alloy: str                # "7075-T6" or "6061-T6"
    max_torque_Nm: float      # highest torque through the coupling
    safety_factor: float
    cbore_allowance_mm: float  # counterbore diameter = head + this (workbook design rule)
    head_gap_mm: float         # screw spacing = head + this (workbook design rule)
    length_step_mm: float      # screw lengths come in multiples of this (workbook design rule)


def clamp_geometry(s: ClampSetup, screw: IsoScrew) -> dict:
    """Screw offset, wall outside the hole, head seat, grip, thread available, counterbore depth.
    Quantities that do not exist when the head seat does not fit are None."""
    R = s.boss_od_mm / 2.0
    e = s.shaft_mm / 2.0 + s.ligament_mm + screw.hole / 2.0
    wall = R - (e + screw.hole / 2.0)
    head_edge = e + screw.head_dk / 2.0
    fits = head_edge <= R
    x_od = math.sqrt(R ** 2 - e ** 2) if e < R else None
    x_seat = math.sqrt(R ** 2 - head_edge ** 2) if fits else None
    engagement = s.engagement_x_d * screw.d
    return {
        "offset_mm": e,
        "wall_out_mm": wall,
        "head_fits": 1 if fits else 0,
        "grip_mm": (x_seat - s.slit_mm / 2.0) if fits else None,
        "thread_avail_mm": (x_od - s.slit_mm / 2.0) if x_od is not None else None,
        "engagement_req_mm": engagement,
        "cbore_dia_mm": screw.head_dk + s.cbore_allowance_mm,
        "cbore_depth_mm": (x_od - x_seat) if fits else None,
    }


def geometry_ok(s: ClampSetup, geo: dict) -> int:
    """All four geometric rules: wall, head seat, grip, thread length available."""
    ok = (geo["wall_out_mm"] >= s.wall_min_mm and geo["head_fits"] == 1
          and geo["grip_mm"] >= s.grip_min_mm
          and geo["thread_avail_mm"] is not None and geo["thread_avail_mm"] >= geo["engagement_req_mm"])
    return 1 if ok else 0


def engagement_achieved(s: ClampSetup, geo: dict, length_mm: float) -> float:
    """Thread a seated screw of this under-head length engages past the head-side jaw and the open slit [mm]."""
    return length_mm - geo["grip_mm"] - s.slit_mm


def max_length_inside(s: ClampSetup, geo: dict) -> float:
    """Longest under-head length that ends inside the far jaw: the seat-to-far-OD chord, grip + slit + avail [mm]."""
    return geo["grip_mm"] + s.slit_mm + geo["thread_avail_mm"]


def screw_tip_inside(s: ClampSetup, geo: dict, length_mm: float) -> int:
    """1 when a screw of this under-head length ends inside the far jaw."""
    return 1 if length_mm <= max_length_inside(s, geo) + LENGTH_TOL_MM else 0


def valid_screw_lengths(s: ClampSetup, geo: dict) -> list[float]:
    """Every length-step multiple that engages the required thread and ends inside the far jaw
    (empty when the head seat does not fit, or when no step multiple meets both rules)."""
    if geo["head_fits"] != 1:
        return []
    shortest = round_up(geo["grip_mm"] + s.slit_mm + geo["engagement_req_mm"], s.length_step_mm)
    out = []
    length = shortest
    while length <= max_length_inside(s, geo) + LENGTH_TOL_MM:
        out.append(length)
        length += s.length_step_mm
    return out


def size_row(s: ClampSetup, screw: IsoScrew) -> dict:
    """The complete independent evaluation of one screw size (lengths are the valid_screw_lengths list)."""
    geo = clamp_geometry(s, screw)
    As = iso_stress_area(screw.d, screw.pitch)
    F_strength = preload_proof_share(s.preload_fraction, PROOF_STRESS_MPA[s.screw_class], As)
    F_strip = stripping_capacity(AL_SHEAR_MPA[s.alloy], min_material_shear_area(screw, geo["engagement_req_mm"]),
                                 s.strip_sf)
    F = min(F_strength, F_strip)
    t_per = clamp_torque_per_screw(s.friction, F, s.shaft_mm, s.clamp_factor)
    need = screws_needed(s.max_torque_Nm * s.safety_factor, t_per)
    spacing = screw.head_dk + s.head_gap_mm
    fit = screws_fit(s.clamp_length_mm, s.axial_margin_mm, geo["cbore_dia_mm"], spacing)
    geo_ok = geometry_ok(s, geo)
    return {
        **geo,
        "As_mm2": As,
        "geometry_ok": geo_ok,
        "preload_strength_N": F_strength,
        "preload_strip_N": F_strip,
        "preload_N": F,
        "head_pressure_MPa": head_bearing_pressure(F, screw.head_dk, screw.hole),
        "torque_per_screw_Nm": t_per,
        "screws_needed": need,
        "pitch_axial_mm": spacing,
        "screws_fit": fit,
        "works": 1 if (geo_ok == 1 and need <= fit) else 0,
        "clamp_torque_Nm": need * t_per,
        "sf_coupling": need * t_per / s.max_torque_Nm,
        "tightening_Nm": tightening_torque(s.nut_factor, F, screw.d),
        "valid_lengths_mm": valid_screw_lengths(s, geo),
        "vent_port_ok": 1 if hex_key_passes_port(screw.hex_s) else 0,
    }


def recommend(s: ClampSetup) -> tuple[int, list[dict]]:
    """(1-based index of the first size that works, 0 when none; all rows)."""
    rows = [size_row(s, screw) for screw in ISO_SCREWS]
    for i, row in enumerate(rows):
        if row["works"] == 1:
            return i + 1, rows
    return 0, rows
