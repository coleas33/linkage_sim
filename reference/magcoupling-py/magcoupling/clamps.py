"""Shaft clamp sizing ('Shaft clamps' and 'Clamp screw sizes' sheets).

Recommended construction: a one-piece slotted clamp in 7075-T6 on a keyed shaft.
One axial slit through one wall plus a transverse relief cut, closed by
tangential ISO 4762 cap screws threaded straight into the far jaw. The clamp
alone holds the coupling's maximum torque with margin; the key is the backup.
Never slit the magnet-carrying 4140 hub or cup — clamps go on aluminium
collars/adapters outside the magnet zone.

Approach for each metric screw size (M2.5 … M6):
  geometry  - screw offset from the shaft axis (half bore + ligament + half hole),
              wall outside the hole, whether a full head seat fits inside the boss,
              head-side jaw (grip) and thread length available in the far jaw;
  strength  - preload = share of proof load × stress area, capped by aluminium
              thread stripping (0.6·π·d·engagement·τ / SF);
  capacity  - clamp torque per screw = µ · preload · shaft diameter · clamp factor;
  fit       - screws needed vs screws that fit along the clamp length.
The first size that passes everything is recommended.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ._fields import ceiling, floor_, out, param


@dataclass(frozen=True)
class ScrewSize:
    name: str
    d_mm: float          # nominal diameter
    pitch_mm: float      # coarse pitch
    As_mm2: float        # tensile stress area
    hole_mm: float       # clearance hole, ISO 273 medium
    head_mm: float       # ISO 4762 head diameter (max)
    head_h_mm: float     # head height
    hex_mm: float        # hex key


SCREW_SIZES = (
    ScrewSize("M2.5", 2.5, 0.45, 3.39, 2.9, 4.5, 2.5, 2),
    ScrewSize("M3", 3, 0.5, 5.03, 3.4, 5.5, 3, 2.5),
    ScrewSize("M4", 4, 0.7, 8.78, 4.5, 7.0, 4, 3),
    ScrewSize("M5", 5, 0.8, 14.2, 5.5, 8.5, 5, 4),
    ScrewSize("M6", 6, 1.0, 20.1, 6.6, 10.0, 6, 5),
)
TABLE_COLUMNS = ("C", "D", "E", "F", "G")   # 'Clamp screw sizes' columns for M2.5 … M6


@dataclass
class ClampInputs:
    safety_factor: float = param(2.0, "-", "Safety factor, clamp alone", "Reversing torque and vibration; the key is extra.", "Shaft clamps!C16")
    friction: float = param(0.15, "-", "Friction coefficient, shaft to bore", "Degreased; 0.10 if the bore could be oily.", "Shaft clamps!C18")
    clamp_type: int = param(1, "-", "Clamp type", "1 = one-piece slotted, 2 = two-piece.", "Shaft clamps!C19", {1: "one-piece slotted", 2: "two-piece"})
    factor_one_piece: float = param(0.8, "-", "Clamp factor, one-piece", "Share of screw force pressing the shaft.", "Shaft clamps!C20")
    factor_two_piece: float = param(1.0, "-", "Clamp factor, two-piece", "", "Shaft clamps!C21")
    alloy: int = param(1, "-", "Aluminium", "1 = 7075-T6, 2 = 6061-T6.", "Shaft clamps!C23", {1: "7075-T6", 2: "6061-T6"})
    screw_class: int = param(1, "-", "Screw class", "1 = 12.9, 2 = 10.9, 3 = A4-70.", "Shaft clamps!C27", {1: "12.9", 2: "10.9", 3: "A4-70"})
    preload_fraction: float = param(0.75, "-", "Preload as a share of proof load", "", "Shaft clamps!C29")
    nut_factor: float = param(0.20, "-", "Nut factor", "Dry or with Loctite 243.", "Shaft clamps!C30")
    engagement_x_d: float = param(2.0, "-", "Thread engagement in aluminium (× screw diameter)", "", "Shaft clamps!C31")
    strip_sf: float = param(1.5, "-", "Safety factor on thread stripping", "", "Shaft clamps!C32")
    boss_od_mm: float = param(25, "mm", "Boss outside diameter", "At 22 mm only M3 fits; two of them need a 14.5 mm clamp.", "Shaft clamps!C35")
    clamp_length_mm: float = param(10, "mm", "Clamp length, free end to relief cut", "", "Shaft clamps!C36")
    slit_mm: float = param(0.8, "mm", "Slit width", "", "Shaft clamps!C37")
    ligament_mm: float = param(1.0, "mm", "Minimum ligament, bore to screw hole", "", "Shaft clamps!C38")
    wall_out_mm: float = param(1.0, "mm", "Minimum wall outside the screw hole", "", "Shaft clamps!C39")
    grip_min_mm: float = param(3.0, "mm", "Minimum head-side jaw (grip)", "", "Shaft clamps!C40")
    axial_margin_mm: float = param(1.0, "mm", "Axial margin, clamp end to counterbore edge", "", "Shaft clamps!C41")
    relief_mm: float = param(1.0, "mm", "Relief cut width", "", "Shaft clamps!C42")
    hinge_mm: float = param(8.0, "mm", "Hinge left under the relief cut", "Measured up from the far side of the boss.", "Shaft clamps!C43")
    key_width_mm: float = param(4, "mm", "Key width", "", "Shaft clamps!C58")
    key_contact_mm: float = param(1.5, "mm", "Key contact height in the hub", "", "Shaft clamps!C59")
    joint_screws: int = param(3, "-", "Adapter joint screws", "Pilot + 3 × M3 + dowel.", "Shaft clamps!C64")
    joint_bolt_circle_mm: float = param(26, "mm", "Bolt circle diameter", "", "Shaft clamps!C65")
    joint_friction: float = param(0.15, "-", "Friction coefficient, nickel plate on aluminium", "", "Shaft clamps!C66")


@dataclass
class ScrewRow:
    """One column of 'Clamp screw sizes' (rows 6–38)."""
    size: str
    d_mm: float
    pitch_mm: float
    As_mm2: float
    hole_mm: float
    head_mm: float
    head_h_mm: float
    hex_mm: float
    tap_drill_mm: float
    offset_mm: float
    wall_out_mm: float
    head_fits: int
    grip_mm: float
    thread_avail_mm: float
    engagement_req_mm: float
    geometry_ok: int
    preload_strength_N: float
    preload_strip_N: float
    preload_N: float
    head_pressure_MPa: float
    head_check: str
    torque_per_screw_Nm: float
    screws_needed: int
    pitch_axial_mm: float
    screws_fit: int
    works: int
    clamp_torque_Nm: float
    sf_coupling: float
    tightening_Nm: float
    length_mm: float
    length_ok: int
    cbore_dia_mm: float
    cbore_depth_mm: float
    vent_port_ok: int


# row number of each ScrewRow field in 'Clamp screw sizes' (for traceability and parity tests)
TABLE_ROWS = dict(zip([f for f in ScrewRow.__dataclass_fields__ if f != "size"], range(6, 39)))


@dataclass
class ClampResults:
    shaft_mm: float = out("mm", "Shaft diameter", cell="Shaft clamps!C14")
    max_torque_Nm: float = out("N·m", "Highest torque through the coupling", "Cold pull-out with +variation.", "Shaft clamps!C15")
    required_Nm: float = out("N·m", "Torque the clamp must hold", cell="Shaft clamps!C17")
    clamp_factor: float = out("-", "Clamp factor used", cell="Shaft clamps!C22")
    al_shear_MPa: float = out("MPa", "Aluminium shear strength", cell="Shaft clamps!C24")
    al_head_limit_MPa: float = out("MPa", "Limiting pressure under the screw head", cell="Shaft clamps!C25")
    al_key_allow_MPa: float = out("MPa", "Key bearing allowable", cell="Shaft clamps!C26")
    screw_proof_MPa: float = out("MPa", "Screw proof stress", cell="Shaft clamps!C28")
    boss_radius_mm: float = out("mm", "Boss radius", cell="Shaft clamps!C44")
    table: list = None
    index: int = out("-", "Size index in the table", "0 = none fits.", "Shaft clamps!C47")
    recommended: str = out("", "Recommended screw", cell="Shaft clamps!C48")
    screws: object = out("-", "Screws per clamp", cell="Shaft clamps!C49")
    tightening_Nm: object = out("N·m", "Tightening torque", "With Loctite 243.", "Shaft clamps!C50")
    hex_mm: object = out("mm", "Hex key", cell="Shaft clamps!C51")
    capacity_Nm: object = out("N·m", "Clamp torque capacity", cell="Shaft clamps!C52")
    sf_coupling: object = out("-", "Safety factor on the coupling torque", "Clamp alone, before the key.", "Shaft clamps!C53")
    head_check: object = out("", "Head pressure on the aluminium", cell="Shaft clamps!C54")
    vent_port: object = out("", "Through the M6 vent port", cell="Shaft clamps!C55")
    key_pressure_MPa: float = out("MPa", "Key bearing pressure on the hub at the clamp design torque", cell="Shaft clamps!C60")
    key_sf: float = out("x", "Key alone: allowable over actual", cell="Shaft clamps!C61")
    joint_preload_N: float = out("N", "Allowable preload per M3", cell="Shaft clamps!C67")
    joint_torque_Nm: float = out("N·m", "Joint slip torque", cell="Shaft clamps!C68")
    joint_sf: float = out("x", "Joint torque over the clamp design torque", cell="Shaft clamps!C69")
    layout_offset_mm: object = out("mm", "Screw offset from the shaft axis", cell="Shaft clamps!C72")
    layout_pitch_mm: object = out("mm", "Screw spacing", cell="Shaft clamps!C73")
    layout_first_mm: object = out("mm", "First screw from the free end", cell="Shaft clamps!C74")
    layout_cbore_dia_mm: object = out("mm", "Counterbore diameter (head side)", cell="Shaft clamps!C75")
    layout_cbore_depth_mm: object = out("mm", "Counterbore depth at the screw axis, from the OD", cell="Shaft clamps!C76")
    layout_grip_mm: object = out("mm", "Head-side jaw (grip)", cell="Shaft clamps!C77")
    layout_tap_drill_mm: object = out("mm", "Tap drill, far jaw", cell="Shaft clamps!C78")
    layout_thread_avail_mm: object = out("mm", "Thread length available in the far jaw", cell="Shaft clamps!C79")
    layout_slit: str = out("", "Slit", cell="Shaft clamps!C80")
    layout_relief: str = out("", "Relief cut", cell="Shaft clamps!C81")


MACHINING_STEPS = (
    "1. Turn the OD and bore (H7) in one setup; cut the keyway 90° from where the slit will go.",
    "2. With the part still solid, drill the clearance hole and counterbore on the head-side jaw, then drill and tap the far jaw, "
    "square to the slit plane at the screw offset. Drilling before slitting keeps the holes aligned.",
    "3. Cut the relief slot, then the slit (slitting saw), leaving the hinge.",
    "4. Deburr, anodize, chase the thread, and assemble dry with Loctite 243 on the screw at the tightening torque.",
)


def _fmt_num(x: float) -> str:
    """Excel-style number-to-text in concatenation (integers without a decimal point)."""
    return str(int(x)) if float(x).is_integer() else repr(x)


def compute(ci: ClampInputs, shaft_mm: float, max_torque_Nm: float, al_props, screw_proof_MPa: float) -> ClampResults:
    """al_props: materials.AluminiumAlloy for the selected alloy."""
    R = ci.boss_od_mm / 2
    T_req = max_torque_Nm * ci.safety_factor
    k = ci.factor_one_piece if ci.clamp_type == 1 else ci.factor_two_piece
    rows = []
    for s in SCREW_SIZES:
        e = shaft_mm / 2 + ci.ligament_mm + s.hole_mm / 2
        wall = R - e - s.hole_mm / 2
        fits = 1 if e + s.head_mm / 2 <= R else 0
        grip = math.sqrt(R ** 2 - (e + s.head_mm / 2) ** 2) - ci.slit_mm / 2 if fits else 0
        avail = math.sqrt(R ** 2 - e ** 2) - ci.slit_mm / 2 if e < R else 0
        ereq = ci.engagement_x_d * s.d_mm
        geo = 1 if (wall >= ci.wall_out_mm and fits == 1 and grip >= ci.grip_min_mm and avail >= ereq) else 0
        Fb = ci.preload_fraction * screw_proof_MPa * s.As_mm2
        Fs = 0.6 * math.pi * s.d_mm * ereq * al_props.shear_MPa / ci.strip_sf
        F = min(Fb, Fs)
        p = F / (math.pi / 4 * (s.head_mm ** 2 - s.hole_mm ** 2))
        Tper = ci.friction * F * shaft_mm / 1000 * k
        need = int(ceiling(T_req / Tper, 1))
        pitch = s.head_mm + 1
        span = ci.clamp_length_mm - 2 * ci.axial_margin_mm - (s.head_mm + 0.5)
        fit = int(floor_(span / pitch, 1)) + 1 if span >= 0 else 0
        works = 1 if (geo == 1 and need <= fit) else 0
        length = ceiling(grip + ereq, 2) if fits else 0
        rows.append(ScrewRow(
            s.name, s.d_mm, s.pitch_mm, s.As_mm2, s.hole_mm, s.head_mm, s.head_h_mm, s.hex_mm, s.d_mm - s.pitch_mm, e, wall,
            fits, grip, avail, ereq, geo, Fb, Fs, F, p, "OK" if p <= al_props.head_pressure_limit_MPa else "Use a hardened washer",
            Tper, need, pitch, fit, works, need * Tper, need * Tper / max_torque_Nm, ci.nut_factor * F * s.d_mm / 1000,
            length, 1 if (fits and length <= grip + avail) else 0, s.head_mm + 0.5,
            (math.sqrt(R ** 2 - e ** 2) - math.sqrt(R ** 2 - (e + s.head_mm / 2) ** 2)) if fits else 0,
            1 if s.hex_mm <= 4 else 0))

    idx = next((i + 1 for i, r in enumerate(rows) if r.works == 1), 0)
    cls = {1: "12.9", 2: "10.9", 3: "A4-70"}[ci.screw_class]
    if idx:
        r = rows[idx - 1]
        rec = f"ISO 4762 {r.size} x {_fmt_num(r.length_mm)}, class {cls}"
        vent = "Yes: the key fits the 4 mm limit" if r.hex_mm <= 4 else "No: key too large"
        layout = dict(layout_offset_mm=r.offset_mm, layout_pitch_mm=r.pitch_axial_mm,
                      layout_first_mm=(ci.clamp_length_mm - (r.screws_needed - 1) * r.pitch_axial_mm) / 2,
                      layout_cbore_dia_mm=r.cbore_dia_mm, layout_cbore_depth_mm=r.cbore_depth_mm, layout_grip_mm=r.grip_mm,
                      layout_tap_drill_mm=r.tap_drill_mm, layout_thread_avail_mm=r.thread_avail_mm)
        sel = dict(screws=r.screws_needed, tightening_Nm=r.tightening_Nm, hex_mm=r.hex_mm, capacity_Nm=r.clamp_torque_Nm,
                   sf_coupling=r.sf_coupling, head_check=r.head_check, vent_port=vent)
    else:
        rec = "None: enlarge the boss or the clamp length"
        layout = {k_: "" for k_ in ("layout_offset_mm", "layout_pitch_mm", "layout_first_mm", "layout_cbore_dia_mm",
                                    "layout_cbore_depth_mm", "layout_grip_mm", "layout_tap_drill_mm", "layout_thread_avail_mm")}
        sel = {k_: "" for k_ in ("screws", "tightening_Nm", "hex_mm", "capacity_Nm", "sf_coupling", "head_check", "vent_port")}

    key_p = 2 * T_req / ((shaft_mm / 1000) * (ci.key_contact_mm / 1000) * (ci.clamp_length_mm / 1000)) / 1e6
    m3 = rows[1]
    joint_T = ci.joint_friction * ci.joint_screws * m3.preload_N * ci.joint_bolt_circle_mm / 2000
    return ClampResults(
        shaft_mm=shaft_mm, max_torque_Nm=max_torque_Nm, required_Nm=T_req, clamp_factor=k, al_shear_MPa=al_props.shear_MPa,
        al_head_limit_MPa=al_props.head_pressure_limit_MPa, al_key_allow_MPa=al_props.key_bearing_allow_MPa,
        screw_proof_MPa=screw_proof_MPa, boss_radius_mm=R, table=rows, index=idx, recommended=rec,
        key_pressure_MPa=key_p, key_sf=al_props.key_bearing_allow_MPa / key_p, joint_preload_N=m3.preload_N,
        joint_torque_Nm=joint_T, joint_sf=joint_T / T_req,
        layout_slit=f"{ci.slit_mm:.1f} mm wide, bore to OD on one side, free end to the relief cut",
        layout_relief=(f"{ci.relief_mm:.1f} mm wide at {ci.clamp_length_mm:.1f} mm from the free end, "
                       f"{ci.boss_od_mm - ci.hinge_mm:.1f} mm deep from the slit side"),
        **sel, **layout)
