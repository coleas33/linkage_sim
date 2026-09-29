"""Task 7: independent checks of the shaft-clamp sizing (engine clamps.py) against audit.references.clamp_ref.

One check per formula (one per root cause), each looping over every scenario and screw size with assert_all, so a
wrong formula yields one candidate that lists all of its wrong cells. Stage-wise where a formula takes an upstream
value (it is fed the engine's own upstream cell), so one wrong stage fails one check; end-to-end only for the
recommendation and the README claims.

Workbook layout rules that are design choices, not physics, are taken as given (not independently checkable):
counterbore = head + 0.5 mm, screw spacing = head + 1 mm, screw lengths in 2 mm steps, the one-/two-piece clamp
factors 0.8/1.0, and the aluminium allowables (head pressure, key bearing).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import pytest

from audit.common import TOL_ALGEBRA, ScenarioCache, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import clamp_ref as ref
# Workbook cell addresses only, used to label failures; no engine arithmetic is reused.
from magcoupling.clamps import TABLE_COLUMNS, TABLE_ROWS

CBORE_ALLOWANCE_MM = 0.5
HEAD_GAP_MM = 1.0
LENGTH_STEP_MM = 2.0

SCENARIOS = {
    "defaults": {},
    "boss22_len14.5": {"clamps.boss_od_mm": 22, "clamps.clamp_length_mm": 14.5},
    "al6061_cl10.9": {"clamps.alloy": 2, "clamps.screw_class": 2},
    "two_piece_oily_A4-70": {"clamps.clamp_type": 2, "clamps.friction": 0.10, "clamps.screw_class": 3},
    "strip_governs": {"clamps.alloy": 2, "clamps.engagement_x_d": 1.0},
}
CASES = ScenarioCache(SCENARIOS)
CLASS_TEXT = {1: "12.9", 2: "10.9", 3: "A4-70"}   # ClampInputs.screw_class selector codes
ALLOY_TEXT = {1: "7075-T6", 2: "6061-T6"}          # ClampInputs.alloy selector codes
SELECTION_CELLS = {
    "screws": "C49", "tightening_Nm": "C50", "hex_mm": "C51", "capacity_Nm": "C52", "sf_coupling": "C53",
    "head_check": "C54", "vent_port": "C55", "layout_offset_mm": "C72", "layout_pitch_mm": "C73",
    "layout_first_mm": "C74", "layout_cbore_dia_mm": "C75", "layout_cbore_depth_mm": "C76", "layout_grip_mm": "C77",
    "layout_tap_drill_mm": "C78", "layout_thread_avail_mm": "C79",
}


def _setup(inp, res) -> ref.ClampSetup:
    """Reference inputs built from the workbook inputs; the only upstream value is the cold-high torque."""
    c = inp.clamps
    return ref.ClampSetup(
        shaft_mm=inp.coupling.bore_mm, boss_od_mm=c.boss_od_mm, slit_mm=c.slit_mm, ligament_mm=c.ligament_mm,
        wall_min_mm=c.wall_out_mm, grip_min_mm=c.grip_min_mm, axial_margin_mm=c.axial_margin_mm,
        clamp_length_mm=c.clamp_length_mm, engagement_x_d=c.engagement_x_d, preload_fraction=c.preload_fraction,
        strip_sf=c.strip_sf, friction=c.friction,
        clamp_factor=c.factor_one_piece if c.clamp_type == 1 else c.factor_two_piece,
        nut_factor=c.nut_factor, screw_class=CLASS_TEXT[c.screw_class], alloy=ALLOY_TEXT[c.alloy],
        max_torque_Nm=res.metal.torque_cold_high_Nm, safety_factor=c.safety_factor,
        cbore_allowance_mm=CBORE_ALLOWANCE_MM, head_gap_mm=HEAD_GAP_MM, length_step_mm=LENGTH_STEP_MM)


def _cell(field: str, i: int) -> str:
    return f"Clamp screw sizes!{TABLE_COLUMNS[i]}{TABLE_ROWS[field]}"


def _na0(x):
    """The workbook prints 0 for a quantity that does not exist (e.g. grip when the head seat does not fit)."""
    return 0.0 if x is None else x


class Case(NamedTuple):
    tag: str            # "<scenario> <size>", for failure labels
    i: int              # size index 0..4 (table column C..G)
    inp: object
    res: object
    setup: ref.ClampSetup
    screw: ref.IsoScrew
    row: object         # engine table row
    geo: dict           # reference geometry


def _cases():
    """Every (scenario, screw size) pair with its engine row and reference geometry."""
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for i, screw in enumerate(ref.ISO_SCREWS):
            yield Case(f"{sc} {screw.name}", i, inp, res, setup, screw, res.clamps.table[i],
                       ref.clamp_geometry(setup, screw))


def _pick_name(index: int):
    return ref.ISO_SCREWS[index - 1].name if index else None


# --------------------------------------------------------------------------- screw data (literature)
@pytest.mark.family("clamps")
def test_screw_catalogue_matches_iso():
    """d, coarse pitch (ISO 261), medium clearance hole (ISO 273), head diameter/height and hex key (ISO 4762),
    tap drill (DIN 336) equal the standard values for M2.5..M6."""
    table = CASES["defaults"][1].clamps.table
    assert_all([(f"{s.name} {f}", _cell(f, i), getattr(table[i], f), v, TOL_ALGEBRA)
                for i, s in enumerate(ref.ISO_SCREWS)
                for f, v in (("d_mm", s.d), ("pitch_mm", s.pitch), ("hole_mm", s.hole), ("head_mm", s.head_dk),
                             ("head_h_mm", s.head_k), ("hex_mm", s.hex_s), ("tap_drill_mm", s.tap_drill))])


@pytest.mark.family("clamps")
def test_stress_area_is_iso898():
    """As equals the ISO 898-1 formula (pi/4)·((d2 + d3)/2)^2 with d2, d3 from the ISO 68-1 profile and the ISO 261
    pitch, printed to 3 significant figures as ISO 898-1 tabulates it."""
    table = CASES["defaults"][1].clamps.table
    assert_all([(f"{s.name} stress area", _cell("As_mm2", i), table[i].As_mm2, ref.iso_stress_area(s.d, s.pitch),
                 TOL_ALGEBRA) for i, s in enumerate(ref.ISO_SCREWS)])


# --------------------------------------------------------------------------- geometry
@pytest.mark.family("clamps")
def test_clamp_geometry():
    """Screw offset, wall outside the hole, head seat, grip, thread available, engagement, counterbore and the
    geometry verdict, re-derived from the chord geometry of the boss (clamp_ref module docstring), for every
    scenario and size."""
    items = []
    for c in _cases():
        items += [(f"{c.tag} {f}", _cell(f, c.i), getattr(c.row, f), _na0(c.geo[f]), TOL_ALGEBRA)
                  for f in ("offset_mm", "wall_out_mm", "head_fits", "grip_mm", "thread_avail_mm", "engagement_req_mm",
                            "cbore_dia_mm", "cbore_depth_mm")]
        items.append((f"{c.tag} geometry_ok", _cell("geometry_ok", c.i), c.row.geometry_ok,
                      ref.geometry_ok(c.setup, c.geo), 0.0))
    assert_all(items)


@pytest.mark.family("clamps")
def test_screw_length_meets_engagement_and_stays_inside():
    """Row 34 (screw length), property check for every scenario and size whose head seat fits: the engine's
    under-head length must engage at least the required thread (engagement_x_d·d) past the head-side jaw and the
    open slit (engaged = length - grip - slit), and must end inside the far jaw (length <= grip + slit + thread
    available, the seat-to-far-OD chord). Each failure also lists which 2 mm-step lengths meet both rules ('none'
    means no standard length satisfies the engagement rule without protruding). Where the head does not fit the
    workbook prints 0. One check for the matrix: every case uses the same row-34 formula."""
    items = []
    for c in _cases():
        cell = _cell("length_mm", c.i)
        if not c.geo["head_fits"]:
            items.append((f"{c.tag} length when the head seat does not fit", cell, c.row.length_mm, 0.0, 0.0))
            continue
        L = c.row.length_mm
        engaged = ref.engagement_achieved(c.setup, c.geo, L)
        need = c.geo["engagement_req_mm"]
        longest = ref.max_length_inside(c.setup, c.geo)
        valid = ref.valid_screw_lengths(c.setup, c.geo)
        options = f"valid 2 mm-step lengths: {', '.join(f'{v:g}' for v in valid) or 'none'}"
        items.append(flag_item(f"{c.tag}: the {L:g} mm screw engages {engaged:.3f} mm >= required {need:.3f} mm "
                               f"({options})", cell, engaged >= need - ref.LENGTH_TOL_MM))
        items.append(flag_item(f"{c.tag}: the {L:g} mm screw ends inside the far jaw (<= {longest:.3f} mm)", cell,
                               L <= longest + ref.LENGTH_TOL_MM))
    assert_all(items)


@pytest.mark.family("clamps")
def test_length_ok_flag():
    """Row 35 (length OK): 1 exactly when the engine's own length ends inside the far jaw (stage-wise: fed the
    engine's length, so a wrong length does not also fail here). 0 where the head seat does not fit."""
    items = []
    for c in _cases():
        expected = ref.screw_tip_inside(c.setup, c.geo, c.row.length_mm) if c.geo["head_fits"] else 0
        items.append((f"{c.tag} length_ok for the {c.row.length_mm:g} mm screw", _cell("length_ok", c.i),
                      c.row.length_ok, expected, 0.0))
    assert_all(items)


# --------------------------------------------------------------------------- strength
@pytest.mark.family("clamps")
def test_stripping_capacity_not_above_min_material():
    """Row 22 (stripping-limited preload). The allowable force before the tapped aluminium strips must not exceed
    tau·A_n/SF with the FED-STD-H28/2B internal-thread shear area at the ISO 965-2 minimum-material limits (6g screw
    major diameter min, 6H pitch diameter max), the smallest area any in-tolerance 6H/6g pair has. One-sided: a
    smaller engine value is conservative and passes. The engine uses a constant 0.6·pi·d·Le; the minimum-material
    area is 0.570 (M2.5) to 0.647 (M6)·pi·d·Le. Evaluated at defaults: the ratio does not depend on the alloy or the
    engagement factor (both scale engine and reference alike; test_stripping_capacity_scales_with_shear_and_engagement
    checks how the engine uses them)."""
    inp, res = CASES["defaults"]
    setup = _setup(inp, res)
    items = []
    for i, s in enumerate(ref.ISO_SCREWS):
        le = setup.engagement_x_d * s.d
        tau = ref.AL_SHEAR_MPA[setup.alloy]
        area = ref.min_material_shear_area(s, le)
        cap = ref.stripping_capacity(tau, area, setup.strip_sf)
        eng = res.clamps.table[i].preload_strip_N
        eng_factor = eng * setup.strip_sf / (tau * math.pi * s.d * le)      # engine area / (pi·d·Le)
        items.append(flag_item(f"{s.name} stripping capacity {eng:.1f} N <= minimum-material {cap:.1f} N "
                               f"(ratio {eng / cap:.4f}; area / (pi·d·Le): engine {eng_factor:.4f}, "
                               f"minimum-material {area / (math.pi * s.d * le):.4f})",
                               _cell("preload_strip_N", i), eng <= cap * (1.0 + TOL_ALGEBRA)))
    assert_all(items)


@pytest.mark.family("clamps")
def test_stripping_capacity_scales_with_shear_and_engagement():
    """Row 22 across scenarios, scaling law: the stripping capacity is proportional to the alloy's shear strength
    (MMPDS 331/207 MPa) and to the engagement length, and inversely to the stripping SF, so
    capacity(scenario)/capacity(defaults) = (tau·Le/SF)(scenario)/(tau·Le/SF)(defaults), whatever the area factor."""
    d_inp, d_res = CASES["defaults"]
    d_setup = _setup(d_inp, d_res)
    items = []
    for c in _cases():
        ratio_ref = ((ref.AL_SHEAR_MPA[c.setup.alloy] * c.setup.engagement_x_d / c.setup.strip_sf)
                     / (ref.AL_SHEAR_MPA[d_setup.alloy] * d_setup.engagement_x_d / d_setup.strip_sf))
        items.append((f"{c.tag} stripping capacity over defaults", _cell("preload_strip_N", c.i),
                      c.row.preload_strip_N / d_res.clamps.table[c.i].preload_strip_N, ratio_ref, TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("clamps")
def test_preload():
    """Rows 21 and 23: preload at the proof share = fraction·Sp·As with the ISO 898-1 tabulated As and the ISO 898-1 /
    ISO 3506-1 proof stress (Shigley Eq. 8-31: 75 % of proof for reusable joints); preload used = the smaller of
    that and the stripping capacity (stage-wise: the engine's row-22 value)."""
    items = []
    for c in _cases():
        strength = ref.preload_proof_share(c.setup.preload_fraction, ref.PROOF_STRESS_MPA[c.setup.screw_class],
                                           ref.iso_stress_area(c.screw.d, c.screw.pitch))
        items += [(f"{c.tag} preload at proof share", _cell("preload_strength_N", c.i), c.row.preload_strength_N,
                   strength, TOL_ALGEBRA),
                  (f"{c.tag} preload used", _cell("preload_N", c.i), c.row.preload_N,
                   min(strength, c.row.preload_strip_N), TOL_ALGEBRA)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_capacity_chain_given_preload():
    """Rows 24-27 and 31-33, fed the engine's preload: head pressure on the annulus and its verdict, torque per
    screw (line-contact bound mu·F·d times the clamp factor), screws needed, clamp torque, safety factor on the
    coupling torque and tightening torque K·F·d."""
    items = []
    for c in _cases():
        F = c.row.preload_N
        p = ref.head_bearing_pressure(F, c.screw.head_dk, c.screw.hole)
        t_per = ref.clamp_torque_per_screw(c.setup.friction, F, c.setup.shaft_mm, c.setup.clamp_factor)
        need = ref.screws_needed(c.setup.max_torque_Nm * c.setup.safety_factor, c.row.torque_per_screw_Nm)
        head_text = "OK" if p <= c.res.clamps.al_head_limit_MPa else "Use a hardened washer"
        items += [
            (f"{c.tag} head pressure", _cell("head_pressure_MPa", c.i), c.row.head_pressure_MPa, p, TOL_ALGEBRA),
            text_item(f"{c.tag} head check", _cell("head_check", c.i), c.row.head_check, head_text),
            (f"{c.tag} torque per screw", _cell("torque_per_screw_Nm", c.i), c.row.torque_per_screw_Nm, t_per,
             TOL_ALGEBRA),
            (f"{c.tag} screws needed", _cell("screws_needed", c.i), c.row.screws_needed, need, 0.0),
            (f"{c.tag} clamp torque", _cell("clamp_torque_Nm", c.i), c.row.clamp_torque_Nm, c.row.screws_needed * t_per,
             TOL_ALGEBRA),
            (f"{c.tag} SF on coupling torque", _cell("sf_coupling", c.i), c.row.sf_coupling,
             c.row.screws_needed * t_per / c.setup.max_torque_Nm, TOL_ALGEBRA),
            (f"{c.tag} tightening torque", _cell("tightening_Nm", c.i), c.row.tightening_Nm,
             ref.tightening_torque(c.setup.nut_factor, F, c.screw.d), TOL_ALGEBRA),
        ]
    assert_all(items)


# --------------------------------------------------------------------------- fit, verdicts, recommendation
@pytest.mark.family("clamps")
def test_screw_fit_and_works():
    """Rows 28-30 and 38: spacing, screws that fit (greedy placement with the axial margin at both ends), 'works'
    (geometry OK and needed <= fit, fed the engine's needed count) and the vent-port flag (hex key across corners
    below the M6 port's minor diameter)."""
    items = []
    for c in _cases():
        spacing = c.screw.head_dk + HEAD_GAP_MM
        fit = ref.screws_fit(c.setup.clamp_length_mm, c.setup.axial_margin_mm, c.geo["cbore_dia_mm"], spacing)
        works = 1 if (ref.geometry_ok(c.setup, c.geo) == 1 and c.row.screws_needed <= fit) else 0
        items += [
            (f"{c.tag} screw spacing", _cell("pitch_axial_mm", c.i), c.row.pitch_axial_mm, spacing, TOL_ALGEBRA),
            (f"{c.tag} screws that fit", _cell("screws_fit", c.i), c.row.screws_fit, fit, 0.0),
            (f"{c.tag} works", _cell("works", c.i), c.row.works, works, 0.0),
            (f"{c.tag} vent port ok", _cell("vent_port_ok", c.i), c.row.vent_port_ok,
             1 if ref.hex_key_passes_port(c.screw.hex_s) else 0, 0.0),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_recommendation():
    """Shaft clamps C47-C49, end to end from the reference (reference preload, count, fit): the recommended size is
    the first that works, with the same screw count and class. The screw length inside the text is the subject of
    test_screw_length_meets_engagement_and_stays_inside, so the text is built with the engine's own length."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c = res.clamps
        idx, rows = ref.recommend(_setup(inp, res))
        items.append(text_item(f"{sc} recommended size", "Shaft clamps!C47", str(_pick_name(c.index)),
                               str(_pick_name(idx))))
        if idx and c.index == idx:
            length = c.table[idx - 1].length_mm
            text = f"ISO 4762 {ref.ISO_SCREWS[idx - 1].name} x {length:g}, class {CLASS_TEXT[inp.clamps.screw_class]}"
            items += [text_item(f"{sc} recommendation text", "Shaft clamps!C48", c.recommended, text),
                      (f"{sc} screws per clamp", "Shaft clamps!C49", c.screws, rows[idx - 1]["screws_needed"], 0.0)]
        elif not idx:
            items.append(text_item(f"{sc} recommendation text", "Shaft clamps!C48", c.recommended,
                                   "None: enlarge the boss or the clamp length"))
    assert_all(items)


@pytest.mark.family("clamps")
def test_selected_size_summary_and_layout():
    """Shaft clamps C50-C55 and C72-C81 for the recommended size: tightening torque, capacity and SF (fed the
    engine's preload), head check, hex key, vent-port text and cut layout (offset, spacing, first screw centred in
    the clamp length with its counterbore keeping the axial margin, counterbore, grip, tap drill, thread available).
    With no recommendation every selection and layout cell must be blank. The relief-cut depth is boss OD - hinge."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        relief = (f"{ci.relief_mm:.1f} mm wide at {ci.clamp_length_mm:.1f} mm from the free end, "
                  f"{ci.boss_od_mm - ci.hinge_mm:.1f} mm deep from the slit side")
        items.append(text_item(f"{sc} relief cut", "Shaft clamps!C81", c.layout_relief, relief))
        if not c.index:
            items += [text_item(f"{sc} blank {f} when nothing fits", f"Shaft clamps!{cell}", str(getattr(c, f)), "")
                      for f, cell in SELECTION_CELLS.items()]
            continue
        i = c.index - 1
        s, row = ref.ISO_SCREWS[i], c.table[i]
        setup = _setup(inp, res)
        geo = ref.clamp_geometry(setup, s)
        n = row.screws_needed
        t_per = ref.clamp_torque_per_screw(setup.friction, row.preload_N, setup.shaft_mm, setup.clamp_factor)
        spacing = s.head_dk + HEAD_GAP_MM
        first = (ci.clamp_length_mm - (n - 1) * spacing) / 2.0
        vent = "Yes: the key fits the 4 mm limit" if ref.hex_key_passes_port(s.hex_s) else "No: key too large"
        head = ("OK" if ref.head_bearing_pressure(row.preload_N, s.head_dk, s.hole) <= c.al_head_limit_MPa
                else "Use a hardened washer")
        items += [
            (f"{sc} tightening torque", "Shaft clamps!C50", c.tightening_Nm,
             ref.tightening_torque(setup.nut_factor, row.preload_N, s.d), TOL_ALGEBRA),
            (f"{sc} clamp capacity", "Shaft clamps!C52", c.capacity_Nm, n * t_per, TOL_ALGEBRA),
            (f"{sc} SF on coupling torque", "Shaft clamps!C53", c.sf_coupling, n * t_per / setup.max_torque_Nm,
             TOL_ALGEBRA),
            text_item(f"{sc} head check", "Shaft clamps!C54", c.head_check, head),
            (f"{sc} hex key", "Shaft clamps!C51", c.hex_mm, s.hex_s, 0.0),
            text_item(f"{sc} vent port", "Shaft clamps!C55", c.vent_port, vent),
            (f"{sc} layout offset", "Shaft clamps!C72", c.layout_offset_mm, geo["offset_mm"], TOL_ALGEBRA),
            (f"{sc} layout spacing", "Shaft clamps!C73", c.layout_pitch_mm, spacing, TOL_ALGEBRA),
            (f"{sc} layout first screw", "Shaft clamps!C74", c.layout_first_mm, first, TOL_ALGEBRA),
            flag_item(f"{sc} first counterbore keeps the {ci.axial_margin_mm:g} mm axial margin "
                      f"(edge at {c.layout_first_mm - geo['cbore_dia_mm'] / 2:.3f} mm)", "Shaft clamps!C74",
                      c.layout_first_mm - geo["cbore_dia_mm"] / 2 >= ci.axial_margin_mm - ref.LENGTH_TOL_MM),
            (f"{sc} layout counterbore dia", "Shaft clamps!C75", c.layout_cbore_dia_mm, geo["cbore_dia_mm"],
             TOL_ALGEBRA),
            (f"{sc} layout counterbore depth", "Shaft clamps!C76", c.layout_cbore_depth_mm, geo["cbore_depth_mm"],
             TOL_ALGEBRA),
            (f"{sc} layout grip", "Shaft clamps!C77", c.layout_grip_mm, geo["grip_mm"], TOL_ALGEBRA),
            (f"{sc} layout tap drill", "Shaft clamps!C78", c.layout_tap_drill_mm, s.tap_drill, TOL_ALGEBRA),
            (f"{sc} layout thread available", "Shaft clamps!C79", c.layout_thread_avail_mm, geo["thread_avail_mm"],
             TOL_ALGEBRA),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_head_check_threshold():
    """Row 25 verdict: 'OK' exactly when the head pressure <= the aluminium limit (M4 row): OK at equality,
    'Use a hardened washer' when the limit is one ulp lower."""
    p = run().clamps.table[2].head_pressure_MPa
    path = "materials.aluminium.al7075.head_pressure_limit_MPa"
    at = run(vary(defaults(), {path: p}))
    below = run(vary(defaults(), {path: math.nextafter(p, -math.inf)}))
    assert_all([
        text_item("limit = M4 head pressure", _cell("head_check", 2), at.clamps.table[2].head_check, "OK"),
        text_item("limit one ulp below the M4 head pressure", _cell("head_check", 2), below.clamps.table[2].head_check,
                  "Use a hardened washer"),
    ])


# --------------------------------------------------------------------------- loads, key, joint
@pytest.mark.family("clamps")
def test_design_torque_factor_and_material_constants():
    """Shaft clamps C15, C17, C22, C24, C28, C44: clamp design torque = safety factor × the cold-high coupling
    torque (README: the clamp alone holds twice it); clamp factor by type; boss radius; aluminium shear (MMPDS) and
    screw proof stress (ISO 898-1 / 3506-1)."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, setup = res.clamps, _setup(inp, res)
        items += [
            (f"{sc} max torque = cold high with +variation", "Shaft clamps!C15 / Metal design!C10", c.max_torque_Nm,
             res.metal.torque_cold_high_Nm, TOL_ALGEBRA),
            (f"{sc} torque the clamp must hold", "Shaft clamps!C17", c.required_Nm,
             setup.safety_factor * setup.max_torque_Nm, TOL_ALGEBRA),
            (f"{sc} clamp factor used", "Shaft clamps!C22", c.clamp_factor, setup.clamp_factor, TOL_ALGEBRA),
            (f"{sc} boss radius", "Shaft clamps!C44", c.boss_radius_mm, setup.boss_od_mm / 2, TOL_ALGEBRA),
            (f"{sc} {setup.alloy} shear strength", "Shaft clamps!C24", c.al_shear_MPa, ref.AL_SHEAR_MPA[setup.alloy],
             TOL_ALGEBRA),
            (f"{sc} class {setup.screw_class} proof stress", "Shaft clamps!C28", c.screw_proof_MPa,
             ref.PROOF_STRESS_MPA[setup.screw_class], TOL_ALGEBRA),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_key_backup():
    """Shaft clamps C60-C61: key bearing pressure at the clamp design torque, p = 2T/(d·h·L) in SI, and
    allowable over actual."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        p = ref.key_bearing_pressure_MPa(c.required_Nm, inp.coupling.bore_mm, ci.key_contact_mm, ci.clamp_length_mm)
        items += [(f"{sc} key bearing pressure", "Shaft clamps!C60", c.key_pressure_MPa, p, TOL_ALGEBRA),
                  (f"{sc} key allowable over actual", "Shaft clamps!C61", c.key_sf, c.al_key_allow_MPa / p,
                   TOL_ALGEBRA)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_adapter_joint():
    """Shaft clamps C67-C69, stage-wise: the allowable preload per M3 is the M3 row's preload used (Clamp screw
    sizes!D23, itself checked by test_preload); slip torque mu·n·F·D_bc/2 and its ratio to the clamp design
    torque."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        t_joint = ref.flange_friction_torque_Nm(ci.joint_friction, ci.joint_screws, c.joint_preload_N,
                                                ci.joint_bolt_circle_mm)
        items += [(f"{sc} allowable preload per M3", "Shaft clamps!C67", c.joint_preload_N, c.table[1].preload_N,
                   TOL_ALGEBRA),
                  (f"{sc} joint slip torque", "Shaft clamps!C68", c.joint_torque_Nm, t_joint, TOL_ALGEBRA),
                  (f"{sc} joint torque over clamp design torque", "Shaft clamps!C69", c.joint_sf,
                   t_joint / c.required_Nm, TOL_ALGEBRA)]
    assert_all(items)


# --------------------------------------------------------------------------- README claims
def _pick_at(boss_mm: float, length_mm: float):
    """(reference pick, reference screws, engine pick, engine screws, geometry-OK sizes by reference and engine)."""
    inp = vary(defaults(), {"clamps.boss_od_mm": boss_mm, "clamps.clamp_length_mm": length_mm})
    res = run(inp)
    setup = _setup(inp, res)
    idx, rows = ref.recommend(setup)
    ref_fit = [s.name for s, r in zip(ref.ISO_SCREWS, rows) if r["geometry_ok"]]
    eng_fit = [s.name for s, r in zip(ref.ISO_SCREWS, res.clamps.table) if r.geometry_ok]
    return (_pick_name(idx), rows[idx - 1]["screws_needed"] if idx else 0, _pick_name(res.clamps.index),
            res.clamps.screws if res.clamps.index else 0, ref_fit, eng_fit)


@pytest.mark.family("clamps")
def test_readme_22mm_two_m3_need_14_5mm():
    """README 'What the default design currently shows' and the Shaft clamps!C35 help: at a 22 mm boss two M3
    screws need a 14.5 mm clamp. Reference and engine agree on M3 x 2 at 14.5 mm and on 'none' at 14.4 mm."""
    items = []
    for length, size, screws in ((14.5, "M3", 2), (14.4, None, 0)):
        ref_pick, ref_n, eng_pick, eng_n, _, _ = _pick_at(22, length)
        items += [flag_item(f"22 mm boss, {length} mm clamp: reference pick {ref_pick} x {ref_n}, "
                            f"README {size} x {screws}", "README / Shaft clamps!C35",
                            ref_pick == size and ref_n == screws),
                  text_item(f"22 mm boss, {length} mm clamp: engine pick", "Shaft clamps!C48", str(eng_pick),
                            str(size)),
                  (f"22 mm boss, {length} mm clamp: engine screws", "Shaft clamps!C49", eng_n, screws, 0.0)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_readme_22mm_only_m3_fits():
    """README ('At 22 mm only M3 fits') and the Shaft clamps!C35 help say only M3 fits a 22 mm boss. Checked two
    ways: the sizes whose geometry passes at 22 mm (wall, head seat, grip, thread), and the first pick at a clamp
    long enough for three screws (18 mm), which the claim implies is M3. The engine is compared with the reference
    in the same check, so a disagreement between them would show here too."""
    _, _, _, _, ref_fit, eng_fit = _pick_at(22, 10)
    ref_pick, ref_n, eng_pick, eng_n, _, _ = _pick_at(22, 18)
    assert_all([
        flag_item(f"sizes whose geometry passes at a 22 mm boss: reference {ref_fit}, README ['M3']",
                  "README / Shaft clamps!C35", ref_fit == ["M3"]),
        text_item("sizes whose geometry passes at a 22 mm boss: engine vs reference",
                  f"Clamp screw sizes!C{TABLE_ROWS['geometry_ok']}:G{TABLE_ROWS['geometry_ok']}", str(eng_fit),
                  str(ref_fit)),
        flag_item(f"first pick at a 22 mm boss, 18 mm clamp: reference {ref_pick} x {ref_n}, README implies M3",
                  "README / Shaft clamps!C35", ref_pick == "M3"),
        text_item(f"first pick at a 22 mm boss, 18 mm clamp: engine ({eng_pick} x {eng_n}) vs reference",
                  "Shaft clamps!C48:C49", f"{eng_pick} x {eng_n}", f"{ref_pick} x {ref_n}"),
    ])
