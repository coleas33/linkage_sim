"""M1 Task 2: the Calculator torque model against independent references.

Groups:
  * chain  - Calculator!C66:C94 re-derived term by term from the textbook planar formulas
             (``audit.references.planar``), same algebra, TOL_ALGEBRA; the sole same-algebra owner
             of those cells.
  * 2D     - the engine's 2D pull-out, torque-angle harmonics and quarter-pitch evaluation against
             the exact 2D field of the real flat-block section (``audit.references.field2d``), in
             free space and with polygonal steel, across face gaps 0.5, 1.4 and 3.0 mm, TOL_MODEL.
  * S_n    - the planar (unrolled) geometry factors against the exact cylindrical solution, and
             their planar limit.

The reference sanity tests live in ``test_torque2d_reference_sanity.py``. A failing check is a
candidate finding for the report (Task 8); the engine is never edited.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from audit.common import TOL_ALGEBRA, TOL_MODEL, TOL_PEAK, ScenarioCache, defaults, mismatch, rel_err, run, vary
from audit.references.field2d import MM, CouplingSection, IronPolygons, cylindrical_s_factor, pullout
from audit.references.planar import planar_s_factor, square_wave_harmonic

#: Planar limit at R_gap = 28 m with t_i = t_o and R_gap mid-gap: the first-order curvature
#: terms cancel between the rings, so the residual is O((dr / R_gap)^2) = (4.3 mm / 28 m)^2 ~ 2e-8.
TOL_PLANAR_LIMIT = 1e-6

CIRCUITS = {0: "free space", 1: "steel"}
#: Face gaps (Metal design!C119). The inner block corners stand 0.373 mm proud of the face radius, so a
#: flat-block section exists only above that (0.3 mm would overlap the blocks: corner gap C9 = -0.07 mm).
#: 0.5 mm leaves a 0.127 mm corner gap, the smallest buildable case; 1.4 mm is the default; 3.0 mm is wide.
FACE_GAPS_MM = (0.5, 1.4, 3.0)
DEFAULT_GAP_MM = 1.4
#: Steel placement that decides the steel verdicts (see ``steel_from``); ignored in free space.
VERDICT_STEEL = "as-built"
#: Calculator rows per harmonic: k_n, B_in, B_on, S_n steel, S_n free, tau_n.
HARMONIC_ROWS = {1: ("C71", "C72", "C73", "C74", "C75", "C76"),
                 3: ("C77", "C78", "C79", "C80", "C81", "C82"),
                 5: ("C83", "C84", "C85", "C86", "C87", "C88")}
#: Term-by-term cases: both circuits at defaults, plus branches defaults never reach: another pole
#: count at the cold end, the fill clamp min(1, ...) (20 poles: blocks wider than the pitch), and
#: unequal inner/outer magnets (outer thinner, narrower and weaker) at the hot end with steel and at
#: 50 °C in free space with manual magnets in both rings, so an inner/outer mix-up would show in
#: either circuit's S_n branch.
CHAIN_CASES = {
    "steel": {},
    "free": {"coupling.backiron": 0},
    "steel-12-poles-cold": {"coupling.npole": 12, "coupling.op_temp_C": -40},
    "free-20-poles-fill-clamped": {"coupling.backiron": 0, "coupling.npole": 20},
    "steel-unequal-rings-hot": {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_thickness_mm": 2.5,
                                "coupling.magnets.manual_outer_width_mm": 5.0,
                                "coupling.magnets.manual_outer_br_T": 1.2, "coupling.op_temp_C": 100},
    "free-unequal-rings": {"coupling.backiron": 0, "coupling.magnets.part_inner": "",
                           "coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 5.0,
                           "coupling.magnets.manual_outer_thickness_mm": 2.5,
                           "coupling.magnets.manual_outer_br_T": 1.2},
}


# --------------------------------------------------------------------------- helpers
#: One engine run per (circuit, face gap) pair the 2D and S_n checks use.
ENGINE_CASES = ScenarioCache({f"{CIRCUITS[b]}, face gap {g} mm": {"coupling.backiron": b, "metal.face_gap_mm": g}
                              for b in CIRCUITS for g in FACE_GAPS_MM})


def engine_case(backiron: int, face_gap_mm: float):
    """(inputs, Calculator results) at defaults with the back-iron circuit (Calculator!C6) and face gap
    (Metal design!C119) changed. Only the pairs in ENGINE_CASES exist (KeyError otherwise)."""
    inp, res = ENGINE_CASES[f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm"]
    return inp, res.model


def section_from(m, n_poles: int) -> CouplingSection:
    """Reference geometry from the engine's resolved fields, remanence at 20 °C."""
    return CouplingSection(
        n_poles=n_poles,
        inner_back_mm=m.inner_face_radius_mm - m.inner_thickness_mm,                  # C54 - C20
        inner_thickness_mm=m.inner_thickness_mm, inner_width_mm=m.inner_width_mm,     # C20, C19
        outer_face_mm=m.outer_face_apothem_mm,                                        # C56
        outer_thickness_mm=m.outer_thickness_mm, outer_width_mm=m.outer_width_mm,     # C30, C29
        br_inner_T=m.inner_br_T, br_outer_T=m.outer_br_T)                             # C21, C31


def steel_from(inp, m, placement: str) -> IronPolygons:
    """Polygonal steel, one flat per pole, turning with its ring.
    "as-built" (decides the verdict): the machined hub flats at C8 minus the inner bondline
    (Metal design!C120) and the cup pocket flats at C60 plus the outer bondline (C121), the steel the
    engine's own geometry and mass rows describe. "at-backs": steel touching the magnet backs
    (hub C54 - C20, cup C60), the idealisation inside the engine's S_iron (t_i + t_o + g, no bondline)."""
    if placement == "as-built":
        hub = inp.coupling.inner_back_apothem_mm - inp.metal.bond_inner_mm
        cup = m.outer_back_apothem_mm + inp.metal.bond_outer_mm
    elif placement == "at-backs":
        hub, cup = m.inner_face_radius_mm - m.inner_thickness_mm, m.outer_back_apothem_mm
    else:
        raise ValueError(f"unknown steel placement {placement!r}")
    return IronPolygons(hub, cup, inp.coupling.npole)


@functools.lru_cache(maxsize=None)
def reference_pullout(backiron: int, face_gap_mm: float, placement: str):
    """Exact 2D pull-out of the engine's flat-block section (cached; a polygonal-steel case takes about 0.6 s).
    Always pass ``placement`` positionally so every caller shares one cache entry per case."""
    inp, m = engine_case(backiron, face_gap_mm)
    iron = steel_from(inp, m, placement) if backiron else None
    return pullout(section_from(m, inp.coupling.npole), iron)


def reference_2d_nm(backiron: int, face_gap_mm: float, placement: str) -> float:
    """Reference pull-out per metre times the active length C33 [N·m]."""
    _, m = engine_case(backiron, face_gap_mm)
    return reference_pullout(backiron, face_gap_mm, placement).torque_per_m * m.active_length_mm * MM


def engine_2d_torque_20c(m) -> float:
    """Engine pull-out at 20 °C before the end-effect and calibration factors: C94 / (C92 * C42)."""
    return m.pullout_20C_Nm / (m.f_end * m.f_cal)


def engine_harmonic_torque_20c(m, n: int) -> float:
    """Engine torque-angle amplitude of harmonic n at 20 °C: tau_n / sin(n pi/2) * C90, times
    (C21 * C31) / (C69 * C70) to take Br from the operating temperature back to 20 °C."""
    tau = {1: m.tau1_Pa, 3: m.tau3_Pa, 5: m.tau5_Pa}[n]
    return (tau / math.sin(n * math.pi / 2) * m.area_lever_m3
            * (m.inner_br_T * m.outer_br_T) / (m.br_inner_T_op * m.br_outer_T_op))


@functools.lru_cache(maxsize=None)
def prototype_ratio() -> float:
    """Engine / exact 2D torque of the Calibration sheet's no-iron prototype: Calibration!C43 against the
    flat-block section (20 x B842SH on a 9.85 mm apothem, flat gap C32, Br at the test temperature C36)
    times the magnet length C18."""
    inp = defaults()
    c, ci = run(inp).calibration, inp.calibration
    sec = CouplingSection(int(c.poles_per_ring), c.face_radius_mm - ci.magnet_thickness_mm,       # C13, C28 - C20
                          ci.magnet_thickness_mm, ci.magnet_width_mm, c.outer_face_apothem_mm,   # C20, C19, C31
                          ci.magnet_thickness_mm, ci.magnet_width_mm, c.br_test_T, c.br_test_T)  # C36
    return c.torque_2d_Nm / (pullout(sec).torque_per_m * ci.magnet_length_mm * MM)


def s_factor_pair(m, n: int, n_poles: int, backiron: int) -> tuple[float, float, str]:
    """(engine S_n, exact cylindrical S for arcs with the engine's radii, workbook cell).
    Arcs: inner C54 - C20 .. C54, outer C56 .. C60; steel circles at C54 - C20 and C60."""
    r1a = m.inner_face_radius_mm - m.inner_thickness_mm
    radii = (r1a, m.inner_face_radius_mm, m.outer_face_apothem_mm, m.outer_back_apothem_mm)
    kind = "iron" if backiron else "free"
    cyl = cylindrical_s_factor(n * n_poles // 2, *radii, m.gap_radius_mm, iron=(radii[0], radii[3]) if backiron else None)
    cell = HARMONIC_ROWS[n][3] if backiron else HARMONIC_ROWS[n][4]
    return getattr(m, f"s{n}_{kind}"), cyl, cell


def torque_chain_reference(inp, m) -> dict:
    """Calculator!C66:C94 re-derived from first principles: {cell: (engine field, reference value)}.

    Inputs: C5, C6, C8, C10, C41, C43 (inputs); C19-C21, C29-C31, C33, C35, C42 (resolved values);
    and the geometry rows C56, C57, C64, owned by the geometry checks. Fill = block width / pole pitch
    at the block's mid-thickness radius, clamped at 1; Br(T) = Br20 (1 + alpha (T - 20)); pole pitch
    tau_p = 2 pi R_g / N and k_n = n pi / tau_p; B_n from ``square_wave_harmonic``; S_n from
    ``planar_s_factor``; tau_n = B_in B_on / (2 mu0) S_n sin(n pi/2) (the engine's quarter-pitch
    evaluation, examined separately by the peak checks); torque = sum tau_n * 2 pi R_g^2 L;
    f_end = 1 - c_end tau_p / L; pull-out = T2D f_end f_cal, and at 20 °C times Br20^2 / Br(T)^2."""
    c = inp.coupling
    n_poles, backed = c.npole, c.backiron == 1
    t_i, w_i, t_o, w_o = m.inner_thickness_mm, m.inner_width_mm, m.outer_thickness_mm, m.outer_width_mm
    fill_i = min(1.0, w_i * n_poles / (2 * math.pi * (c.inner_back_apothem_mm + t_i / 2)))
    fill_o = min(1.0, w_o * n_poles / (2 * math.pi * (m.outer_face_apothem_mm + t_o / 2)))
    temp_factor = 1 + m.alpha_br_per_C * (c.op_temp_C - 20)
    br_i, br_o = m.inner_br_T * temp_factor, m.outer_br_T * temp_factor
    pitch_mm = 2 * math.pi * m.gap_radius_mm / n_poles
    ref = {"C66": ("fill_inner", fill_i), "C67": ("fill_outer", fill_o),
           "C69": ("br_inner_T_op", br_i), "C70": ("br_outer_T_op", br_o)}
    tau = 0.0
    for n, (c_k, c_bi, c_bo, c_si, c_sf, c_tau) in HARMONIC_ROWS.items():
        k = n * math.pi / (pitch_mm * MM)
        b_i, b_o = square_wave_harmonic(br_i, n, fill_i), square_wave_harmonic(br_o, n, fill_o)
        s_iron = planar_s_factor(k, t_i * MM, t_o * MM, m.face_gap_mm * MM, backed=True)
        s_free = planar_s_factor(k, t_i * MM, t_o * MM, m.face_gap_mm * MM, backed=False)
        tau_n = b_i * b_o / (2 * c.mu0) * (s_iron if backed else s_free) * math.sin(n * math.pi / 2)
        tau += tau_n
        ref.update({c_k: (f"k{n}", k), c_bi: (f"b_i{n}", b_i), c_bo: (f"b_o{n}", b_o),
                    c_si: (f"s{n}_iron", s_iron), c_sf: (f"s{n}_free", s_free), c_tau: (f"tau{n}_Pa", tau_n)})
    area_lever = 2 * math.pi * (m.gap_radius_mm * MM) ** 2 * (m.active_length_mm * MM)
    t2d = tau * area_lever
    f_end = 1 - c.c_end * pitch_mm / m.active_length_mm
    pull = t2d * f_end * m.f_cal
    ref.update({"C89": ("tau_Pa", tau), "C90": ("area_lever_m3", area_lever), "C91": ("torque_2d_Nm", t2d),
                "C92": ("f_end", f_end), "C93": ("pullout_Nm", pull),
                "C94": ("pullout_20C_Nm", pull * (m.inner_br_T * m.outer_br_T) / (br_i * br_o))})
    return ref


# --------------------------------------------------------------------------- chain: term by term
@pytest.mark.parametrize("case", list(CHAIN_CASES))
@pytest.mark.family("torque")
def test_calculator_torque_chain_rederived(case):
    """Every Calculator row C66:C94 (fill, Br(T), k_n, B_n, S_n for both circuits, tau_n, sum, area x
    lever, 2D torque, end factor, pull-out at T and at 20 °C) equals its first-principles value
    (``torque_chain_reference``). Same algebra: TOL_ALGEBRA. One check per case lists every failing row."""
    inp = vary(defaults(), CHAIN_CASES[case])
    m = run(inp).model
    failures = [mismatch(f"{field}, {case}", f"Calculator!{cell}", getattr(m, field), value, TOL_ALGEBRA)
                for cell, (field, value) in torque_chain_reference(inp, m).items()
                if rel_err(getattr(m, field), value) > TOL_ALGEBRA]
    assert not failures, " | ".join(failures)


# --------------------------------------------------------------------------- 2D: pull-out
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_pullout_2d_matches_flat_block_reference(backiron, face_gap_mm):
    """2D pull-out at 20 °C before end effect and calibration, C94 / (C92 * C42), against the true
    pull-out (max over angle) of the exact 2D flat-block section times C33. Steel: as-built polygonal
    steel with the bondlines (``steel_from``), which decides the steel verdict."""
    _, m = engine_case(backiron, face_gap_mm)
    engine, reference = engine_2d_torque_20c(m), reference_2d_nm(backiron, face_gap_mm, VERDICT_STEEL)
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"2D pull-out at 20 °C, {CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C94/(C92*C42) vs 2D field x C33", engine, reference, TOL_MODEL)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.family("torque")
def test_pullout_2d_steel_at_magnet_backs(face_gap_mm):
    """Diagnostic companion of the steel verdict above: the same comparison with the steel moved onto
    the magnet backs, the idealisation inside the engine's S_iron. The as-built check decides the
    verdict; comparing the two separates the formula's own error from the bondline it omits. TOL_MODEL."""
    _, m = engine_case(1, face_gap_mm)
    engine, reference = engine_2d_torque_20c(m), reference_2d_nm(1, face_gap_mm, "at-backs")
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"2D pull-out at 20 °C, steel at the magnet backs (no bondline), face gap {face_gap_mm} mm",
        "Calculator!C94/(C92*C42) vs 2D field x C33", engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_calibration_prototype_2d_matches_flat_block_reference():
    """The Calibration sheet's model of the no-iron prototype (20 x B842SH on a 9.85 mm apothem, 1.4 mm
    flat-face gap, 20 °C): its 2D torque before end effect and calibration (Calibration!C43) against
    the exact 2D flat-block section times the magnet length (C18). This 2D value sets the one-point
    correction Calibration!C8 and C9. TOL_MODEL."""
    ratio = prototype_ratio()
    assert abs(ratio - 1) <= TOL_MODEL, mismatch(
        "prototype 2D torque at the test temperature, free space, engine / exact 2D",
        "Calibration!C43 vs 2D field x C18", ratio, 1.0, TOL_MODEL)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.family("torque")
def test_free_space_2d_error_transfers_from_prototype(face_gap_mm):
    """The free-space design takes the prototype's one-point correction (C42 = Calibration!C9), which
    cancels the 2D model error only if that error is the same at the design as at the prototype. So
    (engine / exact 2D at the design) / (engine / exact 2D at the prototype) must be 1 within TOL_MODEL:
    this is the 2D part of the calibrated free-space prediction's error."""
    _, m = engine_case(0, face_gap_mm)
    design = engine_2d_torque_20c(m) / reference_2d_nm(0, face_gap_mm, VERDICT_STEEL)
    assert abs(design / prototype_ratio() - 1) <= TOL_MODEL, mismatch(
        f"free-space 2D error at face gap {face_gap_mm} mm relative to the prototype's",
        "Calculator!C94/(C92*C42) and Calibration!C43 vs 2D field", design, prototype_ratio(), TOL_MODEL)


# --------------------------------------------------------------------------- 2D: angle and harmonics
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_quarter_pitch_evaluation_is_true_peak(backiron, face_gap_mm):
    """The engine evaluates every harmonic at the fundamental's pull-out angle (theta_e = pi/2, hence
    sin(n pi/2)). In the exact 2D section the torque there must be within TOL_PEAK of the true max over
    angle; the observed peak angle is in the message."""
    ref = reference_pullout(backiron, face_gap_mm, VERDICT_STEEL)
    assert abs(ref.at_quarter / ref.torque_per_m - 1) <= TOL_PEAK, mismatch(
        f"2D torque at theta_e=90° vs true peak at {math.degrees(ref.theta_e_peak):.3f}° electrical, "
        f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C76,C82,C88 sin(n·pi/2)", ref.at_quarter, ref.torque_per_m, TOL_PEAK)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_engine_harmonic_series_peaks_at_quarter_pitch(backiron, face_gap_mm):
    """The engine's own curve tau(theta) = sum of tau_n / sin(n pi/2) * sin(n theta) must peak where it is
    evaluated, or C89 is not the pull-out of its own model (a positive third-harmonic amplitude above
    a_1 / 9 would make the top double-humped). Its max over angle (dense scan + bounded search) must
    equal C89. TOL_ALGEBRA."""
    _, m = engine_case(backiron, face_gap_mm)
    amp = {n: tau / math.sin(n * math.pi / 2) for n, tau in ((1, m.tau1_Pa), (3, m.tau3_Pa), (5, m.tau5_Pa))}
    curve = lambda th: sum(a * math.sin(n * th) for n, a in amp.items())
    grid = np.linspace(0.0, math.pi, 2001)
    i = int(np.argmax([curve(x) for x in grid]))
    best = minimize_scalar(lambda x: -curve(x), bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 2000)]),
                           method="bounded", options={"xatol": 1e-12})
    assert rel_err(m.tau_Pa, -best.fun) <= TOL_ALGEBRA, mismatch(
        f"engine shear at 90° vs max of its own harmonic curve (at {math.degrees(best.x):.4f}°), "
        f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C89 vs C76,C82,C88", m.tau_Pa, -best.fun, TOL_ALGEBRA)


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_harmonic_torque_amplitudes(backiron, n):
    """Per-harmonic torque-angle amplitude at 20 °C (engine tau_n / sin(n pi/2) * C90 scaled to 20 °C)
    against the sine coefficient a_n of the exact 2D torque-angle curve times C33 (default gap, as-built
    steel). The difference is measured against the fundamental a_1: harmonics 3 and 5 are about 1 % of
    a_1, so a relative bound on them would flag differences that cannot move the pull-out.
    |a_n engine - a_n reference| <= TOL_MODEL * a_1 reference."""
    _, m = engine_case(backiron, DEFAULT_GAP_MM)
    scale = m.active_length_mm * MM
    harmonics = reference_pullout(backiron, DEFAULT_GAP_MM, VERDICT_STEEL).harmonics
    engine, reference, a1 = engine_harmonic_torque_20c(m, n), harmonics[n] * scale, harmonics[1] * scale
    cell = HARMONIC_ROWS[n][5]
    assert abs(engine - reference) <= TOL_MODEL * abs(a1), mismatch(
        f"harmonic {n}, {CIRCUITS[backiron]}: (engine a_n - reference a_n) / reference a_1; "
        f"engine a_n = {engine:.6g} N·m, reference a_n = {reference:.6g} N·m, reference a_1 = {a1:.6g} N·m",
        f"Calculator!{cell}*C90*(C21*C31)/(C69*C70) vs 2D field x C33", (engine - reference) / a1, 0.0, TOL_MODEL)


@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_omitted_harmonics_negligible(backiron):
    """The engine keeps harmonics 1, 3, 5 only (C89 = C76 + C82 + C88). The reference torque at 90° minus
    its own n <= 5 terms is the part the engine drops (n >= 7, plus the even cogging terms, which vanish
    at 90°); it must stay within TOL_MODEL of the true pull-out (default gap, as-built steel)."""
    ref = reference_pullout(backiron, DEFAULT_GAP_MM, VERDICT_STEEL)
    kept = sum(ref.harmonics[n] * math.sin(n * math.pi / 2) for n in range(1, 6))
    tail = (ref.at_quarter - kept) / ref.torque_per_m
    assert abs(tail) <= TOL_MODEL, mismatch(
        f"reference harmonic tail n >= 7 at 90° as a fraction of the true pull-out, {CIRCUITS[backiron]}",
        "Calculator!C89 (harmonics 1, 3, 5 only)", tail, 0.0, TOL_MODEL)


# --------------------------------------------------------------------------- S_n: curvature and planar limit
@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_planar_s_factor_matches_cylindrical(backiron, face_gap_mm, n):
    """Planar (unrolled, k = n (N/2) / R_gap) S_n against the exact cylindrical factor for arcs with the
    engine's radii (inner C54 - C20 .. C54, outer C56 .. C60, steel at C54 - C20 and C60), same
    normalisation: torque = B_i B_o / (2 mu0) * S * 2 pi C64^2. TOL_MODEL."""
    inp, m = engine_case(backiron, face_gap_mm)
    engine, reference, cell = s_factor_pair(m, n, inp.coupling.npole, backiron)
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"S_{n} planar vs cylindrical, {CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        f"Calculator!{cell} (C71, C54, C56, C60, C64)", engine, reference, TOL_MODEL)


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_planar_s_factor_is_exact_planar_limit(backiron, n):
    """At a planar-like design (20000 poles on a 28 m hub, R_gap 28.004 m, k1 = 357 /m like the default)
    the engine's S_n, including the free-space factor's /2 and the steel sinh form, must equal the exact
    cylindrical factor to O((dr / R_gap)^2). TOL_PLANAR_LIMIT."""
    m = run(vary(defaults(), {"coupling.npole": 20000, "coupling.inner_back_apothem_mm": 28000.0,
                              "coupling.backiron": backiron})).model
    engine, reference, cell = s_factor_pair(m, n, 20000, backiron)
    assert rel_err(engine, reference) <= TOL_PLANAR_LIMIT, mismatch(
        f"S_{n} planar limit (R_gap 28 m), {CIRCUITS[backiron]}", f"Calculator!{cell}", engine, reference,
        TOL_PLANAR_LIMIT)
