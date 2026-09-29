"""Temperature design, part A: demagnetization onsets and ratings, adhesive limits and bond loads, Volkersen screen.

Under test: ``magcoupling.temperature`` (``volkersen_peak_shear_MPa`` and the demag, adhesive, mismatch and
summary-margin parts of ``compute()``), the Calculator magnet temperature checks (C22, C32, C107, C108) and
the magnet library's ratings and Br. This file is the sole owner of Calculator C22, C32, C107, C108 and of the
Temperature design summary margins C13, C15 and the hot-day note F16. References:
``audit.references.demag_adhesive``, the slab forces Task 5 appends to Task 4's ``audit.references.slab_field2d``,
and Task 4's ``remanence`` and ``metal_stack``; none of them imports the engine. Engine values owned by other
tasks (model geometry, Calculator C21 Br20, C93/C94 pull-out, Metal design C10 cold-high torque, Temperature design
C82 block mass) are consumed as inputs and verified there. The reference self-tests live in
``test_temperature_demag_adhesive_reference_sanity.py``.

A failing check is a candidate finding for Task 8, not a bug to fix here.
"""
from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest
from scipy.optimize import brentq

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, defaults, mismatch, rel_err, run, text_item, vary
from audit.references import demag_adhesive as ref
from audit.references.metal_stack import pole_pair_frequency_Hz
from audit.references.remanence import br_ratio, torque_at
from audit.references.slab_field2d import MagnetLayer, layer_block_forces_N
from magcoupling.library import MAGNET_LIBRARY
from magcoupling.temperature import volkersen_peak_shear_MPa

#: Reverse-field case -> engine result field (DemagResults / SummaryResults).
ONSET_FIELDS = {"aligned": "onset_aligned_C", "pullout": "onset_pullout_C",
                "likepole": "onset_skipping_C", "single_ring": "onset_single_ring_C"}

#: Poisson's ratios used only for the biaxial-stiffness sensitivity (typical handbook values).
NU_NDFEB, NU_STEEL = 0.24, 0.29

HOT_DAY_NOTE_MEETS, HOT_DAY_NOTE_BELOW = "Meets it nominally (no variation allowance)", "Below it"
READING_ABOVE, READING_BELOW = "Above the lap-shear strength at the block ends", "Below the lap-shear strength"


# =========================================================================== helpers
def _cell(result_obj, name: str) -> str:
    """Workbook cell recorded in a result field's metadata."""
    return next(f.metadata["cell"] for f in fields(result_obj) if f.name == name)


def _cells(result_obj, *names: str) -> str:
    return ", ".join(_cell(result_obj, n) for n in names)


def _excel_text0(x: float) -> str:
    """Excel TEXT(x, "0") for x >= 0: round half away from zero."""
    return str(int(math.floor(x + 0.5)))


def _is_number(x) -> bool:
    return isinstance(x, (int, float))


def _demag_reference(inp, res) -> dict:
    """Reference demagnetization chain from engine inputs, Calculator C21 (Br20) and the K&J part rating."""
    d = inp.temperature.demag
    knee_args = (d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, inp.calibration.alpha_br_per_C)
    h_ref = ref.load_line_reverse_field_kA_m(res.model.inner_br_T, 1.0, inp.coupling.mu0)
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    offset = ref.calibration_offset_C(h_ref, rating, *knee_args)
    out = {"h_ref": h_ref, "t_ref": ref.knee_crossing_C(h_ref, *knee_args), "offset": offset, "rating": rating}
    for case in ONSET_FIELDS:
        out[case] = ref.knee_crossing_C(getattr(d, f"h_rev_{case}_kA_m"), *knee_args) - offset
    out["magnet_limit"] = out["likepole"] - d.design_margin_C
    return out


def _selected_candidate(inp):
    return inp.temperature.adhesive.candidates[inp.temperature.adhesive.selected - 1]


def _governing_reference(inp, res) -> float:
    return min(_demag_reference(inp, res)["magnet_limit"], _selected_candidate(inp).design_limit_C)


def _hot_day_start_C(inp) -> float:
    return inp.temperature.duty.hot_ambient_C + inp.temperature.duty.driving_rise_C


def _swing_reference(inp, res) -> float:
    cure = _selected_candidate(inp).cure_C
    return max(cure - inp.metal.min_temp_C, _governing_reference(inp, res) - cure)


def _volkersen_inputs_SI(inp, res, bondline_mm: float, biaxial: bool = False, ndfeb_cte: float | None = None) -> dict:
    """Keyword arguments for the reference Volkersen functions, in SI, from engine inputs."""
    mm = inp.temperature.mismatch
    e1, e2 = mm.ndfeb_modulus_GPa * 1e9, inp.materials.steel.modulus_GPa * 1e9
    if biaxial:
        e1, e2 = e1 / (1 - NU_NDFEB), e2 / (1 - NU_STEEL)
    cte = mm.ndfeb_cte_per_C if ndfeb_cte is None else ndfeb_cte
    return dict(G_Pa=mm.adhesive_shear_modulus_GPa * 1e9, eta_m=bondline_mm / 1000,
                E1_Pa=e1, t1_m=res.model.inner_thickness_mm / 1000, E2_Pa=e2, t2_m=res.model.hub_wall_mm / 1000,
                d_alpha_per_C=inp.materials.steel.cte_per_C - cte, dT_C=_swing_reference(inp, res),
                overlap_m=res.model.inner_length_mm / 1000)


def _reading_reference(peak_MPa: float, lap_MPa: float) -> str:
    return READING_ABOVE if peak_MPa > lap_MPa else READING_BELOW


def _inner_mid_radius_mm(inp, res) -> float:
    return inp.coupling.inner_back_apothem_mm + res.model.inner_thickness_mm / 2


def _bond_area_mm2(res) -> float:
    return res.model.inner_length_mm * res.model.inner_width_mm


def _bond_shear_reference(inp, res, lever_arm_mm: float | None = None) -> float:
    arm = _inner_mid_radius_mm(inp, res) if lever_arm_mm is None else lever_arm_mm
    return ref.bond_shear_MPa(res.metal.torque_cold_high_Nm, inp.coupling.npole, arm, _bond_area_mm2(res))


def _fatigue_screen_reference(fatigue_margin: float) -> str:
    return f"OK: {_excel_text0(fatigue_margin)}x margin" if fatigue_margin >= 4 else "CHECK"


# =========================================================================== demagnetization
@pytest.mark.family("temperature")
def test_h_ref_matches_permeance_coefficient_one_load_line():
    """C48 = |H| of a recoil-permeability-1 magnet on a Pc = 1 load line, Br20/(2 mu0) (workbook mu0 input)."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)["h_ref"]
    got = res.temperature.demag.h_ref_kA_m
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("reference-magnet reverse field", _cell(res.temperature.demag, "h_ref_kA_m"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_calibration_reference_onset_and_offset():
    """C49 (uncalibrated model onset of the Pc = 1 magnet) and C50 (offset = C49 - K&J 150 degC rating) by root-finding."""
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    dm = res.temperature.demag
    assert_all([("reference-magnet model onset", _cell(dm, "t_ref_model_C"), dm.t_ref_model_C, r["t_ref"], TOL_ALGEBRA),
                ("calibration offset", _cell(dm, "calibration_offset_C"), dm.calibration_offset_C, r["offset"], TOL_ALGEBRA)])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("case", list(ONSET_FIELDS))
def test_demag_onset_matches_root_find(case):
    """C56-C59: calibrated onset = knee crossing of the Br-scaled 3D reverse field minus the calibration offset."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)[case]
    got = getattr(res.temperature.demag, ONSET_FIELDS[case])
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch(f"demag onset ({case})", _cell(res.temperature.demag, ONSET_FIELDS[case]),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_magnet_limit():
    """C60 = skipping onset - 10 degC design margin (C51) at defaults."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)["magnet_limit"]
    got = res.temperature.demag.magnet_limit_C
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("magnet design limit", _cell(res.temperature.demag, "magnet_limit_C"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {"temperature.demag.hcj20_kA_m": 1400.0},
    {"temperature.demag.beta_hcj_per_C": -0.006},
    {"temperature.demag.knee_fraction": 0.95},
    {"calibration.alpha_br_per_C": -0.001},
    {"temperature.demag.h_rev_likepole_kA_m": 700.0, "temperature.demag.design_margin_C": 15.0},
    {"temperature.demag.h_rev_likepole_kA_m": 1500.0},     # reverse field above the 20 degC knee: onset below 20 degC
    {"coupling.mu0": MU0_EXACT},
    {"coupling.magnets.part_inner": "B842-N52"},            # Br 1.45 T, K&J N52 rating 80 degC
    {"coupling.magnets.part_inner": ""},                    # not in the library: uncalibrated (offset 0)
], ids=["hcj20", "beta", "knee", "alpha_br", "likepole_margin", "h_above_knee", "mu0_exact", "n52_part", "no_library"])
def test_demag_chain_tracks_root_find_under_varied_inputs(changes):
    """C48-C60 follow the independent chain when inputs change (including the uncalibrated not-in-library branch)."""
    inp = vary(defaults(), changes)
    res = run(inp)
    r = _demag_reference(inp, res)
    dm = res.temperature.demag
    pairs = [("h_ref_kA_m", "h_ref"), ("calibration_offset_C", "offset"), ("magnet_limit_C", "magnet_limit")]
    pairs += [(ONSET_FIELDS[c], c) for c in ONSET_FIELDS]
    assert_all([(f"{field_name} with {changes}", _cell(dm, field_name), getattr(dm, field_name), r[key], TOL_ALGEBRA)
                for field_name, key in pairs])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "B842-N52"}, {"coupling.magnets.part_inner": ""}],
                         ids=["defaults", "n52_part", "no_library"])
def test_demag_inputs_link_to_calculator_and_kj_rating(changes):
    """C42 = Calculator C21 (Br20 used), C43 = Calibration C22 (alpha_Br), C47 = K&J rating of the inner part
    (= Calculator C22); a part outside the library has no numeric rating."""
    inp = vary(defaults(), changes)
    res = run(inp)
    dm = res.temperature.demag
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    items = [("Br20 used by demag vs Calculator C21", _cell(dm, "br20_T") + ", Calculator!C21", dm.br20_T, res.model.inner_br_T, 0.0),
             ("alpha_Br vs Calibration C22 input", _cell(dm, "alpha_br") + ", Calibration!C22", dm.alpha_br,
              inp.calibration.alpha_br_per_C, 0.0)]
    if rating is None:
        items.append((f"library rating must be non-numeric for a blank part (C47 {dm.tmax_lib_C!r})", _cell(dm, "tmax_lib_C"),
                      0.0 if _is_number(dm.tmax_lib_C) else 1.0, 1.0, 0.0))
    else:
        items.append(("library rating vs K&J grade rating", _cell(dm, "tmax_lib_C"),
                      dm.tmax_lib_C if _is_number(dm.tmax_lib_C) else math.nan, rating, 0.0))
    assert_all(items)


@pytest.mark.family("temperature")
def test_doc_consistency_readme_demag_numbers():
    """Documentation consistency, not an independent check: README 'Temperature design' figures (10.4 degC shift,
    102.6 degC skipping onset, 92.6 / 92.55 degC limit) and 'skipping is the lowest onset' match the engine."""
    dm = run().temperature.demag
    items = [(f"README {what}", _cell(dm, field_name), getattr(dm, field_name), printed, half_digit / printed)
             for what, field_name, printed, half_digit in [("calibration shift", "calibration_offset_C", 10.4, 0.05),
                                                           ("skipping onset", "onset_skipping_C", 102.6, 0.05),
                                                           ("magnet limit", "magnet_limit_C", 92.6, 0.05),
                                                           ("magnet limit (quick start)", "magnet_limit_C", 92.55, 0.005)]]
    lowest = min(getattr(dm, f) for f in ONSET_FIELDS.values())
    items.append(("README: skipping is the lowest onset", _cell(dm, "onset_skipping_C"), dm.onset_skipping_C, lowest, 0.0))
    assert_all(items)


@pytest.mark.family("temperature")
def test_torque_at_magnet_limit_scales_with_br_squared():
    """C61 = pull-out(20 degC, Calculator C94) * (Br(T_limit)/Br20)^2; C62 repeats the service pull-out (Calculator C93).
    The C93/C94 Br^2 law itself is owned by the torque tasks."""
    inp, res = defaults(), run()
    lim = _demag_reference(inp, res)["magnet_limit"]
    dm = res.temperature.demag
    assert_all([("pull-out at magnet limit", _cell(dm, "torque_at_limit_Nm"), dm.torque_at_limit_Nm,
                 torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, lim), TOL_ALGEBRA),
                ("pull-out at service maximum vs Calculator C93", _cell(dm, "torque_at_service_Nm") + ", Calculator!C93",
                 dm.torque_at_service_Nm, res.model.pullout_Nm, 0.0)])


@pytest.mark.family("temperature")
def test_summary_repeats_demag_and_adhesive_values():
    """C7-C11 and C24 (cure margin = single-ring onset - cure temperature) equal the independent values."""
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    s = res.temperature.summary
    expected = {"onset_aligned_C": r["aligned"], "onset_pullout_C": r["pullout"], "onset_skipping_C": r["likepole"],
                "magnet_limit_C": r["magnet_limit"], "adhesive_limit_C": _selected_candidate(inp).design_limit_C,
                "cure_margin_C": r["single_ring"] - _selected_candidate(inp).cure_C}
    assert_all([(f"summary {name}", _cell(s, name), getattr(s, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"coupling.op_temp_C": 80.0}, {"metal.required_min_Nm": 2.7},
                                     {"temperature.adhesive.selected": 4}, {"temperature.duty.hot_ambient_C": 30.0}],
                         ids=["defaults", "op_80C", "requirement_2p7", "dp460_governs", "mild_day"])
def test_summary_margins_and_hot_day_note(changes):
    """C13 = governing limit - service maximum (Calculator C10); C15 = governing limit - hot-day start (C35 + C36);
    F16 = 'Meets it nominally...' when pull-out at the hot-day start (Br^2 from Calculator C94) >= required (Metal C7).
    Sole owner of C13, C15 and F16; both branches of F16 are exercised (defaults meets, requirement_2p7 is below)."""
    inp = vary(defaults(), changes)
    res = run(inp)
    s = res.temperature.summary
    gov, t0 = _governing_reference(inp, res), _hot_day_start_C(inp)
    torque_hot = torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, t0)
    note = HOT_DAY_NOTE_MEETS if torque_hot >= inp.metal.required_min_Nm else HOT_DAY_NOTE_BELOW
    assert_all([(f"margin above service maximum with {changes}", _cell(s, "margin_service_C"), s.margin_service_C,
                 gov - inp.coupling.op_temp_C, TOL_ALGEBRA),
                (f"margin above hot-day start with {changes}", _cell(s, "margin_hot_day_C"), s.margin_hot_day_C, gov - t0, TOL_ALGEBRA),
                text_item(f"hot-day torque note (reference {torque_hot:.6f} vs required {inp.metal.required_min_Nm} N.m)",
                          _cell(s, "torque_hot_day_note"), s.torque_hot_day_note, note)])


def _alternative_skipping_onset(form: str, inp, res) -> float:
    """Skipping onset under another reasonable way of calibrating the knee model to the supplier rating."""
    d = inp.temperature.demag
    alpha, br20, mu0 = inp.calibration.alpha_br_per_C, res.model.inner_br_T, inp.coupling.mu0
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    n42sh = ref.ARNOLD_N42SH
    mu_rec = n42sh["br_nominal_T"] / (MU0_EXACT * n42sh["hcb_nominal_kA_m"] * 1000)       # 1.056 (Arnold N42SH)
    h_ref = ref.load_line_reverse_field_kA_m(br20, 1.0, mu0, mu_rec if "recoil_permeability" in form else 1.0)
    h_lp = d.h_rev_likepole_kA_m
    if form.startswith("knee_fraction"):
        knee = brentq(lambda k: ref.knee_crossing_C(h_ref, d.hcj20_kA_m, d.beta_hcj_per_C, k, alpha) - rating, 0.3, 1.2)
        return ref.knee_crossing_C(h_lp, d.hcj20_kA_m, d.beta_hcj_per_C, knee, alpha)
    if form == "beta_hcj":
        beta = brentq(lambda b: ref.knee_crossing_C(h_ref, d.hcj20_kA_m, b, d.knee_fraction, alpha) - rating, -0.02, -0.001)
        return ref.knee_crossing_C(h_lp, d.hcj20_kA_m, beta, d.knee_fraction, alpha)
    args = (d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, alpha)
    return ref.knee_crossing_C(h_lp, *args) - ref.calibration_offset_C(h_ref, rating, *args)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("form", ["knee_fraction", "beta_hcj", "recoil_permeability", "knee_fraction+recoil_permeability"])
def test_onset_calibration_form_sensitivity_within_design_margin(form):
    """Model form: other reasonable ways to calibrate to the 150 degC rating move the skipping onset by < the 10 degC margin.

    The engine shifts every onset by a constant temperature. Alternatives: scale the knee fraction, or scale the Hcj
    coefficient, so the Pc = 1 magnet reaches its knee at exactly 150 degC; keep the offset but place the Pc = 1
    operating point with the datasheet recoil permeability mu_rec = Br/(mu0 HcB) = 1.056 (Arnold N42SH); or both the
    knee-fraction calibration and the datasheet recoil permeability. Tolerance: the design margin (C51), whose job is to
    cover model-form uncertainty.
    """
    inp, res = defaults(), run()
    alt = _alternative_skipping_onset(form, inp, res)
    got = res.temperature.demag.onset_skipping_C
    margin = inp.temperature.demag.design_margin_C
    assert abs(got - alt) <= margin, mismatch(f"skipping onset vs {form} calibration (tol = {margin} degC margin)",
                                              _cells(res.temperature.demag, "onset_skipping_C", "calibration_offset_C")
                                              + ", Temperature design!C51", got, alt, margin / abs(alt))


@pytest.mark.family("temperature")
def test_knee_model_coefficients_vs_n42sh_datasheet():
    """Literature: alpha_Br and Hcj20 match the Arnold N42SH datasheet; with its beta_Hcj = -0.55 %/degC (engine -0.50)
    the calibrated skipping onset is not lower than the engine's (engine conservative) and within the 10 degC margin."""
    inp, res = defaults(), run()
    d, n42sh = inp.temperature.demag, ref.ARNOLD_N42SH
    dm = res.temperature.demag
    lit = _demag_reference(vary(inp, {"temperature.demag.beta_hcj_per_C": n42sh["beta_hcj_per_C"]}), res)["likepole"]
    got = dm.onset_skipping_C
    assert_all([("alpha_Br vs datasheet -0.12 %/degC", _cell(dm, "alpha_br"), inp.calibration.alpha_br_per_C,
                 n42sh["alpha_br_per_C"], TOL_ALGEBRA),
                ("Hcj20 vs datasheet minimum", "Temperature design!C44", d.hcj20_kA_m, n42sh["hcj20_min_kA_m"], TOL_ALGEBRA),
                ("skipping onset vs datasheet beta_Hcj (engine must be <= datasheet onset, within margin)",
                 _cell(dm, "onset_skipping_C") + ", Temperature design!C45", got,
                 min(max(got, lit - d.design_margin_C), lit), 0.0)])


@pytest.mark.family("temperature")
def test_knee_evaluated_inside_coefficient_validity_range():
    """Literature: the linear Hcj(T) and Br(T) coefficients are measured over 20-150 degC (Arnold N42SH note 1).

    The knee is evaluated at the uncalibrated model temperature (C49 for the Pc = 1 magnet, reported onset + C50
    offset for the four cases). One check for the one root cause; the message lists every case outside the range.
    """
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    lo, hi = ref.ARNOLD_N42SH["coefficient_range_C"]
    dm = res.temperature.demag
    cases = [("reference magnet (Pc = 1)", r["t_ref"], _cell(dm, "t_ref_model_C"))]
    cases += [(case, r[case] + r["offset"], _cells(dm, ONSET_FIELDS[case], "calibration_offset_C")) for case in ONSET_FIELDS]
    assert_all([(f"knee evaluation temperature ({case}) outside {lo:.0f}-{hi:.0f} degC", cells, t_model,
                 min(max(t_model, lo), hi), 0.0) for case, t_model, cells in cases])


# =========================================================================== magnet ratings and library
@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {},
    {"coupling.op_temp_C": 150.0},                                          # exactly at the rating: OK (<=)
    {"coupling.op_temp_C": math.nextafter(150.0, math.inf)},                # one ulp above: over
    {"coupling.magnets.part_inner": "B842-N52", "coupling.op_temp_C": 90.0},
    {"coupling.magnets.part_outer": "B842", "coupling.op_temp_C": 100.0},  # N42 (80 degC) outer: catches swapped rings
    {"coupling.magnets.part_inner": ""},
    {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""},
], ids=["defaults", "at_rating", "above_rating", "n52_inner_90C", "n42_outer_100C", "no_library_inner",
        "no_library_both"])
def test_magnet_temperature_checks_follow_kj_rating(changes):
    """Calculator C22/C32 carry the K&J rating of each ring's part ('n/a' for a blank part); C107/C108 read 'OK' when
    the operating temperature (C10) is at or below it, 'OVER the magnet rating' above it, and 'unknown' for a part
    outside the library. Sole owner of C22, C32, C107 and C108."""
    inp = vary(defaults(), changes)
    m = run(inp).model
    items = []
    for ring, part, tmax, check, c_rating, c_check in (
            ("inner", inp.coupling.magnets.part_inner, m.inner_tmax_C, m.inner_temp_check, "Calculator!C22", "Calculator!C107"),
            ("outer", inp.coupling.magnets.part_outer, m.outer_tmax_C, m.outer_temp_check, "Calculator!C32", "Calculator!C108")):
        rating = ref.part_rating_C(part)
        if rating is None:
            items.append(text_item(f"{ring} rating of a blank part (no library row)", c_rating, tmax, "n/a"))
            expected = "unknown"
        else:
            items.append((f"{ring} rating of {part} vs K&J", c_rating, tmax if _is_number(tmax) else math.nan, rating, 0.0))
            expected = "OK" if inp.coupling.op_temp_C <= rating else "OVER the magnet rating"
        items.append(text_item(f"{ring} temperature check at {inp.coupling.op_temp_C!r} degC", c_check, check, expected))
    assert_all(items)


@pytest.mark.family("temperature")
def test_library_ratings_follow_kj_grade_table():
    """Every magnet-library row's max operating temperature equals the K&J rating of its grade suffix."""
    assert_all([(f"library {p.part} ({p.grade}) max operating temperature", "Magnet library", p.tmax_C,
                 ref.grade_max_operating_C(p.grade), 0.0) for p in MAGNET_LIBRARY.values()])


@pytest.mark.family("materials")
def test_library_br_within_kj_grade_range():
    """Literature: library Br20 (Calculator C21/C31 source) lies inside K&J's published Br range for its grade.

    Rows whose grade K&J's specification page does not list (N50, N50M) are not checked.
    """
    items = []
    for p in MAGNET_LIBRARY.values():
        if p.grade in ref.KJ_BR_RANGE_T:
            lo, hi = ref.KJ_BR_RANGE_T[p.grade]
            items.append((f"library {p.part} ({p.grade}) Br20 vs K&J {lo}-{hi} T", "Magnet library / Calculator!C21",
                          p.br_T, min(max(p.br_T, lo), hi), 0.0))
    assert_all(items)


# =========================================================================== adhesive
@pytest.mark.family("temperature")
@pytest.mark.parametrize("selected", [1, 2, 3, 4])
def test_adhesive_selection_and_governing_limit(selected):
    """C75-C78 copy the selected candidate row; C12 = min(magnet limit, adhesive limit); F12 names the governing limit."""
    inp = vary(defaults(), {"temperature.adhesive.selected": selected})
    res = run(inp)
    cand = _selected_candidate(inp)
    a, s = res.temperature.adhesive, res.temperature.summary
    mag_lim = _demag_reference(inp, res)["magnet_limit"]
    note = "Magnets govern (skipping case)." if mag_lim <= cand.design_limit_C else "Adhesive governs."
    items = [(f"selected adhesive {name} ({cand.name})", _cell(a, name), getattr(a, name), expect, 0.0)
             for name, expect in (("design_limit_C", cand.design_limit_C), ("cure_C", cand.cure_C), ("lap_shear_MPa", cand.lap_shear_MPa))]
    items += [text_item("selected adhesive name", "Temperature design!C75", a.selected_name, cand.name),
              (f"governing limit ({cand.name})", _cell(s, "governing_limit_C"), s.governing_limit_C,
               min(mag_lim, cand.design_limit_C), TOL_ALGEBRA),
              text_item(f"governing note (magnet {mag_lim:.4f} vs adhesive {cand.design_limit_C} degC)",
                        _cell(s, "governing_note"), s.governing_note, note)]
    assert_all(items)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("tds", ref.ADHESIVE_TDS, ids=[t.name for t in ref.ADHESIVE_TDS])
def test_adhesive_design_limit_follows_tds(tds):
    """Literature: candidate design limit (C65:C74 data) = min(TDS service maximum, Tg - 20 degC) for the TDSs read."""
    cand = next(c for c in defaults().temperature.adhesive.candidates if c.name == tds.name)
    expect = ref.adhesive_design_limit_C(tds.service_max_C, tds.tg_C)
    assert cand.design_limit_C == expect, mismatch(f"design limit of {tds.name} ({tds.source})", "Temperature design!C65:C74",
                                                   cand.design_limit_C, expect, 0.0)


@pytest.mark.family("temperature")
def test_bond_shear_from_cold_high_torque():
    """C81 area = L*W, C83 = Metal C10, C84 r_mid = back apothem + t/2, C85 F = T_cold_high/(N r_mid), C86 tau = F/area."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    torque = res.metal.torque_cold_high_Nm
    r_mid = _inner_mid_radius_mm(inp, res)
    expected = {"cold_high_torque_Nm": torque, "bond_area_mm2": _bond_area_mm2(res), "inner_mid_radius_mm": r_mid,
                "tangential_force_N": torque / (inp.coupling.npole * r_mid / 1000), "bond_shear_MPa": _bond_shear_reference(inp, res)}
    assert_all([(name, _cell(a, name), getattr(a, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
def test_centrifugal_force_from_block_mass():
    """C88 = m omega^2 r_mid at the wheel-rotor speed (C34). The block mass C82 is owned by Task 7's density checks
    and consumed here as an input."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    force = ref.centrifugal_force_N(a.block_mass_g, inp.temperature.duty.wheel_rotor_rpm, _inner_mid_radius_mm(inp, res))
    assert rel_err(a.centrifugal_force_N, force) < TOL_ALGEBRA, mismatch(
        "centrifugal force per block", _cell(a, "centrifugal_force_N"), a.centrifugal_force_N, force, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_static_ratio_and_fatigue_screen_at_defaults():
    """C89 = lap shear / bond shear; C91 = 'OK: Nx margin' when fatigue endurance * lap shear / bond shear >= 4."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    tau = _bond_shear_reference(inp, res)
    lap = _selected_candidate(inp).lap_shear_MPa
    fat = inp.temperature.adhesive_life.fatigue_endurance * lap / tau
    assert_all([("static strength ratio", _cell(a, "static_ratio"), a.static_ratio, lap / tau, TOL_ALGEBRA),
                text_item(f"fatigue screen (reference margin {fat:.4f})", _cell(a, "fatigue_screen"), a.fatigue_screen,
                          _fatigue_screen_reference(fat))])


@pytest.mark.family("temperature")
def test_fatigue_screen_uses_fatigue_endurance_input():
    """C91 should use the fatigue-endurance input (C195) that C196 uses; vary it to 0.1 and compare the screen."""
    inp = vary(defaults(), {"temperature.adhesive_life.fatigue_endurance": 0.1})
    res = run(inp)
    a = res.temperature.adhesive
    fat_ref = inp.temperature.adhesive_life.fatigue_endurance * _selected_candidate(inp).lap_shear_MPa / _bond_shear_reference(inp, res)
    assert_all([text_item(f"fatigue screen with endurance 0.1 (reference margin {fat_ref:.4f})",
                          _cell(a, "fatigue_screen") + ", Temperature design!C195", a.fatigue_screen,
                          _fatigue_screen_reference(fat_ref))])


@pytest.mark.family("temperature")
def test_shear_reversals_equal_pole_pair_passes():
    """C90 = pole-pair passes over life while slipping: (N/2) * slip rpm / 60 * event duration * events (one full shear
    reversal per pole-pair pass). Task 5 owns C90; Task 6 owns the separately computed C168 and C191."""
    inp, res = defaults(), run()
    md = inp.metal
    expect = pole_pair_frequency_Hz(inp.coupling.npole, md.slip_rpm) * md.slip_event_s * md.life_events
    got = res.temperature.adhesive.shear_reversals
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("shear reversals", _cell(res.temperature.adhesive, "shear_reversals"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_fatigue_screen_robust_to_bond_plane_lever_arm():
    """Model form: taking the lever arm at the bond plane (back apothem) instead of the block mid radius raises bond
    shear 16 %; the fatigue screen verdict (C91, OK or CHECK) must not change."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    tau_back = _bond_shear_reference(inp, res, lever_arm_mm=inp.coupling.inner_back_apothem_mm)
    fat_back = inp.temperature.adhesive_life.fatigue_endurance * _selected_candidate(inp).lap_shear_MPa / tau_back
    verdict = "OK" if fat_back >= 4 else "CHECK"
    assert_all([text_item(f"fatigue verdict with the bond-plane lever arm (tau {tau_back:.4f} MPa, margin {fat_back:.3f})",
                          _cells(a, "fatigue_screen", "inner_mid_radius_mm"), a.fatigue_screen.split(":")[0], verdict)])


def _press_on_forces(inp, res):
    """(inner F_y list, outer F_y list) over one pole pair at Br(governing limit), 2D multi-layer slab model."""
    m = res.model
    br = m.inner_br_T * br_ratio(inp.calibration.alpha_br_per_C, _governing_reference(inp, res))
    y_outer = inp.metal.bond_inner_mm + m.inner_thickness_mm + m.face_gap_mm
    h = y_outer + m.outer_thickness_mm + inp.metal.bond_outer_mm
    inner_fy, outer_fy = [], []
    for shift in np.linspace(0.0, 2 * m.pole_pitch_mm, 33):
        layers = [MagnetLayer(inp.metal.bond_inner_mm, m.inner_thickness_mm, br, m.fill_inner, 0.0),
                  MagnetLayer(y_outer, m.outer_thickness_mm, br * m.outer_br_T / m.inner_br_T, m.fill_outer, shift)]
        (_, fy_in), (_, fy_out) = layer_block_forces_N(layers, h, m.pole_pitch_mm, m.active_length_mm)
        inner_fy.append(fy_in)
        outer_fy.append(fy_out)
    return inner_fy, outer_fy


@pytest.mark.family("temperature")
def test_magnetics_press_inner_blocks_onto_hub():
    """README claim 'the magnetics press the blocks onto the steel', inner ring: the net magnetic radial force on an
    inner block points into the hub at every relative angle over a pole pair and exceeds the centrifugal force at
    wheel speed (C88, verified above). 2D two-plate model at Br(governing limit), the weakest case. Sign check: the
    planar model omits curvature, end leakage and finite steel permeability, so only a margin of the whole
    centrifugal force counts."""
    inp, res = defaults(), run()
    inner_fy, _ = _press_on_forces(inp, res)
    fc = res.temperature.adhesive.centrifugal_force_N
    weakest = max(inner_fy)                                       # least negative = weakest press-on
    assert weakest < -fc, mismatch("weakest inward magnetic force on an inner block vs centrifugal (must be < -Fc)",
                                   _cell(res.temperature.adhesive, "centrifugal_force_N"), weakest, -fc, 0.0)


@pytest.mark.family("temperature")
def test_magnetics_press_outer_blocks_onto_cup():
    """Same README claim, outer ring: the net magnetic radial force on an outer block points into the cup at every angle."""
    inp, res = defaults(), run()
    _, outer_fy = _press_on_forces(inp, res)
    weakest = min(outer_fy)
    assert weakest > 0.0, mismatch("weakest outward magnetic force on an outer block (must be > 0)",
                                   "README Temperature design claim; no workbook cell", weakest, 0.0, 0.0)


# =========================================================================== thermal mismatch (Volkersen)
@pytest.mark.family("temperature")
def test_mismatch_inputs_and_worst_swing():
    """C94, C98-C102: steel CTE and modulus, hub wall, cold limit, worst swing max(cure - cold, governing - cure), bondline."""
    inp, res = defaults(), run()
    mm = res.temperature.mismatch
    expected = {"steel_cte": inp.materials.steel.cte_per_C, "steel_E_GPa": inp.materials.steel.modulus_GPa,
                "steel_thickness_mm": res.model.hub_wall_mm, "cold_limit_C": inp.metal.min_temp_C,
                "worst_swing_C": _swing_reference(inp, res), "current_bondline_mm": inp.metal.bond_inner_mm}
    assert_all([(name, _cell(mm, name), getattr(mm, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("which", ["current", "recommended"])
def test_volkersen_peak_shear_matches_derivation(which):
    """C104/C105 and volkersen_peak_shear_MPa equal the independently derived closed form (TOL_ALGEBRA)."""
    inp, res = defaults(), run()
    bond = inp.metal.bond_inner_mm if which == "current" else inp.temperature.mismatch.recommended_bondline_mm
    k = _volkersen_inputs_SI(inp, res, bond)
    expect = ref.volkersen_thermal_peak_shear_Pa(**k) / 1e6
    fn = volkersen_peak_shear_MPa(k["G_Pa"] / 1e9, k["d_alpha_per_C"], k["dT_C"], bond, inp.temperature.mismatch.ndfeb_modulus_GPa,
                                  res.model.inner_thickness_mm, inp.materials.steel.modulus_GPa, res.model.hub_wall_mm,
                                  res.model.inner_length_mm)
    field_name = f"peak_shear_{which}_MPa"
    cell = _cell(res.temperature.mismatch, field_name)
    assert_all([("volkersen_peak_shear_MPa at engine inputs", cell, fn, expect, TOL_ALGEBRA),
                (f"peak end shear ({which} bondline)", cell, getattr(res.temperature.mismatch, field_name), expect, TOL_ALGEBRA)])


@pytest.mark.family("temperature")
def test_volkersen_peak_shear_matches_finite_difference():
    """C104 against the finite-difference shear-lag solution (20,001 nodes, O((lam dx)^2) ~ 1e-8; tolerance 1e-6)."""
    inp, res = defaults(), run()
    _, _, tau = ref.volkersen_thermal_fd(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm))
    expect = float(np.max(np.abs(tau))) / 1e6
    got = res.temperature.mismatch.peak_shear_current_MPa
    assert rel_err(got, expect) < 1e-6, mismatch("peak end shear vs FD", _cell(res.temperature.mismatch, "peak_shear_current_MPa"),
                                                 got, expect, 1e-6)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {"temperature.adhesive.selected": 2},                   # heat cure at 120 degC: cold side governs the swing
    {"temperature.adhesive.selected": 3},
    {"metal.bond_inner_mm": 0.2},
    {"temperature.mismatch.adhesive_shear_modulus_GPa": 1.2, "temperature.mismatch.recommended_bondline_mm": 0.25},
    {"metal.min_temp_C": -20.0, "temperature.mismatch.ndfeb_cte_per_C": -1.5e-6},
], ids=["ea9514", "2214", "bond_0p2", "stiff_glue", "mild_cold"])
def test_volkersen_tracks_derivation_under_varied_inputs(changes):
    """C101, C104, C105 follow the independent chain when adhesive, bondline, modulus, cold limit or CTE change."""
    inp = vary(defaults(), changes)
    res = run(inp)
    mm = res.temperature.mismatch
    items = [(f"worst swing with {changes}", _cell(mm, "worst_swing_C"), mm.worst_swing_C, _swing_reference(inp, res), TOL_ALGEBRA)]
    for which, bond in (("current", inp.metal.bond_inner_mm), ("recommended", inp.temperature.mismatch.recommended_bondline_mm)):
        expect = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, bond)) / 1e6
        items.append((f"peak shear ({which}) with {changes}", _cell(mm, f"peak_shear_{which}_MPa"),
                      getattr(mm, f"peak_shear_{which}_MPa"), expect, TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"temperature.mismatch.adhesive_shear_modulus_GPa": 0.107}], ids=["defaults", "soft_glue"])
def test_mismatch_reading_threshold(changes):
    """C106 reads 'Above the lap-shear strength at the block ends' exactly when C104 exceeds the selected lap shear."""
    inp = vary(defaults(), changes)
    res = run(inp)
    s1 = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm)) / 1e6
    lap = _selected_candidate(inp).lap_shear_MPa
    assert_all([text_item(f"mismatch reading (reference peak {s1:.4f} vs lap shear {lap} MPa)",
                          _cells(res.temperature.mismatch, "reading", "peak_shear_current_MPa"),
                          res.temperature.mismatch.reading, _reading_reference(s1, lap))])


@pytest.mark.family("temperature")
def test_adhesive_shear_modulus_matches_default_adhesive_tds():
    """Literature: the adhesive shear modulus (C96) should match the selected adhesive's TDS tensile modulus through
    G = E/(2(1+nu)); any nu in 0.3-0.5 is accepted, so the tolerance is the half-width of that G band. C96 is one
    input shared by every candidate (not a per-candidate column), so only the default selection (AA 326) is tested."""
    inp = defaults()
    cand = _selected_candidate(inp)
    tds = next(t for t in ref.ADHESIVE_TDS if t.name == cand.name)
    g_lo, g_hi = ref.shear_modulus_GPa(tds.tensile_modulus_GPa, 0.5), ref.shear_modulus_GPa(tds.tensile_modulus_GPa, 0.3)
    g_mid, half = (g_lo + g_hi) / 2, (g_hi - g_lo) / 2
    got = inp.temperature.mismatch.adhesive_shear_modulus_GPa
    assert g_lo <= got <= g_hi, mismatch(f"adhesive shear modulus vs {tds.name} TDS ({tds.source})", "Temperature design!C96",
                                         got, g_mid, half / g_mid)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("variant", ["biaxial_stiffness", "datasheet_ndfeb_cte"])
def test_mismatch_reading_robust_to_model_details(variant):
    """Model form: plane-stress biaxial adherend stiffness E/(1-nu) (nu 0.24 NdFeB, 0.29 steel), or the Arnold NdFeB
    CTE perpendicular to magnetization (-1.0e-6/degC instead of -0.8e-6), must not flip the C106 reading."""
    inp, res = defaults(), run()
    kwargs = {"biaxial": True} if variant == "biaxial_stiffness" else {"ndfeb_cte": ref.ARNOLD_N42SH["cte_perpendicular_per_C"]}
    s1_alt = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm, **kwargs)) / 1e6
    lap = _selected_candidate(inp).lap_shear_MPa
    mm = res.temperature.mismatch
    assert_all([text_item(f"mismatch reading with {variant} (peak {s1_alt:.4f} vs engine {mm.peak_shear_current_MPa:.4f} MPa)",
                          _cells(mm, "reading", "peak_shear_current_MPa"), mm.reading, _reading_reference(s1_alt, lap))])
