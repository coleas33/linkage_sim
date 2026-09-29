"""Torque model: independent checks of the engine (M1, Task 3).

Groups:
  * 3D: real magpylib 3D free-space rings vs the engine's free-space pull-out (Calculator!C93/C42), its 2D torque
    (C91), its end factor (C92), and the Calibration sheet's prototype model, correction and supplied 3D results.
  * laws: scaling and limit laws on the engine via vary(): length, Br, gap, fill, thick and thin magnets.
  * temperature: Br(T) linear at alpha_Br, pull-out ~ Br^2, the 20 °C pull-out.
  * calibration: the one-point prototype correction re-derived, C95/C96, the factor-selection rule.
  * constants: rounded mu0 vs exact.

A failing check is a CANDIDATE FINDING for Task 8, not a bug to fix here. Checks that can fail for the same reason
carry the same "Root-cause group" line in their docstrings (T3-RC1 to T3-RC6), so Task 8 can review each root cause
once. Sweeps that share one root cause are a single check that lists every point in its message.

Ownership (one owner per cell): this task owns Calculator C95, C96 and the Calibration sheet. The term-by-term
re-derivation of the torque chain C66-C94 is Task 2's (test_torque_field2d.py::test_calculator_torque_chain_rederived);
the gearbox rows C99/C100 are Task 4's (test_metal_design.py::test_gearbox_equivalents); the magnet temperature checks
C22/C32/C107/C108 are Task 5's (test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating);
Task 4 also owns the geometry rows C64/C65 and Metal design C8-C10 (hot/cold torque band). The 3D, law and
temperature-scaling checks here test the model behind C69-C94 by other methods (3D fields, limits, scaling across
temperature); they do not re-derive those cells.
"""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, TOL_MODEL, TOL_PEAK, defaults, mismatch, rel_err, run, vary
from audit.references import planar
from audit.references import torque_ref as tr

MM = tr.MM

#: Free-space comparison state: 20 °C (Br = 1.29 T) and no back iron (magpylib has no permeable material; the
#: restriction comes from magpylib, not from the tolerance). Relative model error does not depend on temperature,
#: because pull-out ~ Br_i*Br_o exactly (test_pullout_scales_as_br_squared_with_temperature), so comparing at 20 °C is
#: equivalent to 50 °C. The 3D reference is converged to <= 1e-5 (test_torque_reference_sanity.py), so the whole
#: deviation from it belongs to the engine and is judged against audit.common.TOL_MODEL (checks here) and TOL_PEAK.
FREE_20C = {"coupling.op_temp_C": 20, "coupling.backiron": 0}
#: Blank part numbers make the engine use the manual magnet dimensions (their defaults equal B842SH).
MANUAL = {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""}
#: Lengths for the 3D long-length (2D) slope. The end deficit is constant beyond ~20 mm (sanity test).
L_SLOPE = (30 * MM, 60 * MM)
#: Face gaps of the 3D gap-sweep check [mm]: corner gaps 0.23-4.03 mm, i.e. the engine's gap sweep (corner
#: 0.5-4 mm) plus one point below it.
GAP_SWEEP_FACE_GAPS_MM = (0.6, 0.9, 1.4, 2.4, 3.4, 4.4)
#: Face gaps beyond the sweep for the decay law [mm].
DECAY_FACE_GAPS_MM = (6.0, 10.0, 30.0, 100.0)
#: The pole sweep's pole counts (sweeps.POLE_SWEEP_POLES).
POLE_SWEEP = (6, 8, 10, 12, 14, 16)
#: Quadrature for the gap sweep: 0.25 mm / 6-point panels are converged to 1e-10 at the 0.23 mm corner clearance.
PANEL_FINE_M, ORDER_FINE = 0.25 * MM, 6
#: Corner gaps of the two supplied 3D results, from the labels of their input cells: Calibration!C47 "3D result at
#: 1.0 mm corner gap" and C48 "3D result at 1.5 mm corner gap". Those cells hold only the torques. The dataclass
#: attributes fea_gap1_mm and fea_gap2_mm are not inputs (no param() metadata) and the engine does not read them.
FEA_LABEL_GAPS_MM = (1.0, 1.5)


# --------------------------------------------------------------------------- helpers
def _assert_all_within(what: str, cells: str, rows: list[tuple[str, float, float]], tol: float) -> None:
    """One check over a sweep (one root cause, one candidate): every (label, engine, reference) row must agree within
    ``tol``. The message is mismatch() for the worst row, followed by every row's signed error."""
    errors = [rel_err(eng, ref) for _, eng, ref in rows]
    worst = max(range(len(rows)), key=errors.__getitem__)
    label, eng, ref = rows[worst]
    table = "; ".join(f"{lab}: {(e - r) / abs(r):+.3%}" if r != 0 else f"{lab}: {e:+.3e} (reference 0)"
                      for lab, e, r in rows)
    assert errors[worst] <= tol, mismatch(f"{what}; worst at {label}; all rows (engine/reference - 1): {table}",
                                          cells, eng, ref, tol)


def _cell(results_obj, field: str) -> str:
    """Workbook cell of a result field, from its field metadata."""
    for f in dataclasses.fields(results_obj):
        if f.name == field:
            return f.metadata["cell"]
    raise KeyError(f"{type(results_obj).__name__} has no field {field!r}")


@lru_cache(maxsize=None)
def _free_case(changes: tuple = ()) -> tuple:
    """(inputs, results, 3D ring pair) at FREE_20C plus ``changes``, a tuple of (path, value) pairs.

    Deliberately not an audit.common.ScenarioCache: the keys are open-ended change tuples (the pole sweep's hub
    apothem, _hub_that_fits, comes from an engine run at call time, so no fixed named table exists at import) and
    the value also carries the 3D ring pair, which _t3d and the sweeps reuse under the same key."""
    inp = vary(defaults(), {**FREE_20C, **dict(changes)})
    res = run(inp)
    return inp, res, tr.ring_pair_from_engine(inp, res)


def _engine_raw(res) -> float:
    """Engine free-space pull-out without its calibration factor: 2D torque x end factor = C93 / C42."""
    return res.model.pullout_Nm / res.model.f_cal


@lru_cache(maxsize=None)
def _t3d(changes: tuple = ()) -> float:
    """3D torque [N*m] at the engine's pull-out angle (half a pole pitch) for _free_case(changes), default quadrature."""
    _, _, pair = _free_case(changes)
    return tr.ring_torque_3d(pair, pair.half_pitch)


@lru_cache(maxsize=None)
def _default_per_length() -> float:
    """3D long-length (2D) torque per metre of the default cross-section at 20 °C, half a pole pitch."""
    _, _, pair = _free_case()
    return tr.torque_per_length_3d(pair, *L_SLOPE, pair.half_pitch)


@lru_cache(maxsize=None)
def _prototype_3d() -> float:
    """3D pull-out (max over angle) of the free-space prototype, from the Calibration inputs alone."""
    t3d, _ = tr.ring_pullout_3d(tr.ring_pair_from_prototype(defaults().calibration))
    return t3d


def _hub_that_fits(npole: int) -> float:
    """The default hub apothem where the inner blocks fit its flats. Otherwise, the smallest apothem whose flats hold
    the block plus 0.05 mm (the pole sweep's fit rule). This keeps every 3D pole case physically buildable."""
    width = run().model.inner_width_mm                                  # Calculator!C19
    return max(defaults().coupling.inner_back_apothem_mm, width / (2 * math.tan(math.pi / npole)) + 0.05)


def _pole_changes(npole: int) -> tuple:
    return (("coupling.npole", npole), ("coupling.inner_back_apothem_mm", _hub_that_fits(npole)))


# =========================================================================== 3D cross-checks
@pytest.mark.family("torque")
def test_3d_free_space_amplitude_across_gap_sweep():
    """Engine free-space pull-out without its calibration factor (C93/C42 = C91*C92) vs real 3D magpylib rings at the
    same angle (half a pole pitch), face gaps 0.6-4.4 mm (corner gaps 0.23-4.03 mm), 20 °C. Because the angle is
    matched, this measures the harmonic model's amplitude only; the angle is the peak check. The panels are
    0.25 mm / 6-point, converged to 1e-10 at the smallest gap. One check for the whole sweep. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    rows = []
    for g in GAP_SWEEP_FACE_GAPS_MM:
        _, res, pair = _free_case((("metal.face_gap_mm", g),))
        ref = tr.ring_torque_3d(pair, pair.half_pitch, panel_m=PANEL_FINE_M, order=ORDER_FINE)
        rows.append((f"face gap {g} mm (corner {res.model.corner_gap_mm:.2f} mm)", _engine_raw(res), ref))
    _assert_all_within("free-space pull-out vs 3D at half a pole pitch, 20 °C, gap sweep",
                       "Calculator!C93/C42 = C91*C92 (backiron 0), Metal design!C119", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_free_space_decay_beyond_gap_sweep():
    """Limit law g -> infinity: pull-out must vanish at the physical rate. Engine ratio pull-out(g) / pull-out(1.4 mm)
    vs the same 3D ratio at face gaps 6, 10, 30 and 100 mm (free space, 20 °C, half a pole pitch). Ratios cancel the
    engine's error at the design gap. This replaces a loose "ratio < 1e-3" bound. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (the planar approximation's error grows with gap / radius)."""
    _, base_res, _ = _free_case()
    e0, t0 = _engine_raw(base_res), _t3d()
    rows = []
    for g in DECAY_FACE_GAPS_MM:
        changes = (("metal.face_gap_mm", g),)
        _, res, _ = _free_case(changes)
        rows.append((f"face gap {g} mm", _engine_raw(res) / e0, _t3d(changes) / t0))
    _assert_all_within("free-space pull-out ratio T(g)/T(1.4 mm) vs 3D ratio, 20 °C",
                       "Calculator!C93/C42 (backiron 0), Metal design!C119", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_free_space_amplitude_across_pole_sweep():
    """Engine free-space pull-out without its calibration factor vs 3D rings at half a pole pitch, for the pole sweep's
    pole counts 6-16. The hub is the default apothem where the blocks fit, otherwise the pole sweep's smallest fitting
    apothem. Face gap 1.4 mm, 20 °C, angle-matched. One check. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    rows = []
    for n in POLE_SWEEP:
        _, res, _ = _free_case(_pole_changes(n))
        rows.append((f"{n} poles (hub {_hub_that_fits(n):.3f} mm)", _engine_raw(res), _t3d(_pole_changes(n))))
    _assert_all_within("free-space pull-out vs 3D at half a pole pitch, 20 °C, pole sweep",
                       "Calculator!C93/C42 = C91*C92 (backiron 0), C5, C8", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_peak_at_half_pole_pitch_across_pole_sweep():
    """The engine evaluates every harmonic at half a pole pitch (sin(n pi/2) in each tau_n); the spec names this as a
    known candidate. Half a pitch is always a stationary point by symmetry. This checks that it is the maximum: 3D
    torque at half a pitch vs the 3D peak of the torque-angle curve, poles 6-16. One check. Tolerance TOL_PEAK.
    Root-cause group: T3-RC2 (pull-out evaluated at half a pole pitch)."""
    rows = []
    for n in POLE_SWEEP:
        _, _, pair = _free_case(_pole_changes(n))
        peak, theta = tr.ring_pullout_3d(pair)
        rows.append((f"{n} poles (3D peak at {math.degrees(theta):.2f} deg, engine angle {180 / n:.2f} deg)",
                     _t3d(_pole_changes(n)), peak))
    _assert_all_within("3D torque at the engine's pull-out angle vs 3D peak (engine value = 3D at half a pitch)",
                       "Calculator!C76, C82, C88 (sin(n pi/2)) -> C93", rows, TOL_PEAK)


@pytest.mark.family("torque")
def test_3d_long_length_limit_matches_engine_torque_2d():
    """Engine 2D (infinite-length) free-space torque vs the 3D per-length torque of the same cross-section x L.

    The reference is the slope of 3D torque vs length between 30 and 60 mm, which removes the end deficit exactly
    (it agrees with Task 2's exact 2D solution to 1.1e-5, sanity test). This isolates the 2D harmonic model from the
    end factor. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space).
    """
    _, res, _ = _free_case()
    reference = _default_per_length() * res.model.active_length_mm * MM
    engine = res.model.torque_2d_Nm
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "2D free-space torque at 20 °C vs 3D long-length slope x L", "Calculator!C91 (backiron 0), C33",
        engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_end_effect_factor_across_lengths():
    """Engine end factor 1 - c_end * pitch / L vs the 3D factor T3D(L) / (2D torque per length x L), default cross-
    section, both rings of length L (manual magnets), L = 3.17-50.8 mm, at half a pole pitch. One check. Tolerance
    TOL_MODEL. Root-cause group: none expected (the empirical form's magnitude)."""
    _, _, pair = _free_case()
    rows = []
    for length_mm in (3.17, 6.35, 12.7, 25.4, 50.8):
        inp = vary(defaults(), {**FREE_20C, **MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})
        engine = run(inp).model.f_end
        reference = tr.end_factor_3d(pair, length_mm * MM, _default_per_length(), pair.half_pitch)
        rows.append((f"L = {length_mm} mm", engine, reference))
    _assert_all_within("end-effect factor vs 3D", "Calculator!C92, C41, C65, C33", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_end_effect_factor_short_magnet():
    """Edge case L = 1 mm (below c_end * pitch = 1.32 mm): a physical end factor lies in (0, 1] for every L > 0.
    Engine f_end vs the 3D factor at half a pole pitch. Tolerance TOL_MODEL.
    Root-cause group: T3-RC3 (the linear end-factor form turns negative for short magnets)."""
    inp = vary(defaults(), {**FREE_20C, **MANUAL, "coupling.magnets.manual_inner_length_mm": 1.0,
                            "coupling.magnets.manual_outer_length_mm": 1.0})
    m = run(inp).model
    _, _, pair = _free_case()
    reference = tr.end_factor_3d(pair, 1.0 * MM, _default_per_length(), pair.half_pitch)
    assert rel_err(m.f_end, reference) <= TOL_MODEL, mismatch(
        f"end-effect factor at L = 1 mm (engine pull-out {m.pullout_Nm:.4f} N·m)", "Calculator!C92, C93",
        m.f_end, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_axial_overhang_effect():
    """Outer ring BX042SH (25.4 mm long) over inner B842SH (12.7 mm). The engine uses the active length min(L_i, L_o)
    (Calculator!C33), so its pull-out ratio to the equal-length case is 1. The reference is the same ratio in 3D, at
    half a pole pitch. Ratios cancel the 2D model error. Tolerance TOL_MODEL.
    Root-cause group: T3-RC4 (active length = min(L_i, L_o) ignores the overhang)."""
    changes = (("coupling.magnets.part_outer", "BX042SH"),)
    _, base_res, _ = _free_case()
    _, over_res, _ = _free_case(changes)
    engine = _engine_raw(over_res) / _engine_raw(base_res)
    reference = _t3d(changes) / _t3d()
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "pull-out ratio, 25.4 mm outer over 12.7 mm inner vs equal lengths", "Calculator!C33, C93/C42",
        engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_raw_model():
    """Calibration sheet's original model before the 0.95 factor (2D torque x end factor) vs the 3D pull-out of the
    free-space prototype (20 x B842SH, hub apothem 9.85 mm, 1.4 mm flat gap, 20 °C), built from the Calibration inputs
    alone. Tolerance TOL_MODEL. Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    cal = run().calibration
    engine = cal.torque_2d_Nm * cal.f_end
    reference = _prototype_3d()
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "prototype model before the calibration factor vs 3D", "Calibration!C43*C38", engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_measured_correction():
    """The one-point correction C9 = 0.95 x measured / (raw model x 0.95) = measured / raw model is meant to absorb
    model error only. If it did only that, it would equal the physics-implied correction 3D pull-out / raw model.
    The difference is bench vs physics: measurement, the ASSUMED 20 °C test temperature (C16), the actual Br vs the
    nominal 1.29 T. Tolerance TOL_MODEL. Root-cause group: T3-RC5 (bench measurement vs physics; calibration premise)."""
    cal = run().calibration
    reference = _prototype_3d() / (cal.torque_2d_Nm * cal.f_end)
    assert rel_err(cal.f_cal_updated, reference) <= TOL_MODEL, mismatch(
        f"updated calibration factor vs 3D / raw model (measured {defaults().calibration.measured_torque_Nm} N·m, "
        f"3D {_prototype_3d():.4f} N·m)", "Calibration!C9, C5, C43, C38", cal.f_cal_updated, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_vs_supplied_3d_results():
    """The Calibration sheet's two supplied 3D results (C47: 2.06 N·m at a 1.0 mm corner gap; C48: 1.7 N·m at 1.5 mm;
    "supplied context, not rerun") feed the interpolation C49/C50. They are compared with the 3D pull-out (max over
    angle) of the same free-space prototype at those corner gaps, built from the Calibration inputs alone. One check.
    Tolerance TOL_MODEL. Root-cause group: T3-RC6 (supplied 3D inputs vs free-space physics)."""
    c = defaults().calibration
    rows = []
    for gap_mm, supplied in zip(FEA_LABEL_GAPS_MM, (c.fea_torque1_Nm, c.fea_torque2_Nm)):
        proto = dataclasses.replace(c, gap_definition=0, spacing_mm=gap_mm)
        t3d, _ = tr.ring_pullout_3d(tr.ring_pair_from_prototype(proto))
        rows.append((f"corner gap {gap_mm} mm", supplied, t3d))
    _assert_all_within("supplied 3D prototype results (engine column) vs 3D pull-out", "Calibration!C47, C48", rows,
                       TOL_MODEL)


# =========================================================================== scaling and limit laws (engine only)
@pytest.mark.family("torque")
@pytest.mark.parametrize("length_mm", [1.0, 6.35, 25.4, 50.8])
def test_pullout_affine_in_active_length(length_mm):
    """T2D ~ L and f_end = 1 - c_end*pitch/L give T(L) = a*(L - c_end*pitch), so pull-out(L) / pull-out(12.7 mm) =
    (L - c_end*pitch) / (12.7 - c_end*pitch). Same algebra: TOL_ALGEBRA."""
    base = run(vary(defaults(), MANUAL)).model
    new = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})).model
    c_end, pitch = defaults().coupling.c_end, base.pole_pitch_mm
    expected = base.pullout_Nm * (length_mm - c_end * pitch) / (base.active_length_mm - c_end * pitch)
    assert rel_err(new.pullout_Nm, expected) <= TOL_ALGEBRA, mismatch(
        f"pull-out at L = {length_mm} mm", "Calculator!C33, C65, C93", new.pullout_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("length_mm", [1.0, 6.35, 25.4, 50.8])
def test_torque_2d_proportional_to_active_length(length_mm):
    """2D torque = tau * 2 pi R_g^2 L is exactly proportional to L at a fixed cross-section. TOL_ALGEBRA."""
    base = run(vary(defaults(), MANUAL)).model
    new = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})).model
    expected = base.torque_2d_Nm * length_mm / base.active_length_mm
    assert rel_err(new.torque_2d_Nm, expected) <= TOL_ALGEBRA, mismatch(
        f"2D torque at L = {length_mm} mm", "Calculator!C33, C91", new.torque_2d_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("changes", [
    {**MANUAL, "coupling.magnets.manual_inner_br_T": 1.1, "coupling.magnets.manual_outer_br_T": 1.1},
    {**MANUAL, "coupling.magnets.manual_inner_br_T": 1.1},
    {"coupling.magnets.part_inner": "B842-N52", "coupling.magnets.part_outer": "B842-N52"},
], ids=["both-rings-1.1T", "inner-only-1.1T", "library-N52"])
def test_pullout_proportional_to_br_product(changes):
    """Linear magnetostatics: every field harmonic is proportional to its ring's Br, so pull-out ~ Br_i * Br_o
    (~ Br^2 for equal rings). Steel circuit, f_cal 0.95 in every case. TOL_ALGEBRA."""
    base = run().model
    new = run(vary(defaults(), changes)).model
    expected = base.pullout_Nm * (new.inner_br_T * new.outer_br_T) / (base.inner_br_T * base.outer_br_T)
    assert rel_err(new.pullout_Nm, expected) <= TOL_ALGEBRA, mismatch(
        "pull-out vs Br_i*Br_o", "Calculator!C21, C31, C93", new.pullout_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
def test_pullout_decreases_monotonically_with_face_gap(backiron):
    """Field harmonics decay with distance (e^{-k g}), so pull-out must fall strictly as the face gap grows. The rate
    is checked against 3D in test_3d_free_space_decay_beyond_gap_sweep (free space only)."""
    gaps = [0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0, 100.0]
    vals = [run(vary(defaults(), {"coupling.backiron": backiron, "metal.face_gap_mm": g})).model.pullout_Nm for g in gaps]
    for (g0, v0), (g1, v1) in zip(zip(gaps, vals), zip(gaps[1:], vals[1:])):
        assert v1 < v0, mismatch(f"pull-out at face gap {g1} mm vs {g0} mm (must be lower), backiron {backiron}",
                                 "Metal design!C119 -> Calculator!C93", v1, v0, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
def test_pullout_increases_monotonically_with_fill(backiron):
    """At half a pitch the other ring's fundamental tangential field has one sign across each whole pole, so widening
    the blocks (fill 0.27-0.95, below the cap of 1) must raise pull-out. Width changes nothing else, because the outer
    face apothem is the inner face radius plus the face gap."""
    widths = [2.0, 3.0, 4.0, 5.0, 6.0, 6.35, 7.0]
    rows = [run(vary(defaults(), {**MANUAL, "coupling.backiron": backiron, "coupling.magnets.manual_inner_width_mm": w,
                                  "coupling.magnets.manual_outer_width_mm": w})).model for w in widths]
    max_fill = max(max(r.fill_inner, r.fill_outer) for r in rows)
    assert max_fill < 1, mismatch("precondition: every fill below the cap of 1 (the law holds only there)",
                                  "Calculator!C66, C67", max_fill, 1.0, 0.0)
    for (w0, r0), (w1, r1) in zip(zip(widths, rows), zip(widths[1:], rows[1:])):
        assert r1.pullout_Nm > r0.pullout_Nm, mismatch(
            f"pull-out at width {w1} mm (fill {r1.fill_inner:.3f}/{r1.fill_outer:.3f}) vs {w0} mm (must be higher), "
            f"backiron {backiron}", "Calculator!C66, C67, C93", r1.pullout_Nm, r0.pullout_Nm, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("npole, n", [(60, 5), (200, 1)])
@pytest.mark.parametrize("circuit", ["iron", "free"])
def test_geometry_factor_thick_magnet_limit(npole, n, circuit):
    """Thick-magnet limit k*t -> infinity: two semi-infinite arrays give S_n = e^{-k g} / 2 with or without back iron.
    S_n depends on thickness only through k_n*t, so raising the pole count reaches k_n*t > 22 at the default
    thickness and gap. The remaining relative deviation, about 2 e^{-k t} < 6e-10, is below TOL_ALGEBRA."""
    m = run(vary(defaults(), {"coupling.npole": npole})).model
    k = getattr(m, f"k{n}")
    kt = k * min(m.inner_thickness_mm, m.outer_thickness_mm) * MM
    field = f"s{n}_{circuit}"
    cells = f"{_cell(m, field)}, {_cell(m, f'k{n}')}, Calculator!C57"
    assert kt > 22, mismatch(f"precondition: k{n}*t in the thick-magnet regime, {npole} poles", cells, kt, 22.0, 0.0)
    s = getattr(m, field)
    expected = math.exp(-k * m.face_gap_mm * MM) / 2
    assert rel_err(s, expected) <= TOL_ALGEBRA, mismatch(
        f"S{n} ({circuit}) thick-magnet limit, {npole} poles, k*t = {kt:.1f}", cells, s, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("n", [1, 3, 5])
def test_geometry_factor_thin_magnet_iron_gain(n):
    """Thin-magnet limit (t = 1e-4 mm): back iron multiplies the free-space interaction by 4 / (1 - e^{-2 k g}).
    By the method of images, each thin ring's image in its own backing doubles its field (2 x 2). Repeated
    reflections between the two iron surfaces, a distance g apart, sum to 1 / (1 - e^{-2 k g}). The first-order
    correction k(t_i + t_o)|coth(k g) - 1/2| is below 3e-4, so the tolerance is 1e-3."""
    m = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_thickness_mm": 1e-4,
                              "coupling.magnets.manual_outer_thickness_mm": 1e-4})).model
    k, g = getattr(m, f"k{n}"), m.face_gap_mm * MM
    ratio = getattr(m, f"s{n}_iron") / getattr(m, f"s{n}_free")
    expected = 4 / (1 - math.exp(-2 * k * g))
    assert rel_err(ratio, expected) <= 1e-3, mismatch(
        f"S{n} iron/free ratio, thin magnets", f"{_cell(m, f's{n}_iron')}/{_cell(m, f's{n}_free')}",
        ratio, expected, 1e-3)


# =========================================================================== temperature scaling
@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 50, 80, 150])
@pytest.mark.parametrize("ring", ["inner", "outer"])
def test_br_linear_in_temperature(temp_C, ring):
    """Br(T) = Br20 * (1 + alpha_Br * (T - 20)), the linear reversible coefficient (planar.br_at). TOL_ALGEBRA."""
    m = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model
    alpha = defaults().calibration.alpha_br_per_C
    br_t = getattr(m, f"br_{ring}_T_op")
    expected = planar.br_at(getattr(m, f"{ring}_br_T"), alpha, temp_C)
    assert rel_err(br_t, expected) <= TOL_ALGEBRA, mismatch(
        f"{ring} Br at {temp_C} °C", f"{_cell(m, f'br_{ring}_T_op')}, Calibration!C22", br_t, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
def test_alpha_br_within_sintered_ndfeb_band():
    """Literature: sintered NdFeB datasheets give a reversible Br coefficient of about -0.09 to -0.13 %/°C
    (N42SH is typically quoted at -0.12 %/°C, e.g. in the Arnold Magnetic Technologies and K&J Magnetics grade
    tables). Check: the default alpha lies inside that band (centre -0.11 %/°C, half-width 0.02 %/°C)."""
    alpha = defaults().calibration.alpha_br_per_C
    centre, half = -0.0011, 0.0002
    assert rel_err(alpha, centre) <= half / abs(centre), mismatch(
        "alpha_Br vs sintered NdFeB band", "Calibration!C22", alpha, centre, half / abs(centre))


@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 80, 150])
def test_pullout_scales_as_br_squared_with_temperature(temp_C):
    """Both rings share alpha_Br, so pull-out(T) = pull-out(20 °C) * (1 + alpha (T - 20))^2. TOL_ALGEBRA.
    (This task owns the Br^2 temperature law on C93.)"""
    alpha = defaults().calibration.alpha_br_per_C
    t20 = run(vary(defaults(), {"coupling.op_temp_C": 20})).model.pullout_Nm
    t_t = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model.pullout_Nm
    expected = t20 * (planar.br_at(1.0, alpha, temp_C)) ** 2
    assert rel_err(t_t, expected) <= TOL_ALGEBRA, mismatch(
        f"pull-out at {temp_C} °C", "Calculator!C93, Calibration!C22", t_t, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 50, 80, 150])
def test_pullout_20C_field_equals_pullout_at_20C(temp_C):
    """The 20 °C pull-out reported at any operating temperature (rescaled by Br20^2 / Br(T)^2) equals the pull-out
    obtained by operating at 20 °C. TOL_ALGEBRA."""
    reported = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model.pullout_20C_Nm
    expected = run(vary(defaults(), {"coupling.op_temp_C": 20})).model.pullout_Nm
    assert rel_err(reported, expected) <= TOL_ALGEBRA, mismatch(
        f"20 °C pull-out reported at {temp_C} °C", "Calculator!C94", reported, expected, TOL_ALGEBRA)


# =========================================================================== calibration
_CAL_SCENARIOS = {
    "default": {},
    "corner-gap-1.0mm-35C-measured-2.0": {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.0,
                                          "calibration.test_temp_C": 35, "calibration.measured_torque_Nm": 2.0},
}
_CAL_FIELDS = [  # (engine CalibrationResults field, value from the re-derivation)
    ("poles_per_ring", lambda d: d["geometry"].poles),
    ("face_radius_mm", lambda d: d["geometry"].face_radius_mm),
    ("corner_radius_mm", lambda d: d["geometry"].corner_radius_mm),
    ("corner_gap_mm", lambda d: d["geometry"].corner_gap_mm),
    ("outer_face_apothem_mm", lambda d: d["geometry"].outer_face_apothem_mm),
    ("flat_gap_mm", lambda d: d["geometry"].flat_gap_mm),
    ("gap_radius_mm", lambda d: d["geometry"].gap_radius_mm),
    ("fill_inner", lambda d: d["geometry"].fill_inner),
    ("fill_outer", lambda d: d["geometry"].fill_outer),
    ("br_test_T", lambda d: d["br_test_T"]),
    ("pole_pitch_mm", lambda d: d["geometry"].pole_pitch_mm),
    ("f_end", lambda d: d["f_end"]),
    ("tau1_Pa", lambda d: d["tau_n_Pa"][1]),
    ("tau3_Pa", lambda d: d["tau_n_Pa"][3]),
    ("tau5_Pa", lambda d: d["tau_n_Pa"][5]),
    ("torque_2d_Nm", lambda d: d["torque_2d_Nm"]),
    ("model_torque_Nm", lambda d: d["model_torque_Nm"]),
    ("original_model_Nm", lambda d: d["model_torque_Nm"]),
    ("model_error", lambda d: d["model_error"]),
    ("measured_over_model", lambda d: d["measured_over_model"]),
    ("f_cal_updated", lambda d: d["f_cal_updated"]),
]


@pytest.mark.family("torque")
@pytest.mark.parametrize("scenario", list(_CAL_SCENARIOS))
@pytest.mark.parametrize("field, pick", _CAL_FIELDS, ids=[f[0] for f in _CAL_FIELDS])
def test_prototype_calibration_rederived(scenario, field, pick):
    """One-point prototype correction re-derived from the Calibration inputs (reference prototype_calibration:
    polygon geometry, free-space planar shear at half a pitch, 2 pi R_g^2 L, end factor, 0.95, measured/model).
    Same algebra: TOL_ALGEBRA."""
    inp = vary(defaults(), _CAL_SCENARIOS[scenario])
    cal = run(inp).calibration
    engine = getattr(cal, field)
    reference = pick(tr.prototype_calibration(inp.calibration))
    assert rel_err(engine, reference) <= TOL_ALGEBRA, mismatch(
        f"calibration {field} ({scenario})", _cell(cal, field), engine, reference, TOL_ALGEBRA)


@pytest.mark.family("torque")
def test_calculator_reproduces_prototype_model():
    """The Calculator run on the prototype layout (no iron, hub 9.85 mm, 1.4 mm gap, 20 °C) with the original 0.95
    factor gives the re-derived prototype model torque, so the two sheets agree. TOL_ALGEBRA."""
    inp = vary(defaults(), {"coupling.backiron": 0, "coupling.inner_back_apothem_mm": 9.85,
                            "metal.face_gap_mm": 1.4, "coupling.op_temp_C": 20})
    m = run(inp).model
    engine = m.pullout_Nm / m.f_cal * inp.calibration.f_cal_original
    reference = tr.prototype_calibration(inp.calibration)["model_torque_Nm"]
    assert rel_err(engine, reference) <= TOL_ALGEBRA, mismatch(
        "Calculator on the prototype layout x 0.95", "Calculator!C93/C42 vs Calibration!C6", engine, reference,
        TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("changes", [
    {},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.3},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.5},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 0.8},
    {"calibration.fea_torque1_Nm": 2.3, "calibration.fea_torque2_Nm": 1.5},
], ids=["default", "corner-1.3mm", "corner-1.5mm-span-end", "corner-0.8mm-outside", "other-3D-torques"])
def test_fea_interpolation(changes):
    """Linear interpolation of the two supplied 3D results at the prototype corner gap, and its error vs the
    measurement. Off the 1.0-1.5 mm span (ends included) the engine reports 'outside range' / 'n.a.'. The reference is
    numpy.interp over FEA_LABEL_GAPS_MM with the fea_torque inputs. TOL_ALGEBRA."""
    inp = vary(defaults(), changes)
    c, cal = inp.calibration, run(inp).calibration
    cg = tr.prototype_geometry(c).corner_gap_mm
    if FEA_LABEL_GAPS_MM[0] <= cg <= FEA_LABEL_GAPS_MM[1]:
        ref = float(np.interp(cg, FEA_LABEL_GAPS_MM, [c.fea_torque1_Nm, c.fea_torque2_Nm]))
        assert rel_err(cal.fea_interp_Nm, ref) <= TOL_ALGEBRA, mismatch(
            f"3D interpolation at corner gap {cg:.4f} mm", "Calibration!C49, C47, C48", cal.fea_interp_Nm, ref,
            TOL_ALGEBRA)
        ref_err = ref / c.measured_torque_Nm - 1
        assert rel_err(cal.fea_interp_error, ref_err) <= TOL_ALGEBRA, mismatch(
            "3D interpolation error vs measured", "Calibration!C50, C5", cal.fea_interp_error, ref_err, TOL_ALGEBRA)
    else:
        ok = (cal.fea_interp_Nm, cal.fea_interp_error) == ("outside range", "n.a.")
        assert ok, mismatch(
            f"3D interpolation off the span at corner gap {cg:.4f} mm: engine {cal.fea_interp_Nm!r} / "
            f"{cal.fea_interp_error!r}, expected 'outside range' / 'n.a.' (1 = match)", "Calibration!C49, C50",
            float(ok), 1.0, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
@pytest.mark.parametrize("field, backed", [("pullout_iron_Nm", True), ("pullout_noiron_Nm", False)],
                         ids=["steel-C95", "free-C96"])
def test_circuit_variant_pullouts_rederived(backiron, field, backed):
    """C95 (same layout, steel circuit, original 0.95 factor) and C96 (same layout, free space, the selected factor
    C42), re-derived end to end from the engine's geometry fields with planar.planar_shear_stress: shear at half a
    pitch (harmonics 1, 3, 5) x 2 pi R_g^2 L x f_end x factor. Same algebra: TOL_ALGEBRA."""
    inp = vary(defaults(), {"coupling.backiron": backiron})
    m = run(inp).model
    tau = planar.planar_shear_stress(m.br_inner_T_op, m.br_outer_T_op, m.fill_inner, m.fill_outer, m.pole_pitch_mm * MM,
                                     m.inner_thickness_mm * MM, m.outer_thickness_mm * MM, m.face_gap_mm * MM,
                                     m.pole_pitch_mm * MM / 2, inp.coupling.mu0, backed, (1, 3, 5))
    factor = inp.calibration.f_cal_original if backed else m.f_cal
    expected = (planar.torque_on_cylinder(tau, m.gap_radius_mm * MM, m.active_length_mm * MM)
                * planar.end_factor(inp.coupling.c_end, m.pole_pitch_mm, m.active_length_mm) * factor)
    engine = getattr(m, field)
    assert rel_err(engine, expected) <= TOL_ALGEBRA, mismatch(
        f"{field}, backiron {backiron}", f"{_cell(m, field)}, Calculator!C42, C64-C70, C92", engine, expected,
        TOL_ALGEBRA)


_RULE_CASES = [  # (label, changes, measured correction expected?)
    ("steel-circuit", {}, False),
    ("prototype-circuit", {"coupling.backiron": 0}, True),
    ("no-iron-8-poles", {"coupling.backiron": 0, "coupling.npole": 8}, False),
    ("no-iron-inner-B842", {"coupling.backiron": 0, "coupling.magnets.part_inner": "B842"}, False),
    ("no-iron-outer-B842", {"coupling.backiron": 0, "coupling.magnets.part_outer": "B842"}, False),
    ("no-iron-8-poles-8-pole-prototype", {"coupling.backiron": 0, "coupling.npole": 8, "calibration.total_magnets": 16},
     True),
]


@pytest.mark.family("torque")
@pytest.mark.parametrize("label, changes, measured", _RULE_CASES, ids=[c[0] for c in _RULE_CASES])
def test_calibration_factor_selection_rule(label, changes, measured):
    """README / model docstring rule: the measured correction applies only to the prototype's circuit (no back iron,
    the same pole count as the prototype, B842SH in both rings); otherwise the original 0.95 applies. The expected
    value comes from the re-derived correction. TOL_ALGEBRA."""
    inp = vary(defaults(), changes)
    engine = run(inp).model.f_cal
    expected = tr.prototype_calibration(inp.calibration)["f_cal_updated"] if measured else inp.calibration.f_cal_original
    assert rel_err(engine, expected) <= TOL_ALGEBRA, mismatch(
        f"calibration factor, {label}", "Calculator!C42, Calibration!C9, C24", engine, expected, TOL_ALGEBRA)


# =========================================================================== constants
@pytest.mark.family("constants")
def test_pullout_inversely_proportional_to_mu0():
    """tau = B_i B_o / (2 mu0) * ..., so pull-out(exact mu0) / pull-out(rounded) = MU0_rounded / MU0_exact.
    TOL_ALGEBRA."""
    rounded = run().model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.mu0": MU0_EXACT})).model.pullout_Nm
    expected = rounded * defaults().coupling.mu0 / MU0_EXACT
    assert rel_err(exact, expected) <= TOL_ALGEBRA, mismatch(
        "pull-out with exact mu0", "Calculator!C43, C93", exact, expected, TOL_ALGEBRA)


@pytest.mark.family("constants")
def test_rounded_mu0_effect_on_pullout_is_negligible():
    """Rounded MU0 = 1.256637e-6 vs 4 pi 1e-7: the effect on pull-out must be negligible, i.e. below 1e-6. For
    comparison, the Calculator shows 4 significant figures (5e-4) and the 3D comparison bar is 2e-2."""
    rounded = run().model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.mu0": MU0_EXACT})).model.pullout_Nm
    assert rel_err(rounded, exact) < 1e-6, mismatch("pull-out, rounded vs exact mu0", "Calculator!C43, C93",
                                                    rounded, exact, 1e-6)


@pytest.mark.family("constants")
def test_calibrated_noiron_pullout_independent_of_mu0():
    """With the measured correction in use (prototype circuit), the correction absorbs mu0: changing mu0 in both
    sheets leaves the calibrated pull-out unchanged. TOL_ALGEBRA."""
    base = run(vary(defaults(), {"coupling.backiron": 0})).model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.backiron": 0, "coupling.mu0": MU0_EXACT,
                                  "calibration.mu0": MU0_EXACT})).model.pullout_Nm
    assert rel_err(exact, base) <= TOL_ALGEBRA, mismatch(
        "calibrated no-iron pull-out, exact vs rounded mu0 in both sheets", "Calculator!C43, Calibration!C25, C93",
        exact, base, TOL_ALGEBRA)
