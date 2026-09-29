"""Task 7: every gap-sweep and pole-sweep row against an independent single evaluation of the README torque model
(audit.references.sweep_ref: Task 3's planar formulas and Task 4's coupling geometry, SI units, no engine
arithmetic), including the fit/status flags.

Upstream values taken from the engine as data, not re-derived here: the magnet dimensions and Br20 resolved from the
library (res.model.*), the Calculator's calibration factor (res.model.f_cal) and the Calculator's corner gap
(res.model.corner_gap_mm; Task 4 checks both of the latter's geometry).
"""
from __future__ import annotations

import math

import pytest

from audit.common import TOL_ALGEBRA, ScenarioCache, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import sweep_ref as ref
# Sweep grids (the rows to check) and workbook column letters (failure labels); no engine arithmetic is reused.
from magcoupling.sweeps import GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES, SWEEP_COLUMNS

SCENARIOS = {
    "defaults": {},
    "no_iron": {"coupling.backiron": 0},                 # free-space S_n, measured calibration factor in the gap sweep
    "arcs": {"coupling.faceted": 0},
    "npole16": {"coupling.npole": 16},                   # inner flats too narrow at the current hub apothem
    "wide_outer": {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 12.0},
    "hot80": {"coupling.op_temp_C": 80},
}
CASES = ScenarioCache(SCENARIOS)
NUMERIC_FIELDS = [f for f in SWEEP_COLUMNS if f not in ("variable", "status")]
FIRST_ROW = 6   # workbook row of the first sweep row


def _setup(inp, res) -> ref.SweepSetup:
    ci, md, m = inp.coupling, inp.metal, res.model
    return ref.SweepSetup(
        faceted=ci.faceted, backiron=ci.backiron, t_i_mm=m.inner_thickness_mm, w_i_mm=m.inner_width_mm,
        t_o_mm=m.outer_thickness_mm, w_o_mm=m.outer_width_mm, length_mm=min(m.inner_length_mm, m.outer_length_mm),
        br_i20_T=m.inner_br_T, br_o20_T=m.outer_br_T, alpha_per_C=inp.calibration.alpha_br_per_C,
        op_temp_C=ci.op_temp_C,
        bond_inner_mm=md.bond_inner_mm, bond_outer_mm=md.bond_outer_mm, cup_wall_corner_mm=md.cup_wall_corner_mm,
        bore_mm=ci.bore_mm, keyway_mm=ci.keyway_depth_mm, c_end=ci.c_end, mu0=ci.mu0, gear_ratio=ci.gear_ratio,
        gear_eff=ci.gear_efficiency,
        required_floor_Nm=ref.required_floor(ci.drive_torque_Nm, ci.drive_safety_factor, md.required_min_Nm),
        max_diameter_mm=md.max_diameter_mm)


def _row_items(sheet: str, scenario: str, j: int, row, variable: float, expected: dict) -> list:
    """The row's variable, every numeric column at TOL_ALGEBRA and the status text."""
    r = FIRST_ROW + j
    items = [(f"{scenario} row {j} variable", f"{sheet}!B{r}", row.variable, variable, 0.0)]
    items += [(f"{scenario} row {j} {f}", f"{sheet}!{SWEEP_COLUMNS[f]}{r}", getattr(row, f), expected[f], TOL_ALGEBRA)
              for f in NUMERIC_FIELDS]
    items.append(text_item(f"{scenario} row {j} status", f"{sheet}!AA{r}", row.status, expected["status"]))
    return items


@pytest.mark.family("sweeps")
def test_gap_sweep_rows():
    """Every gap-sweep row, in every scenario, equals a single evaluation at that corner gap with the current poles
    and hub apothem and the Calculator's calibration factor: all 25 columns and the status."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for j, gap in enumerate(GAP_SWEEP_CORNER_GAPS_MM):
            expected = ref.sweep_row(setup, inp.coupling.npole, inp.coupling.inner_back_apothem_mm, gap,
                                     res.model.f_cal)
            items += _row_items("Gap sweep", sc, j, res.gap_sweep[j], gap, expected)
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_rows_given_apothem():
    """Every pole-sweep row, in every scenario, equals a single evaluation at that pole count and the row's own inner
    apothem, at the Calculator's corner gap and the original calibration factor (the measured correction does not
    transfer). The apothem rule itself is checked by the two apothem tests below."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for j, npole in enumerate(POLE_SWEEP_POLES):
            row = res.pole_sweep[j]
            expected = ref.sweep_row(setup, npole, row.inner_apothem_mm, res.model.corner_gap_mm,
                                     inp.calibration.f_cal_original)
            items += _row_items("Pole sweep", sc, j, row, npole, expected)
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_apothem_rule_as_stated():
    """Pole sweep column C against the rule as the sheet note states it (engine sweeps.py module and pole_sweep
    docstrings, ported from the workbook): 'the smallest inner apothem that fits the block width (+0.05 mm) and the
    keyed bore wall (2.5 mm)', i.e. max(w_i/(2·tan(pi/N)) + 0.05, bore/2 + keyway + 2.5), with the wall measured from
    the block-back apothem because the note does not mention the bondline. Whether that wall matches the
    Calculator's own wall definition is test_pole_sweep_apothem_keeps_calculator_hub_wall."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        ci = inp.coupling
        for j, npole in enumerate(POLE_SWEEP_POLES):
            items.append((f"{sc} N={npole} inner apothem", f"Pole sweep!C{FIRST_ROW + j}",
                          res.pole_sweep[j].inner_apothem_mm,
                          ref.stated_min_inner_apothem(npole, res.model.inner_width_mm, ci.bore_mm, ci.keyway_depth_mm),
                          TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_apothem_keeps_calculator_hub_wall():
    """Definition consistency. The Calculator defines the hub wall past the keyway as apothem - inner bondline -
    bore/2 - keyway (Calculator!C53, engine field hub_wall_past_key_mm, help text 'Keep ≥ ~2.5 mm'; Task 4 verifies
    that definition, and Task 4's coupling_geometry computes it here). The pole sweep's rule uses the same 2.5 mm
    but measures it from the block-back apothem, without the bondline. This check asks whether each pole-sweep
    apothem keeps the Calculator's C53 wall at 2.5 mm; a shortfall equal to the bondline means the two sheets use
    different wall definitions, not an arithmetic slip. Every pole count in one check (one root cause)."""
    inp, res = CASES["defaults"]
    setup = _setup(inp, res)
    items = []
    for j, npole in enumerate(POLE_SWEEP_POLES):
        a = res.pole_sweep[j].inner_apothem_mm
        wall = ref.geometry_at_corner_gap(setup, npole, a, res.model.corner_gap_mm).hub_wall_past_key
        items.append(flag_item(f"N={npole}: apothem {a:.3f} mm leaves {wall:.3f} mm of hub wall past the keyway "
                               f"(Calculator C53 definition, bondline {setup.bond_inner_mm:g} mm) >= "
                               f"{ref.KEYED_WALL_MM:g} mm", f"Pole sweep!C{FIRST_ROW + j} / Calculator!C53",
                               wall >= ref.KEYED_WALL_MM - 1e-9))
    assert_all(items)


@pytest.mark.family("sweeps")
def test_sweep_status_covers_every_branch():
    """Across the scenarios the sweeps reach all five status texts, so every status branch was compared above."""
    seen = {r.status for sc in SCENARIOS for r in CASES[sc][1].gap_sweep + CASES[sc][1].pole_sweep}
    missing = sorted({ref.STATUS_INNER_NARROW, ref.STATUS_OUTER_NARROW, ref.STATUS_OD, ref.STATUS_BELOW_MIN,
                      ref.STATUS_NOMINAL} - seen)
    assert_all([flag_item(f"every status branch reached (missing: {missing})", "Gap sweep!AA, Pole sweep!AA",
                          not missing)])


@pytest.mark.family("sweeps")
def test_sweep_below_hot_minimum_threshold():
    """Sweep status 'below hot minimum' exactly when the row's nominal pull-out < the required floor: pinned on the
    1.0 mm corner-gap row (fits, inside the envelope) with the floor equal to the row's pull-out and one ulp above."""
    j = GAP_SWEEP_CORNER_GAPS_MM.index(1)
    x = run().gap_sweep[j].pullout_op_Nm
    at = run(vary(defaults(), {"metal.required_min_Nm": x}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(x, math.inf)}))
    cell = f"Gap sweep!AA{FIRST_ROW + j}"
    assert_all([text_item("floor = row pull-out", cell, at.gap_sweep[j].status, ref.STATUS_NOMINAL),
                text_item("floor one ulp above the row pull-out", cell, above.gap_sweep[j].status,
                          ref.STATUS_BELOW_MIN)])
