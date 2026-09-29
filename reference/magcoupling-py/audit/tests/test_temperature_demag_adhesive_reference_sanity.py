"""Reference sanity for Task 5 (not engine checks): the demagnetization, rating, adhesive and Volkersen references in
``audit.references.demag_adhesive`` and the force and energy functions Task 5 appends to Task 4's
``audit.references.slab_field2d``, each against a known answer. Task 4's sanity file covers the slab field itself
(long-pitch circuit limit, mirror symmetry, iron-face limit). Nothing here calls the engine.
"""
from __future__ import annotations

import math

import pytest
from scipy.integrate import trapezoid

from audit.common import MU0_EXACT, TOL_ALGEBRA, mismatch, rel_err
from audit.references import demag_adhesive as ref
from audit.references.slab_field2d import MagnetLayer, field_energy_J_per_m, layer_block_forces_N, strip_stress_N_per_m


@pytest.mark.family("temperature")
def test_sanity_knee_crossing_hand_cases():
    """Sanity: knee-crossing root-find reproduces hand-solved linear intersections; parallel lines raise."""
    # alpha = 0: 1000 (1 - 0.005 dT) = 500  ->  dT = 100
    t = ref.knee_crossing_C(500.0, 1000.0, -0.005, 1.0, 0.0)
    assert rel_err(t, 120.0) < TOL_ALGEBRA, mismatch("hand case alpha=0", "reference", t, 120.0, TOL_ALGEBRA)
    # 500 (1 - 0.001 dT) = 1000 (1 - 0.005 dT)  ->  4.5 dT = 500
    t = ref.knee_crossing_C(500.0, 1000.0, -0.005, 1.0, -0.001)
    expect = 20.0 + 500.0 / 4.5
    assert rel_err(t, expect) < TOL_ALGEBRA, mismatch("hand case alpha=-0.001", "reference", t, expect, TOL_ALGEBRA)
    # zero reverse field: the knee itself reaches zero at 20 + 1/0.005
    t = ref.knee_crossing_C(0.0, 1592.0, -0.005, 0.9, -0.0012)
    assert rel_err(t, 220.0) < TOL_ALGEBRA, mismatch("hand case H=0", "reference", t, 220.0, TOL_ALGEBRA)
    with pytest.raises(ValueError):
        ref.knee_crossing_C(500.0, 1000.0, -0.0025, 1.0, -0.005)   # equal slopes (1000*0.0025 = 500*0.005): never cross


@pytest.mark.family("temperature")
def test_sanity_load_line_textbook():
    """Sanity: Pc = 1, mu_rec = 1 puts the operating point at H = -Br/(2 mu0), B = Br/2; Pc -> inf gives H -> 0."""
    h = ref.load_line_reverse_field_kA_m(1.2, 1.0, MU0_EXACT)
    expect = 1.2 / (2 * MU0_EXACT) / 1000                       # 477.4648 kA/m
    assert rel_err(h, expect) < TOL_ALGEBRA, mismatch("H at Pc=1", "reference", h, expect, TOL_ALGEBRA)
    b_on_load_line = 1.0 * MU0_EXACT * h * 1000                 # |B| = Pc mu0 |H|
    assert rel_err(b_on_load_line, 0.6) < TOL_ALGEBRA, mismatch("B at Pc=1", "reference", b_on_load_line, 0.6, TOL_ALGEBRA)
    h_big = ref.load_line_reverse_field_kA_m(1.2, 1e9, MU0_EXACT)
    assert h_big < 1e-6, mismatch("H as Pc -> inf", "reference", h_big, 0.0, 1e-6)


@pytest.mark.family("temperature")
def test_sanity_grade_rating_table():
    """Sanity: grade parsing gives the K&J ratings quoted on its pages (N42 and N52 80 C, N42SH 150 C, N35AH 220 C)."""
    for grade, rating in (("N42", 80.0), ("N52", 80.0), ("N42SH", 150.0), ("N50M", 100.0), ("N35AH", 220.0)):
        got = ref.grade_max_operating_C(grade)
        assert got == rating, mismatch(f"K&J rating of {grade}", "reference", got, rating, 0.0)
    assert ref.part_rating_C("B842SH") == 150.0, mismatch("B842SH rating", "reference", ref.part_rating_C("B842SH"), 150.0, 0.0)
    assert ref.part_rating_C("") is None, mismatch("blank part has no rating", "reference", 1.0, 0.0, 0.0)
    for bad in ("N42XX", "SmCo26", "42SH"):
        with pytest.raises(ValueError):
            ref.grade_max_operating_C(bad)
    with pytest.raises(KeyError):
        ref.part_rating_C("B999")


@pytest.mark.family("temperature")
def test_sanity_adhesive_limit_and_bond_formulas():
    """Sanity: min(service, Tg - 20) with missing data, unit bond shear, unit centrifugal force, G = E/(2(1+nu))."""
    assert ref.adhesive_design_limit_C(120.0, None) == 120.0, mismatch("service only", "reference", 120.0, 120.0, 0.0)
    assert ref.adhesive_design_limit_C(200.0, 133.0) == 113.0, mismatch("Tg governs", "reference", 113.0, 113.0, 0.0)
    assert ref.adhesive_design_limit_C(100.0, 133.0) == 100.0, mismatch("service governs", "reference", 100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        ref.adhesive_design_limit_C(None, None)
    tau = ref.bond_shear_MPa(1.0, 1, 1000.0, 1.0)               # 1 N over 1 mm^2
    assert rel_err(tau, 1.0) < TOL_ALGEBRA, mismatch("unit bond shear", "reference", tau, 1.0, TOL_ALGEBRA)
    f = ref.centrifugal_force_N(1000.0, 60.0 / (2 * math.pi), 1000.0)   # 1 kg, 1 rad/s, 1 m
    assert rel_err(f, 1.0) < TOL_ALGEBRA, mismatch("unit centrifugal", "reference", f, 1.0, TOL_ALGEBRA)
    g = ref.shear_modulus_GPa(2.6, 0.3)
    assert rel_err(g, 1.0) < TOL_ALGEBRA, mismatch("G from E, nu", "reference", g, 1.0, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_sanity_volkersen_limits_and_numerical_solution():
    """Sanity: Volkersen closed form hits its short- and long-overlap limits and matches the finite-difference BVP.

    Short overlap (lam L -> 0): rigid adherends, the adhesive takes the free mismatch u = da dT L/2 at each end,
    tau = G da dT L / (2 eta). Long overlap (lam L -> inf): tau = G da dT / (eta lam). Finite differences on
    20,001 nodes carry O((lam dx)^2) ~ 1e-8 error, so 1e-6 is a safe tolerance. The force balance
    integral(0..L/2) tau dx = N1(0) is also checked on the numerical profile.
    """
    base = dict(G_Pa=0.5e9, eta_m=1e-4, E1_Pa=160e9, t1_m=3e-3, E2_Pa=205e9, t2_m=5e-3, d_alpha_per_C=13e-6, dT_C=70.0)
    lam = ref.volkersen_lambda_per_m(base["G_Pa"], base["eta_m"], base["E1_Pa"], base["t1_m"], base["E2_Pa"], base["t2_m"])
    short = 1e-3 / lam
    got = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=short)
    expect = base["G_Pa"] * base["d_alpha_per_C"] * base["dT_C"] * short / (2 * base["eta_m"])
    assert rel_err(got, expect) < 1e-6, mismatch("short-overlap limit", "reference", got, expect, 1e-6)
    long_ = 60.0 / lam
    got = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=long_)
    expect = base["G_Pa"] * base["d_alpha_per_C"] * base["dT_C"] / (base["eta_m"] * lam)
    assert rel_err(got, expect) < 1e-12, mismatch("long-overlap limit", "reference", got, expect, 1e-12)
    L = 12.7e-3
    closed = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=L)
    x, n1, tau = ref.volkersen_thermal_fd(**base, overlap_m=L)
    assert rel_err(tau[-1], closed) < 1e-6, mismatch("FD end shear", "reference", tau[-1], closed, 1e-6)
    assert rel_err(-tau[0], closed) < 1e-6, mismatch("FD antisymmetry", "reference", -tau[0], closed, 1e-6)
    mid = len(x) // 2
    balance = trapezoid(tau[mid:], x[mid:])
    assert rel_err(balance, n1[mid]) < 1e-6, mismatch("FD force balance", "reference", balance, n1[mid], 1e-6)


@pytest.mark.family("temperature")
def test_sanity_plate_pressure_is_b_squared_over_two_mu0():
    """Sanity: a fully filled layer between the plates with a very long period presses on each plate with the textbook
    magnetic pressure B^2/(2 mu0), B = Br t / h (magnetic circuit), and the strip stress is the same in the air gap.

    Each pitch holds one polarity transition, which lowers the strip-averaged pressure by a term linear in h/pitch
    (4.5e-4 at pitch 20 m); a two-pitch Richardson extrapolation (20 m and 40 m) removes it. The remainder is below
    1e-12 in practice, so the tolerance is 1e-8.
    """
    layer = [MagnetLayer(0.05, 3.17, 1.29, 1.0, 0.0)]
    h = 7.84
    p20 = strip_stress_N_per_m(h, layer, h, 20000.0, 200001)[0] / 20.0          # N/m per m of pitch = Pa
    p40 = strip_stress_N_per_m(h, layer, h, 40000.0, 200001)[0] / 40.0
    got = 2.0 * p40 - p20
    expect = (1.29 * 3.17 / h) ** 2 / (2.0 * MU0_EXACT)
    assert rel_err(got, expect) < 1e-8, mismatch("uniform-limit plate pressure", "reference", got, expect, 1e-8)
    in_gap = strip_stress_N_per_m(3.5, layer, h, 20000.0, 200001)[0] / 20.0
    assert rel_err(in_gap, p20) < 1e-12, mismatch("pressure at the plate vs in the gap", "reference", in_gap, p20, 1e-12)


@pytest.mark.family("temperature")
def test_sanity_slab_stress_is_divergence_free_and_matches_virtual_work():
    """Sanity: strip stress is the same at any height in air, and the plate force equals -dU/dh (virtual work).

    Maxwell stress is divergence-free in source-free air, so the strip integrals at two heights in the gap agree to
    round-off. The top-plate force from stress must equal minus the derivative of the field energy with respect to
    the plate position (central difference, step 1e-5 mm, truncation ~1e-10); tolerance 1e-8. Likewise the
    tangential force on the outer block must equal minus the derivative of the energy with respect to its shift.
    """
    layers = [MagnetLayer(0.05, 3.17, 1.29, 0.86, 0.0), MagnetLayer(4.62, 3.17, 1.29, 0.62, 2.0)]
    h, pitch = 7.84, 8.809
    a = strip_stress_N_per_m(3.5, layers, h, pitch)
    b = strip_stress_N_per_m(4.4, layers, h, pitch)
    for i, what in enumerate(("T_yy", "T_xy")):
        assert abs(a[i] - b[i]) < 1e-9 * max(abs(a[0]), 1.0), mismatch(f"{what} height invariance", "reference", a[i], b[i], 1e-9)
    dh = 1e-5
    f_energy = -(field_energy_J_per_m(layers, h + dh, pitch, 4001)
                 - field_energy_J_per_m(layers, h - dh, pitch, 4001)) / (2 * dh / 1000)
    f_stress = -strip_stress_N_per_m(h, layers, h, pitch, 4001)[0]
    assert rel_err(f_energy, f_stress) < 1e-8, mismatch("plate force: energy vs stress", "reference", f_energy, f_stress, 1e-8)
    forces = layer_block_forces_N(layers, h, pitch, 1000.0, 4001)
    ds = 1e-4

    def with_outer_shift(shift_mm: float) -> list[MagnetLayer]:
        return [layers[0], MagnetLayer(4.62, 3.17, 1.29, 0.62, shift_mm)]

    fx_energy = -(field_energy_J_per_m(with_outer_shift(2.0 + ds), h, pitch, 4001)
                  - field_energy_J_per_m(with_outer_shift(2.0 - ds), h, pitch, 4001)) / (2 * ds / 1000)
    assert rel_err(fx_energy, forces[1][0]) < 1e-8, mismatch("tangential force: energy vs stress", "reference",
                                                             fx_energy, forces[1][0], 1e-8)
    with pytest.raises(ValueError):
        layer_block_forces_N([MagnetLayer(0.0, 5.0, 1.29, 0.86), MagnetLayer(4.0, 3.0, 1.29, 0.62)], 8.0, pitch, 1.0)
