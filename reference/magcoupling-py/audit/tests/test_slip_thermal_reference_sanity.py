"""Sanity tests for what Task 6 adds besides engine checks: audit/references/slip_thermal.py and the
shared multi-cell helpers from audit/common.py (Task 1) that the Task 6 checks rely on.

These test the references and helpers, not the engine: every function the Task 6 checks rely on is
pinned to a published value, a known limit, or an independent numerical solution. Every test name
contains 'sanity' so the coverage table can leave them out of the engine confirmations.
"""
import math

import pytest
from scipy.integrate import solve_ivp

from audit.common import MU0_EXACT, ScenarioCache, assert_all, mismatch, rel_err, text_item
from audit.references import slip_thermal as ref


@pytest.mark.family("thermal")
def test_sanity_skin_depth_copper():
    """[HB] Copper, sigma = 5.8e7 S/m, at 60 Hz: delta = 66.1/sqrt(f) mm = 8.53 mm. Tolerance 1e-3 (the constant has 3 figures)."""
    got = ref.skin_depth_m(2 * math.pi * 60, 1.0, 5.8e7, MU0_EXACT)
    want = 66.1e-3 / math.sqrt(60)
    assert rel_err(got, want) <= 1e-3, mismatch("copper skin depth at 60 Hz", "reference", got, want, 1e-3)


@pytest.mark.family("thermal")
def test_sanity_halfspace_thin_skin_limit():
    """[S] At k*delta = 1e-3 the exact half-space loss equals the thin-skin form; the gap is (k delta)^2/4 = 2.5e-7, tolerance 1e-6."""
    omega, mu_r, sigma = 1000.0, 200.0, 4.5e6
    delta = ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    k = 1e-3 / delta
    got = ref.halfspace_loss_per_area(0.2, omega, mu_r, sigma, k, MU0_EXACT)
    want = ref.halfspace_loss_thin_skin(0.2, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-6, mismatch("half-space loss, k*delta -> 0", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
def test_sanity_halfspace_resistance_limited_limit():
    """[S] At k*delta = 1e3 (field decays as e^{-ky}, currents resistance limited) P/A -> sigma omega^2 bn^2 / (4 k^3); tolerance 1e-9."""
    omega, mu_r, sigma, bn = 1000.0, 200.0, 4.5e6, 0.2
    k = 1e3 / ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    want = sigma * omega ** 2 * bn ** 2 / (4 * k ** 3)
    assert rel_err(got, want) <= 1e-9, mismatch("half-space loss, k*delta -> inf", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_halfspace_matches_jackson_surface_loss():
    """[J] Thin-skin limit (k*delta = 1e-3): loss = mu omega delta |H_t|^2 / 4 with |H_t| = |gamma| bn / (k mu); tolerance 1e-6."""
    omega, mu_r, sigma, bn = 1000.0, 200.0, 4.5e6, 0.2
    mu = MU0_EXACT * mu_r
    delta = ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    k = 1e-3 / delta
    h_t = abs(ref.halfspace_gamma(omega, mu_r, sigma, k, MU0_EXACT)) * bn / (k * mu)
    want = mu * omega * delta * h_t ** 2 / 4
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-6, mismatch("half-space loss vs Jackson surface loss", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("k_delta", [0.1, 0.643, 3.0])
def test_sanity_halfspace_poynting_equals_ohmic_loss(k_delta):
    """[J] Energy conservation at finite k*delta: Poynting flux into the surface = integral of |J|^2/(2 sigma). Tolerance 1e-12."""
    omega, mu_r, sigma, bn = 1047.0, 200.0, 4.5e6, 0.207
    k = k_delta / ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    want = ref.halfspace_poynting_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-12, mismatch(f"ohmic loss vs Poynting flux, k*delta={k_delta}", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_surface_bn_infinite_permeability_is_the_image():
    """[S] mu_r -> inf: the surface field is the image-method (doubled) field. The correction |gamma|/(mu_r k)
    falls only as mu_r^-1/2 (1.4e-7 at mu_r = 1e12), so mu_r = 1e24 is used (1.4e-13); tolerance 1e-9."""
    got = ref.halfspace_surface_bn(0.4, 1047.0, 1e24, 4.5e6, 400.0, MU0_EXACT)
    assert rel_err(got, 0.4) <= 1e-9, mismatch("surface Bn, mu_r -> inf", "reference", got, 0.4, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_surface_bn_static_image_coefficient():
    """Non-conducting permeable half-space: image coefficient (mu_r - 1)/(mu_r + 1), so Bn = 2S mu_r/(mu_r + 1); tolerance 1e-12."""
    mu_r, source = 200.0, 0.2
    got = ref.halfspace_surface_bn(2 * source, 1047.0, mu_r, 0.0, 400.0, MU0_EXACT)
    want = source * (1 + (mu_r - 1) / (mu_r + 1))
    assert rel_err(got, want) <= 1e-12, mismatch("surface Bn, sigma = 0", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_thin_sheet_drag_peaks_at_reitz_speed():
    """[R] Thin-sheet eddy drag F = P/v peaks at v = w = 2/(mu0 sigma t); grid search, tolerance one grid step (1e-3)."""
    sigma, t = 2.5e7, 1e-3
    w = 2 / (MU0_EXACT * sigma * t)
    speeds = [w * (0.5 + 1e-3 * i) for i in range(1001)]
    drag = [ref.thin_sheet_loss_low_speed(0.3, v, sigma, t) * ref.thin_sheet_reaction_factor(v, sigma, t, MU0_EXACT) / v
            for v in speeds]
    v_peak = speeds[drag.index(max(drag))]
    assert rel_err(v_peak, w) <= 1e-3, mismatch("thin-sheet drag peak speed", "reference", v_peak, w, 1e-3)


@pytest.mark.family("thermal")
def test_sanity_thin_disk_matches_sheet_integrated_over_annulus():
    """A uniform field amplitude bn over an annulus: the disk loss (with <Bz^2> = bn^2/2, int r^2 dA = pi (r1^4 - r0^4)/2)
    equals the thin-sheet loss at v = omega r integrated over the annulus (midpoint rule, 20000 rings); tolerance 1e-8."""
    omega, sigma, t, bn, r0, r1 = 209.4, 2.5e7, 8e-4, 0.05, 0.0145, 0.0214
    n = 20000
    dr = (r1 - r0) / n
    rings = [r0 + (i + 0.5) * dr for i in range(n)]
    want = sum(ref.thin_sheet_loss_low_speed(bn, omega * r, sigma, t) * 2 * math.pi * r * dr for r in rings)
    got = ref.thin_disk_loss_low_speed(omega, sigma, t, bn ** 2 / 2 * math.pi * (r1 ** 4 - r0 ** 4) / 2)
    assert rel_err(got, want) <= 1e-8, mismatch("thin disk vs integrated thin sheet", "reference", got, want, 1e-8)


@pytest.mark.family("thermal")
def test_sanity_russell_norsworthy_value():
    """[RN] No overhang, a = kL/2 = 1: factor 1 - tanh(1) = 0.238406; tolerance 1e-12."""
    got = ref.sheet_end_factor(2.0, 1.0, 0.0)
    want = 1 - math.tanh(1.0)
    assert rel_err(got, want) <= 1e-12, mismatch("Russell-Norsworthy factor at a = 1", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_end_factor_long_sheet_limit():
    """A very long sheet (a = 1e6) has no end loss reduction: factor -> 1; tolerance 1e-5."""
    got = ref.sheet_end_factor(2.0, 1e6, 0.0)
    assert rel_err(got, 1.0) <= 1e-5, mismatch("end factor, L -> inf", "reference", got, 1.0, 1e-5)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("k,length,overhang", [(363.2, 0.0127, 0.0009), (200.0, 0.01, 0.0), (1.0, 1.0, 2.0)])
def test_sanity_end_factor_matches_finite_volume(k, length, overhang):
    """The closed-form end factor (own derivation, overhang included) matches a finite-volume solution that
    integrates |J|^2 directly. The FV error is second order: ~3e-7 at 4000 cells; tolerance 1e-5."""
    got = ref.sheet_end_factor(k, length, overhang)
    want = ref.sheet_end_factor_numeric(k, length, overhang, cells=4000)
    assert rel_err(got, want) <= 1e-5, mismatch(f"end factor vs finite volume (k={k}, L={length}, h={overhang})",
                                                "reference", got, want, 1e-5)


@pytest.mark.family("thermal")
def test_sanity_strip_loss_is_bertotti_classical_loss():
    """[B] sigma omega^2 B^2 d^2 / 24 equals pi^2 sigma d^2 f^2 B^2 / 6; tolerance 1e-12."""
    sigma, f, b, d = 2e6, 50.0, 1.5, 0.35e-3
    got = ref.strip_loss_per_volume(b, 2 * math.pi * f, sigma, d)
    want = math.pi ** 2 * sigma * d ** 2 * f ** 2 * b ** 2 / 6
    assert rel_err(got, want) <= 1e-12, mismatch("thin-strip loss vs Bertotti", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("aspect,beta", [(1.0, 0.141), (2.0, 0.229), (3.0, 0.263), (10.0, 0.312), (math.inf, 1 / 3)])
def test_sanity_rect_section_factor_matches_timoshenko(aspect, beta):
    """[TG] rect_section_factor = 3 beta with Timoshenko's torsion coefficients; tolerance half a unit in
    the table's third figure (0.0005/beta)."""
    got = ref.rect_section_factor(aspect)
    want = 3 * beta
    tol = 0.0005 / beta
    assert rel_err(got, want) <= tol, mismatch(f"rectangular-section factor at b/a={aspect}", "reference", got, want, tol)


@pytest.mark.family("thermal")
def test_sanity_loglog_slope():
    """y = 3 x^1.5 through x = 1000 and 2000 has slope 1.5; tolerance 1e-12."""
    got = ref.loglog_slope(1000.0, 3 * 1000.0 ** 1.5, 2000.0, 3 * 2000.0 ** 1.5)
    assert rel_err(got, 1.5) <= 1e-12, mismatch("log-log slope", "reference", got, 1.5, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_heat_capacity_water_and_steel():
    """[I] 1 kg water (4186 J/(kg K)) plus 500 g steel (473 J/(kg K)) = 4422.5 J/K; tolerance 1e-12."""
    got = ref.heat_capacity_J_K([(1000.0, 4186.0), (500.0, 473.0)])
    assert rel_err(got, 4422.5) <= 1e-12, mismatch("heat capacity sum", "reference", got, 4422.5, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_first_order_rise_at_one_time_constant():
    """[I] After one time constant the rise is (1 - 1/e) of the steady rise; tolerance 1e-12."""
    p, g, c = 7.0, 0.3, 82.0
    got = ref.first_order_rise(p, g, c, c / g)
    want = (1 - math.exp(-1)) * p / g
    assert rel_err(got, want) <= 1e-12, mismatch("first-order rise at t = tau", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_time_to_95_percent_is_tau_ln20():
    """[I] Time to 95 % of the steady rise = tau ln 20 = 2.9957 tau; tolerance 1e-12."""
    p, g, c = 7.0, 0.3, 82.0
    got = ref.time_to_rise(0.95 * p / g, p, g, c)
    want = c / g * math.log(20)
    assert rel_err(got, want) <= 1e-12, mismatch("time to 95 %", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("target,want", [(0.0, 0.0), (-5.0, 0.0), (7.0 / 0.3, math.inf), (30.0, math.inf)])
def test_sanity_time_to_rise_edge_cases(target, want):
    """Target already reached (<= 0) takes 0 s; a target at or above the steady rise is never reached (inf)."""
    got = ref.time_to_rise(target, 7.0, 0.3, 82.0)
    assert got == want, mismatch(f"time to rise, target {target}", "reference", got, want, 0.0)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("t_s", [0.1, 2.0, 273.0, 1000.0])
def test_sanity_ode_rise_matches_closed_form(t_s):
    """Numerical integration agrees with the first-order closed form; DOP853 at rtol 1e-12, tolerance 1e-9."""
    got = ref.ode_rise(7.0, 0.3, 82.0, t_s)
    want = ref.first_order_rise(7.0, 0.3, 82.0, t_s)
    assert rel_err(got, want) <= 1e-9, mismatch(f"ODE rise at {t_s} s", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_ode_time_to_rise_matches_closed_form():
    """Event location in the numerical integration agrees with -tau ln(1 - target/steady); tolerance 1e-9."""
    got = ref.ode_time_to_rise(20.0, 7.0, 0.3, 82.0, 1e5)
    want = ref.time_to_rise(20.0, 7.0, 0.3, 82.0)
    assert rel_err(got, want) <= 1e-9, mismatch("ODE time to rise", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_event_train_matches_brute_force_integration():
    """The periodic-steady-state mean from event_train_mean_rise matches brute-force integration of 80 periods
    (tau = 10 s, pulse 1 s every 4 s; the start-up transient has decayed by e^-32); tolerance 1e-8."""
    p, g, c, t_on, period = 5.0, 0.5, 5.0, 1.0, 4.0
    theta = 0.0
    for _ in range(80):
        on = solve_ivp(lambda t, y: [(p - g * y[0]) / c, y[0]], (0, t_on), [theta, 0.0], method="DOP853", rtol=1e-12, atol=1e-14)
        off = solve_ivp(lambda t, y: [(-g * y[0]) / c, y[0]], (0, period - t_on), [on.y[0, -1], 0.0], method="DOP853",
                        rtol=1e-12, atol=1e-14)
        theta = off.y[0, -1]
    got = (on.y[1, -1] + off.y[1, -1]) / period
    want = ref.event_train_mean_rise(p, g, c, t_on, period)
    assert rel_err(got, want) <= 1e-8, mismatch("event-train mean rise vs brute force", "reference", got, want, 1e-8)


@pytest.mark.family("thermal")
def test_sanity_event_train_continuous_limit():
    """A pulse that fills the whole period is continuous heating: mean rise = P/G; tolerance 1e-12."""
    got = ref.event_train_mean_rise(5.0, 0.5, 5.0, 4.0, 4.0)
    assert rel_err(got, 10.0) <= 1e-12, mismatch("event-train mean rise, duty 1", "reference", got, 10.0, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("x0", [0.3, -2.0, 20000.0])
def test_sanity_central_difference_matches_analytic_derivative(x0):
    """d/dx (sin(x) + x^3) = cos(x) + 3 x^2. With the relative step 1e-4 the truncation (h^2/6) f'''/f' is at most
    4e-9 and the round-off eps |f| / h at most 2e-12 of f' at these points; tolerance 1e-6."""
    got = ref.central_difference(lambda x: math.sin(x) + x ** 3, x0)
    want = math.cos(x0) + 3 * x0 ** 2
    assert rel_err(got, want) <= 1e-6, mismatch(f"central difference at x0={x0}", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
def test_sanity_central_difference_rejects_zero_point():
    """A relative step has no size at x0 = 0: the function refuses instead of dividing by zero."""
    with pytest.raises(ValueError):
        ref.central_difference(math.sin, 0.0)


@pytest.mark.family("constants")
def test_sanity_assert_all_passes_within_tolerance():
    """assert_all accepts items whose relative error is within tolerance, including an exact zero reference."""
    assert_all([("a", "reference", 1.0 + 1e-12, 1.0, 1e-9), ("b", "reference", 0.0, 0.0, 0.0)])


@pytest.mark.family("constants")
def test_sanity_assert_all_lists_every_failure():
    """assert_all fails once and names every failing item (each formatted by mismatch); passing items are omitted."""
    with pytest.raises(AssertionError) as err:
        assert_all([("first", "X!C1", 2.0, 1.0, 0.1), ("fine", "X!C2", 1.0, 1.0, 0.1), ("second", "X!C3", 5.0, 1.0, 0.1)])
    msg = str(err.value)
    got = 1.0 if ("first [X!C1]" in msg and "second [X!C3]" in msg and "fine" not in msg) else 0.0
    assert got == 1.0, mismatch(f"assert_all message {msg!r}", "reference", got, 1.0, 0.0)


@pytest.mark.family("constants")
@pytest.mark.parametrize("engine,reference", [(math.nan, 1.0), (math.inf, math.inf)])
def test_sanity_assert_all_never_passes_nan(engine, reference):
    """NaN never passes, and inf against inf gives rel_err NaN, so infinite values must be compared as flags."""
    with pytest.raises(AssertionError):
        assert_all([("nan", "reference", engine, reference, 1.0)])


@pytest.mark.family("constants")
@pytest.mark.parametrize("engine_text,expected_text,want", [("OK", "OK", 0.0), ("CHECK", "OK", 1.0)])
def test_sanity_text_item(engine_text, expected_text, want):
    """text_item passes equal texts and fails different ones (rel_err 0 or 1 against the flag 1.0)."""
    what, cells, engine, reference, tol = text_item("verdict", "X!C1", engine_text, expected_text)
    got = rel_err(engine, reference)
    assert got == want, mismatch(f"text_item {engine_text!r} vs {expected_text!r}", cells, got, want, tol)


@pytest.mark.family("constants")
def test_sanity_scenario_cache_runs_each_scenario_once():
    """ScenarioCache applies the named changes and returns the same objects on the second lookup (one engine run)."""
    cache = ScenarioCache({"poles12": {"coupling.npole": 12}})
    first, second = cache["poles12"], cache["poles12"]
    same = 1.0 if (first[0].coupling.npole == 12 and first is second) else 0.0
    assert same == 1.0, mismatch("scenario cache identity and change", "reference", same, 1.0, 0.0)
