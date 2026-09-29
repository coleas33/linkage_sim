"""Task 7: constants, dimensional consistency, scaling laws and text-verdict thresholds.

Dimensional checks use audit.references.units_ref, a quantity algebra that rejects mixed dimensions:
- formula level: each formula in FORMULAS, written as the README / spec / literature states it, has the dimensions
  of its result (test_formula_is_dimensionally_consistent; independent of the engine code). Covered (FORMULAS groups):
  torque chain (wave-number argument k·t, harmonic amplitude B_n, Br(T), S_n steel and free, shear tau_n, torque
  sum, end factor, calibration correction, sweep-row torque); Metal design (hot/cold torque band, gearbox input
  T/(i·eta), clearance stack); Materials (back-iron thickness, plating offsets); clamps (preload, stripping, torque
  per screw, tightening torque, head pressure, key pressure, joint torque); thermal network (heat capacity, time
  constant, steady rise, event rise, time to limit, critical drag, slip power); slip losses (skin depth, half-space,
  thin shell, thin strip); demagnetization (knee, load line, calibration offset); adhesive (Volkersen lambda and
  shear, bond shear, centrifugal force); masses (density, inertia). Formulas not in FORMULAS are not
  dimension-checked here;
- engine level: the numeric unit factors the engine applies in the torque chain equal the SI conversions
  (test_torque_chain_unit_factors); elsewhere the owning task's SI re-derivation at TOL_ALGEBRA verifies them;
- scaling laws that follow from dimensions: geometric similarity of the Calculator chain and the thermal network's
  response to its conductance.
Verdict checks pin each comparison operator at its exact boundary with math.nextafter.

Ownership, as the other tasks' drafts assign it: Task 7 owns the verdict thresholds Metal design C11 and C37 and the
Calculator verdict C102 (the values they compare are checked by Tasks 3 and 4), the NdFeB density use in
Temperature design C82 (Task 5 consumes it), and units/scaling on Temperature design C141-C153 (Task 6 owns
their values). Calculator C110 (magnet mass) belongs to Task 4.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import units_ref as u
# The constants under test.
from magcoupling.constants import MU0, NDFEB_DENSITY_G_MM3


# --------------------------------------------------------------------------- constants
@pytest.mark.family("constants")
def test_mu0_rounded_constant():
    """The workbook's 1.256637e-6 is 4·pi·1e-7 to 7 significant figures: allowed error is half a unit in the 7th
    figure, 0.5e-12/1.256637e-6 = 3.98e-7 relative. (Torque scales with 1/mu0, so the rounding moves it by the same
    relative amount; re-derivations at TOL_ALGEBRA must use the engine's mu0 input, not MU0_EXACT.)"""
    tol = 0.5e-12 / 1.256637e-6
    d = defaults()
    assert_all([("MU0 constant", "constants.MU0", MU0, MU0_EXACT, tol),
                ("vacuum permeability input", "Calculator!C43", d.coupling.mu0, MU0_EXACT, tol),
                ("vacuum permeability input", "Calibration!C25", d.calibration.mu0, MU0_EXACT, tol)])


@pytest.mark.family("constants")
def test_ndfeb_density_constant():
    """0.0075 g/mm^3 is 7.5 g/cm^3, the nominal density of sintered NdFeB (supplier datasheets span about
    7.4-7.6 g/cm^3, so tol = 0.1/7.5)."""
    got = (NDFEB_DENSITY_G_MM3 * u.GRAM / u.MM ** 3).to(u.GRAM / (10 * u.MM) ** 3)
    assert_all([("NdFeB density in g/cm^3", "constants.NDFEB_DENSITY_G_MM3", got, 7.5, 0.1 / 7.5)])


@pytest.mark.family("constants")
def test_block_mass_uses_ndfeb_density():
    """Temperature design C82 (block mass, which temperature.py computes with a hard-coded 0.0075 rather than the
    shared constant) = inner block volume × 7.5 g/cm^3, computed with units."""
    res = run()
    m = res.model
    rho = 7.5 * u.GRAM / (10 * u.MM) ** 3
    volume = (m.inner_length_mm * u.MM) * (m.inner_width_mm * u.MM) * (m.inner_thickness_mm * u.MM)
    assert_all([("block mass", "Temperature design!C82", res.temperature.adhesive.block_mass_g,
                 (volume * rho).to(u.GRAM), TOL_ALGEBRA)])


# --------------------------------------------------------------------------- formula-level dimensions
#: One of each unit, named by physical quantity; only the dimensions matter.
Q = SimpleNamespace(
    B=1.0 * u.TESLA, mu0=1.0 * u.H_PER_M, H=1.0 * u.A_PER_M, k=1.0 / u.M, length=1.0 * u.M, area=1.0 * u.M ** 2,
    F=1.0 * u.N, T=1.0 * u.NM, stress=1.0 * u.PA, P=1.0 * u.W, time=1.0 * u.S, dT=1.0 * u.K, alpha=1.0 * u.PER_K,
    omega=1.0 * u.RAD_PER_S, rpm=1.0 * u.RPM, v=1.0 * u.M / u.S, f=1.0 / u.S, sigma=1.0 * u.S_PER_M,
    mass=1.0 * u.KG, rho=1.0 * u.KG / u.M ** 3, c=1.0 * u.J_PER_KG_K, C=1.0 * u.J_PER_K, G=1.0 * u.W_PER_K)


def _f(fn, arg: u.Quantity) -> u.Quantity:
    """fn (sin, sinh, exp, tanh, log) of an argument that must be dimensionless; raises DimensionError otherwise."""
    return fn(arg.dimensionless()) * u.ONE


# (id, formula as stated, source, builder of the result from Q, unit of the result)
FORMULAS = [
    # torque model (README 'Torque model')
    ("torque-wave-number", "k·t with k = n·(poles/2)/R_gap", "README step 2",
     lambda q: (3 * (10 / 2) / q.length) * q.length, u.ONE),
    ("torque-harmonic-amplitude", "B_n = 4·Br/(n·pi)·sin(n·pi·fill/2), fill = block width/pole pitch",
     "README step 1",
     lambda q: 4 * q.B / (3 * math.pi) * _f(math.sin, 3 * math.pi * (q.length / q.length) / 2), u.TESLA),
    ("torque-br-temperature", "Br(T) = Br20·(1 + alpha·(T - 20)), alpha [1/K]", "README step 4",
     lambda q: q.B * (u.ONE + q.alpha * (q.dT - 20 * q.dT)), u.TESLA),
    ("torque-s-steel", "S_n = sinh(k·t_i)·sinh(k·t_o)/sinh(k·(t_i + t_o + g))", "README step 2",
     lambda q: _f(math.sinh, q.k * q.length) ** 2 / _f(math.sinh, q.k * (q.length + q.length + q.length)), u.ONE),
    ("torque-s-free", "S_n = (1 - e^(-k·t_i))·(1 - e^(-k·t_o))·e^(-k·g)/2", "README step 2",
     lambda q: (u.ONE - _f(math.exp, -q.k * q.length)) ** 2 * _f(math.exp, -q.k * q.length) / 2, u.ONE),
    ("torque-shear", "tau_n = B_in·B_on/(2·mu0)·S_n·sin(n·pi/2)", "README step 2",
     lambda q: q.B * q.B / (2 * q.mu0) * math.sin(math.pi / 2), u.PA),
    ("torque-sum", "T = sum(tau_n)·2·pi·R_gap^2·L", "README step 3",
     lambda q: q.stress * 2 * math.pi * q.length ** 2 * q.length, u.NM),
    ("torque-end-factor", "1 - c_end·pole pitch/L", "README step 3",
     lambda q: u.ONE - 0.7 * q.length / q.length, u.ONE),
    ("torque-calibration-correction", "f_cal = 0.95·T_measured/T_model", "README step 3 (Calibration!C9)",
     lambda q: 0.95 * q.T / q.T, u.ONE),
    ("sweep-row-torque", "T_row = sum(tau_n)·2·pi·R_gap^2·L·f_end·f_cal",
     "Gap sweep / Pole sweep rows (README steps 1-3)",
     lambda q: q.stress * 2 * math.pi * q.length ** 2 * q.length * (u.ONE - 0.7 * q.length / q.length) * 0.95, u.NM),
    # metal design and materials (README 'Metal design', 'Materials')
    ("metal-torque-band", "T_band = T·(1 ± v)·(Br(T_2)/Br(T_1))^2 (same dimensions for either sign)",
     "README 'Metal design', torque range",
     lambda q: q.T * (u.ONE - 0.15 * u.ONE) * (q.B / q.B) ** 2, u.NM),
    ("metal-gearbox-input", "T_in = T_out/(i·eta)", "power balance (Calculator!C99; Metal design!C156, C158)",
     lambda q: q.T / (5 * 0.95), u.NM),
    ("metal-clearance-stack", "c = corner gap - sleeve - liner - beddings - allowances",
     "README 'Metal design', running clearance",
     lambda q: q.length - 0.1 * q.length - 0.2 * q.length - 2 * (0.025 * q.length) - 0.05 * q.length, u.M),
    ("materials-backiron-thickness", "t = B·tau_p/(pi·B_sat)", "Hanselman ch. 4 (Calculator!C104)",
     lambda q: q.B * q.length / (math.pi * q.B), u.M),
    ("materials-plating-offset", "machined = finished ± surfaces·t_plate",
     "README 'Materials', plating (Materials!C27-C30)",
     lambda q: 10 * q.length - 2 * (0.015 * q.length), u.M),
    # clamps (README 'Shaft clamps'; clamp_ref sources)
    ("clamp-preload", "F = fraction·Sp·As", "ISO 898-1",
     lambda q: 0.75 * q.stress * q.area, u.N),
    ("clamp-stripping", "F = tau·A_n/SF, A_n = pi·n·Le·Ds·(1/(2n) + tan30·(Ds - En)), n = 1/P", "FED-STD-H28/2B",
     lambda q: q.stress * (math.pi * q.k * q.length * q.length * (1 / (2 * q.k) + 0.57735 * (q.length - q.length))),
     u.N),
    ("clamp-torque-per-screw", "T = mu·F·d·clamp factor", "README",
     lambda q: 0.15 * q.F * q.length * 0.8, u.NM),
    ("clamp-tightening", "T = K·F·d", "Shigley Eq. 8-27",
     lambda q: 0.2 * q.F * q.length, u.NM),
    ("clamp-head-pressure", "p = F/((pi/4)·(d_k^2 - d_h^2))", "bearing annulus",
     lambda q: q.F / (math.pi / 4 * (q.area - q.area / 4)), u.PA),
    ("clamp-key-pressure", "p = 2T/(d·h·L)", "Shigley, keys",
     lambda q: 2 * q.T / (q.length * q.length * q.length), u.PA),
    ("clamp-joint-torque", "T = mu·n·F·D_bc/2", "flange friction",
     lambda q: 0.15 * 4 * q.F * q.length / 2, u.NM),
    # thermal network (README 'Thermal network'; Incropera ch. 5)
    ("thermal-heat-capacity", "C = sum(m·c)", "README",
     lambda q: q.mass * q.c, u.J_PER_K),
    ("thermal-time-constant", "tau = C/G", "Incropera",
     lambda q: q.C / q.G, u.S),
    ("thermal-steady-rise", "dT = P/G", "Incropera",
     lambda q: q.P / q.G, u.K),
    ("thermal-event-rise", "dT = P·t_event/C", "README",
     lambda q: q.P * q.time / q.C, u.K),
    ("thermal-time-to-limit", "t = -tau·ln(1 - dT_allow/dT_steady)", "Incropera",
     lambda q: -(q.C / q.G) * _f(math.log, u.ONE - 0.5 * q.dT / q.dT), u.S),
    ("thermal-critical-drag", "T = (T_limit - T_start)·G/omega", "README",
     lambda q: q.dT * q.G / q.omega, u.NM),
    ("thermal-slip-power", "P = T·omega (omega from rpm)", "README",
     lambda q: q.T * q.rpm, u.W),
    # slip losses (README 'Slip heating'; Jackson 8.1, Stoll, Reitz, Bertotti)
    ("slip-skin-depth", "delta^2 = 2/(omega·mu0·mu_r·sigma)", "Jackson 8.1",
     lambda q: 2 / (q.omega * q.mu0 * 300 * q.sigma), u.M ** 2),
    ("slip-halfspace", "P/A = sigma·omega^2·B^2·delta/(4·k^2) (speed^1.5 law)", "Stoll",
     lambda q: q.sigma * q.omega ** 2 * q.B ** 2 * q.length / (4 * q.k ** 2), u.W / u.M ** 2),
    ("slip-thin-shell", "P/A = sigma·t·v^2·B^2/2 (speed^2 law)", "Reitz / Stoll",
     lambda q: q.sigma * q.length * q.v ** 2 * q.B ** 2 / 2, u.W / u.M ** 2),
    ("slip-thin-strip", "P/V = pi^2·sigma·d^2·f^2·B^2/6", "Bertotti",
     lambda q: math.pi ** 2 * q.sigma * q.length ** 2 * q.f ** 2 * q.B ** 2 / 6, u.W / u.M ** 3),
    # temperature design (README 'Demagnetization', 'Adhesive'; Volkersen)
    ("demag-knee", "Hk = 0.9·Hcj20·(1 - beta·(T - 20))", "README",
     lambda q: 0.9 * q.H * (u.ONE - 0.005 * q.alpha * q.dT), u.A_PER_M),
    ("demag-load-line", "H = Br/mu0·1/(1 + Pc)", "permeance-coefficient load line",
     lambda q: q.B / q.mu0 / (1 + 1.0), u.A_PER_M),
    ("demag-calibration-offset", "offset = T_ref - T_rating, T_ref = 20 + (Hk - H_ref)/(Hk·|beta| - H_ref·|alpha|)",
     "README 'Demagnetization' (Temperature design!C49, C50)",
     lambda q: 20 * q.dT + (0.9 * q.H - q.H) / (0.9 * q.H * (0.005 * q.alpha) - q.H * (0.0012 * q.alpha)) - 150 * q.dT,
     u.K),
    ("adhesive-volkersen-lambda", "lambda^2 = G/eta·(1/(E1·t1) + 1/(E2·t2))", "Volkersen",
     lambda q: q.stress / q.length * (1 / (q.stress * q.length) + 1 / (q.stress * q.length)), 1 / u.M ** 2),
    ("adhesive-volkersen-shear", "tau = G·d_alpha·dT·tanh(lambda·L/2)/(eta·lambda)", "Volkersen",
     lambda q: q.stress * q.alpha * q.dT * _f(math.tanh, q.k * q.length / 2) / (q.length * q.k), u.PA),
    ("adhesive-bond-shear", "tau = T/(n·r·A)", "README",
     lambda q: q.T / (10 * q.length * q.area), u.PA),
    ("adhesive-centrifugal", "F = m·omega^2·r", "Newton",
     lambda q: q.mass * q.omega ** 2 * q.length, u.N),
    # masses and inertia (README 'Masses and envelope')
    ("mass-density", "m = rho·V", "README",
     lambda q: q.rho * q.length ** 3, u.KG),
    ("mass-inertia", "J = sum(m·r^2)", "README",
     lambda q: q.mass * q.length ** 2, u.KG_M2),
]


@pytest.mark.family("constants")
@pytest.mark.parametrize("key, formula, source, build, unit", FORMULAS, ids=[f[0] for f in FORMULAS])
def test_formula_is_dimensionally_consistent(key, formula, source, build, unit):
    """Formula-level check: the formula as stated has the dimensions of its result (units_ref raises if a sum or a
    transcendental argument mixes dimensions). Independent of the engine code; the engine's numeric unit factors
    for the same formula are verified by the SI re-derivation of the owning check (torque chain:
    test_torque_chain_unit_factors)."""
    q = build(Q)
    assert_all([flag_item(f"{key}: {formula} gives {q.describe()}, expected {unit.describe()}", source,
                          q.dims == unit.dims)])


# --------------------------------------------------------------------------- engine-level unit factors
@pytest.mark.family("constants")
def test_torque_chain_unit_factors():
    """The engine's numeric unit factors in the torque chain, observed from its outputs, equal the SI conversions:
    k = n·(p/2)/R needs mm -> m (x1000), area_lever = 2·pi·R^2·L needs mm^3 -> m^3 (1e-9), and
    torque = tau·area_lever needs Pa·m^3 = N·m (1)."""
    m, ci = run().model, defaults().coupling
    r, L = m.gap_radius_mm, m.active_length_mm
    assert_all([
        ("k1 unit factor (1/mm -> 1/m)", "Calculator!C71", m.k1 * r / (ci.npole / 2), (1 / u.MM).to(1 / u.M),
         TOL_ALGEBRA),
        ("area x lever unit factor (mm^3 -> m^3)", "Calculator!C90", m.area_lever_m3 / (2 * math.pi * r ** 2 * L),
         (u.MM ** 3).to(u.M ** 3), TOL_ALGEBRA),
        ("2D torque unit factor (Pa·m^3 -> N·m)", "Calculator!C91", m.torque_2d_Nm / (m.tau_Pa * m.area_lever_m3),
         (u.PA * u.M ** 3).to(u.NM), TOL_ALGEBRA),
    ])


SIMILARITY_LENGTHS = (
    "coupling.inner_back_apothem_mm", "coupling.bore_mm", "coupling.keyway_depth_mm",
    "coupling.magnets.manual_inner_length_mm", "coupling.magnets.manual_inner_width_mm",
    "coupling.magnets.manual_inner_thickness_mm", "coupling.magnets.manual_outer_length_mm",
    "coupling.magnets.manual_outer_width_mm", "coupling.magnets.manual_outer_thickness_mm",
    "metal.face_gap_mm", "metal.bond_inner_mm", "metal.bond_outer_mm", "metal.cup_wall_corner_mm",
)


def _scaled(lam: float):
    """Defaults with manual magnets (same values as the library B842SH) and every length multiplied by lam."""
    base = vary(defaults(), {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""})
    changes = {}
    for path in SIMILARITY_LENGTHS:
        obj = base
        for p in path.split("."):
            obj = getattr(obj, p)
        changes[path] = obj * lam
    return run(vary(base, changes)).model


@pytest.mark.family("constants")
def test_geometric_similarity_scaling():
    """Scaling every length by lam = 1.7 (Br fixed) leaves every k·t, k·g and pitch/L unchanged, so shear stress is
    invariant and torque grows as lam^3; gap radius and cup OD grow as lam; wave number shrinks as 1/lam. Catches
    mixed-unit sums, hard-coded lengths and wrong exponents anywhere in the Calculator chain."""
    lam = 1.7
    a, b = _scaled(1.0), _scaled(lam)
    assert_all([
        ("shear stress invariant", "Calculator!C89", b.tau_Pa, a.tau_Pa, TOL_ALGEBRA),
        ("2D torque ~ lam^3", "Calculator!C91", b.torque_2d_Nm, lam ** 3 * a.torque_2d_Nm, TOL_ALGEBRA),
        ("end factor invariant", "Calculator!C92", b.f_end, a.f_end, TOL_ALGEBRA),
        ("pull-out ~ lam^3", "Calculator!C93", b.pullout_Nm, lam ** 3 * a.pullout_Nm, TOL_ALGEBRA),
        ("gap radius ~ lam", "Calculator!C64", b.gap_radius_mm, lam * a.gap_radius_mm, TOL_ALGEBRA),
        ("cup OD ~ lam", "Calculator!C62", b.cup_od_mm, lam * a.cup_od_mm, TOL_ALGEBRA),
        ("wave number ~ 1/lam", "Calculator!C71", b.k1, a.k1 / lam, TOL_ALGEBRA),
    ])


@pytest.mark.family("thermal")
def test_thermal_scaling_with_conductance():
    """Dimensions fix how the lumped network responds to its conductance G (Incropera ch. 5): doubling G halves
    tau = C/G and both steady rises P/G, doubles the critical drag (limit - start)·G/omega, and leaves the heat
    capacity and the per-event rise P·t/C unchanged."""
    a = run().temperature.thermal
    g2 = 2 * defaults().temperature.thermal.conductance_W_K
    b = run(vary(defaults(), {"temperature.thermal.conductance_W_K": g2})).temperature.thermal
    assert_all([
        ("heat capacity unchanged", "Temperature design!C141", b.heat_capacity_J_K, a.heat_capacity_J_K, TOL_ALGEBRA),
        ("time constant halves", "Temperature design!C143", b.time_constant_s, a.time_constant_s / 2, TOL_ALGEBRA),
        ("steady rise (estimate) halves", "Temperature design!C146", b.steady_rise_est_C, a.steady_rise_est_C / 2,
         TOL_ALGEBRA),
        ("steady rise (high) halves", "Temperature design!C147", b.steady_rise_high_C, a.steady_rise_high_C / 2,
         TOL_ALGEBRA),
        ("critical drag doubles", "Temperature design!C153", b.critical_drag_Nm, 2 * a.critical_drag_Nm, TOL_ALGEBRA),
        ("rise per event unchanged", "Temperature design!C145", b.rise_per_event_C, a.rise_per_event_C, TOL_ALGEBRA),
    ])


# --------------------------------------------------------------------------- verdict thresholds
@pytest.mark.family("metal")
def test_hot_min_check_threshold():
    """Metal design C11: 'Below hot minimum' exactly when the hot-low torque (C9: pull-out at the operating
    temperature × (1 - variation)) < the required minimum: at defaults (2.25 < 2.5 N·m, as the README states),
    'Estimate covers hot min' at equality, 'Below hot minimum' one ulp above."""
    base = run()
    hot_low = base.metal.torque_hot_low_Nm
    at = run(vary(defaults(), {"metal.required_min_Nm": hot_low}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(hot_low, math.inf)}))
    assert_all([
        text_item(f"defaults (hot low {hot_low:.4f} N·m)", "Metal design!C11", base.metal.hot_min_check,
                  "Below hot minimum" if hot_low < defaults().metal.required_min_Nm else "Estimate covers hot min"),
        text_item("required = hot low", "Metal design!C11", at.metal.hot_min_check, "Estimate covers hot min"),
        text_item("required one ulp above hot low", "Metal design!C11", above.metal.hot_min_check,
                  "Below hot minimum"),
    ])


@pytest.mark.family("metal")
def test_clearance_check_threshold():
    """Metal design C37: 'Below target' exactly when the minimum running clearance (C35) < the residual target: at
    defaults (-0.10 < 0.2 mm, as the README states), 'Meets assumed target' at equality, 'Below target' one ulp
    above."""
    base = run()
    c = base.metal.min_running_clearance_mm
    at = run(vary(defaults(), {"metal.residual_target_mm": c}))
    above = run(vary(defaults(), {"metal.residual_target_mm": math.nextafter(c, math.inf)}))
    assert_all([
        text_item(f"defaults (clearance {c:.4f} mm)", "Metal design!C37", base.metal.clearance_check,
                  "Below target" if c < defaults().metal.residual_target_mm else "Meets assumed target"),
        text_item("target = clearance", "Metal design!C37", at.metal.clearance_check, "Meets assumed target"),
        text_item("target one ulp above clearance", "Metal design!C37", above.metal.clearance_check, "Below target"),
    ])


@pytest.mark.family("torque")
def test_calculator_verdict_threshold():
    """Calculator verdict compares the NOMINAL pull-out (no variation) with the floor max(drive·SF, required min):
    'Nominal only: hot test' at defaults (2.647 >= 2.5) and at equality, 'Below hot minimum' one ulp above."""
    base = run()
    ci, md = defaults().coupling, defaults().metal
    floor = max(ci.drive_torque_Nm * ci.drive_safety_factor, md.required_min_Nm)
    at = run(vary(defaults(), {"metal.required_min_Nm": base.model.pullout_Nm}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(base.model.pullout_Nm, math.inf)}))
    assert_all([
        ("required floor", "Calculator!C101", base.model.required_floor_Nm, floor, TOL_ALGEBRA),
        text_item("defaults", "Calculator!C102", base.model.verdict,
                  "Below hot minimum" if base.model.pullout_Nm < floor else "Nominal only: hot test"),
        text_item("floor = pull-out", "Calculator!C102", at.model.verdict, "Nominal only: hot test"),
        text_item("floor one ulp above pull-out", "Calculator!C102", above.model.verdict, "Below hot minimum"),
    ])
