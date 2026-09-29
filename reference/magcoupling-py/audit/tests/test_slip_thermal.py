"""Task 6 engine checks: slip losses, the slip thermal network, slip life, magnet and adhesive life, and the
part-B summary rows of the Temperature design sheet.

References: audit/references/slip_thermal.py (this task), Task 4's audit/references/metal_stack.py
(omega_rad_s, pole_pairs, pole_pair_frequency_Hz, shaft_power_W) and audit/references/remanence.py (torque_at).
Multi-cell checks use the shared helpers in audit/common.py (assert_all, text_item, ScenarioCache).

One test per root cause: cells that one engine formula or one missing guard would get wrong together are
checked in one test (assert_all lists every failing cell), so Task 8 receives one candidate per root cause.

Tolerances
  TOL_ALGEBRA  the engine's closed form re-derived here (same algebra).
  TOL_MODEL    (audit.common) engine closed form vs the exact solution of the idealized problem it approximates:
               2 %. A failure is a model-approximation candidate.
  TOL_NUMERIC  engine closed form vs numerical integration (DOP853, rtol 1e-12): 1e-7.
  TOL_LABEL    a label that states '95 %' is met when the reached fraction rounds to 95 %.
Re-derivations use the engine's mu0 input (Calculator!C43, rounded 1.256637e-6), so a same-algebra check isolates
transcription; the rounding itself belongs to the constants family.

Inputs taken from engine rows that other tasks verify: part masses and the rotating mass (Task 4), the governing
limit, magnet limit, onsets, cure margin and the Volkersen worst-swing shear (Task 5), pull-out at 20 C (Tasks 2-4).
Ownership: this file owns the Temperature design result rows C6, C14, C16-C23, C25, C28-C33, C37 and C109-C202
(C13, C15 and F16 belong to Task 5's test_summary_margins_and_hot_day_note; C24 and C90 to Task 5). Metal design
C86, C89, C91 and C92 (slip duty) belong to Task 4 (test_slip_duty, test_slip_loss_from_measured_drag). Task 7's
test_thermal_chain_units and test_thermal_scaling_with_conductance check units and scaling on C141-C153, a
different method.

Not independently checkable here (listed for the report): the cap end factor (0.7 in C128; the radial profile of the
cap end field is collapsed into cap_integral_T2m4), the web's r^2 weighting ((r_mid/pp)^2 * int B^2 dA instead of
int B^2 (r/pp)^2 dA; profile collapsed into web_integral_T2m2), the seven 3D field inputs C116-C122 (fields3d, M3
scope), harmonics above the fundamental, the finite cup wall (1.8 mm at the corners against delta 1.30 mm) versus the
half-space, the single-lump network (the split of the 0.3 W/K between the two shafts is not an input), and the
judgement inputs high_multiplier (C133) and end_factor (C114), whose sensitivities audit/tools/placeholder_sensitivity.py
prints for the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from audit.common import (TOL_ALGEBRA, TOL_MODEL, ScenarioCache, assert_all, defaults, mismatch, rel_err, run,
                          text_item, vary)
from audit.references import slip_thermal as ref
from audit.references.metal_stack import omega_rad_s, pole_pair_frequency_Hz, pole_pairs, shaft_power_W
from audit.references.remanence import torque_at

TOL_NUMERIC = 1e-7
TOL_LABEL = 0.5 / 95

CASES = ScenarioCache({
    "defaults": {},
    "half_speed": {"metal.slip_rpm": 1000},
    "measured_drag": {"metal.measured_drag_Nm": 0.02},
    "low_conductance": {"temperature.thermal.conductance_W_K": 0.08},
    "start_above_limit": {"temperature.duty.driving_rise_C": 40},
    "high_requirement": {"metal.required_min_Nm": 2.6},
    "low_hot_strength": {"temperature.adhesive_life.hot_strength_retained": 0.4},
    "small_daily_swing": {"temperature.adhesive_life.daily_swing_C": 3},
})


@dataclass(frozen=True)
class Slip:
    """Slip kinematics and geometry, from inputs and from geometry rows verified by other tasks."""
    omega: float          # mechanical slip speed, rad/s
    rev_s: float          # relative revolutions per second
    pp: float             # pole pairs per ring
    omega_e: float        # field angular frequency seen by the opposite ring, rad/s
    L: float              # active length, m
    r_hub: float          # hub steel surface radius, m
    r_cup: float          # cup steel surface radius, m
    r_mid: float          # inner block mid radius, m
    r_sleeve: float       # mean sleeve radius, m
    r_liner: float        # mean liner radius, m
    r_cap_out: float      # cap outer radius, m
    overhang: float       # retainer length past the active length at each end, m
    block_short: float    # short in-plane block side (circumferential width), m
    block_long: float     # long in-plane block side (axial length), m
    block_volume: float   # m^3


@dataclass(frozen=True)
class Heat:
    """Thermal network quantities re-derived from inputs, the re-derived slip loss and engine part masses."""
    p_estimate: float     # sum of the re-derived part losses, W
    p_use: float          # bench drag x Omega when entered, else p_estimate, W
    p_high: float         # W
    capacity: float       # J/K
    conductance: float    # W/K
    tau: float            # s
    t0: float             # hot-day start, C
    t_limit: float        # governing limit, C
    rise_limit: float     # t_limit - t0, C (negative when the start is above the limit)
    steady_est: float     # continuous-slip steady temperature, estimate, C
    steady_high: float    # continuous-slip steady temperature, high case, C
    time_to_limit_est: float   # s (inf: never; 0: already there)
    time_to_limit_high: float  # s
    critical_drag: float  # drag whose steady state equals the limit, N·m (0 when the start is above the limit)
    fault_rise: float     # high-case rise at the fault trip time, C
    event_rise: float     # high-case adiabatic rise per event, C
    duty: float           # share of life spent slipping
    avg_rise_high: float  # life-average slip heating, high case, C
    rotations: float      # relative slip rotations over life
    peak: float           # hot-day peak magnet / bond temperature, C


def slip_ctx(inp, res) -> Slip:
    ci, md, m, ret = inp.coupling, inp.metal, res.model, res.retainers
    w, l = m.inner_width_mm / 1000, m.inner_length_mm / 1000
    return Slip(omega=omega_rad_s(md.slip_rpm), rev_s=md.slip_rpm / 60, pp=pole_pairs(ci.npole),
                omega_e=2 * math.pi * pole_pair_frequency_Hz(ci.npole, md.slip_rpm), L=m.active_length_mm / 1000,
                r_hub=(ci.inner_back_apothem_mm - md.bond_inner_mm) / 1000,
                r_cup=(m.outer_back_apothem_mm + md.bond_outer_mm) / 1000,
                r_mid=(ci.inner_back_apothem_mm + m.inner_thickness_mm / 2) / 1000,
                r_sleeve=(ret.sleeve_id_mm + ret.sleeve_od_mm) / 4 / 1000,
                r_liner=(ret.liner_id_mm + ret.liner_od_mm) / 4 / 1000,
                r_cap_out=md.cap_od_mm / 2 / 1000,
                overhang=(md.retainer_span_mm - m.active_length_mm) / 2 / 1000,
                block_short=min(w, l), block_long=max(w, l),
                block_volume=m.inner_length_mm * m.inner_width_mm * m.inner_thickness_mm * 1e-9)


def steel_losses(inp, res, exact: bool, web_image: float) -> dict[str, float]:
    """Hub, cup and web loss of a travelling field over steel.

    exact=False: the thin-skin limit (k*delta -> 0, the form the engine uses); exact=True: Stoll's half-space with
    finite k*delta and the finite-permeability surface reflection. b_hub_T and b_cup_T are already doubled at the
    steel surface (fields3d.run doubles both; the b_hub_T help text, C116, says so). web_image multiplies the web
    end-field amplitude taken from web_integral_T2m2 (C121): 1 uses it as given, 2 doubles it at the steel web as hub
    and cup are doubled. The web wave number is taken at the block mid radius, as the engine takes it."""
    c, sl = slip_ctx(inp, res), inp.temperature.slip_loss
    st = inp.materials.steel
    steel = dict(omega=c.omega_e, mu_r=st.mu_r_incremental, sigma=st.conductivity_S_m, mu0=inp.coupling.mu0)

    def per_area(b_image, k):
        if exact:
            return ref.halfspace_loss_per_area(ref.halfspace_surface_bn(b_image, k=k, **steel), k=k, **steel)
        return ref.halfspace_loss_thin_skin(b_image, k=k, **steel)

    return {"hub": per_area(sl.b_hub_T, c.pp / c.r_hub) * 2 * math.pi * c.r_hub * c.L,
            "cup": per_area(sl.b_cup_T, c.pp / c.r_cup) * 2 * math.pi * c.r_cup * c.L,
            # per-area loss for the unit end-field amplitude (times web_image) times the end-field integral int B^2 dA
            "web": per_area(web_image, c.pp / c.r_mid) * sl.web_integral_T2m2}


def rederived_losses(inp, res) -> dict[str, float]:
    """The engine's slip-loss closed forms re-derived as limits of the reference models: thin-skin travelling field
    (hub, cup, web as given), long thin sheet at low speed x the end-factor input (sleeve, liner, cap), thin strip
    (magnets)."""
    c, sl = slip_ctx(inp, res), inp.temperature.slip_loss
    al = inp.materials.aluminium.al6061.conductivity_S_m

    def shell(b, r, t_mm):
        return sl.end_factor * ref.thin_sheet_loss_low_speed(b, c.omega * r, sl.sigma_316_S_m, t_mm / 1000) * 2 * math.pi * r * c.L

    return {
        **steel_losses(inp, res, exact=False, web_image=1.0),
        "sleeve": shell(sl.b_sleeve_T, c.r_sleeve, inp.metal.sleeve_mm),
        "liner": shell(sl.b_liner_T, c.r_liner, inp.metal.liner_mm),
        "cap": sl.end_factor * ref.thin_disk_loss_low_speed(c.omega, al, inp.metal.cap_axial_mm / 1000, sl.cap_integral_T2m4),
        # the engine takes the block width as the strip thickness whichever side is shorter
        "magnets": ref.strip_loss_per_volume(sl.b_magnet_T, c.omega_e, sl.sigma_ndfeb_S_m, res.model.inner_width_mm / 1000)
                   * c.block_volume * 2 * inp.coupling.npole,
    }


def literature_losses(inp, res) -> dict[str, float]:
    """Exact solutions of the idealized problems the engine's closed forms approximate: Stoll's half-space (hub, cup,
    web with its end field doubled at the steel web), Russell-Norsworthy end factor with the retainer overhang times
    Reitz's sheet reaction (sleeve, liner), sheet reaction only (cap), rectangular-section factor (magnets)."""
    c, sl, mu0 = slip_ctx(inp, res), inp.temperature.slip_loss, inp.coupling.mu0
    al = inp.materials.aluminium.al6061.conductivity_S_m
    t_cap = inp.metal.cap_axial_mm / 1000

    def shell(b, r, t_mm):
        t, v = t_mm / 1000, c.omega * r
        return (ref.sheet_end_factor(c.pp / r, c.L, c.overhang) * ref.thin_sheet_loss_low_speed(b, v, sl.sigma_316_S_m, t)
                * ref.thin_sheet_reaction_factor(v, sl.sigma_316_S_m, t, mu0) * 2 * math.pi * r * c.L)

    return {
        **steel_losses(inp, res, exact=True, web_image=2.0),
        "sleeve": shell(sl.b_sleeve_T, c.r_sleeve, inp.metal.sleeve_mm),
        "liner": shell(sl.b_liner_T, c.r_liner, inp.metal.liner_mm),
        # the cap end factor has no independent reference (radial end-field profile unknown): the engine's is kept
        # and only the sheet reaction is added, taken at the cap outer radius where it is largest
        "cap": (sl.end_factor * ref.thin_disk_loss_low_speed(c.omega, al, t_cap, sl.cap_integral_T2m4)
                * ref.thin_sheet_reaction_factor(c.omega * c.r_cap_out, al, t_cap, mu0)),
        "magnets": (ref.strip_loss_per_volume(sl.b_magnet_T, c.omega_e, sl.sigma_ndfeb_S_m, c.block_short)
                    * ref.rect_section_factor(c.block_long / c.block_short) * c.block_volume * 2 * inp.coupling.npole),
    }


def heat_ctx(inp, res) -> Heat:
    c, md, tt = slip_ctx(inp, res), inp.metal, inp.temperature
    th, mass, ret = tt.thermal, res.mass, res.retainers
    steel_c = inp.materials.steel.specific_heat_J_kgK
    measured = md.measured_drag_Nm is not None
    p_estimate = sum(rederived_losses(inp, res).values())
    p_use = shaft_power_W(md.measured_drag_Nm, md.slip_rpm) if measured else p_estimate
    p_high = p_use if measured else p_use * tt.slip_loss.high_multiplier
    capacity = ref.heat_capacity_J_K([
        (mass.magnets_g, th.c_ndfeb), (mass.cup_g, steel_c), (mass.hub_g, steel_c), (mass.boss_g, steel_c),
        (md.hardware_g, steel_c), (ret.retainers_g, th.c_316), (ret.endplates_g, th.c_316), (ret.cap_g, th.c_aluminium)])
    g = th.conductance_W_K
    t0 = tt.duty.hot_ambient_C + tt.duty.driving_rise_C
    t_limit = res.temperature.summary.governing_limit_C
    fault_rise = ref.first_order_rise(p_high, g, capacity, tt.duty.fault_trip_s)
    event_rise = p_high * md.slip_event_s / capacity
    duty = md.life_events * md.slip_event_s / 3600 / tt.duty.life_hours
    return Heat(p_estimate=p_estimate, p_use=p_use, p_high=p_high, capacity=capacity, conductance=g, tau=capacity / g,
                t0=t0, t_limit=t_limit, rise_limit=t_limit - t0,
                steady_est=t0 + p_use / g, steady_high=t0 + p_high / g,
                time_to_limit_est=ref.time_to_rise(t_limit - t0, p_use, g, capacity),
                time_to_limit_high=ref.time_to_rise(t_limit - t0, p_high, g, capacity),
                critical_drag=max(0.0, t_limit - t0) * g / c.omega,
                fault_rise=fault_rise, event_rise=event_rise, duty=duty, avg_rise_high=duty * p_high / g,
                rotations=md.life_events * c.rev_s * md.slip_event_s,
                peak=t0 + max(fault_rise, event_rise) + duty * p_high / g)


def case(name: str):
    """(inputs, results, Slip, Heat) for a named input case; CASES runs the engine once per case."""
    inp, res = CASES[name]
    return inp, res, slip_ctx(inp, res), heat_ctx(inp, res)


def pullout_at(inp, res, temp_C: float) -> float:
    """Pull-out torque at temp_C from the 20 C value (torque ~ Br^2, Br linear in temperature)."""
    return torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, temp_C)


def shear_amplitude(inp, res, h: Heat) -> float:
    """Bond shear per reversal at the peak temperature with +variation: F = T / (N r_mid) per block over L x w (MPa)."""
    m = res.model
    r_mid_mm = inp.coupling.inner_back_apothem_mm + m.inner_thickness_mm / 2
    torque = pullout_at(inp, res, h.peak) * (1 + inp.metal.variation)
    return torque / (inp.coupling.npole * r_mid_mm / 1000) / (m.inner_length_mm * m.inner_width_mm)


def selected_adhesive(inp):
    return inp.temperature.adhesive.candidates[inp.temperature.adhesive.selected - 1]


def hot_fatigue_margin(inp, res, h: Heat) -> float:
    """Hot fatigue strength (lap shear x share retained hot x fatigue endurance) over the shear amplitude."""
    al = inp.temperature.adhesive_life
    return selected_adhesive(inp).lap_shear_MPa * al.hot_strength_retained * al.fatigue_endurance / shear_amplitude(inp, res, h)


def daily_peak_shear(inp, res) -> float:
    """Volkersen shear is linear in the temperature swing, so the daily-cycle shear scales the worst-swing value (MPa)."""
    mm = res.temperature.mismatch
    return mm.peak_shear_recommended_MPa * inp.temperature.adhesive_life.daily_swing_C / mm.worst_swing_C


def as_seconds(value) -> float:
    """Engine time rows are text ('never: ...') when the limit is not reached; that means infinite time."""
    return math.inf if isinstance(value, str) else value


def never_item(what: str, cell: str, engine_value, reference_s: float) -> tuple:
    """A 'limit never reached' comparison as an assert_all item: 1.0 when the engine gives its 'never' text, against
    1.0 when the reference time is infinite (inf against inf has no relative error)."""
    return (f"{what}: engine {engine_value!r}", cell, 1.0 if isinstance(engine_value, str) else 0.0,
            1.0 if math.isinf(reference_s) else 0.0, 0.0)


# ============================================================ slip losses vs first principles and literature
PARTS = {
    "hub": ("hub_W", "Temperature design!C123"),
    "cup": ("cup_W", "Temperature design!C124"),
    "web": ("web_W", "Temperature design!C125"),
    "sleeve": ("sleeve_W", "Temperature design!C126"),
    "liner": ("liner_W", "Temperature design!C127"),
    "cap": ("cap_W", "Temperature design!C128"),
    "magnets": ("magnets_W", "Temperature design!C129"),
}


def model_items(parts, reference, extra_cells: str = "") -> list[tuple]:
    """assert_all items comparing each part's engine loss and its 1000 -> 2000 rpm log-log slope with
    reference(inp, res)[part], at TOL_MODEL. extra_cells names a shared input cell behind the parts."""
    inp, res, _, _ = case("defaults")
    inp_h, res_h, _, _ = case("half_speed")
    want, want_h = reference(inp, res), reference(inp_h, res_h)
    items = []
    for part in parts:
        field, cell = PARTS[part]
        cell += extra_cells
        eng, eng_h = getattr(res.temperature.slip_loss, field), getattr(res_h.temperature.slip_loss, field)
        items.append((f"{part} loss", cell, eng, want[part], TOL_MODEL))
        items.append((f"{part} speed exponent 1000-2000 rpm", cell, ref.loglog_slope(1000.0, eng_h, 2000.0, eng),
                      ref.loglog_slope(1000.0, want_h[part], 2000.0, want[part]), TOL_MODEL))
    return items


@pytest.mark.family("thermal")
def test_steel_skin_depth():
    """Steel skin depth at the field frequency, delta = sqrt(2/(omega_e mu0 mu_r sigma)) [Jackson], same algebra."""
    inp, res, c, _ = case("defaults")
    st = inp.materials.steel
    want = ref.skin_depth_m(c.omega_e, st.mu_r_incremental, st.conductivity_S_m, inp.coupling.mu0) * 1000
    eng = res.temperature.slip_loss.skin_depth_mm
    assert rel_err(eng, want) <= TOL_ALGEBRA, mismatch("steel skin depth", "Temperature design!C115", eng, want, TOL_ALGEBRA)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("part", list(PARTS))
def test_slip_loss_rederivation(part):
    """Each slip-loss row equals its closed form re-derived from first principles: travelling field over a permeable
    conductor in the thin-skin limit (hub, cup, web), long thin sheet at low speed times the end factor (sleeve,
    liner, cap), thin strip (magnets). Same algebra: TOL_ALGEBRA."""
    inp, res, _, _ = case("defaults")
    field, cell = PARTS[part]
    eng = getattr(res.temperature.slip_loss, field)
    want = rederived_losses(inp, res)[part]
    assert rel_err(eng, want) <= TOL_ALGEBRA, mismatch(f"{part} slip loss, re-derived closed form", cell, eng, want, TOL_ALGEBRA)


@pytest.mark.family("thermal")
def test_steel_surface_losses_vs_exact_halfspace():
    """Hub, cup and web losses and their speed exponents against Stoll's exact half-space for the same surface field
    (TOL_MODEL). One root cause: the thin-skin form (loss ~ speed^1.5) needs k*delta << 1, but k*delta is 0.64 (hub),
    0.36 (cup) and 0.55 (web) at defaults, and the finite-permeability reflection |1 + gamma/(mu_r k)|^-2 is left out.
    The web's missing steel image is a separate root cause (test_web_end_field_doubled_at_steel_web)."""
    assert_all(model_items(("hub", "cup", "web"), lambda i, r: steel_losses(i, r, exact=True, web_image=1.0)))


@pytest.mark.family("thermal")
def test_web_end_field_doubled_at_steel_web():
    """Web loss (C125) against the same thin-skin model with the end field doubled at the steel web (TOL_MODEL).

    Why doubled: at a highly permeable surface the normal field is twice the incident field (image method), and the
    engine doubles hub and cup for that reason: fields3d.run sets b_hub = 2 * _fundamental_br(g, g.outer(0), g.hub_surf)
    and b_cup = 2 * _fundamental_br(g, g.inner(0), g.cup_surf), and the b_hub_T help text reads "3D, doubled at the steel
    surface" (Temperature design!C116). The web integral is web_int = disk(9.0, 18.0, g.L / 2 + web_gap_mm, True), the
    fundamental of Bz from g.inner(0) alone (the inner ring and its hub images) at the web plane, with no factor 2 and
    no mirror across that plane; its help text reads only "3D." (C121). fields3d.run at defaults returns 1.0346e-5,
    the default input 1.035e-5, so the input is this free-space value. temperature.py then applies the same per-area
    steel formula to the web as to hub and cup. The web is steel in the engine (steel sigma and mu_r in C125).
    Consistent treatment multiplies the web loss by 4; with the exact half-space as well it is 0.332 W."""
    inp, res, _, _ = case("defaults")
    want = steel_losses(inp, res, exact=False, web_image=2.0)["web"]
    eng = res.temperature.slip_loss.web_W
    assert rel_err(eng, want) <= TOL_MODEL, mismatch("web loss with the end field doubled at the steel web",
                                                     "Temperature design!C125, C121", eng, want, TOL_MODEL)


@pytest.mark.family("thermal")
def test_shell_losses_vs_russell_norsworthy():
    """Sleeve and liner losses and their speed exponents against a finite thin shell (TOL_MODEL): Russell-Norsworthy end
    factor extended to the retainer overhang, times Reitz's sheet reaction. One root cause: the end-factor input 0.7
    (Temperature design!C114) shared by both shells."""
    assert_all(model_items(("sleeve", "liner"), literature_losses, extra_cells=", C114"))


@pytest.mark.family("thermal")
def test_cap_loss_vs_sheet_reaction():
    """Cap loss and its speed exponent with Reitz's sheet reaction at the cap outer radius (TOL_MODEL); the cap end
    factor itself is not independently checkable (module docstring)."""
    assert_all(model_items(("cap",), literature_losses))


@pytest.mark.family("thermal")
def test_magnet_loss_vs_rectangular_section():
    """Magnet eddy loss and its speed exponent against the exact low-frequency loss of a rectangular prism (Prandtl
    torsion analogy, Timoshenko-Goodier) instead of the thin-strip formula on the 2:1 block face (TOL_MODEL)."""
    assert_all(model_items(("magnets",), literature_losses))


@pytest.mark.family("thermal")
def test_total_slip_loss_vs_literature():
    """Total estimated slip loss against the sum of the literature part losses (TOL_MODEL)."""
    inp, res, _, _ = case("defaults")
    eng = res.temperature.slip_loss.total_W
    want = sum(literature_losses(inp, res).values())
    assert rel_err(eng, want) <= TOL_MODEL, mismatch("total slip loss vs literature", "Temperature design!C130", eng, want, TOL_MODEL)


# ============================================================ thermal network vs numerical and exact references
@pytest.mark.family("thermal")
def test_consistency_heat_capacity_mass_closure():
    """Internal consistency, not an independent check (both sides are engine rows): the parts in the lumped heat
    capacity (C141) add up to the modelled rotating mass (Calculator!C114, which Task 4 checks independently), so
    every rotating part is counted once."""
    inp, res, _, _ = case("defaults")
    m, ret = res.mass, res.retainers
    parts = m.magnets_g + m.cup_g + m.hub_g + m.boss_g + inp.metal.hardware_g + ret.retainers_g + ret.endplates_g + ret.cap_g
    eng = m.total_g
    assert rel_err(eng, parts) <= TOL_ALGEBRA, mismatch("rotating mass vs parts in the heat capacity",
                                                        "Calculator!C114 / Temperature design!C141", eng, parts, TOL_ALGEBRA)


@pytest.mark.family("thermal")
def test_time_to_limit_vs_ode():
    """Unbroken-slip time to the limit, estimate and high case, against numerical integration of
    C dT/dt = P - G (T - T0) with an event at the limit. Conductance 0.08 W/K so both cases reach it. TOL_NUMERIC."""
    inp, res, _, h = case("low_conductance")
    th = res.temperature.thermal
    assert_all([
        (f"time to limit ({which}) vs ODE", cell, as_seconds(eng),
         ref.ode_time_to_rise(h.rise_limit, p, h.conductance, h.capacity, 50 * h.tau), TOL_NUMERIC)
        for which, cell, eng, p in (("estimate", "Temperature design!C152", th.time_to_limit_est, h.p_use),
                                    ("high", "Temperature design!C150", th.time_to_limit_high, h.p_high))])


@pytest.mark.family("thermal")
def test_time_to_limit_never_branch():
    """At defaults the steady rise stays below the limit, so the limit is never reached: infinite time, which the
    engine reports as text (C150, C151, C152 and the summary copy C19)."""
    inp, res, c, h = case("defaults")
    th, s = res.temperature.thermal, res.temperature.summary
    assert_all([never_item("time to limit, high", "Temperature design!C150", th.time_to_limit_high, h.time_to_limit_high),
                never_item("rotations to limit, high", "Temperature design!C151", th.rotations_to_limit_high,
                           h.time_to_limit_high * c.rev_s),
                never_item("time to limit, estimate", "Temperature design!C152", th.time_to_limit_est, h.time_to_limit_est),
                never_item("summary time to limit", "Temperature design!C19", s.time_to_limit_high, h.time_to_limit_high)])


@pytest.mark.family("thermal")
def test_temperature_at_fault_trip_vs_ode():
    """Magnet temperature at the fault trip time (high case) against numerical integration. TOL_NUMERIC."""
    inp, res, _, h = case("defaults")
    want = h.t0 + ref.ode_rise(h.p_high, h.conductance, h.capacity, inp.temperature.duty.fault_trip_s)
    eng = res.temperature.thermal.temp_at_fault_C
    assert rel_err(eng, want) <= TOL_NUMERIC, mismatch("temperature at fault trip vs ODE", "Temperature design!C161", eng, want, TOL_NUMERIC)


@pytest.mark.family("thermal")
def test_rise_per_event_vs_first_order_response():
    """The per-event rise is adiabatic (P t / C); against the exact first-order step response the error is t/(2 tau)
    = 1.8e-4 at defaults. TOL_MODEL."""
    inp, res, _, h = case("defaults")
    t = inp.metal.slip_event_s
    est = ref.first_order_rise(h.p_use, h.conductance, h.capacity, t)
    high = ref.first_order_rise(h.p_high, h.conductance, h.capacity, t)
    assert_all([("rise per event (estimate)", "Temperature design!C145", res.temperature.thermal.rise_per_event_C, est, TOL_MODEL),
                ("rise per event (estimate)", "Temperature design!C171", res.temperature.slip_life.rise_per_event_est_C, est, TOL_MODEL),
                ("rise per event (high)", "Temperature design!C172", res.temperature.slip_life.rise_per_event_high_C, high, TOL_MODEL)])


@pytest.mark.family("thermal")
def test_average_slip_rise_vs_event_train():
    """Average slip heating over life (duty x steady rise) against the periodic steady state of an explicit train of
    slip events (one event every life_hours x 3600 / events seconds). TOL_NUMERIC."""
    inp, res, _, h = case("defaults")
    period = inp.temperature.duty.life_hours * 3600 / inp.metal.life_events
    t_on = inp.metal.slip_event_s
    sl = res.temperature.slip_life
    assert_all([
        (f"average slip rise ({which}) vs event train", cell, eng,
         ref.event_train_mean_rise(p, h.conductance, h.capacity, t_on, period), TOL_NUMERIC)
        for which, cell, eng, p in (("estimate", "Temperature design!C175", sl.avg_rise_est_C, h.p_use),
                                    ("high", "Temperature design!C176", sl.avg_rise_high_C, h.p_high))])


@pytest.mark.family("thermal")
def test_time_to_95_percent_label():
    """'Time / rotations to 95 % of the steady rise' use 3 tau; the fraction reached there must round to 95 %
    (TOL_LABEL). The exact time is tau ln 20 = 2.996 tau."""
    inp, res, c, h = case("defaults")
    th = res.temperature.thermal
    assert_all([(f"share of the steady rise reached at {what}", cell, 1 - math.exp(-t / h.tau), 0.95, TOL_LABEL)
                for what, cell, t in (("t95_s", "Temperature design!C159", th.t95_s),
                                      ("rev95", "Temperature design!C160", th.rev95 / c.rev_s))])


@pytest.mark.family("thermal")
def test_critical_drag_closure():
    """Critical drag is the drag whose steady state equals the limit: entered as the bench drag it must give a steady
    high-case temperature equal to the governing limit (Temperature design!C149 vs C12)."""
    inp, res, _, h = case("defaults")
    res_d = run(vary(inp, {"metal.measured_drag_Nm": res.temperature.thermal.critical_drag_Nm}))
    eng = res_d.temperature.thermal.steady_high_C
    assert rel_err(eng, h.t_limit) <= TOL_ALGEBRA, mismatch("steady temperature at the critical drag", "Temperature design!C149, C153",
                                                          eng, h.t_limit, TOL_ALGEBRA)


# ============================================================ edge cases
@pytest.mark.family("thermal")
def test_start_above_limit_gives_zero_time_and_drag():
    """Edge case, one root cause (no guard for a hot-day start above the governing limit): driving rise 40 C puts the
    start (95 C) above the 92.55 C limit. The limit is reached at t = 0 (0 s, 0 rev), and any drag exceeds it, so the
    critical drag is 0 N·m. Negative times and a negative drag (slip that cools) are unphysical."""
    inp, res, c, h = case("start_above_limit")
    th, s = res.temperature.thermal, res.temperature.summary
    assert_all([
        ("time to limit, high", "Temperature design!C150", as_seconds(th.time_to_limit_high), h.time_to_limit_high, TOL_ALGEBRA),
        ("rotations to limit, high", "Temperature design!C151", as_seconds(th.rotations_to_limit_high),
         h.time_to_limit_high * c.rev_s, TOL_ALGEBRA),
        ("time to limit, estimate", "Temperature design!C152", as_seconds(th.time_to_limit_est), h.time_to_limit_est, TOL_ALGEBRA),
        ("summary time to limit", "Temperature design!C19", as_seconds(s.time_to_limit_high), h.time_to_limit_high, TOL_ALGEBRA),
        ("critical drag", "Temperature design!C153", th.critical_drag_Nm, h.critical_drag, TOL_ALGEBRA),
        ("summary critical drag", "Temperature design!C23", s.critical_drag_Nm, h.critical_drag, TOL_ALGEBRA)])


@pytest.mark.family("thermal")
def test_zero_measured_drag_runs():
    """Edge case: a bench drag of 0 N·m means no slip heating; rotations per degree are unbounded. The engine must
    return a value (inf) rather than raise."""
    try:
        eng = run(vary(defaults(), {"metal.measured_drag_Nm": 0.0})).temperature.thermal.rev_per_C_est
    except ZeroDivisionError:
        eng = math.nan
    assert eng == math.inf, mismatch("rotations per degree at zero measured drag", "Temperature design!C156", eng, math.inf, 0.0)


# ============================================================ row re-derivations (same algebra)
def _row(what, family, engines, reference, case_name="defaults"):
    """engines: [(cell, getter), ...]; a section row and its copies (summary, link cells) share one reference."""
    return pytest.param(what, engines, reference, case_name, marks=pytest.mark.family(family), id=what)


ROWS = [
    # links to inputs
    _row("service_max", "temperature", [("Temperature design!C6", lambda r: r.temperature.summary.service_max_C)],
         lambda i, r, c, h: i.coupling.op_temp_C),
    _row("slip_rpm", "thermal", [("Temperature design!C28", lambda r: r.temperature.duty.slip_rpm)], lambda i, r, c, h: i.metal.slip_rpm),
    _row("slip_event_s", "thermal", [("Temperature design!C33", lambda r: r.temperature.duty.slip_event_s)],
         lambda i, r, c, h: i.metal.slip_event_s),
    _row("steel_conductivity", "thermal", [("Temperature design!C109", lambda r: r.temperature.slip_loss.steel_sigma_S_m)],
         lambda i, r, c, h: i.materials.steel.conductivity_S_m),
    _row("steel_mu_r", "thermal", [("Temperature design!C110", lambda r: r.temperature.slip_loss.steel_mu_r)],
         lambda i, r, c, h: i.materials.steel.mu_r_incremental),
    _row("cap_conductivity", "thermal", [("Temperature design!C112", lambda r: r.temperature.slip_loss.cap_sigma_S_m)],
         lambda i, r, c, h: i.materials.aluminium.al6061.conductivity_S_m),
    _row("steel_specific_heat", "thermal", [("Temperature design!C137", lambda r: r.temperature.thermal.steel_c)],
         lambda i, r, c, h: i.materials.steel.specific_heat_J_kgK),
    # duty and kinematics
    _row("slip_rad_s", "thermal", [("Temperature design!C29", lambda r: r.temperature.duty.slip_rad_s)], lambda i, r, c, h: c.omega),
    _row("pole_pairs", "thermal", [("Temperature design!C30", lambda r: r.temperature.duty.pole_pairs)], lambda i, r, c, h: c.pp),
    _row("field_freq_Hz", "thermal", [("Temperature design!C31", lambda r: r.temperature.duty.field_freq_Hz)],
         lambda i, r, c, h: pole_pair_frequency_Hz(i.coupling.npole, i.metal.slip_rpm)),
    _row("field_omega", "thermal", [("Temperature design!C32", lambda r: r.temperature.duty.field_omega_rad_s)], lambda i, r, c, h: c.omega_e),
    _row("hot_day_start", "thermal", [("Temperature design!C37", lambda r: r.temperature.duty.hot_day_start_C),
                                      ("Temperature design!C14", lambda r: r.temperature.summary.hot_day_start_C),
                                      ("Temperature design!C144", lambda r: r.temperature.thermal.start_C)], lambda i, r, c, h: h.t0),
    # the temperature margins C13 and C15 are Task 5's (test_summary_margins_and_hot_day_note)
    # slip-loss bookkeeping
    _row("total_W", "thermal", [("Temperature design!C130", lambda r: r.temperature.slip_loss.total_W)], lambda i, r, c, h: h.p_estimate),
    _row("drag_from_total", "thermal", [("Temperature design!C131", lambda r: r.temperature.slip_loss.drag_Nm)],
         lambda i, r, c, h: h.p_estimate / c.omega),
    _row("used_W_estimate", "thermal", [("Temperature design!C132", lambda r: r.temperature.slip_loss.used_W)], lambda i, r, c, h: h.p_use),
    _row("high_W_estimate", "thermal", [("Temperature design!C134", lambda r: r.temperature.slip_loss.high_W)], lambda i, r, c, h: h.p_high),
    _row("used_and_high_W_measured", "thermal", [("Temperature design!C132", lambda r: r.temperature.slip_loss.used_W),
                                                 ("Temperature design!C134", lambda r: r.temperature.slip_loss.high_W)],
         lambda i, r, c, h: shaft_power_W(0.02, i.metal.slip_rpm), "measured_drag"),
    # thermal network
    _row("heat_capacity", "thermal", [("Temperature design!C141", lambda r: r.temperature.thermal.heat_capacity_J_K)],
         lambda i, r, c, h: h.capacity),
    _row("time_constant", "thermal", [("Temperature design!C143", lambda r: r.temperature.thermal.time_constant_s)], lambda i, r, c, h: h.tau),
    _row("steady_rise_est", "thermal", [("Temperature design!C146", lambda r: r.temperature.thermal.steady_rise_est_C)],
         lambda i, r, c, h: h.p_use / h.conductance),
    _row("steady_rise_high", "thermal", [("Temperature design!C147", lambda r: r.temperature.thermal.steady_rise_high_C)],
         lambda i, r, c, h: h.p_high / h.conductance),
    _row("steady_est", "thermal", [("Temperature design!C148", lambda r: r.temperature.thermal.steady_est_C),
                                   ("Temperature design!C17", lambda r: r.temperature.summary.steady_estimate_C)],
         lambda i, r, c, h: h.steady_est),
    _row("steady_high", "thermal", [("Temperature design!C149", lambda r: r.temperature.thermal.steady_high_C),
                                    ("Temperature design!C18", lambda r: r.temperature.summary.steady_high_C)],
         lambda i, r, c, h: h.steady_high),
    _row("time_to_limit_high", "thermal", [("Temperature design!C150", lambda r: as_seconds(r.temperature.thermal.time_to_limit_high)),
                                           ("Temperature design!C19", lambda r: as_seconds(r.temperature.summary.time_to_limit_high))],
         lambda i, r, c, h: h.time_to_limit_high, "low_conductance"),
    _row("rotations_to_limit_high", "thermal",
         [("Temperature design!C151", lambda r: as_seconds(r.temperature.thermal.rotations_to_limit_high))],
         lambda i, r, c, h: h.time_to_limit_high * c.rev_s, "low_conductance"),
    _row("time_to_limit_est", "thermal", [("Temperature design!C152", lambda r: as_seconds(r.temperature.thermal.time_to_limit_est))],
         lambda i, r, c, h: h.time_to_limit_est, "low_conductance"),
    _row("critical_drag", "thermal", [("Temperature design!C153", lambda r: r.temperature.thermal.critical_drag_Nm),
                                      ("Temperature design!C23", lambda r: r.temperature.summary.critical_drag_Nm)],
         lambda i, r, c, h: h.critical_drag),
    _row("heating_rate_est", "thermal", [("Temperature design!C154", lambda r: r.temperature.thermal.heating_rate_est_C_s)],
         lambda i, r, c, h: h.p_use / h.capacity),
    _row("heating_rate_high", "thermal", [("Temperature design!C155", lambda r: r.temperature.thermal.heating_rate_high_C_s)],
         lambda i, r, c, h: h.p_high / h.capacity),
    _row("rev_per_C_est", "thermal", [("Temperature design!C156", lambda r: r.temperature.thermal.rev_per_C_est)],
         lambda i, r, c, h: c.rev_s * h.capacity / h.p_use),
    _row("rev_per_C_high", "thermal", [("Temperature design!C157", lambda r: r.temperature.thermal.rev_per_C_high)],
         lambda i, r, c, h: c.rev_s * h.capacity / h.p_high),
    _row("rev_per_tau", "thermal", [("Temperature design!C158", lambda r: r.temperature.thermal.rev_per_tau)],
         lambda i, r, c, h: h.tau * c.rev_s),
    _row("temp_at_fault", "thermal", [("Temperature design!C161", lambda r: r.temperature.thermal.temp_at_fault_C)],
         lambda i, r, c, h: h.t0 + h.fault_rise),
    # slip life
    _row("life_events", "temperature", [("Temperature design!C164", lambda r: r.temperature.slip_life.events)],
         lambda i, r, c, h: i.metal.life_events),
    _row("rev_per_event", "temperature", [("Temperature design!C165", lambda r: r.temperature.slip_life.rev_per_event)],
         lambda i, r, c, h: c.rev_s * i.metal.slip_event_s),
    _row("life_rotations", "temperature", [("Temperature design!C166", lambda r: r.temperature.slip_life.rotations),
                                           ("Temperature design!C21", lambda r: r.temperature.summary.life_rotations)],
         lambda i, r, c, h: h.rotations),
    _row("slip_hours", "temperature", [("Temperature design!C167", lambda r: r.temperature.slip_life.slip_hours)],
         lambda i, r, c, h: i.metal.life_events * i.metal.slip_event_s / 3600),
    # one full field and shear reversal per pole-pair pass
    _row("pole_pair_passes", "temperature", [("Temperature design!C168", lambda r: r.temperature.slip_life.like_pole_passes),
                                             ("Temperature design!C191", lambda r: r.temperature.adhesive_life.reversals)],
         lambda i, r, c, h: h.rotations * c.pp),
    _row("heat_per_event_est", "temperature", [("Temperature design!C169", lambda r: r.temperature.slip_life.heat_per_event_est_J)],
         lambda i, r, c, h: h.p_use * i.metal.slip_event_s),
    _row("heat_per_event_high", "temperature", [("Temperature design!C170", lambda r: r.temperature.slip_life.heat_per_event_high_J)],
         lambda i, r, c, h: h.p_high * i.metal.slip_event_s),
    _row("life_heat_high_MJ", "temperature", [("Temperature design!C173", lambda r: r.temperature.slip_life.life_heat_high_MJ)],
         lambda i, r, c, h: i.metal.life_events * h.p_high * i.metal.slip_event_s / 1e6),
    _row("slip_duty", "temperature", [("Temperature design!C174", lambda r: r.temperature.slip_life.slip_duty)], lambda i, r, c, h: h.duty),
    _row("rise_per_pct_duty", "temperature", [("Temperature design!C177", lambda r: r.temperature.slip_life.rise_per_pct_duty_C)],
         lambda i, r, c, h: 0.01 * h.p_high / h.conductance),
    _row("summary_avg_slip_heating", "temperature",
         [("Temperature design!C22", lambda r: r.temperature.summary.avg_slip_heating_high_C)], lambda i, r, c, h: h.avg_rise_high),
    # magnet life
    _row("peak", "temperature", [("Temperature design!C180", lambda r: r.temperature.magnet_life.peak_C),
                                 ("Temperature design!C20", lambda r: r.temperature.summary.peak_with_fault_C),
                                 ("Temperature design!C189", lambda r: r.temperature.adhesive_life.peak_C)], lambda i, r, c, h: h.peak),
    _row("margin_to_skipping_onset", "temperature", [("Temperature design!C181", lambda r: r.temperature.magnet_life.margin_onset_C)],
         lambda i, r, c, h: r.temperature.demag.onset_skipping_C - h.peak),
    _row("margin_to_magnet_limit", "temperature", [("Temperature design!C182", lambda r: r.temperature.magnet_life.margin_limit_C)],
         lambda i, r, c, h: r.temperature.demag.magnet_limit_C - h.peak),
    _row("torque_hot_day", "temperature", [("Temperature design!C184", lambda r: r.temperature.magnet_life.torque_hot_day_Nm),
                                           ("Temperature design!C16", lambda r: r.temperature.summary.torque_hot_day_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.t0)),
    _row("torque_at_peak", "temperature", [("Temperature design!C186", lambda r: r.temperature.magnet_life.torque_peak_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.peak)),
    # adhesive life
    _row("bond_margin", "temperature", [("Temperature design!C190", lambda r: r.temperature.adhesive_life.margin_C)],
         lambda i, r, c, h: selected_adhesive(i).design_limit_C - h.peak),
    _row("torque_peak_with_variation", "temperature",
         [("Temperature design!C192", lambda r: r.temperature.adhesive_life.torque_peak_var_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.peak) * (1 + i.metal.variation)),
    _row("shear_amplitude", "temperature", [("Temperature design!C193", lambda r: r.temperature.adhesive_life.shear_amplitude_MPa)],
         lambda i, r, c, h: shear_amplitude(i, r, h)),
    _row("hot_fatigue_margin", "temperature", [("Temperature design!C196", lambda r: r.temperature.adhesive_life.hot_fatigue_margin)],
         lambda i, r, c, h: hot_fatigue_margin(i, r, h)),
    _row("daily_cycles", "temperature", [("Temperature design!C200", lambda r: r.temperature.adhesive_life.daily_cycles)],
         lambda i, r, c, h: i.temperature.adhesive_life.service_years * 365),
    _row("daily_peak_shear", "temperature", [("Temperature design!C201", lambda r: r.temperature.adhesive_life.daily_peak_shear_MPa)],
         lambda i, r, c, h: daily_peak_shear(i, r)),
]


@pytest.mark.parametrize("what,engines,reference,case_name", ROWS)
def test_row_rederivation(what, engines, reference, case_name):
    """Engine row (and its summary or link copies) equals the closed form re-derived here from first principles
    (first-order RC network, P = T Omega, life totals from events x duration x speed, torque ~ Br^2 with Br linear in
    T). TOL_ALGEBRA."""
    inp, res, c, h = case(case_name)
    want = reference(inp, res, c, h)
    assert_all([(what, cell, getter(res), want, TOL_ALGEBRA) for cell, getter in engines])


# ============================================================ screens and verdict (text rows)
def _torque_meets_requirement(inp, res, c, h) -> bool:
    return pullout_at(inp, res, h.t0) >= inp.metal.required_min_Nm


def _hot_fatigue_ok(inp, res, c, h) -> bool:
    return hot_fatigue_margin(inp, res, h) >= 4


def _daily_shear_below_endurance(inp, res, c, h) -> bool:
    endurance = selected_adhesive(inp).lap_shear_MPa * inp.temperature.adhesive_life.fatigue_endurance
    return daily_peak_shear(inp, res) < endurance


def _temperature_ok(inp, res, c, h) -> bool:
    """Positive hot-day margin, magnet and bond margins at the peak, and cure margin of at least 10 C (C24, Task 5)."""
    return (h.rise_limit > 0 and res.temperature.demag.magnet_limit_C - h.peak > 0
            and selected_adhesive(inp).design_limit_C - h.peak > 0 and res.temperature.summary.cure_margin_C >= 10)


SCREENS = {  # rule: ([(cell, engine text getter), ...], passing text, other text, re-derived pass criterion)
    # C185 only: its summary copy, the hot-day note F16, is Task 5's (test_summary_margins_and_hot_day_note)
    "hot_day_torque": ([("Temperature design!C185", lambda r: r.temperature.magnet_life.torque_hot_day_check)],
                       "Meets it nominally (no variation allowance)", "Below it", _torque_meets_requirement),
    "hot_fatigue": ([("Temperature design!C197", lambda r: r.temperature.adhesive_life.hot_fatigue_screen)],
                    "OK", "CHECK: get hot fatigue data", _hot_fatigue_ok),
    "daily_cycle": ([("Temperature design!C202", lambda r: r.temperature.adhesive_life.daily_screen)],
                    "Below the fatigue endurance", "Above the fatigue endurance: qualify by thermal cycling",
                    _daily_shear_below_endurance),
    "verdict": ([("Temperature design!C25", lambda r: r.temperature.summary.verdict)],
                "OK on temperature. Confirm drag torque and thermal cycling by test.", "CHECK: see the rows above.",
                _temperature_ok),
}
SCREEN_CASES = [("hot_day_torque", "defaults"), ("hot_day_torque", "high_requirement"), ("hot_fatigue", "defaults"),
                ("hot_fatigue", "low_hot_strength"), ("daily_cycle", "defaults"), ("daily_cycle", "small_daily_swing"),
                ("verdict", "defaults"), ("verdict", "start_above_limit")]


@pytest.mark.family("temperature")
@pytest.mark.parametrize("rule,case_name", SCREEN_CASES, ids=[f"{rule}-{name}" for rule, name in SCREEN_CASES])
def test_screen_text(rule, case_name):
    """Life screens, the hot-day torque check (C185) and the temperature verdict pass exactly when the re-derived
    quantities meet the stated criterion; each rule is exercised on both branches."""
    engines, pass_text, other_text, passes = SCREENS[rule]
    inp, res, c, h = case(case_name)
    expected = pass_text if passes(inp, res, c, h) else other_text
    assert_all([text_item(f"{rule} ({case_name})", cell, getter(res), expected) for cell, getter in engines])
