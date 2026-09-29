"""Materials checks: back-iron thickness by flux conservation (Calculator C103–C106, Materials C20–C22),
the no-back-iron mode, electroless-nickel pre-plate offsets (Materials C27–C30) and material constants
against literature. The magnet temperature ratings (Calculator C22, C32, C107, C108) are Task 5's
(test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating).
A failing check is a candidate finding for Task 8."""
from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import pytest

from audit.common import TOL_ALGEBRA, TOL_MODEL, defaults, mismatch, rel_err, run, vary
from audit.references.backiron_flux import (backiron_flux_per_depth, flat_circuit_flux_density,
                                            sinusoidal_backiron_thickness, unrolled_coupling_slab)
from audit.references.block_geometry import geometry_for_design, rotating_parts_for_design
from audit.references.metal_stack import preplate_size
from audit.references.remanence import br_at

#: 100 % IACS = 58.0 MS/m (International Annealed Copper Standard, IEC 60028).
IACS_S_M = 58.0e6


def _br_op(inp, res) -> tuple[float, float]:
    """Inner and outer remanence at the operating temperature, from the 20 °C values (Calculator C21, C31)."""
    a, t = inp.calibration.alpha_br_per_C, inp.coupling.op_temp_C
    return br_at(res.model.inner_br_T, a, t), br_at(res.model.outer_br_T, a, t)


def _flat_circuit_b(inp, res) -> float:
    br_i, br_o = _br_op(inp, res)
    return flat_circuit_flux_density(br_i, br_o, res.model.inner_thickness_mm, res.model.outer_thickness_mm,
                                     geometry_for_design(inp, res).face_gap)


def _required_thickness_sinusoidal(inp, res) -> float:
    return sinusoidal_backiron_thickness(_flat_circuit_b(inp, res), geometry_for_design(inp, res).pole_pitch,
                                         inp.materials.steel.bsat_T)


def _required_thickness_2d(inp, res, surface: str) -> float:
    """Back-iron thickness from the exact 2D field of the steel-backed slab, both rings unrolled at the gap-radius
    pole pitch with each flat block at its real width (unrolled_coupling_slab: fill = w / pitch, 0.7209 on both
    rings at defaults). Not the arc-length fills C66/C67, which at that pitch make the inner blocks 19.5 % too wide
    and the outer ones 14 % too narrow. C104 has no fill in it, so the block width enters only through this
    reference. Flat blocks only: an arc magnet has no single width to unroll."""
    if inp.coupling.faceted != 1:
        raise ValueError("the unrolled slab reference is defined for flat blocks (coupling.faceted = 1) only")
    br_i, br_o = _br_op(inp, res)
    g, m = geometry_for_design(inp, res), res.model
    layers, h = unrolled_coupling_slab(g, br_i, br_o, m.inner_width_mm, m.outer_width_mm, m.inner_thickness_mm,
                                       m.outer_thickness_mm)
    return backiron_flux_per_depth(surface, layers, h, g.pole_pitch) / inp.materials.steel.bsat_T


def _ceiling_tenth(x: float) -> float:
    """Round up to the next 0.1 mm in decimal arithmetic."""
    return float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_CEILING))


# --------------------------------------------------------------------------- back iron
@pytest.mark.family("materials")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "B842-N52", "coupling.magnets.part_outer": "B861"}],
                         ids=["defaults", "N52 inner, 1.59 mm outer"])
def test_gap_flux_density_flat_circuit(changes):
    """Calculator!C103 against the ideal-iron series circuit B = (Br_i·t_i + Br_o·t_o)/(t_i + t_o + g) at the
    operating temperature (Furlani 2001 §3.3). The engine uses the mean Br times (t_i + t_o), which equals
    the MMF sum only when the rings have equal Br or equal thickness."""
    inp = vary(defaults(), changes)
    res = run(inp)
    ref = _flat_circuit_b(inp, res)
    got = res.model.gap_flux_density_T
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"flat-circuit gap flux density ({changes or 'defaults'})", "Calculator!C103",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
@pytest.mark.parametrize("field,cell", [("model", "Calculator!C104"), ("materials", "Materials!C20")])
def test_backiron_requirement_sinusoidal(field, cell):
    """Calculator!C104 / Materials!C20 against flux conservation for a sinusoidal gap field: one pole carries
    2·B̂·τ/π per unit length, half turns each way in the back iron, so t = B̂·τ/(π·B_design), with B̂ the
    flat-circuit value and τ the reference pole pitch at the gap radius."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_sinusoidal(inp, res)
    got = res.model.backiron_needed_mm if field == "model" else res.materials.backiron_thickness_needed_mm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("back-iron thickness, sinusoidal flux", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_cup_backiron_requirement_vs_2d_slab():
    """Calculator!C104 (compared with the cup corner wall) against the flux entering the cup steel between a
    pole centre and the inter-pole line in the exact 2D field of the steel-backed slab (ideal iron, square-wave
    magnetization, 4001 harmonics, aligned rings = maximum flux), divided by B_design. The inter-pole line is
    at the pocket corners, where the wall is thinnest. Blocks at their real width, as in _required_thickness_2d.
    Tolerance TOL_MODEL."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_2d(inp, res, "outer")
    got = res.model.backiron_needed_mm
    assert rel_err(got, ref) < TOL_MODEL, mismatch("cup back-iron thickness vs 2D slab flux", "Calculator!C104, Materials!C20",
                                                    got, ref, TOL_MODEL)


@pytest.mark.family("materials")
def test_hub_backiron_requirement_vs_2d_slab():
    """The hub check (Calculator!C106) reuses the single requirement C104. Reference: the flux entering the hub
    steel in the same 2D slab solution (blocks at their real width, as in _required_thickness_2d). At defaults
    both rings are the same magnet (width, thickness and Br), so the slab carries the same flux into the hub as
    into the cup and this check shares the cup check's residual. Tolerance TOL_MODEL."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_2d(inp, res, "inner")
    got = res.model.backiron_needed_mm
    assert rel_err(got, ref) < TOL_MODEL, mismatch("hub back-iron thickness vs 2D slab flux", "Calculator!C104, Calculator!C106",
                                                    got, ref, TOL_MODEL)


@pytest.mark.family("materials")
@pytest.mark.parametrize("wall", [1.8, 2.0, "required", 1.0])
def test_cup_wall_check_logic(wall):
    """Materials!C22: 'OK' when the corner wall ≥ the requirement (boundary included), otherwise ask for the
    requirement rounded up to 0.1 mm. Materials!C21 must carry the wall input. Logic only: the threshold is
    the engine's own requirement C104, whose value test_backiron_requirement_sinusoidal verifies."""
    inp = defaults()
    t_req = run(inp).model.backiron_needed_mm
    wall_mm = t_req if wall == "required" else wall
    res = run(vary(inp, {"metal.cup_wall_corner_mm": wall_mm}))
    want = "OK" if wall_mm >= t_req else f"Too thin: raise Metal design C122 to at least {_ceiling_tenth(t_req):.1f} mm"
    got = res.materials.cup_wall_check
    assert got == want, mismatch(f"cup-wall verdict {got!r}, reference {want!r}", "Materials!C22", float(got == want), 1.0, 0.0)
    assert rel_err(res.materials.cup_wall_corner_mm, wall_mm) < TOL_ALGEBRA, \
        mismatch("cup wall link", "Materials!C21", res.materials.cup_wall_corner_mm, wall_mm, TOL_ALGEBRA)


@pytest.mark.family("materials")
@pytest.mark.parametrize("changes", [{}, {"coupling.backiron": 0}, {"metal.cup_wall_corner_mm": 2.0}],
                         ids=["defaults", "no back iron", "2.0 mm wall"])
def test_calculator_thickness_checks(changes):
    """Calculator!C105 (cup, corner wall) and C106 (hub, wall under the flats): 'No back iron' without steel,
    otherwise 'Thickness OK' when the wall ≥ the requirement, else 'Too thin'. Logic only: the threshold is
    the engine's C104; the hub wall is the reference hub wall."""
    inp = vary(defaults(), changes)
    res = run(inp)
    t_req = res.model.backiron_needed_mm
    g = geometry_for_design(inp, res)
    for got, wall, cell in ((res.model.cup_ring_check, inp.metal.cup_wall_corner_mm, "Calculator!C105"),
                            (res.model.hub_check, g.hub_wall, "Calculator!C106")):
        want = "No back iron" if inp.coupling.backiron == 0 else ("Thickness OK" if wall >= t_req else "Too thin")
        assert got == want, mismatch(f"thickness verdict {got!r}, reference {want!r}", cell, float(got == want), 1.0, 0.0)


@pytest.mark.family("materials")
def test_no_back_iron_mode_treats_the_cup_consistently():
    """coupling.backiron = 0 selects 'no intentional back iron' (Calculator!C6). The torque model then uses the
    free-space factor S_n (Calculator!C93 equals its free-space variant C96) and Calculator!C105 reports 'No back
    iron'. Free-space S_n is valid only when no ferromagnetic material sits behind either ring, yet in the same
    run two outputs keep a flux-carrying steel cup directly behind the outer magnets:
    - Calculator!C111 prices the one-piece cup as steel: C111 divided by the independent cup volume is the
      steel density Metal design!C132, unchanged from the backiron = 1 run (the hub, by contrast, becomes
      aluminium, C112);
    - Materials!C22 sizes that cup wall as back iron, with the same verdict as the backiron = 1 run.
    Reference: with the selector at 0, no output treats the cup as back iron (count 0). Which side needs the
    correction (a non-magnetic cup, or a torque model with steel behind the outer ring) is for the finding to
    decide. The message also gives the pull-out of the same run in both circuits, free space (C96) and steel
    (C95): a steel ring behind only the outer magnets puts the true value between them."""
    base = run(defaults())
    inp = vary(defaults(), {"coupling.backiron": 0})
    res = run(inp)
    cup_volume_mm3 = rotating_parts_for_design(inp, res, cup_density=1.0)["cup"].mass_g
    implied_density = res.mass.cup_g / cup_volume_mm3
    steel = inp.metal.steel_density_g_mm3
    as_back_iron = {
        f"C111 cup at {implied_density * 1000:.3f} g/cm³ (steel {steel * 1000:.3f}), unchanged from backiron = 1":
            rel_err(implied_density, steel) < TOL_ALGEBRA and rel_err(res.mass.cup_g, base.mass.cup_g) < TOL_ALGEBRA,
        f"Materials C22 {res.materials.cup_wall_check!r}, same as backiron = 1":
            res.materials.cup_wall_check == base.materials.cup_wall_check,
    }
    as_absent = {
        "C93 = C96 (free-space S_n)": rel_err(res.model.pullout_Nm, res.model.pullout_noiron_Nm) < TOL_ALGEBRA,
        f"C105 {res.model.cup_ring_check!r}": res.model.cup_ring_check == "No back iron",
    }
    n_iron = sum(as_back_iron.values())
    what = (f"backiron = 0: cup treated as back iron by [{'; '.join(k for k, v in as_back_iron.items() if v)}] "
            f"and as absent by [{'; '.join(k for k, v in as_absent.items() if v)}]; pull-out C96 = "
            f"{res.model.pullout_noiron_Nm:.4f} N·m (free space), C95 = {res.model.pullout_iron_Nm:.4f} N·m (steel)")
    assert n_iron == 0, mismatch(what, "Calculator!C6, C93, C95, C96, C105, C111; Materials!C22; Metal design!C132",
                                 float(n_iron), 0.0, 0.0)


# --------------------------------------------------------------------------- plating
@pytest.mark.family("materials")
@pytest.mark.parametrize("thickness", [0.015, 0.025])
def test_preplate_offsets(thickness):
    """Materials!C27–C30: each machined size is the one that plates to the finished size, with the coating
    growing out of the material: hub flat apothem (external, one surface) under by t, pocket apothem
    (internal, one surface) over by t, bores (internal, two surfaces) over by 2t on diameter, outside
    diameters (external, two surfaces) under by 2t."""
    inp = vary(defaults(), {"materials.nickel.thickness_mm": thickness})
    res = run(inp)
    g, bore = geometry_for_design(inp, res), inp.coupling.bore_mm
    cases = ((res.materials.hub_flats_under_mm, g.hub_apothem - preplate_size(g.hub_apothem, thickness, 1, True), "Materials!C27"),
             (res.materials.cup_pockets_over_mm, preplate_size(g.pocket_apothem, thickness, 1, False) - g.pocket_apothem, "Materials!C28"),
             (res.materials.bores_over_dia_mm, preplate_size(bore, thickness, 2, False) - bore, "Materials!C29"),
             (res.materials.ods_under_dia_mm, g.cup_od - preplate_size(g.cup_od, thickness, 2, True), "Materials!C30"))
    for got, ref, cell in cases:
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"pre-plate offset at {thickness} mm", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- constants
@pytest.mark.family("materials")
@pytest.mark.parametrize("cell", ["Calculator!C36", "Calculator!C37"])
def test_calculator_material_links(cell):
    """Calculator!C36 carries the back-iron design flux density (Materials!C13) and C37 the corner wall input
    (Metal design!C122)."""
    inp = defaults()
    m = run(inp).model
    got, ref = {"Calculator!C36": (m.bsat_T, inp.materials.steel.bsat_T),
                "Calculator!C37": (m.cup_wall_corner_mm, inp.metal.cup_wall_corner_mm)}[cell]
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("linked value", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_steel_density_units_consistent():
    """Materials!C19 (g/cm³) and the mass model's Metal design!C132 (g/mm³) must be the same density."""
    inp = defaults()
    got, ref = inp.metal.steel_density_g_mm3, inp.materials.steel.density_g_cm3 / 1000
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("steel density units", "Metal design!C132, Materials!C19", got, ref, TOL_ALGEBRA)


LITERATURE = [
    # (label, cell, getter, literature value, tolerance, source)
    ("4140 density g/cm3", "Materials!C19", lambda i: i.materials.steel.density_g_cm3, 7.85, 1e-2, "ASM Handbook Vol. 1, AISI 4140"),
    ("4140 conductivity S/m", "Materials!C14", lambda i: i.materials.steel.conductivity_S_m, 1 / 0.22e-6, 3e-2,
     "AISI 4140 annealed resistivity 0.22 µΩ·m (MatWeb/ASM), two significant figures"),
    ("4140 specific heat J/kgK", "Materials!C16", lambda i: i.materials.steel.specific_heat_J_kgK, 473, 2e-2, "MatWeb AISI 4140: 0.473 J/g·°C"),
    ("4140 CTE 1/C", "Materials!C17", lambda i: i.materials.steel.cte_per_C, 12.3e-6, 2e-2, "ASM: 12.2–12.3 µm/m·°C, 20–100 °C"),
    ("4140 modulus GPa", "Materials!C18", lambda i: i.materials.steel.modulus_GPa, 205, 3e-2, "MatWeb AISI 4140: 190–210 GPa, typical 205"),
    ("7075-T6 yield MPa", "Materials!C34", lambda i: i.materials.aluminium.al7075.yield_MPa, 503, 1e-2, "ASM Handbook Vol. 2, 7075-T6"),
    ("7075-T6 shear MPa", "Materials!C35", lambda i: i.materials.aluminium.al7075.shear_MPa, 331, 1e-2, "ASM Handbook Vol. 2, 7075-T6"),
    ("7075-T6 conductivity S/m", "Materials!C38", lambda i: i.materials.aluminium.al7075.conductivity_S_m, 0.33 * IACS_S_M, 2e-2,
     "7075-T6: 33 % IACS"),
    ("6061-T6 yield MPa", "Materials!C39", lambda i: i.materials.aluminium.al6061.yield_MPa, 276, 1e-2, "ASM Handbook Vol. 2, 6061-T6"),
    ("6061-T6 shear MPa", "Materials!C40", lambda i: i.materials.aluminium.al6061.shear_MPa, 207, 1e-2, "ASM Handbook Vol. 2, 6061-T6"),
    ("6061-T6 conductivity S/m", "Materials!C43", lambda i: i.materials.aluminium.al6061.conductivity_S_m, 0.43 * IACS_S_M, 2e-2,
     "6061-T6: 43 % IACS"),
    ("class 12.9 proof MPa", "Materials!C48", lambda i: i.materials.screws.proof_12_9_MPa, 970, TOL_ALGEBRA, "ISO 898-1:2013 Table 3"),
    ("class 10.9 proof MPa", "Materials!C49", lambda i: i.materials.screws.proof_10_9_MPa, 830, TOL_ALGEBRA, "ISO 898-1:2013 Table 3"),
    ("A4-70 yield MPa", "Materials!C50", lambda i: i.materials.screws.yield_A4_70_MPa, 450, TOL_ALGEBRA, "ISO 3506-1:2020, Rp0.2 min"),
    ("6061 cap density g/cm3", "Metal design!C42", lambda i: i.metal.al_density_g_mm3 * 1000, 2.70, 1e-2, "Aluminum Association, 6061"),
    ("316L density g/cm3", "Metal design!C44", lambda i: i.metal.sleeve_density_g_mm3 * 1000, 8.0, 1e-2, "ASM Handbook Vol. 1, 316L: 7.99–8.0"),
]


@pytest.mark.family("materials")
@pytest.mark.parametrize("label,cell,getter,value,tol,source", LITERATURE, ids=[row[0] for row in LITERATURE])
def test_material_constants_vs_literature(label, cell, getter, value, tol, source):
    """Material inputs against handbook/standard values; tolerance is the spread or rounding of the
    published value (see ``source``)."""
    got = getter(defaults())
    assert rel_err(got, value) < tol, mismatch(f"{label} vs {source}", cell, got, value, tol)
