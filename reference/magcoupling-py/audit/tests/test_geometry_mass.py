"""Geometry, mass and inertia checks (Calculator rows 9, 33, 38, 51–67, 110–115) against the
independent block-geometry reference. A failing check is a candidate finding for Task 8."""
from __future__ import annotations

import math
import re

import pytest

from audit.common import TOL_ALGEBRA, defaults, mismatch, rel_err, run, vary
from audit.references.block_geometry import (axial_overlap, geometry_for_design, retainer_geometry_for_design,
                                             retainer_parts_for_design, ring_min_clearance, rotating_parts_for_design)

#: Physically valid layouts (blocks fit on both polygons) that exercise N, apothem and gap.
CONFIGS = {
    "defaults": {},
    "8 poles": {"coupling.npole": 8},
    "12 poles, 12.5 mm apothem": {"coupling.npole": 12, "coupling.inner_back_apothem_mm": 12.5},
    "1.0 mm gap": {"metal.face_gap_mm": 1.0},
    "2.0 mm gap": {"metal.face_gap_mm": 2.0},
}

#: (engine field on res.model, workbook cell, reference attribute on CouplingGeometry)
GEOMETRY_FIELDS = [
    ("corner_gap_mm", "Calculator!C9", "corner_gap"),
    ("hub_wall_mm", "Calculator!C38", "hub_wall"),
    ("inner_flat_width_mm", "Calculator!C51", "inner_flat_width"),
    ("hub_wall_past_key_mm", "Calculator!C53", "hub_wall_past_key"),
    ("inner_face_radius_mm", "Calculator!C54", "inner_face_radius"),
    ("inner_corner_radius_mm", "Calculator!C55", "inner_corner_radius"),
    ("outer_face_apothem_mm", "Calculator!C56", "outer_face_apothem"),
    ("face_gap_mm", "Calculator!C57", "face_gap"),
    ("outer_flat_width_mm", "Calculator!C58", "outer_flat_width"),
    ("outer_back_apothem_mm", "Calculator!C60", "outer_back_apothem"),
    ("pocket_corner_radius_mm", "Calculator!C61", "pocket_corner_radius"),
    ("cup_od_mm", "Calculator!C62", "cup_od"),
    ("gap_radius_mm", "Calculator!C64", "gap_radius"),
    ("pole_pitch_mm", "Calculator!C65", "pole_pitch"),
    ("fill_inner", "Calculator!C66", "fill_inner"),
    ("fill_outer", "Calculator!C67", "fill_outer"),
]

#: Swing-arm radius of the added-inertia estimate (Calculator C115 label: "at 0.25 m from the swing axis").
SWING_RADIUS_M = 0.25


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", list(CONFIGS))
def test_corner_gap_is_minimum_ring_clearance(name):
    """Calculator!C9 equals the smallest 2D distance between the inner and outer block rings over every
    relative rotation, measured numerically on explicit block rectangles placed from the inputs (outer
    faces at inner back apothem + thickness + flat-face gap). Tolerance 1e-8: the bounded Brent search
    locates a smooth quadratic minimum to 1e-12 rad."""
    inp = vary(defaults(), CONFIGS[name])
    m = run(inp).model
    outer_face = inp.coupling.inner_back_apothem_mm + m.inner_thickness_mm + inp.metal.face_gap_mm
    ref = ring_min_clearance(inp.coupling.npole, inp.coupling.inner_back_apothem_mm, m.inner_thickness_mm,
                             m.inner_width_mm, outer_face, m.outer_thickness_mm, m.outer_width_mm)
    assert rel_err(m.corner_gap_mm, ref) < 1e-8, mismatch(f"corner gap vs 2D ring clearance ({name})", "Calculator!C9",
                                                           m.corner_gap_mm, ref, 1e-8)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("field,cell,attr", GEOMETRY_FIELDS, ids=[f[1] for f in GEOMETRY_FIELDS])
def test_block_geometry_rederived(name, field, cell, attr):
    """Every polygon/block dimension re-derived by construction (block rectangles, pocket polygon from
    intersecting side lines, vertex radii, side lengths) from the flat-face gap and block data. The fill
    factors are block width over the pole arc at the block's mid-thickness radius, the planar unrolling the
    workbook's slab model uses."""
    inp = vary(defaults(), CONFIGS[name])
    res = run(inp)
    got, ref = getattr(res.model, field), getattr(geometry_for_design(inp, res), attr)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{field} ({name})", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "BX042SH"}], ids=["defaults", "25.4 mm inner"])
def test_active_length_is_axial_overlap(changes):
    """Calculator!C33: the active length is the axial overlap of the two rings' blocks, both centred on the
    same mid-plane (interval intersection)."""
    res = run(vary(defaults(), changes))
    m = res.model
    ref = axial_overlap(m.inner_length_mm, m.outer_length_mm)
    assert rel_err(m.active_length_mm, ref) < TOL_ALGEBRA, mismatch(f"active length ({changes or 'defaults'})", "Calculator!C33",
                                                                     m.active_length_mm, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_cup_wall_at_flats_is_steel_only():
    """Calculator!C63 'Ring wall at the flats' should be the steel between the pocket flat and the OD. The pocket
    flat sits at the outer block back plus the outer bondline (the engine's own pocket apothem, used for C61),
    so the wall is OD/2 − (C60 + bond_outer), as the inner hub wall C38 excludes the inner bondline. The engine
    subtracts C60 only, counting the bondline as steel."""
    inp = defaults()
    res = run(inp)
    ref = geometry_for_design(inp, res).cup_wall_flat
    got = res.model.cup_wall_flat_mm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("cup wall at the flats", "Calculator!C63, Calculator!C60, Metal design!C121",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_arc_mode_uses_arc_geometry():
    """One root cause, three symptoms. With arcs (coupling.faceted = 0) the inner ring has no block corners: its
    largest radius is the arc OD a + t, and the pocket is round. Reference, by construction:
    (1) the effective gap Calculator!C57 equals the flat-face gap input Metal design!C119, because the arc gap is
        uniform; the engine subtracts the flat-block corner protrusion √((a+t)² + (w/2)²) − (a+t) (C9 'formula
        always uses the corner geometry');
    (2) the sleeve bore Metal design!C175 clears the arc OD, 2·(a + t + bedding), not flat-block corners;
    (3) the cup cavity in Calculator!C111 is the round pocket the engine itself uses for C61 and the OD
        (radius C60 + bond, no 1/cos(π/N)), not the N-gon of the same apothem. Built on the engine's
        placement C60/C62, so this item isolates the cavity shape.
    All three are listed in the failure message, failing or not."""
    inp = vary(defaults(), {"coupling.faceted": 0})
    res = run(inp)
    items = [
        ("(1) effective gap vs the input gap", "Calculator!C57, Calculator!C9, Metal design!C119",
         res.model.face_gap_mm, geometry_for_design(inp, res).face_gap),
        ("(2) sleeve bore over the arc OD", "Metal design!C175",
         res.retainers.sleeve_id_mm, retainer_geometry_for_design(inp, res).sleeve_id),
        ("(3) cup mass with a round pocket", "Calculator!C111, Calculator!C61",
         res.mass.cup_g, rotating_parts_for_design(inp, res)["cup"].mass_g),
    ]
    all_ok = all(rel_err(got, ref) < TOL_ALGEBRA for _, _, got, ref in items)
    assert all_ok, "arc mode (faceted = 0) reuses flat-block geometry: " + " || ".join(
        mismatch(what, cells, got, ref, TOL_ALGEBRA) for what, cells, got, ref in items)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("changes,inner_want,outer_want", [
    ({}, "OK", "OK"),
    ({"coupling.npole": 12}, "TOO NARROW", "OK"),
    ({"coupling.faceted": 0}, "n/a", "n/a"),
])
def test_flat_fit_checks(changes, inner_want, outer_want):
    """Calculator!C52/C59: a block fits its polygon when the polygon side at the block's tight plane (inner: the
    block back; outer: the block face) is at least the block width; the reported slack is side − width.
    Arcs report n/a."""
    inp = vary(defaults(), changes)
    res = run(inp)
    m, g = res.model, geometry_for_design(inp, res)
    for got, want, side, width, cell in ((m.inner_flat_check, inner_want, g.inner_flat_width, m.inner_width_mm, "Calculator!C52"),
                                         (m.outer_flat_check, outer_want, g.outer_flat_width, m.outer_width_mm, "Calculator!C59")):
        fits = side >= width
        assert got.startswith(want), mismatch(f"fit verdict {got!r}, reference {want!r} (fits={fits})", cell,
                                              float(got.startswith(want)), 1.0, 0.0)
        if want == "OK":
            shown = float(re.search(r"(-?\d+\.\d+) mm", got).group(1))
            assert abs(shown - (side - width)) <= 0.005 + 1e-12, \
                mismatch(f"slack shown in {got!r} (2-decimal text, abs tol 0.005 mm)", cell, shown, side - width, 5e-3)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", ["defaults", "8 poles", "12 poles, 12.5 mm apothem"])
@pytest.mark.parametrize("part,field,cell", [("magnets", "magnets_g", "Calculator!C110"), ("cup", "cup_g", "Calculator!C111"),
                                             ("hub", "hub_g", "Calculator!C112"), ("boss", "boss_g", "Calculator!C113")],
                         ids=["C110", "C111", "C112", "C113"])
def test_component_masses(name, part, field, cell):
    """Gross solid masses from explicit shapes: 2N block rectangles (7.5 g/cm³ NdFeB), cup = disk(OD) minus the
    shoelace area of the pocket polygon over the cavity depth plus the web disk minus the bore, hub polygon
    minus bore, boss tube. Built on the engine's ring placement (C56, C60, C62) so only the mass formula is
    tested. Holes, keyways and threads are not subtracted (the workbook's documented simplification)."""
    inp = vary(defaults(), CONFIGS[name])
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)[part].mass_g
    got = getattr(res.mass, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{part} mass ({name})", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_total_mass():
    """Calculator!C114 = magnets + cup + hub + boss + sleeve and liner + hardware allowance + cap + endplates,
    every solid from the independent references."""
    inp = defaults()
    res = run(inp)
    parts = rotating_parts_for_design(inp, res) | retainer_parts_for_design(inp, res)
    ref = math.fsum(p.mass_g for p in parts.values()) + inp.metal.hardware_g
    assert rel_err(res.mass.total_g, ref) < TOL_ALGEBRA, mismatch("total rotating mass", "Calculator!C114", res.mass.total_g, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_no_back_iron_hub_is_aluminium():
    """Calculator!C112 with coupling.backiron = 0 ('no intentional back iron'): the hub is the aluminium carrier
    (Metal design!C42), as the workbook states. (Whether the steel cup is consistent with this mode is
    test_materials.py::test_no_back_iron_mode_treats_the_cup_consistently.)"""
    inp = vary(defaults(), {"coupling.backiron": 0})
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)["hub"].mass_g
    got = res.mass.hub_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("hub mass without back iron", "Calculator!C112, Calculator!C6",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_arc_mode_hub_is_round():
    """Calculator!C112 with arcs (coupling.faceted = 0): the hub is round, π·r² with r the hub apothem."""
    inp = vary(defaults(), {"coupling.faceted": 0})
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)["hub"].mass_g
    got = res.mass.hub_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("arc-mode hub mass", "Calculator!C112", got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_added_inertia_point_mass():
    """Calculator!C115 is a point-mass estimate, m·d² at d = 0.25 m. Reference: parallel-axis theorem,
    I = m·d² + J_own, with J_own the coupling's polar inertia about its own axis (assumed parallel to the
    swing axis) summed over every solid, and the 6 g hardware allowance put at the cup OD radius (upper
    bound). Tolerance 1 %: the check confirms the neglected own-axis term is below 1 % of the result."""
    inp = defaults()
    res = run(inp)
    parts = rotating_parts_for_design(inp, res) | retainer_parts_for_design(inp, res)
    mass_g = math.fsum(p.mass_g for p in parts.values()) + inp.metal.hardware_g
    j_own_g_mm2 = (math.fsum(p.polar_inertia_g_mm2 for p in parts.values())
                   + inp.metal.hardware_g * (res.model.cup_od_mm / 2) ** 2)
    ref = mass_g / 1000 * SWING_RADIUS_M ** 2 + j_own_g_mm2 * 1e-9
    got = res.mass.added_inertia_kgm2
    assert rel_err(got, ref) < 1e-2, mismatch(f"added inertia (J_own = {j_own_g_mm2 * 1e-9:.3e} kg·m²)", "Calculator!C115",
                                              got, ref, 1e-2)
