"""Sanity tests for the Task 4 references against textbook cases with known answers.

These never touch the engine: they prove the references are trustworthy before the engine
is compared against them. Every test name contains 'sanity' so coverage tools can tell them
apart from engine checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import TOL_ALGEBRA, mismatch, rel_err
from audit.references.backiron_flux import (backiron_flux_per_depth, coupling_slab, flat_circuit_flux_density,
                                            sinusoidal_backiron_thickness, unrolled_coupling_slab)
from audit.references.block_geometry import (annulus_part, axial_overlap, block_rectangle, convex_polygon_distance,
                                             coupling_geometry, max_radius, min_radius, point_segment_distance,
                                             polygon_area, polygon_polar_moment, regular_polygon, ring_min_clearance,
                                             side_length)
from audit.references.metal_stack import (gearbox_input_torque, gearbox_output_torque, linear_stack, omega_rad_s,
                                          plated_size, pole_pair_frequency_Hz, pole_pairs, preplate_size, shaft_power_W,
                                          sleeve_liner_clearance)
from audit.references.remanence import br_at, br_ratio, torque_at
from audit.references.slab_field2d import MagnetLayer, harmonic_coefficients, iron_face_coefficients


# --------------------------------------------------------------------------- block_geometry
@pytest.mark.family("geometry")
def test_sanity_regular_polygon_square_and_hexagon():
    """Square of apothem 1: vertices at (±1, ±1), side 2. Hexagon of apothem √3/2: circumradius 1, side 1."""
    sq = regular_polygon(1.0, 4, phase=0.0)
    for x, y in sq:
        assert abs(abs(x) - 1) < 1e-12 and abs(abs(y) - 1) < 1e-12, \
            mismatch(f"square vertex ({x}, {y})", "reference", max(abs(x), abs(y)), 1.0, 1e-12)
    assert abs(side_length(sq) - 2) < 1e-12, mismatch("square side", "reference", side_length(sq), 2.0, 1e-12)
    hx = regular_polygon(math.sqrt(3) / 2, 6)
    assert rel_err(max_radius(hx), 1.0) < 1e-12, mismatch("hexagon circumradius", "reference", max_radius(hx), 1.0, 1e-12)
    assert rel_err(side_length(hx), 1.0) < 1e-12, mismatch("hexagon side", "reference", side_length(hx), 1.0, 1e-12)
    assert rel_err(min_radius(hx), math.sqrt(3) / 2) < 1e-12, \
        mismatch("hexagon apothem", "reference", min_radius(hx), math.sqrt(3) / 2, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_area_and_polar_moment():
    """Unit square about its centre: A = 1, J = 1/6. A 2×1 rectangle centred at (3, 0): J_O = J_c + A·d²
    = 2·(4 + 1)/12 + 2·9 (parallel-axis theorem). A 20000-gon of apothem 1 approaches the unit disk:
    A → π, J → π/2 (polygon error ~ (π/n)²/3 ≈ 8e-9)."""
    sq = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    assert rel_err(polygon_area(sq), 1.0) < 1e-12, mismatch("square area", "reference", polygon_area(sq), 1.0, 1e-12)
    assert rel_err(polygon_polar_moment(sq), 1 / 6) < 1e-12, \
        mismatch("square J", "reference", polygon_polar_moment(sq), 1 / 6, 1e-12)
    rect = [(2.0, -0.5), (4.0, -0.5), (4.0, 0.5), (2.0, 0.5)]
    want = 2 * (4 + 1) / 12 + 2 * 9
    assert rel_err(polygon_polar_moment(rect), want) < 1e-12, \
        mismatch("offset rectangle J", "reference", polygon_polar_moment(rect), want, 1e-12)
    disk = regular_polygon(1.0, 20000)
    assert rel_err(polygon_area(disk), math.pi) < 1e-7, mismatch("disk area", "reference", polygon_area(disk), math.pi, 1e-7)
    assert rel_err(polygon_polar_moment(disk), math.pi / 2) < 1e-7, \
        mismatch("disk J", "reference", polygon_polar_moment(disk), math.pi / 2, 1e-7)


@pytest.mark.family("geometry")
def test_sanity_annulus_part():
    """Tube OD 4, ID 2, length 3, ρ 1: m = π(4 − 1)·3 = 9π; J = m·(R² + r²)/2 = 9π·(4 + 1)/2."""
    p = annulus_part(4.0, 2.0, 3.0, 1.0)
    assert rel_err(p.mass_g, 9 * math.pi) < 1e-12, mismatch("tube mass", "reference", p.mass_g, 9 * math.pi, 1e-12)
    want = 9 * math.pi * 5 / 2
    assert rel_err(p.polar_inertia_g_mm2, want) < 1e-12, mismatch("tube J", "reference", p.polar_inertia_g_mm2, want, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_distances():
    """Point (0, 2) to segment (−1, 0)–(1, 0): 2; to (3, 0)–(5, 0): hypot(3, 2). Unit squares at x ∈ [0, 1] and
    [3, 4]: 2. A square rotated 45° with its corner at (1, 0) against a square whose face is at x = 1.5: 0.5."""
    mid = point_segment_distance((0, 2), (-1, 0), (1, 0))
    assert abs(mid - 2) < 1e-15, mismatch("point-segment interior", "reference", mid, 2.0, 1e-15)
    d = point_segment_distance((0, 2), (3, 0), (5, 0))
    assert rel_err(d, math.hypot(3, 2)) < 1e-15, mismatch("point-segment end", "reference", d, math.hypot(3, 2), 1e-15)
    a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b = [(3, 0), (4, 0), (4, 1), (3, 1)]
    assert abs(convex_polygon_distance(a, b) - 2) < 1e-15, mismatch("square gap", "reference", convex_polygon_distance(a, b), 2, 1e-15)
    diamond = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    box = [(1.5, -2), (3, -2), (3, 2), (1.5, 2)]
    got = convex_polygon_distance(diamond, box)
    assert abs(got - 0.5) < 1e-15, mismatch("corner to face", "reference", got, 0.5, 1e-15)


@pytest.mark.family("geometry")
def test_sanity_axial_overlap():
    """Centred 12.7 and 25.4 mm blocks overlap by 12.7 mm; two 10 mm blocks offset by 4 mm overlap by 6 mm;
    offset by 20 mm they do not overlap."""
    for args, want in (((12.7, 25.4), 12.7), ((10.0, 10.0, 4.0), 6.0), ((10.0, 10.0, 20.0), 0.0)):
        got = axial_overlap(*args)
        assert abs(got - want) < 1e-12, mismatch(f"axial overlap {args}", "reference", got, want, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_block_rectangle_and_ring_clearance():
    """A block at angle 0 with near face at 1, thickness 1, width 2 has corners at radius √(2² + 1²) = √5.
    Four such blocks against outer blocks whose faces sit at apothem 3: the inner corner can face an outer
    face centre, so the smallest clearance over rotation is 3 − √5 (hand geometry). In the thin-block
    limit (width 1e-6) the clearance is the face-to-face gap, 1.0."""
    blk = block_rectangle(1.0, 1.0, 2.0, 0.0)
    assert rel_err(max_radius(blk), math.sqrt(5)) < 1e-15, mismatch("block corner radius", "reference", max_radius(blk), math.sqrt(5), 1e-15)
    got = ring_min_clearance(4, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0)
    assert rel_err(got, 3 - math.sqrt(5)) < 1e-9, mismatch("ring clearance, N=4", "reference", got, 3 - math.sqrt(5), 1e-9)
    thin = ring_min_clearance(10, 10.0, 3.0, 1e-6, 14.0, 3.0, 1e-6)
    assert rel_err(thin, 1.0) < 1e-9, mismatch("thin-block ring clearance", "reference", thin, 1.0, 1e-9)


# --------------------------------------------------------------------------- backiron_flux and slab_field2d
@pytest.mark.family("materials")
def test_sanity_flat_circuit_textbook():
    """Furlani 2001 §3.3: ideal iron, μ_r = 1, Br = 1.2 T, magnet length 4 mm, gap 1 mm → B = 1.2·4/5 = 0.96 T.
    Split between two 2 mm magnets it is the same; with no gap B = Br."""
    got = flat_circuit_flux_density(1.2, 1.2, 2.0, 2.0, 1.0)
    assert rel_err(got, 0.96) < TOL_ALGEBRA, mismatch("flat circuit", "reference", got, 0.96, TOL_ALGEBRA)
    got = flat_circuit_flux_density(1.2, 1.2, 4.0, 0.0, 0.0)
    assert rel_err(got, 1.2) < TOL_ALGEBRA, mismatch("flat circuit, no gap", "reference", got, 1.2, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_sanity_sinusoidal_backiron_thickness():
    """Half the flux of one pole of B̂·sin(πx/τ), integrated numerically (scipy quad), divided by B_design:
    B̂ = 1 T, τ = 10 mm, B_design = 1.5 T → (2·10/π)/2/1.5 = 2.1221 mm."""
    flux, _ = quad(lambda x: math.sin(math.pi * x / 10.0), 0.0, 10.0)
    want = flux / 2 / 1.5
    got = sinusoidal_backiron_thickness(1.0, 10.0, 1.5)
    assert rel_err(got, want) < 1e-12, mismatch("sinusoidal back iron", "reference", got, want, 1e-12)


@pytest.mark.family("materials")
@pytest.mark.parametrize("surface", ["outer", "inner"])
def test_sanity_slab_magnet_fills_space(surface):
    """One magnet filling the whole space between the plates: H = 0 and B = μ0·M exactly, so the flux entering
    either plate over half a pole is Br·fill·τ/2 (square wave). Exercises both terms of iron_face_coefficients
    (the sheet on the plate adds nothing; μ0M of the touching layer is added).
    Tolerance 2e-4: harmonic truncation at n = 4001 (terms fall as 1/n²)."""
    br, fill, tau = 1.3, 0.8, 9.0
    got = backiron_flux_per_depth(surface, [MagnetLayer(0.0, 3.0, br, fill)], 3.0, tau)
    want = br * fill * tau / 2
    assert rel_err(got, want) < 2e-4, mismatch(f"slab, magnet fills space ({surface} plate)", "reference", got, want, 2e-4)


@pytest.mark.family("materials")
def test_sanity_slab_long_pitch_is_flat_circuit():
    """When the pole pitch is 10⁴ times the circuit height the field is the flat circuit's under the magnets and
    zero between them, so the half-pole flux is B_flat·fill·τ/2. Tolerance 1e-3: fringing is O(h/τ)."""
    br, fill, ti, to, g = 1.24, 0.7, 3.17, 3.17, 1.4
    layers, h = coupling_slab(br, br, fill, fill, ti, to, g)
    tau = 1e4 * h
    got = backiron_flux_per_depth("outer", layers, h, tau, n_max=40001)
    want = flat_circuit_flux_density(br, br, ti, to, g) * fill * tau / 2
    assert rel_err(got, want) < 1e-3, mismatch("slab, long pitch", "reference", got, want, 1e-3)


@pytest.mark.family("materials")
def test_sanity_slab_mirror_symmetry():
    """Identical rings: the inner and outer plates carry the same flux (mirror symmetry y → h − y)."""
    layers, h = coupling_slab(1.24, 1.24, 0.75, 0.75, 3.0, 3.0, 1.2)
    outer = backiron_flux_per_depth("outer", layers, h, 8.8)
    inner = backiron_flux_per_depth("inner", layers, h, 8.8)
    assert rel_err(inner, outer) < 1e-12, mismatch("slab symmetry", "reference", inner, outer, 1e-12)


@pytest.mark.family("materials")
@pytest.mark.parametrize("surface", ["outer", "inner"])
def test_sanity_iron_face_is_limit_of_general_solution(surface):
    """iron_face_coefficients against the module's general solution harmonic_coefficients: with air between the
    last magnet face and a plate, B_y at distance d from that plate is B_face·cosh(k d) (the Green's function's
    y-dependence there), harmonic by harmonic. Unequal, opposed and shifted layers exercise both cos and sin
    terms. Tolerance 1e-12 of the largest harmonic: rounding only."""
    layers = [MagnetLayer(0.6, 2.0, 1.2, 0.8, 0.7), MagnetLayer(3.7, 1.1, -1.1, 0.6, -0.4)]
    h, pitch, d = 5.3, 7.0, 0.05
    n, a_face, b_face = iron_face_coefficients(surface, layers, h, pitch, n_max=201)
    _, a_y, b_y, _, _ = harmonic_coefficients(h - d if surface == "outer" else d, layers, h, pitch, n_max=201)
    growth = np.cosh(n * math.pi / pitch * d)
    scale = float(np.max(np.abs(np.concatenate([a_y, b_y]))))
    err = float(np.max(np.abs(np.concatenate([a_y - a_face * growth, b_y - b_face * growth])))) / scale
    assert err < 1e-12, mismatch(f"iron-face coefficients vs general solution ({surface})", "reference", err, 0.0, 1e-12)


def _default_rings_geometry(w_o: float = 6.35):
    """Reference geometry of the default rings: 10 poles, both rings the same 6.35 x 3.17 mm block, inner block
    backs at apothem 10.15 mm, 1.4 mm face gap. Gap radius 10.15 + 3.17 + 0.7 = 14.02 mm, pole pitch
    2*pi*14.02/10 = 8.8090 mm. Bondlines, cup wall, bore and keyway do not enter the slab."""
    return coupling_geometry(npole=10, inner_back_apothem=10.15, t_i=3.17, w_i=6.35, t_o=3.17, w_o=w_o, face_gap=1.4,
                             bond_inner=0.05, bond_outer=0.05, cup_wall_corner=1.8, bore=10.0, keyway_depth=1.7)


@pytest.mark.family("materials")
def test_sanity_unrolled_slab_keeps_block_widths():
    """The back-iron checks unroll both rings onto one slab at the gap-radius pole pitch tau = 2*pi*14.02/10 =
    8.8090 mm. A flat block keeps its real width there, so each layer's block is fill*tau = 6.35 mm wide
    (fill = 6.35/8.8090 = 0.7209 on both rings). The arc-length fills at the blocks' mid radii (Calculator C66/C67,
    0.8612 inner, 0.6198 outer) would scale each block by R_g/r_mid instead: 7.586 mm (inner) and 5.460 mm
    (outer). Both rings are then the same layer, so the slab is mirror-symmetric (y -> h - y) and the hub and cup
    plates carry the same flux. Tolerance 1e-12: rounding only."""
    g = _default_rings_geometry()
    layers, h = unrolled_coupling_slab(g, 1.24356, 1.24356, 6.35, 6.35, 3.17, 3.17)
    for ring, layer in zip(("inner", "outer"), layers):
        width = layer.fill * g.pole_pitch
        assert rel_err(width, 6.35) < 1e-12, mismatch(f"{ring} block width in the unrolled slab (fill {layer.fill:.4f} "
                                                      f"x pitch {g.pole_pitch:.4f} mm)", "reference", width, 6.35, 1e-12)
    assert rel_err(h, 3.17 + 1.4 + 3.17) < 1e-12, mismatch("slab height", "reference", h, 7.74, 1e-12)
    inner = backiron_flux_per_depth("inner", layers, h, g.pole_pitch)
    outer = backiron_flux_per_depth("outer", layers, h, g.pole_pitch)
    assert rel_err(inner, outer) < 1e-12, mismatch("unrolled identical rings: hub vs cup plate flux [T*mm]", "reference",
                                                   inner, outer, 1e-12)


@pytest.mark.family("materials")
@pytest.mark.parametrize("w_o", [9.0, 0.0], ids=["wider than the pitch", "zero width"])
def test_sanity_unrolled_slab_refuses_unrepresentable_blocks(w_o):
    """A 9.0 mm outer block fits its 9.57 mm pocket flat but is wider than the 8.809 mm gap-radius pitch: unrolled
    it would overlap its neighbours, which a slab layer (0 < fill <= 1) cannot represent, so the slab is refused
    rather than clamped. A zero-width block is refused too."""
    with pytest.raises(ValueError):
        unrolled_coupling_slab(_default_rings_geometry(w_o), 1.24356, 1.24356, 6.35, w_o, 3.17, 3.17)


@pytest.mark.family("materials")
def test_sanity_backiron_flux_needs_aligned_layers():
    """The pole-centre symmetry argument only holds for aligned layers; a shifted layer is refused."""
    with pytest.raises(ValueError):
        backiron_flux_per_depth("outer", [MagnetLayer(0.0, 3.0, 1.3, 0.8, 0.5)], 3.0, 9.0)


# --------------------------------------------------------------------------- remanence
@pytest.mark.family("metal")
def test_sanity_br_temperature_scaling():
    """α = −0.12 %/°C: Br(50 °C) = 1.29·(1 − 0.0012·30) = 1.24356 T; Br(100 °C)/Br(20 °C) = 0.904; torque scales
    by 0.904²; moving 20 → 100 → 20 °C returns the start value."""
    assert rel_err(br_at(1.29, -0.0012, 50.0), 1.24356) < TOL_ALGEBRA, \
        mismatch("Br at 50 °C", "reference", br_at(1.29, -0.0012, 50.0), 1.24356, TOL_ALGEBRA)
    assert rel_err(br_ratio(-0.0012, 100.0), 0.904) < TOL_ALGEBRA, \
        mismatch("Br ratio", "reference", br_ratio(-0.0012, 100.0), 0.904, TOL_ALGEBRA)
    t100 = torque_at(2.0, -0.0012, 20.0, 100.0)
    assert rel_err(t100, 2.0 * 0.904 ** 2) < TOL_ALGEBRA, mismatch("torque at 100 °C", "reference", t100, 2.0 * 0.904 ** 2, TOL_ALGEBRA)
    back = torque_at(t100, -0.0012, 100.0, 20.0)
    assert rel_err(back, 2.0) < TOL_ALGEBRA, mismatch("round trip", "reference", back, 2.0, TOL_ALGEBRA)


# --------------------------------------------------------------------------- metal_stack
@pytest.mark.family("metal")
def test_sanity_duty_and_power():
    """20 poles are 10 pole pairs. 2 poles (one pole pair) at 60 rpm pass 1 pole pair per second. 60/(2π) rpm is
    1 rad/s, and 1 N·m at that speed is 1 W."""
    assert rel_err(pole_pairs(20), 10.0) < TOL_ALGEBRA, mismatch("pole pairs", "reference", pole_pairs(20), 10.0, TOL_ALGEBRA)
    assert rel_err(pole_pair_frequency_Hz(2, 60.0), 1.0) < TOL_ALGEBRA, \
        mismatch("pole-pair frequency", "reference", pole_pair_frequency_Hz(2, 60.0), 1.0, TOL_ALGEBRA)
    w = omega_rad_s(60 / (2 * math.pi))
    assert rel_err(w, 1.0) < TOL_ALGEBRA, mismatch("angular speed", "reference", w, 1.0, TOL_ALGEBRA)
    p = shaft_power_W(1.0, 60 / (2 * math.pi))
    assert rel_err(p, 1.0) < TOL_ALGEBRA, mismatch("shaft power", "reference", p, 1.0, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_sanity_gearbox_power_balance():
    """0.6 N·m in at 3000 rpm through 5:1 at η = 0.95: the output turns at 600 rpm and carries 95 % of the input
    power, so T_out = 0.95·P_in/ω_out = 2.85 N·m. The input form inverts the output form."""
    want = 0.95 * shaft_power_W(0.6, 3000.0) / (2 * math.pi * 600.0 / 60)
    got = gearbox_output_torque(0.6, 5.0, 0.95)
    assert rel_err(got, want) < TOL_ALGEBRA, mismatch("gearbox output torque", "reference", got, want, TOL_ALGEBRA)
    back = gearbox_input_torque(got, 5.0, 0.95)
    assert rel_err(back, 0.6) < TOL_ALGEBRA, mismatch("gearbox round trip", "reference", back, 0.6, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_sanity_plating_and_stacks():
    """A 10 mm pin plated 0.015 mm becomes 10.030 mm; a 10 mm bore becomes 9.970 mm; a pin machined with
    preplate_size plates back to 10 mm. A 1.0 mm corner gap minus 0.1 + 0.2 walls and 2 × 0.025 bedding
    leaves 0.65 mm; 0.4 + 0.05 + 0.05 stacks to 0.5."""
    assert rel_err(plated_size(10.0, 0.015, 2, True), 10.03) < TOL_ALGEBRA, \
        mismatch("plated pin", "reference", plated_size(10.0, 0.015, 2, True), 10.03, TOL_ALGEBRA)
    assert rel_err(plated_size(10.0, 0.015, 2, False), 9.97) < TOL_ALGEBRA, \
        mismatch("plated bore", "reference", plated_size(10.0, 0.015, 2, False), 9.97, TOL_ALGEBRA)
    pre = preplate_size(10.0, 0.015, 2, True)
    assert rel_err(plated_size(pre, 0.015, 2, True), 10.0) < TOL_ALGEBRA, \
        mismatch("pre-plate round trip", "reference", plated_size(pre, 0.015, 2, True), 10.0, TOL_ALGEBRA)
    c = sleeve_liner_clearance(1.0, 0.1, 0.2, 0.025, 0.025)
    assert rel_err(c, 0.65) < TOL_ALGEBRA, mismatch("clearance stack", "reference", c, 0.65, TOL_ALGEBRA)
    assert rel_err(linear_stack(0.4, 0.05, 0.05), 0.5) < TOL_ALGEBRA, \
        mismatch("linear stack", "reference", linear_stack(0.4, 0.05, 0.05), 0.5, TOL_ALGEBRA)
