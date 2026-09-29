"""Sanity tests for audit/references/planar.py and audit/references/torque_ref.py (M1, Task 3).

Each reference is checked against a case with a known answer. No engine value is under test
here, so a failure means the reference is wrong, not the engine. Every test name contains
"sanity", so Task 8's coverage table can leave these tests out of the engine confirmations.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import MU0_EXACT, defaults, mismatch, run, vary
from audit.references import planar
from audit.references import torque_ref as tr

MM = tr.MM
REF = "reference only"

#: The workbook-default section written out by hand, so the 3D sanity tests need no engine: 10 poles, B842SH
#: blocks 3.17 mm (radial) x 6.35 mm x 12.7 mm, Br 1.29 T (20 °C), inner block backs on the 10.15 mm hub apothem
#: (faces at 13.32 mm), outer block faces 1.4 mm further out at 14.72 mm.
DEFAULT_PAIR = tr.RingPair(npole=10, r_i_m=(10.15 + 3.17 / 2) * MM, t_i_m=3.17 * MM, w_i_m=6.35 * MM, l_i_m=12.7 * MM,
                           j_i_T=1.29, r_o_m=(14.72 + 3.17 / 2) * MM, t_o_m=3.17 * MM, w_o_m=6.35 * MM, l_o_m=12.7 * MM,
                           j_o_T=1.29)
#: The same section at a 0.6 mm face gap (outer faces at 13.92 mm, corner gap 0.23 mm): the tightest 3D engine case.
TIGHT_PAIR = replace(DEFAULT_PAIR, r_o_m=(13.92 + 3.17 / 2) * MM)


# --------------------------------------------------------------------------- planar.py
@pytest.mark.family("torque")
@pytest.mark.parametrize("n, fill", [(1, 0.8), (3, 0.8), (5, 0.52), (3, 0.37)])
def test_sanity_square_wave_harmonic_by_quadrature(n, fill):
    """square_wave_harmonic vs the Fourier integral b_n = (1/pi) * integral of m(theta) cos(n theta) over one period,
    with m = +1 within fill*pi/2 of 0 and -1 within fill*pi/2 of pi. The integral uses adaptive quadrature with the
    window edges as break points; tolerance 1e-10."""
    half = fill * math.pi / 2

    def m(th: float) -> float:
        if abs(th) < half:
            return 1.0
        if abs(th - math.pi) < half:
            return -1.0
        return 0.0

    val, _ = quad(lambda th: m(th) * math.cos(n * th), -math.pi / 2, 3 * math.pi / 2,
                  points=[-half, half, math.pi - half, math.pi + half], limit=200, epsabs=1e-13)
    expected = val / math.pi
    got = planar.square_wave_harmonic(1.0, n, fill)
    assert abs(got - expected) <= 1e-10, mismatch(f"square-wave harmonic n={n}, fill={fill}", REF, got, expected, 1e-10)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backed", [False, True])
def test_sanity_planar_factors_thick_array_limit(backed):
    """Both circuits reduce to semi-infinite arrays when k t -> infinity: S_1 -> e^{-k g}/2, so at delta = pitch/2
    tau_1 = B_1^2 e^{-k g} / (4 mu0). Here k t = 20 pi, which leaves e^{-20 pi} ~ 5e-28; tolerance 1e-12."""
    p, g, br = 6.0 * MM, 1.0 * MM, 1.3
    tau = planar.planar_shear_stress(br, br, 1.0, 1.0, p, 20 * p, 20 * p, g, p / 2, MU0_EXACT, backed, (1,))
    b1 = 4 * br / math.pi
    expected = b1 * b1 * math.exp(-math.pi * g / p) / (4 * MU0_EXACT)
    assert abs(tau / expected - 1) <= 1e-12, mismatch(f"thick-array shear, backed={backed}", REF, tau, expected, 1e-12)


@pytest.mark.family("torque")
@pytest.mark.parametrize("kg", [0.2, 0.52, 1.5])
def test_sanity_iron_backed_thin_array_limit_is_image_series(kg):
    """Method of images for two thin arrays, each lying on its own permeable backing, a gap g apart. Each array's image
    in its own backing doubles its field (2 x 2 = 4). Reflections between the two backings add the geometric series
    1 / (1 - e^{-2 k g}). So S(backed) / S(free) -> 4 / (1 - e^{-2 k g}) as t -> 0. Evaluated at k t = 1e-7, where
    the first-order correction k t (2 coth(k g) - 1) is at most 9.2e-7 (observed 9.1e-7 at k g = 0.2); tolerance
    1e-5."""
    k, t = 1000.0, 1e-10
    g = kg / k
    ratio = planar.iron_backed_factor(k, t, t, g) / planar.free_space_factor(k, t, t, g)
    expected = 4 / (1 - math.exp(-2 * kg))
    assert abs(ratio / expected - 1) <= 1e-5, mismatch(f"thin-array iron gain at k g = {kg}", REF, ratio, expected, 1e-5)


@pytest.mark.family("torque")
def test_sanity_planar_helper_hand_values():
    """Hand values: Br(50 °C) = 1.29 (1 - 0.0012 * 30) = 1.24356 T; k_3 at a 6 mm pitch = 3 pi / 0.006 m;
    100 kPa on a cylinder of radius 13.7 mm and length 12.7 mm gives 1e5 * 2 pi * 0.0137^2 * 0.0127 N*m;
    end factor 1 - 0.15 * 8.62 / 12.7 = 0.898189. The first two are exact, the rest are rounding-level: 1e-12."""
    cases = [
        ("Br at 50 °C", planar.br_at(1.29, -0.0012, 50.0), 1.24356),
        ("wave number k_3", planar.wave_number(3, 0.006), 1570.7963267948965),
        ("torque on cylinder", planar.torque_on_cylinder(1e5, 0.0137, 0.0127), 1e5 * 2 * math.pi * 0.0137 ** 2 * 0.0127),
        ("end factor", planar.end_factor(0.15, 8.62, 12.7), 1 - 0.15 * 8.62 / 12.7),
    ]
    for name, got, expected in cases:
        assert abs(got / expected - 1) <= 1e-12, mismatch(name, REF, got, expected, 1e-12)


# --------------------------------------------------------------------------- torque_ref.py: charge-model integrator
@pytest.mark.family("torque")
def test_sanity_uniform_field_force_zero_and_torque_m_cross_b():
    """Charge-model integrator in a uniform field: F = 0 and T = m x B with m = J V / mu0 (textbook dipole torque).

    The integrand is constant on each face, so Gauss-Legendre is exact: tolerance 1e-12 of the scale (one face's
    force for F, |m x B| for T).
    """
    blk = tr.Block(1.0 * MM, 2.0 * MM, 0.3, 3.0 * MM, 6.0 * MM, 12.0 * MM, 1.2)
    b0 = np.array([0.05, 0.5, 0.2])
    force, torque = tr.torque_on_blocks([blk], lambda p: np.tile(b0, (len(p), 1)))
    m = blk.j_T * blk.a_m * blk.b_m * blk.c_m / MU0_EXACT * np.array([math.cos(blk.phi), math.sin(blk.phi), 0.0])
    expected = np.cross(m, b0)
    face_force = blk.j_T / MU0_EXACT * blk.b_m * blk.c_m * float(np.linalg.norm(b0))
    torque_scale = float(np.linalg.norm(expected))
    for i in range(3):
        assert abs(force[i]) <= 1e-12 * face_force, mismatch(
            f"uniform-field net force component {i} [N] (scale {face_force:.3e} N)", REF, force[i], 0.0, 1e-12)
        assert abs(torque[i] - expected[i]) <= 1e-12 * torque_scale, mismatch(
            f"uniform-field torque component {i} [N*m]", REF, torque[i], expected[i], 1e-12)


@pytest.mark.family("torque")
def test_sanity_coaxial_cubes_approach_dipole_force():
    """Two coaxial cubes (a = 2 mm, J = 1.3 T) at d = 10 a: F = -3 mu0 m^2 / (2 pi d^4) (textbook dipole-dipole).

    A uniformly magnetized cube has no quadrupole-order correction (cubic symmetry), so the finite-size error is
    O((a/d)^4) = 1e-4 (observed 1.0e-4); tolerance 1e-3. The off-axis components vanish by symmetry: 1e-9 of F.
    """
    a, d, j = 2.0 * MM, 20.0 * MM, 1.3
    src, tgt = tr.Block(0.0, 0.0, 0.0, a, a, a, j), tr.Block(d, 0.0, 0.0, a, a, a, j)
    force, _ = tr.torque_on_blocks([tgt], tr.magpylib_field([src]), panel_m=a / 4, order=6)
    m = j * a ** 3 / MU0_EXACT
    expected = -3 * MU0_EXACT * m * m / (2 * math.pi * d ** 4)
    assert abs(force[0] / expected - 1) <= 1e-3, mismatch("coaxial cube axial force [N]", REF, force[0], expected, 1e-3)
    for i in (1, 2):
        assert abs(force[i]) <= 1e-9 * abs(expected), mismatch(
            f"coaxial cube off-axis force component {i} [N] (scale {abs(expected):.3e} N)", REF, force[i], 0.0, 1e-9)


@pytest.mark.family("torque")
@pytest.mark.parametrize("delta_over_pitch", [0.5, 0.25])
def test_sanity_3d_planar_arrays_reproduce_planar_shear_stress(delta_over_pitch):
    """3D charge model on long planar alternating arrays vs planar.planar_shear_stress (all odd harmonics to 3999).

    Pitch 6 mm, fill 0.8, t = 3 mm, g = 1 mm, Br = 1.3 T, 41 source blocks. The 2D value is the slope of force vs
    length between 30 and 60 mm. Observed agreement 2.4e-6 (array truncation and slope residual); tolerance 1e-4.
    The target's own-array neighbours exert no tangential force on it by symmetry, so only the lower array is a source.
    """
    p, w, t, g, br = 6.0 * MM, 4.8 * MM, 3.0 * MM, 1.0 * MM, 1.3
    delta = delta_over_pitch * p

    def force_x(length):
        lower = [tr.Block(k * p, -t / 2, math.pi / 2, t, w, length, br * (-1) ** k) for k in range(-20, 21)]
        target = tr.Block(delta, g + t / 2, math.pi / 2, t, w, length, br)
        f, _ = tr.torque_on_blocks([target], tr.magpylib_field(lower))
        return f[0]

    tau_3d = -(force_x(60 * MM) - force_x(30 * MM)) / (30 * MM) / p
    tau_ref = planar.planar_shear_stress(br, br, w / p, w / p, p, t, t, g, delta, MU0_EXACT, False, range(1, 4001, 2))
    assert abs(tau_3d / tau_ref - 1) <= 1e-4, mismatch(
        f"planar-array shear stress at delta = {delta_over_pitch} pitch [Pa]", REF, tau_3d, tau_ref, 1e-4)


# --------------------------------------------------------------------------- torque_ref.py: rings
@pytest.mark.family("torque")
def test_sanity_ring_newton_third_law():
    """Torque on the outer ring from the inner ring = -(torque on the inner ring from the outer ring), all blocks.

    The two integrals use different faces and fields, so agreement tests the integrator; tolerance 1e-6.
    """
    pair = DEFAULT_PAIR
    theta = 0.3 * 2 * math.pi / pair.npole
    inner = tr.ring_blocks(pair.npole, pair.r_i_m, pair.t_i_m, pair.w_i_m, pair.l_i_m, pair.j_i_T, 0.0)
    outer = tr.ring_blocks(pair.npole, pair.r_o_m, pair.t_o_m, pair.w_o_m, pair.l_o_m, pair.j_o_T, theta)
    _, t_on_outer = tr.torque_on_blocks(outer, tr.magpylib_field(inner))
    _, t_on_inner = tr.torque_on_blocks(inner, tr.magpylib_field(outer))
    assert abs(t_on_outer[2] + t_on_inner[2]) <= 1e-6 * abs(t_on_outer[2]), \
        mismatch("action-reaction torque [N*m]", REF, t_on_outer[2], -t_on_inner[2], 1e-6)


@pytest.mark.family("torque")
def test_sanity_ring_symmetries():
    """Exact ring symmetries: zero torque when aligned and when like poles face, T(pitch - th) = T(th),
    T(th + pitch) = -T(th), and block 0 x N equals the sum over all outer blocks. Tolerance 1e-9 of the peak."""
    pair = DEFAULT_PAIR
    pitch = 2 * math.pi / pair.npole
    th = 0.37 * pitch
    peak = tr.ring_torque_3d(pair, pair.half_pitch)
    t_th = tr.ring_torque_3d(pair, th)
    cases = [
        ("torque when aligned (theta = 0)", tr.ring_torque_3d(pair, 0.0), 0.0),
        ("torque when like poles face (theta = pitch)", tr.ring_torque_3d(pair, pitch), 0.0),
        ("mirror symmetry T(pitch - th) vs T(th)", tr.ring_torque_3d(pair, pitch - th), t_th),
        ("antisymmetry T(th + pitch) vs -T(th)", tr.ring_torque_3d(pair, th + pitch), -t_th),
        ("block 0 x N vs all outer blocks", t_th, tr.ring_torque_3d(pair, th, all_blocks=True)),
    ]
    for name, got, expected in cases:
        assert abs(got - expected) <= 1e-9 * abs(peak), mismatch(
            f"{name} [N*m] (scale: peak {peak:.4f} N*m)", REF, got, expected, 1e-9)


@pytest.mark.family("torque")
def test_sanity_ring_quadrature_converged_default_gap():
    """Default quadrature (1 mm panels, 4 points) vs 0.25 mm panels, 6 points, at half a pole pitch and the 1.4 mm
    face gap. Observed difference 1e-7; tolerance 1e-5, far below the 2 % engine comparison bar."""
    coarse = tr.ring_torque_3d(DEFAULT_PAIR, DEFAULT_PAIR.half_pitch)
    fine = tr.ring_torque_3d(DEFAULT_PAIR, DEFAULT_PAIR.half_pitch, panel_m=0.25 * MM, order=6)
    assert abs(coarse / fine - 1) <= 1e-5, mismatch("quadrature convergence, 1.4 mm face gap [N*m]", REF, coarse, fine, 1e-5)


@pytest.mark.family("torque")
def test_sanity_ring_quadrature_converged_tightest_gap():
    """At the tightest engine case (0.6 mm face gap, 0.23 mm corner clearance) 1 mm panels are 4.8e-4 off, so the gap
    sweep uses 0.25 mm / 6-point panels. Those agree with 0.1 mm / 6-point panels to 1e-10; tolerance 1e-6."""
    fine = tr.ring_torque_3d(TIGHT_PAIR, TIGHT_PAIR.half_pitch, panel_m=0.25 * MM, order=6)
    finer = tr.ring_torque_3d(TIGHT_PAIR, TIGHT_PAIR.half_pitch, panel_m=0.1 * MM, order=6)
    assert abs(fine / finer - 1) <= 1e-6, mismatch("quadrature convergence, 0.6 mm face gap [N*m]", REF, fine, finer, 1e-6)


@pytest.mark.family("torque")
def test_sanity_long_length_slope_is_length_independent():
    """The 2D (per-length) torque from lengths 30/60 mm equals that from 60/120 mm, because the end deficit is
    constant. Observed difference 1.1e-5; tolerance 1e-4."""
    th = DEFAULT_PAIR.half_pitch
    a_short = tr.torque_per_length_3d(DEFAULT_PAIR, 30 * MM, 60 * MM, th)
    a_long = tr.torque_per_length_3d(DEFAULT_PAIR, 60 * MM, 120 * MM, th)
    assert abs(a_short / a_long - 1) <= 1e-4, mismatch("per-length torque [N*m/m]", REF, a_short, a_long, 1e-4)


@pytest.mark.family("torque")
def test_sanity_3d_long_length_slope_matches_2d_field_reference():
    """Cross-validation of two independent references on the same free-space section at half a pole pitch: the 3D
    long-length slope (magpylib fields, surface-charge force) vs Task 2's exact 2D flat-block solution
    (audit.references.field2d: bound-current sheets, Maxwell stress). The 3D slope residual is 1.1e-5 and the 2D
    free-space solution is exact to about 1e-9, so the tolerance is 1e-4.

    field2d is imported inside the test so that only this test depends on Task 2's module. It uses
    CouplingSection(n_poles, inner_back_mm, inner_thickness_mm, inner_width_mm, outer_face_mm, outer_thickness_mm,
    outer_width_mm, br_inner_T, br_outer_T) and torque_per_length(sec, theta_e), whose positive sign is restoring.
    """
    from audit.references import field2d

    sec = field2d.CouplingSection(10, 10.15, 3.17, 6.35, 14.72, 3.17, 6.35, 1.29, 1.29)
    ref_2d = field2d.torque_per_length(sec, math.pi / 2)
    per_length_3d = tr.torque_per_length_3d(DEFAULT_PAIR, 30 * MM, 60 * MM, DEFAULT_PAIR.half_pitch)
    assert abs(per_length_3d / ref_2d - 1) <= 1e-4, mismatch(
        "3D long-length slope vs Task 2 exact 2D torque per length [N*m/m]", REF, per_length_3d, ref_2d, 1e-4)


# --------------------------------------------------------------------------- torque_ref.py: prototype and builders
@pytest.mark.family("torque")
def test_sanity_prototype_geometry_hand_values():
    """Prototype geometry by hand: face radius 9.85 + 3.17 = 13.02 mm, corner radius hypot(13.02, 3.175), flat gap
    1.4 mm, corner gap 1.4 mm minus the corner overhang, outer face 14.42 mm, gap radius 13.72 mm, pitch
    2 pi 13.72 / 10, fills w N / (2 pi r_mid). Rounding-level agreement: 1e-12."""
    geo = tr.prototype_geometry(defaults().calibration)
    corner = math.sqrt(13.02 ** 2 + 3.175 ** 2)
    cases = [
        ("poles per ring", geo.poles, 10.0),
        ("face radius [mm]", geo.face_radius_mm, 13.02),
        ("corner radius [mm]", geo.corner_radius_mm, corner),
        ("flat gap [mm]", geo.flat_gap_mm, 1.4),
        ("corner gap [mm]", geo.corner_gap_mm, 1.4 - (corner - 13.02)),
        ("outer face apothem [mm]", geo.outer_face_apothem_mm, 14.42),
        ("gap radius [mm]", geo.gap_radius_mm, 13.72),
        ("pole pitch [mm]", geo.pole_pitch_mm, 2 * math.pi * 13.72 / 10),
        ("inner fill", geo.fill_inner, 6.35 * 10 / (2 * math.pi * (9.85 + 3.17 / 2))),
        ("outer fill", geo.fill_outer, 6.35 * 10 / (2 * math.pi * (14.42 + 3.17 / 2))),
    ]
    for name, got, expected in cases:
        assert abs(got - expected) <= 1e-12 * max(1.0, abs(expected)), mismatch(
            f"prototype {name}", REF, got, expected, 1e-12)


@pytest.mark.family("torque")
def test_sanity_prototype_gap_definitions_agree():
    """The same prototype described by its corner gap (definition 0) gives the same calibration as by its flat gap."""
    c = defaults().calibration
    geo = tr.prototype_geometry(c)
    as_corner = replace(c, gap_definition=0, spacing_mm=c.spacing_mm - (geo.corner_radius_mm - geo.face_radius_mm))
    a, b = tr.prototype_calibration(c), tr.prototype_calibration(as_corner)
    assert abs(a["f_cal_updated"] / b["f_cal_updated"] - 1) <= 1e-12, \
        mismatch("corner vs flat gap definition", REF, b["f_cal_updated"], a["f_cal_updated"], 1e-12)


@pytest.mark.family("torque")
def test_sanity_ring_builder_matches_hand_geometry():
    """ring_pair_from_engine at the free-space 20 °C defaults reproduces DEFAULT_PAIR (written out by hand), so the
    builder reads the right engine fields: block centre radius = face radius -/+ t/2, widths, lengths, Br."""
    inp = vary(defaults(), {"coupling.op_temp_C": 20, "coupling.backiron": 0})
    built = tr.ring_pair_from_engine(inp, run(inp))
    for name in ("r_i_m", "t_i_m", "w_i_m", "l_i_m", "j_i_T", "r_o_m", "t_o_m", "w_o_m", "l_o_m", "j_o_T"):
        got, expected = getattr(built, name), getattr(DEFAULT_PAIR, name)
        assert abs(got / expected - 1) <= 1e-12, mismatch(f"ring builder {name}", REF, got, expected, 1e-12)
    assert built.npole == DEFAULT_PAIR.npole, mismatch("ring builder npole", REF, built.npole, DEFAULT_PAIR.npole, 0.0)


@pytest.mark.family("torque")
def test_sanity_ring_builder_rejects_unsupported_geometry():
    """The 3D builder refuses arc magnets (faceted = 0) and odd pole counts instead of silently modelling them."""
    inp_arc = vary(defaults(), {"coupling.faceted": 0})
    with pytest.raises(ValueError):
        tr.ring_pair_from_engine(inp_arc, run(inp_arc))
    inp_odd = vary(defaults(), {"coupling.npole": 9})
    with pytest.raises(ValueError):
        tr.ring_pair_from_engine(inp_odd, run(inp_odd))
