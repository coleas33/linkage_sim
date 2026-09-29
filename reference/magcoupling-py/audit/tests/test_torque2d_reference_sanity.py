"""M1 Task 2: sanity checks of the 2D field reference (``field2d``) and the planar formulas (``planar``).

These tests never run the engine. Each one checks a reference against a textbook result with a
known answer, against a second independent method, or against its own convergence behaviour, so
the engine checks in ``test_torque_field2d.py`` compare the engine with a trustworthy reference.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import MU0_EXACT, mismatch
from audit.references.field2d import (
    MM, CouplingSection, IronCircles, IronPolygons, Magnet, Sheets, b_field, b_from_charges, b_from_currents,
    b_from_inverted_currents, charge_sheets, current_sheets, cylindrical_s_factor, maxwell_torque_per_length,
    polygon_iron_charges, torque_per_length,
)
from audit.references.planar import planar_s_factor, square_wave_harmonic

#: The default section written out (10 poles, B842SH 6.35 x 3.17 mm, inner backs at 10.15 mm,
#: outer faces at 14.72 mm, outer backs at 17.89 mm), so these tests need no engine.
SANITY_SECTION = CouplingSection(10, 10.15, 3.17, 6.35, 14.72, 3.17, 6.35, 1.29, 1.29)
SANITY_CUP_MM = 17.89
#: The same section at a 0.5 mm face gap (outer faces at 13.82 mm): the inner block corners
#: (13.693 mm) clear the outer faces by only 0.127 mm, the hardest case for the stress quadrature.
SMALL_GAP_SECTION = CouplingSection(10, 10.15, 3.17, 6.35, 13.82, 3.17, 6.35, 1.29, 1.29)


def sliced_radial_ring(r_a_mm: float, r_b_mm: float, m: int, b_T: float, phase: float, slices: int) -> list[Magnet]:
    """Annulus of radially magnetized polygon slices, J = b_T cos(m theta - phase) along each
    slice's centre radius: a polygon stand-in for sinusoidal radial magnetization."""
    out = []
    for j in range(slices):
        c, h = 2 * math.pi * (j + 0.5) / slices, math.pi / slices
        lo, hi = np.exp(1j * (c - h)), np.exp(1j * (c + h))
        verts = (r_a_mm * MM * lo, r_b_mm * MM * lo, r_b_mm * MM * hi, r_a_mm * MM * hi)
        out.append(Magnet(tuple(complex(v) for v in verts), b_T * math.cos(m * c - phase) * np.exp(1j * c)))
    return out


# --------------------------------------------------------------------------- field2d: sources
@pytest.mark.family("torque")
def test_sanity_sheet_fields_match_quadrature():
    """Closed-form sheet fields (direct current, inverted-image current, charge) equal direct
    numerical integration of the line-source field over the sheet (scipy quad, 1e-13)."""
    sh = Sheets(np.array([0.001 + 0.002j]), np.array([0.004 + 0.0035j]), np.array([0.7]))
    pts = np.array([0.003 + 0.001j, -0.002 + 0.005j, 0.0025 + 0.0027j])
    rho = 0.0025

    def integrate(source_at, line_field):
        d, out = sh.z2[0] - sh.z1[0], []
        for p in pts:
            def f(t, part):
                v = line_field(p - source_at(sh.z1[0] + t * d)) * abs(d) * sh.k[0]
                return v.real if part == 0 else v.imag
            out.append(quad(f, 0, 1, args=(0,), epsabs=0, epsrel=1e-13, limit=200)[0]
                       + 1j * quad(f, 0, 1, args=(1,), epsabs=0, epsrel=1e-13, limit=200)[0])
        return np.array(out)

    current = lambda r: 1j / (2 * math.pi) * r / abs(r) ** 2    # line current, mu0 I = 1 along +z
    charge = lambda r: 1 / (2 * math.pi) * r / abs(r) ** 2      # line charge, mu0 q = 1
    for name, closed, numeric in (
            ("direct current", b_from_currents(pts, sh), integrate(lambda z: z, current)),
            ("inverted current", b_from_inverted_currents(pts, sh, rho), integrate(lambda z: rho ** 2 / np.conj(z), current)),
            ("charge", b_from_charges(pts, sh), integrate(lambda z: z, charge))):
        err = float(np.max(np.abs(closed - numeric)) / np.max(np.abs(numeric)))
        assert err < 1e-10, mismatch(f"{name} sheet field vs quadrature", "reference only", err, 0.0, 1e-10)


@pytest.mark.family("torque")
def test_sanity_bar_on_axis_textbook():
    """Infinitely long bar, width w, height t, magnetized across its height: on the axis above
    it B_y = (Br/pi) [atan(w / 2(y - t)) - atan(w / 2y)], B_x = 0 (the field of two
    uniformly charged strips, sigma = +/-Br/mu0; e.g. Furlani, Permanent Magnet and
    Electromechanical Devices, 2001, ch. 4)."""
    br, w, t = 1.3, 6e-3, 3e-3
    bar = Magnet((complex(-w / 2, 0), complex(w / 2, 0), complex(w / 2, t), complex(-w / 2, t)), 1j * br)
    ys = np.array([3.5e-3, 5e-3, 1e-2, 3e-2])
    b = b_field(1j * ys, [bar])
    expect = br / math.pi * (np.arctan(w / (2 * (ys - t))) - np.arctan(w / (2 * ys)))
    for got, ref in zip(b, expect):
        assert abs(got.imag - ref) <= 1e-12 * abs(ref), mismatch("bar on-axis B_y", "reference only", got.imag, ref, 1e-12)
        assert abs(got.real) <= 1e-12 * abs(ref), mismatch("bar on-axis B_x", "reference only", got.real, 0.0, 1e-12)


@pytest.mark.family("torque")
def test_sanity_polygon_centre_field_is_half_polarization():
    """At the centre of a uniformly magnetized regular polygon (n >= 3) the 2D demagnetizing
    tensor is isotropic with trace 1, so H = -M/2 and B = J/2 exactly: the transverse
    demagnetizing factor 1/2 of an infinite cylinder (Osborn, Phys. Rev. 67, 1945)."""
    j = 1.2 * np.exp(0.7j)
    for n in (3, 6, 11):
        poly = Magnet(tuple(complex(2e-3 * np.exp(2j * math.pi * k / n)) for k in range(n)), j)
        got = b_field(np.array([0j]), [poly])[0]
        assert abs(got - j / 2) <= 1e-12, mismatch(f"centre field of a {n}-gon", "reference only", abs(got), abs(j / 2), 1e-12)


@pytest.mark.family("torque")
def test_sanity_charge_and_current_models_agree():
    """In air the magnet field from surface charges (used by the BEM) equals the field from
    bound surface currents (used everywhere else), at 64 points around the default gap."""
    sec = SANITY_SECTION
    mags = sec.inner() + sec.outer(0.1)
    z = sec.stress_radius_mm() * MM * np.exp(2j * math.pi * (np.arange(64) + 0.3) / 64)
    bc, bq = b_from_currents(z, current_sheets(mags)), b_from_charges(z, charge_sheets(mags))
    err = float(np.max(np.abs(bc - bq)) / np.max(np.abs(bc)))
    assert err < 1e-12, mismatch("charge vs current field in the gap", "reference only", err, 0.0, 1e-12)


# --------------------------------------------------------------------------- field2d: steel
@pytest.mark.family("torque")
def test_sanity_images_make_field_normal_to_steel():
    """Infinitely permeable steel: B has no tangential part on its surface. With 8 image
    generations |B_t| / max|B| < 1e-9 on both circles; without images it is order 1."""
    sec = SANITY_SECTION
    mags = sec.inner() + sec.outer(0.1)
    iron = IronCircles(10.0, 18.5, 8)
    theta = 2 * math.pi * (np.arange(97) + 0.37) / 97
    for r_mm in (iron.r_inner_mm, iron.r_outer_mm):
        z = r_mm * MM * np.exp(1j * theta)
        for images, bound in ((iron, 1e-9), (None, None)):
            b = b_field(z, mags, images) * np.exp(-1j * theta)
            ratio = float(np.max(np.abs(b.imag)) / np.max(np.abs(b)))
            if bound is None:
                assert ratio > 0.1, mismatch(f"tangential B without images at r={r_mm}", "reference only", ratio, 0.1, 0.1)
            else:
                assert ratio < bound, mismatch(f"tangential B on steel at r={r_mm}", "reference only", ratio, 0.0, bound)


@pytest.mark.family("torque")
def test_sanity_image_series_converged():
    """Truncating the image series at 8 generations changes the torque by < 1e-10 relative
    compared with 12: each generation contributes about (r_inner / r_outer)^(N/2) = 0.055,
    and 0.055^8 is about 8e-11.

    Bound history: the first draft asserted 1e-12, which contradicts this estimate; the bound
    was corrected to match it. The reference output did not change (observed 3.3e-11)."""
    sec = SANITY_SECTION
    r1, r2 = sec.inner_back_mm, sec.outer_back_corner_mm()
    t8 = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2, 8))
    t12 = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2, 12))
    assert abs(t8 / t12 - 1) < 1e-10, mismatch("image series 8 vs 12 generations", "reference only", t8, t12, 1e-10)


@pytest.mark.family("torque")
def test_sanity_bem_matches_images_on_circular_steel():
    """Two independent steel methods agree: the BEM on 200-gons (hub inscribed in r1, cup
    circumscribed about r2) against the exact image series on the circles. The polygons hold
    slightly less steel; that error is O((pi/200)^2), about 2e-4, so the bound is 1e-3."""
    sec = SANITY_SECTION
    r1, r2 = sec.inner_back_mm, sec.outer_back_corner_mm()
    t_img = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2))
    t_bem = torque_per_length(sec, math.pi / 2, IronPolygons(r1 * math.cos(math.pi / 200), r2, 200, 4))
    assert abs(t_bem / t_img - 1) < 1e-3, mismatch("BEM 200-gon vs image circles", "reference only", t_bem, t_img, 1e-3)


@pytest.mark.family("torque")
def test_sanity_bem_antiperiodic_matches_full():
    """The one-pitch BEM (``antiperiod`` = N, used by the stress integral) and the full-boundary BEM
    give the same gap field, for 10-gon steel at the magnet backs and with bondlines and for 200-gon
    steel, at two angles: 1e-10 relative (the discrete systems are equivalent; only rounding differs)."""
    sec = SANITY_SECTION
    z = sec.stress_radius_mm() * MM * np.exp(2j * math.pi * (np.arange(64) + 0.3) / 64)
    for iron in (IronPolygons(10.15, SANITY_CUP_MM, 10), IronPolygons(10.10, 17.94, 10),
                 IronPolygons(10.15 * math.cos(math.pi / 200), sec.outer_back_corner_mm(), 200, 4)):
        for delta in (0.06, 0.22):
            mags = sec.inner() + sec.outer(delta)
            full = b_from_charges(z, polygon_iron_charges(mags, iron, delta))
            reduced = b_from_charges(z, polygon_iron_charges(mags, iron, delta, antiperiod=sec.n_poles))
            err = float(np.max(np.abs(reduced - full)) / np.max(np.abs(full)))
            assert err < 1e-10, mismatch(f"one-pitch vs full BEM, {iron.n_sides}-gon, delta={delta}",
                                         "reference only", err, 0.0, 1e-10)


@pytest.mark.parametrize("hub_mm, cup_mm", [(10.15, 17.89), (10.10, 17.94)], ids=["at-magnet-backs", "with-bondlines"])
@pytest.mark.family("torque")
def test_sanity_bem_converged(hub_mm, cup_mm):
    """BEM discretisation error on the real flat-face steel, for both steel placements the engine
    checks use (steel touching the magnet backs, and 0.05 mm bondlines): 40 vs 80 panels per side
    agree to < 1e-3 relative, the error bound quoted for the steel reference."""
    sec = SANITY_SECTION
    t40 = torque_per_length(sec, math.pi / 2, IronPolygons(hub_mm, cup_mm, 10, 40))
    t80 = torque_per_length(sec, math.pi / 2, IronPolygons(hub_mm, cup_mm, 10, 80))
    assert abs(t40 / t80 - 1) < 1e-3, mismatch("BEM 40 vs 80 panels per side", "reference only", t40, t80, 1e-3)


# --------------------------------------------------------------------------- field2d: stress integral
@pytest.mark.family("torque")
def test_sanity_maxwell_stress_independent_of_radius():
    """The Maxwell stress is divergence-free in source-free air, so the torque is the same on
    any circle in the clear gap (free space, circular steel and polygonal steel, 1e-9)."""
    sec = SANITY_SECTION
    lo, hi = sec.inner_corner_mm(), sec.outer_face_mm
    for iron in (None, IronCircles(sec.inner_back_mm, sec.outer_back_corner_mm()),
                 IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 10)):
        ta = torque_per_length(sec, 1.1, iron, r_mm=lo + 0.25 * (hi - lo))
        tb = torque_per_length(sec, 1.1, iron, r_mm=lo + 0.75 * (hi - lo))
        assert abs(ta / tb - 1) < 1e-9, mismatch(f"Maxwell torque at two gap radii, {type(iron).__name__}",
                                                 "reference only", ta, tb, 1e-9)


@pytest.mark.family("torque")
def test_sanity_stress_quadrature_converged_at_small_gap():
    """At the 0.5 mm face gap the stress circle passes 0.064 mm from the inner block corners, where
    the field varies fastest. The midpoint rule is spectrally accurate for this periodic integrand,
    with an error of at most about exp(-2 pi d / h) (d the corner distance, h the sample spacing):
    that estimate is 1e-5 for 256 samples per pitch, so 256 against 2048 must agree to 1e-4."""
    sec = SMALL_GAP_SECTION
    mags = sec.inner() + sec.outer(math.pi / 2 / (sec.n_poles / 2))
    t256 = maxwell_torque_per_length(mags, sec.stress_radius_mm(), sec.n_poles, n_per_pitch=256)
    t2048 = maxwell_torque_per_length(mags, sec.stress_radius_mm(), sec.n_poles, n_per_pitch=2048)
    assert abs(t256 / t2048 - 1) < 1e-4, mismatch("stress quadrature 256 vs 2048 samples per pitch, 0.5 mm gap",
                                                  "reference only", t256, t2048, 1e-4)


# --------------------------------------------------------------------------- cylindrical harmonics and planar formulas
@pytest.mark.family("torque")
def test_sanity_square_wave_harmonic_matches_numeric_fourier():
    """square_wave_harmonic equals the Fourier coefficient of the alternating pulse train computed
    by quadrature over one pole pair (+B for |x| < fill tau/2, -B for |x - tau| < fill tau/2):
    a_n = (1/tau) * integral of f(x) cos(n pi x / tau) dx, for n = 1..6 (even n vanish), 1e-12."""
    tau, b = 1.0, 1.3
    for fill in (0.3, 0.72, 1.0):
        h = fill * tau / 2
        for n in range(1, 7):
            c = lambda x: math.cos(n * math.pi * x / tau)
            numeric = b / tau * (quad(c, -h, h, epsabs=1e-14)[0] - quad(c, tau - h, tau + h, epsabs=1e-14)[0])
            closed = square_wave_harmonic(b, n, fill)
            assert abs(closed - numeric) <= 1e-12, mismatch(f"square-wave harmonic n={n}, fill={fill}",
                                                            "reference only", closed, numeric, 1e-12)
    with pytest.raises(ValueError, match="must be >= 1"):
        square_wave_harmonic(1.0, 0, 0.5)


@pytest.mark.family("torque")
def test_sanity_planar_s_factor_thick_magnet_limit():
    """Thick magnets make the backing irrelevant: for k t -> infinity both planar factors tend to
    e^-k g / 2 (one semi-infinite array faces another). At k t = 20 the remaining difference is
    of order e^-2kt, so 1e-8 relative."""
    k, g = 357.0, 1.4e-3
    t = 20 / k
    limit = math.exp(-k * g) / 2
    for backed in (False, True):
        s = planar_s_factor(k, t, t, g, backed)
        assert abs(s / limit - 1) < 1e-8, mismatch(f"thick-magnet limit, backed={backed}", "reference only", s, limit, 1e-8)


@pytest.mark.family("torque")
def test_sanity_cylindrical_s_factor_planar_limit():
    """As R_gap grows at fixed k (m = k R_gap) the exact cylindrical factor converges to the planar
    ones in ``planar_s_factor`` (free space incl. the /2, and the steel sinh form). With t_i != t_o
    the residual is a first-order curvature term: it halves when R_gap doubles (ratio 2 +/- 1 %)
    and stays below dr / R_gap, dr = t_i + g + t_o. This one test validates both references.

    Bound history: the first draft asserted a fixed residual bound whose error order was wrong
    (it assumed second-order convergence). It now asserts the first-order rate, which is
    stronger; error x R_gap is constant from 5 m to 80 m. The references did not change."""
    k, ti, to, g = 357.0, 3.17e-3, 2.5e-3, 1.4e-3
    planar = {"free": planar_s_factor(k, ti, to, g, backed=False), "steel": planar_s_factor(k, ti, to, g, backed=True)}

    def residuals(m: int) -> dict:
        rg = m / k
        radii = (rg - g / 2 - ti, rg - g / 2, rg + g / 2, rg + g / 2 + to)
        return {"free": cylindrical_s_factor(m, *radii, rg) / planar["free"] - 1,
                "steel": cylindrical_s_factor(m, *radii, rg, iron=(radii[0], radii[3])) / planar["steel"] - 1}

    e20, e40 = residuals(7140), residuals(14280)           # R_gap = 20 m and 40 m
    bound = (ti + g + to) / (14280 / k)
    for name in planar:
        assert abs(e20[name] / e40[name] - 2) < 0.02, mismatch(
            f"cylindrical S planar residual 20 m / 40 m, {name}", "reference only", e20[name] / e40[name], 2.0, 0.01)
        assert abs(e40[name]) < bound, mismatch(
            f"cylindrical S planar residual at 40 m, {name}", "reference only", e40[name], 0.0, bound)


@pytest.mark.parametrize("steel", [False, True], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_sanity_cylindrical_s_factor_matches_sliced_rings(steel):
    """Two independent solutions of the curved problem agree: the analytic cylindrical harmonic
    factor against Maxwell-stress torque on rings of K radially magnetized polygon slices
    (m = 5, unequal ring thicknesses so an inner/outer mix-up would show, steel as image
    circles at the ring backs). The slicing error is second order in 1/K, so going from
    K = 720 to 360 must quadruple it (ratio 4 +/- 2 %), which shows the analytic value is the
    exact limit; at K = 720 the error must also be below 1e-3.

    Bound history: the first draft asserted only a fixed bound tighter than the K = 720 slicing
    error. It now asserts the second-order rate, which is stronger; error x K^2 is -92.1 from
    K = 180 to 1440. The references did not change."""
    m_ord = 5
    r1a, r1b, r2a, r2b, rg = 10.15, 13.32, 14.72, 17.0, 14.02
    iron = IronCircles(r1a, r2b) if steel else None
    s = cylindrical_s_factor(m_ord, r1a, r1b, r2a, r2b, rg, iron=(r1a, r2b) if steel else None)
    t_analytic = 1.0 / (2 * MU0_EXACT) * s * 2 * math.pi * (rg * MM) ** 2

    def error(slices: int) -> float:
        mags = (sliced_radial_ring(r1a, r1b, m_ord, 1.0, 0.0, slices)
                + sliced_radial_ring(r2a, r2b, m_ord, 1.0, math.pi / 2, slices))
        return maxwell_torque_per_length(mags, 0.5 * (r1b + r2a), 2 * m_ord, iron) / t_analytic - 1

    e360, e720 = error(360), error(720)
    assert abs(e720) < 1e-3, mismatch("sliced rings (K=720) vs cylindrical harmonic torque", "reference only",
                                      e720, 0.0, 1e-3)
    assert abs(e360 / e720 / 4 - 1) < 0.02, mismatch("sliced-ring error ratio K=360 / K=720", "reference only",
                                                     e360 / e720, 4.0, 0.02)


# --------------------------------------------------------------------------- error paths
@pytest.mark.family("torque")
def test_sanity_reference_refuses_invalid_geometry():
    """The reference refuses set-ups where its method is invalid instead of returning a number:
    a steel circle through the outer blocks' back faces (their corners at 18.17 mm lie beyond
    17.89 mm), polygon steel that cuts into a magnet (hub above the block backs, or cup inside
    the outer backs), steel without the pole symmetry, a one-pitch BEM whose anti-period is odd
    or does not divide the steel sides, a stress circle outside the clear gap, inner corners that
    reach the outer faces (0.3 mm face gap), and cylindrical factors with m < 2, unordered radii or
    steel inside a magnet ring."""
    sec = SANITY_SECTION
    with pytest.raises(ValueError, match="outside the air between the steel circles"):
        torque_per_length(sec, math.pi / 2, IronCircles(sec.inner_back_mm, SANITY_CUP_MM))
    with pytest.raises(ValueError, match="enters the steel polygons"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm + 0.05, SANITY_CUP_MM, 10))
    with pytest.raises(ValueError, match="enters the steel polygons"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm, SANITY_CUP_MM - 0.05, 10))
    with pytest.raises(ValueError, match="symmetry"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 15))
    for bad in (4, 5):
        with pytest.raises(ValueError, match="anti-period"):
            polygon_iron_charges(sec.inner() + sec.outer(0.0), IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 10),
                                 0.0, antiperiod=bad)
    with pytest.raises(ValueError, match="clear gap"):
        torque_per_length(sec, math.pi / 2, None, r_mm=13.5)
    with pytest.raises(ValueError, match="no clear gap"):
        CouplingSection(10, 10.15, 3.17, 6.35, 13.62, 3.17, 6.35, 1.29, 1.29).stress_radius_mm()
    with pytest.raises(ValueError, match="m must be >= 2"):
        cylindrical_s_factor(1, 10.0, 13.0, 14.0, 17.0, 13.5)
    with pytest.raises(ValueError, match="r1a < r1b < r2a < r2b"):
        cylindrical_s_factor(5, 10.0, 14.5, 14.0, 17.0, 13.5)
    with pytest.raises(ValueError, match="outside the magnet annuli"):
        cylindrical_s_factor(5, 10.0, 13.0, 14.0, 17.0, 13.5, iron=(10.5, 17.0))
