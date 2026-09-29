"""Independent 2D magnetostatic reference for the coupling cross-section (M1, Task 2).

Nothing here uses the engine (only MU0_EXACT comes from audit.common). The coupling is
treated as infinitely long (2D).

Numerical-exact flat-block model (``b_field``, ``torque_per_length``, ``pullout``):
  * Each block is a uniformly magnetized rectangle, represented by its bound
    surface current mu0 K = (J x n)_z on the edges. The field of a straight sheet
    is integrated in closed form, so the free-space field is exact to rounding.
  * Steel, option 1, ``IronCircles``: infinitely permeable circular surfaces by
    the method of images (a line current I at z has the image I, same sign, at
    R^2 / conj z), a converged series of reflections between the two circles.
  * Steel, option 2, ``IronPolygons``: infinitely permeable polygonal surfaces
    (the real flat hub faces and cup pockets) by a boundary-element method: the
    magnetic scalar potential is constant on each steel body and each body's net
    induced charge is zero. Needed because flat blocks cannot touch a circle.
  * Torque per unit length from the Maxwell stress on a circle in the air gap.

Exact cylindrical harmonic solution (``cylindrical_s_factor``): sinusoidal RADIAL
magnetization M cos(m theta) in two concentric annuli, free space or between
infinitely permeable circles; torque from the force on the outer ring's magnetic
charges. This is the curved counterpart of the engine's planar S_n factors.

Conventions: SI inside; a point is a complex number z = x + iy; a field is the
complex number B_x + i B_y. ``CouplingSection``, ``IronCircles`` and
``IronPolygons`` take lengths in mm, like the engine's fields.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from audit.common import MU0_EXACT

MM = 1e-3


# --------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Magnet:
    """Uniformly magnetized infinite prism: counter-clockwise polygon vertices
    (complex, metres) and polarization J = mu0 * M (complex, tesla)."""
    vertices: tuple
    J: complex


@dataclass(frozen=True)
class Sheets:
    """Straight sheets z1 -> z2 with a uniform strength (tesla): mu0 K for currents
    (K along +z), mu0 sigma for magnetic charge."""
    z1: np.ndarray
    z2: np.ndarray
    k: np.ndarray


def _edge_sheets(magnets: list[Magnet], part: str) -> Sheets:
    """For each edge, with outward normal n: conj(J) n = J.n + i (J x n)_z. The real
    part is the surface charge mu0 sigma = J.n, the imaginary part the bound current."""
    z1, z2, k = [], [], []
    for mag in magnets:
        v = mag.vertices
        for a, b in zip(v, v[1:] + v[:1]):
            n = -1j * (b - a) / abs(b - a)          # outward normal of a CCW polygon edge
            c = np.conj(mag.J) * n
            s = c.real if part == "charge" else c.imag
            if abs(s) > 1e-12 * abs(mag.J):
                z1.append(a), z2.append(b), k.append(s)
    return Sheets(np.array(z1, dtype=complex), np.array(z2, dtype=complex), np.array(k, dtype=float))


def current_sheets(magnets: list[Magnet]) -> Sheets:
    return _edge_sheets(magnets, "current")


def charge_sheets(magnets: list[Magnet]) -> Sheets:
    return _edge_sheets(magnets, "charge")


def _kernel(z: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Integral over the straight segment z1 -> z2 of ds / (z - zeta), shape (points, sheets):
    conj(d) Log((z - z1)/(z - z2)), d the unit direction. z - zeta runs along a straight
    line that misses 0 when z is off the segment, so the principal Log is exact."""
    zz = z[:, None]
    d = (z2 - z1) / np.abs(z2 - z1)
    return np.conj(d)[None, :] * np.log((zz - z1[None, :]) / (zz - z2[None, :]))


def b_from_currents(z: np.ndarray, sh: Sheets, scale: float = 1.0) -> np.ndarray:
    """B_x + i B_y of current sheets, optionally scaled about the origin by ``scale``.
    A line current I at z' gives conj(B) = -i mu0 I / (2 pi (z - z')). Scaling keeps each
    element's current, so the scaled sheet carries density K / scale."""
    cb = -1j / (2 * math.pi) * (_kernel(z, sh.z1 * scale, sh.z2 * scale) @ (sh.k / scale))
    return np.conj(cb)


def b_from_inverted_currents(z: np.ndarray, sh: Sheets, rho: float) -> np.ndarray:
    """B_x + i B_y of the inversion images (radius rho, same sign) of current sheets.
    Element K dt at z'(t) = z1 + t d becomes a line current K dt at w = rho^2 / conj z'(t).
    With e = conj d and u(t) = z conj z'(t) - rho^2 (linear in t):
    integral over [0, l] of dt / (z - w) = l / z + rho^2 / (z^2 e) Log(u(l) / u(0))."""
    zz = z[:, None]
    length = np.abs(sh.z2 - sh.z1)
    e = np.conj((sh.z2 - sh.z1) / length)
    integral = (length[None, :] / zz + rho ** 2 / (zz ** 2 * e[None, :])
                * np.log((zz * np.conj(sh.z2)[None, :] - rho ** 2) / (zz * np.conj(sh.z1)[None, :] - rho ** 2)))
    return np.conj(-1j / (2 * math.pi) * (integral @ sh.k))


def b_from_charges(z: np.ndarray, sh: Sheets) -> np.ndarray:
    """B_x + i B_y (= mu0 H, valid in air) of charge sheets: conj(B) = mu0 sigma / (2 pi) * kernel."""
    return np.conj(_kernel(z, sh.z1, sh.z2) @ sh.k / (2 * math.pi))


def _log_potential(z: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Integral over the segment of ln|z - zeta| ds, shape (points, sheets). In segment
    coordinates (u along, v across, segment on 0 <= s <= l):
    F(s) = x ln sqrt(x^2 + v^2) - x + v atan(x / v), x = s - u; result F(l) - F(0)."""
    length = np.abs(z2 - z1)
    d = (z2 - z1) / length
    w = (z[:, None] - z1[None, :]) / d[None, :]
    u, v = w.real, w.imag

    def f(s):
        x = s - u
        r2 = x * x + v * v
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(r2 > 0, 0.5 * x * np.log(np.where(r2 > 0, r2, 1.0)), 0.0)
            t3 = np.where(v != 0, v * np.arctan(x / np.where(v != 0, v, 1.0)), 0.0)
        return t1 - x + t3

    return f(length[None, :]) - f(0.0)


# --------------------------------------------------------------------------- steel
@dataclass(frozen=True)
class IronCircles:
    """Infinitely permeable steel filling r <= r_inner_mm (hub) and r >= r_outer_mm (cup).

    The image series reflects the sources alternately in the two circles,
    ``generations`` reflections per chain. Two generations move the images by
    q = (r_outer / r_inner)^2 in radius; an alternating ring of N poles has only angular
    harmonics of order >= N/2, so each generation lowers the omitted field by about
    (r_inner / r_outer)^(N/2) (0.055 for the default design; 8 generations leave
    < 1e-10, which ``test_sanity_image_series_converged`` confirms)."""
    r_inner_mm: float
    r_outer_mm: float
    generations: int = 8


def image_maps(iron: IronCircles) -> list[tuple[str, float]]:
    """Transformations taking the real sources to each image: ("scale", s) is z -> s z,
    ("invert", rho) is z -> rho^2 / conj z. Reflection in a circle of radius R maps
    ("scale", s) to ("invert", R / sqrt s) and ("invert", rho) to ("scale", (R / rho)^2)."""
    r1, r2 = iron.r_inner_mm * MM, iron.r_outer_mm * MM
    maps = []
    for first, second in ((r1, r2), (r2, r1)):
        kind, value = "scale", 1.0
        for g in range(iron.generations):
            radius = first if g % 2 == 0 else second
            kind, value = ("invert", radius / math.sqrt(value)) if kind == "scale" else ("scale", (radius / value) ** 2)
            maps.append((kind, value))
    return maps


@dataclass(frozen=True)
class IronPolygons:
    """Infinitely permeable steel inside a regular ``n_sides``-gon of apothem hub_apothem_mm
    (flats centred at angles 2 pi i / n_sides) and outside one of apothem cup_apothem_mm
    (flats centred at cup_phase + 2 pi i / n_sides, cup_phase passed to ``b_field``: the cup
    turns with the outer ring). Solved by collocation with piecewise-constant charge panels,
    graded towards the vertices; ``test_sanity_bem_converged`` bounds the discretisation error."""
    hub_apothem_mm: float
    cup_apothem_mm: float
    n_sides: int
    panels_per_side: int = 40


def _polygon_panels(apothem: float, n_sides: int, per_side: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    rv = apothem / math.cos(math.pi / n_sides)
    t = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, per_side + 1))
    a, b = [], []
    for i in range(n_sides):
        c = phase + 2 * math.pi * i / n_sides
        v0, v1 = rv * np.exp(1j * (c - math.pi / n_sides)), rv * np.exp(1j * (c + math.pi / n_sides))
        pts = v0 + (v1 - v0) * t
        a.append(pts[:-1]), b.append(pts[1:])
    return np.concatenate(a), np.concatenate(b)


def _polygon_support(z: np.ndarray, n_sides: int, phase: float) -> np.ndarray:
    """Largest projection of each point on the flat normals of a regular polygon (flats centred at
    phase + 2 pi i / n_sides): a point lies outside the polygon of apothem a when it exceeds a."""
    normals = np.exp(1j * (phase + 2 * math.pi * np.arange(n_sides) / n_sides))
    return np.max((z[:, None] * np.conj(normals)[None, :]).real, axis=1)


def polygon_iron_charges(magnets: list[Magnet], iron: IronPolygons, cup_phase: float,
                         antiperiod: int | None = None) -> Sheets:
    """Induced charge on the steel surfaces: the scalar potential psi (B = -grad psi) is a
    constant C_hub on the hub surface and C_cup on the cup surface, and each body's net
    charge is zero (div B = 0 inside the steel, H = 0 there). The charge model assumes the
    magnets sit in the air, so a magnet vertex inside the hub polygon or outside the cup polygon
    is refused (touching the steel, as a block on its flat does, is allowed).

    ``antiperiod`` = N (even) declares that turning the sources by 2 pi / N flips their sign, as in
    any coupling section with N alternating poles per ring. The steel is also N-fold symmetric, so
    the induced charge flips sign from one pole pitch to the next and both constants vanish (each
    equals its own negative). Then only the panels of the first pitch are unknowns, with the kernel
    summed over the N pitches with alternating signs: the same discrete solution (the panels of each
    pitch are exact rotated copies), N times fewer collocation points and an N^2 smaller system.
    ``test_sanity_bem_antiperiodic_matches_full`` checks the two solutions agree. None solves the
    full boundary, valid for any sources."""
    verts = np.array([v for mag in magnets for v in mag.vertices], dtype=complex)
    hub, cup = iron.hub_apothem_mm * MM, iron.cup_apothem_mm * MM
    if (_polygon_support(verts, iron.n_sides, 0.0).min() < hub * (1 - 1e-9)
            or _polygon_support(verts, iron.n_sides, cup_phase).max() > cup * (1 + 1e-9)):
        raise ValueError(f"a magnet enters the steel polygons (hub apothem {iron.hub_apothem_mm} mm, "
                         f"cup apothem {iron.cup_apothem_mm} mm)")
    h1, h2 = _polygon_panels(hub, iron.n_sides, iron.panels_per_side, 0.0)
    c1, c2 = _polygon_panels(cup, iron.n_sides, iron.panels_per_side, cup_phase)
    z1, z2 = np.concatenate([h1, c1]), np.concatenate([h2, c2])
    p, nh = len(z1), len(h1)
    mid, length = (z1 + z2) / 2, np.abs(z2 - z1)
    src = charge_sheets(magnets)
    if antiperiod is None:
        a = np.zeros((p + 2, p + 2))
        a[:p, :p] = -_log_potential(mid, z1, z2) / (2 * math.pi)
        a[:nh, p] = -1.0
        a[nh:p, p + 1] = -1.0
        a[p, :nh], a[p + 1, nh:p] = length[:nh], length[nh:]
        rhs = np.zeros(p + 2)
        rhs[:p] = _log_potential(mid, src.z1, src.z2) @ src.k / (2 * math.pi)
        return Sheets(z1, z2, np.linalg.solve(a, rhs)[:p])
    if antiperiod % 2 or iron.n_sides % antiperiod:
        raise ValueError(f"anti-period {antiperiod} needs an even pole count dividing the {iron.n_sides} steel sides")
    per = nh // antiperiod                                          # panels per pitch on each body
    signs = (-1.0) ** np.arange(antiperiod)
    colloc = np.concatenate([mid[:per], mid[nh:nh + per]])
    g = -_log_potential(colloc, z1, z2) / (2 * math.pi)             # (2 per, p): pitch-major panel order

    def fold(block: np.ndarray) -> np.ndarray:                      # sum the N pitches with signs +, -, +, ...
        return (block.reshape(2 * per, antiperiod, per) * signs[None, :, None]).sum(axis=1)

    a = np.concatenate([fold(g[:, :nh]), fold(g[:, nh:])], axis=1)
    rhs = _log_potential(colloc, src.z1, src.z2) @ src.k / (2 * math.pi)
    sol = np.linalg.solve(a, rhs)
    hub_k, cup_k = (np.outer(signs, part).ravel() for part in (sol[:per], sol[per:]))   # back to all pitches
    return Sheets(z1, z2, np.concatenate([hub_k, cup_k]))


# --------------------------------------------------------------------------- field and torque
def b_field(z, magnets: list[Magnet], iron: IronCircles | IronPolygons | None = None,
            cup_phase: float = 0.0, antiperiod: int | None = None) -> np.ndarray:
    """B_x + i B_y [T] at points z (complex, metres), in air, for the magnets and optional steel.

    Each magnet's bound currents sum to zero, so the centre images an infinitely
    permeable circle also needs cancel and are omitted. Images are only valid for sources
    in the air between the circles, so a current outside that annulus is refused.
    ``antiperiod`` is passed to ``polygon_iron_charges`` (polygonal steel only)."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    sh = current_sheets(magnets)
    b = b_from_currents(z, sh)
    if isinstance(iron, IronCircles):
        radii = np.abs(np.concatenate([sh.z1, sh.z2]))
        if radii.min() < iron.r_inner_mm * MM * (1 - 1e-9) or radii.max() > iron.r_outer_mm * MM * (1 + 1e-9):
            raise ValueError(f"bound currents span r = {radii.min() / MM:.4f}..{radii.max() / MM:.4f} mm, outside the "
                             f"air between the steel circles ({iron.r_inner_mm}..{iron.r_outer_mm} mm)")
        for kind, value in image_maps(iron):
            b = b + (b_from_currents(z, sh, value) if kind == "scale" else b_from_inverted_currents(z, sh, value))
    elif isinstance(iron, IronPolygons):
        b = b + b_from_charges(z, polygon_iron_charges(magnets, iron, cup_phase, antiperiod))
    return b


def maxwell_torque_per_length(magnets: list[Magnet], r_mm: float, n_poles: int,
                              iron: IronCircles | IronPolygons | None = None, cup_phase: float = 0.0,
                              n_per_pitch: int = 256) -> float:
    """Torque per unit length [N·m/m] on everything inside the circle of radius r_mm (the
    inner rotor), from the Maxwell stress: T/L = (r^2 / mu0) * integral of B_r B_theta dtheta.

    The integrand repeats every pole pitch (turning the whole section by one pitch
    flips every source, so B -> -B), so one pitch is sampled with the midpoint rule,
    which is spectrally accurate for a smooth periodic integrand. That needs magnets that
    flip sign under a turn of 2 pi / n_poles and steel with n_poles-fold symmetry; the same
    property lets polygonal steel be solved over one pole pitch (``antiperiod``)."""
    if isinstance(iron, IronPolygons) and iron.n_sides % n_poles:
        raise ValueError(f"steel polygon with {iron.n_sides} sides lacks the {n_poles}-fold symmetry the stress integral uses")
    r = r_mm * MM
    theta = 2 * math.pi / n_poles * (np.arange(n_per_pitch) + 0.5) / n_per_pitch
    b = b_field(r * np.exp(1j * theta), magnets, iron, cup_phase, antiperiod=n_poles) * np.exp(-1j * theta)
    return float(2 * math.pi * r ** 2 / MU0_EXACT * np.mean(b.real * b.imag))


def rect_block(back_apothem_m: float, thickness_m: float, width_m: float, angle: float, br_signed: float) -> Magnet:
    """Flat block with its back face on the polygon face at ``angle``, magnetized along the
    face normal (outward for br_signed > 0)."""
    rot = complex(math.cos(angle), math.sin(angle))
    a, t, h = back_apothem_m, thickness_m, width_m / 2
    local = (complex(a, -h), complex(a + t, -h), complex(a + t, h), complex(a, h))
    return Magnet(tuple(p * rot for p in local), br_signed * rot)


@dataclass(frozen=True)
class CouplingSection:
    """2D section: n_poles flat blocks per ring on polygon faces, alternating magnetization,
    block 0 of each ring magnetized outward (aligned rings attract). Lengths in mm, Br in T."""
    n_poles: int
    inner_back_mm: float
    inner_thickness_mm: float
    inner_width_mm: float
    outer_face_mm: float
    outer_thickness_mm: float
    outer_width_mm: float
    br_inner_T: float
    br_outer_T: float

    def _ring(self, back_mm: float, thickness_mm: float, width_mm: float, br: float, phase: float) -> list[Magnet]:
        return [rect_block(back_mm * MM, thickness_mm * MM, width_mm * MM, phase + 2 * math.pi * i / self.n_poles,
                           br if i % 2 == 0 else -br) for i in range(self.n_poles)]

    def inner(self) -> list[Magnet]:
        return self._ring(self.inner_back_mm, self.inner_thickness_mm, self.inner_width_mm, self.br_inner_T, 0.0)

    def outer(self, delta_mech: float) -> list[Magnet]:
        return self._ring(self.outer_face_mm, self.outer_thickness_mm, self.outer_width_mm, self.br_outer_T, delta_mech)

    def inner_corner_mm(self) -> float:
        """Largest radius of the inner blocks (front corners)."""
        return math.hypot(self.inner_back_mm + self.inner_thickness_mm, self.inner_width_mm / 2)

    def outer_back_corner_mm(self) -> float:
        """Largest radius of the outer blocks (back corners)."""
        return math.hypot(self.outer_face_mm + self.outer_thickness_mm, self.outer_width_mm / 2)

    def stress_radius_mm(self) -> float:
        """Middle of the clear air annulus between the inner front corners and the outer faces."""
        if self.inner_corner_mm() >= self.outer_face_mm:
            raise ValueError("inner block corners reach the outer block faces: no clear gap for the stress circle")
        return 0.5 * (self.inner_corner_mm() + self.outer_face_mm)


def torque_per_length(sec: CouplingSection, theta_e: float, iron: IronCircles | IronPolygons | None = None,
                      r_mm: float | None = None) -> float:
    """Torque per unit length [N·m/m] on the inner rotor with the outer ring (and cup) turned
    by the electrical angle theta_e (mechanical theta_e / (N/2)). Positive = restoring."""
    r = sec.stress_radius_mm() if r_mm is None else r_mm
    if not sec.inner_corner_mm() < r < sec.outer_face_mm:
        raise ValueError(f"stress circle r = {r} mm is not in the clear gap "
                         f"({sec.inner_corner_mm():.4f}..{sec.outer_face_mm} mm)")
    delta = theta_e / (sec.n_poles / 2)
    return maxwell_torque_per_length(sec.inner() + sec.outer(delta), r, sec.n_poles, iron, cup_phase=delta)


@dataclass(frozen=True)
class Pullout:
    torque_per_m: float        # true pull-out: max over angle [N·m/m]
    theta_e_peak: float        # electrical angle of that max [rad]
    at_quarter: float          # torque at theta_e = pi/2, where the engine evaluates [N·m/m]
    harmonics: dict            # {n: a_n}, T(theta_e) = sum over n >= 1 of a_n sin(n theta_e) [N·m/m]


def pullout(sec: CouplingSection, iron: IronCircles | IronPolygons | None = None, n_samples: int = 48) -> Pullout:
    """Torque-angle curve over one pole pair, its sine harmonics, and the true pull-out.

    Every configuration here is mirror-symmetric, so T(-theta) = -T(theta) and the curve is
    a pure sine series; midpoint samples on (0, pi) give a_n for n < n_samples exactly
    (discrete sine transform), apart from aliasing of harmonics that fall off like
    exp(-n k1 g). Circles or free space leave only odd n; polygonal steel adds cogging
    between each ring and the other rotor's flats, which appears as even n. The max is
    refined with a bounded scalar search between the neighbours of the best sample."""
    th = math.pi * (np.arange(n_samples) + 0.5) / n_samples
    t = np.array([torque_per_length(sec, x, iron) for x in th])
    harm = {n: float(2 / n_samples * np.sum(t * np.sin(n * th))) for n in range(1, n_samples)}
    i = int(np.argmax(t))
    lo, hi = th[max(i - 1, 0)], th[min(i + 1, n_samples - 1)]
    best = minimize_scalar(lambda x: -torque_per_length(sec, x, iron), bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-7})
    return Pullout(float(-best.fun), float(best.x), torque_per_length(sec, math.pi / 2, iron), harm)


# --------------------------------------------------------------------------- cylindrical harmonics
def cylindrical_s_factor(m: int, r1a: float, r1b: float, r2a: float, r2b: float, r_gap: float,
                         iron: tuple[float, float] | None = None) -> float:
    """Exact 2D geometry factor for order-m sinusoidal RADIAL magnetization in the annuli
    r1a..r1b (inner ring) and r2a..r2b (outer ring), normalised like the engine:
    torque amplitude per length = B_i B_o / (2 mu0) * S * 2 pi r_gap^2. Radii in one unit.

    Free space (``iron=None``): a ring charge s cos(m theta) has potential
    (s / 2m) (r</r>)^m cos(m theta) (the free-space subdomain solution for radial
    magnetization, Zhu & Howe, IEEE Trans. Magn. 29(1), 1993). Integrating both rings'
    surface (M.n) and volume (-div M = -M cos(m theta) / r) charges gives, with x = r / r_gap,
        S = m^2 / (2 (m^2 - 1)) * (x1b^(m+1) - x1a^(m+1)) * (x2a^(1-m) - x2b^(1-m)).
    Steel (``iron=(rho1, rho2)``: infinitely permeable for r <= rho1 and r >= rho2): the
    potential is constant on both circles, the order-m Green's function is
    u1(r<) u2(r>) / W, u1 = (r/rho1)^m - (rho1/r)^m, u2 = (rho2/r)^m - (r/rho2)^m,
    W = 2m (Q - 1/Q), Q = (rho2/rho1)^m, and S = m p_i p_o / W with
    p = [r u(r)] across the ring minus the integral of u across the ring."""
    if m < 2:
        raise ValueError("order m must be >= 2 (m = 1 needs the logarithmic solution)")
    x1a, x1b, x2a, x2b = (r / r_gap for r in (r1a, r1b, r2a, r2b))
    if not x1a < x1b < x2a < x2b:
        raise ValueError("need r1a < r1b < r2a < r2b")
    if iron is None:
        return m * m / (2 * (m * m - 1)) * (x1b ** (m + 1) - x1a ** (m + 1)) * (x2a ** (1 - m) - x2b ** (1 - m))
    rho1, rho2 = iron[0] / r_gap, iron[1] / r_gap
    if rho1 > x1a * (1 + 1e-12) or rho2 < x2b * (1 - 1e-12):
        raise ValueError("steel must lie outside the magnet annuli: rho1 <= r1a and rho2 >= r2b")

    def u1(s):
        return (s / rho1) ** m - (rho1 / s) ** m

    def u1_int(s):
        return s ** (m + 1) / ((m + 1) * rho1 ** m) + rho1 ** m * s ** (1 - m) / (m - 1)

    def u2(r):
        return (rho2 / r) ** m - (r / rho2) ** m

    def u2_int(r):
        return rho2 ** m * r ** (1 - m) / (1 - m) - r ** (m + 1) / ((m + 1) * rho2 ** m)

    p_i = x1b * u1(x1b) - x1a * u1(x1a) - (u1_int(x1b) - u1_int(x1a))
    p_o = x2b * u2(x2b) - x2a * u2(x2a) - (u2_int(x2b) - u2_int(x2a))
    q = (rho2 / rho1) ** m
    return abs(m * p_i * p_o / (2 * m * (q - 1 / q)))
