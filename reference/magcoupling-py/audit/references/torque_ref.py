"""Independent torque references for the magcoupling torque model (M1, Task 3).

Nothing here imports or calls ``magcoupling``. Two references:

1. **3D magnetostatics** (``torque_on_blocks``, ``ring_torque_3d``, ``ring_pullout_3d``,
   ``torque_per_length_3d``, ``end_factor_3d``). Each block is a rigid, uniformly magnetized
   cuboid with relative permeability 1. That is magpylib's model, and the engine's harmonic
   model assumes the same. The source ring's field comes from magpylib's closed-form cuboid
   solution. The force on a target block uses the magnetic surface-charge model: a uniformly
   magnetized body carries charge sigma = M.n = (J.n)/mu0 on its surfaces and no volume
   charge, so in the external field B_ext it feels

       F = sum_faces sigma * integral(B_ext dA),   T = sum_faces sigma * integral(r x B_ext dA).

   This is exact for rigid uniform magnets. With M uniform and B_ext curl-free and
   divergence-free inside the target, the divergence theorem turns the dipole-density force
   integral(M.grad)B_ext dV and torque integral(M x B_ext + r x (M.grad)B_ext) dV into these
   surface integrals. Only the two faces normal to the magnetization carry charge. The face
   integrals use composite Gauss-Legendre quadrature; the integrand is smooth because the
   target faces never touch the sources.

2. **Prototype calibration** (``prototype_geometry``, ``prototype_calibration``). This
   re-derives the Calibration sheet's one-point correction from the prototype description:
   two equal rings of flat blocks on a polygonal hub, in free space, with a bench pull-out
   torque. The planar algebra comes from ``audit.references.planar``.

INDEPENDENCE NOTE: ``ring_pair_from_engine`` reads the ring geometry from the engine's
Calculator geometry fields (C18-C20, C28-C30, C54, C56, C69, C70). The 3D checks built on it
therefore inherit any error in those cells. That is acceptable because Task 4
(test_geometry_mass.py) verifies the faceted geometry cells independently. The prototype
rings (``ring_pair_from_prototype``) are built from the Calibration inputs alone.

Units: SI (m, T, N, N*m) in the 3D code. The calibration helpers take the engine's mm inputs
and convert them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import magpylib as magpy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from audit.common import MU0_EXACT
from audit.references import planar

MM = 1e-3
FieldFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- 3D: blocks and charge-model force
@dataclass(frozen=True)
class Block:
    """Uniformly magnetized cuboid centred in the z = 0 plane, polarized along its local x axis.

    ``phi`` rotates the block about +z, so local x maps to (cos phi, sin phi, 0).
    ``a_m`` is the size along local x (the magnetized direction), ``b_m`` along local y and
    ``c_m`` along z. ``j_T`` is the signed polarization J = mu0 M along local x.
    """
    x_m: float
    y_m: float
    phi: float
    a_m: float
    b_m: float
    c_m: float
    j_T: float

    def to_magpylib(self) -> magpy.magnet.Cuboid:
        return magpy.magnet.Cuboid(dimension=(self.a_m, self.b_m, self.c_m), polarization=(self.j_T, 0.0, 0.0),
                                   position=(self.x_m, self.y_m, 0.0), orientation=Rotation.from_euler("z", self.phi))


def magpylib_field(sources: Iterable[Block]) -> FieldFn:
    """B field [T] of ``sources`` at an (n, 3) array of points [m], from magpylib's analytical cuboid solution."""
    mags = [b.to_magpylib() for b in sources]

    def field(points: np.ndarray) -> np.ndarray:
        return np.asarray(magpy.getB(mags, points, sumup=True), dtype=float).reshape(-1, 3)

    return field


def _composite_gauss(length: float, panel: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of composite Gauss-Legendre quadrature on [-length/2, length/2]."""
    n_panels = max(1, math.ceil(length / panel - 1e-9))
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-length / 2, length / 2, n_panels + 1)
    mid, half = (edges[:-1] + edges[1:]) / 2, (edges[1:] - edges[:-1]) / 2
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def torque_on_blocks(targets: Iterable[Block], field: FieldFn, panel_m: float = 1.0 * MM,
                     order: int = 4, mu0: float = MU0_EXACT) -> tuple[np.ndarray, np.ndarray]:
    """Net force [N] and torque about the origin [N*m] on ``targets`` in the external ``field``.

    Surface-charge model (see the module docstring): charge sigma = +-J/mu0 on the two faces
    normal to each block's magnetization. The faces are integrated with composite
    Gauss-Legendre quadrature, ``order`` points per panel, panels no longer than ``panel_m``.
    """
    pts, dq = [], []                     # quadrature points and charge elements sigma*dA [A*m]
    for blk in targets:
        y, wy = _composite_gauss(blk.b_m, panel_m, order)
        z, wz = _composite_gauss(blk.c_m, panel_m, order)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        W = np.outer(wy, wz).ravel()
        c, s = math.cos(blk.phi), math.sin(blk.phi)
        for side in (1.0, -1.0):         # +x face carries +J/mu0, -x face carries -J/mu0
            lx = side * blk.a_m / 2
            gx = blk.x_m + c * lx - s * Y.ravel()
            gy = blk.y_m + s * lx + c * Y.ravel()
            pts.append(np.c_[gx, gy, Z.ravel()])
            dq.append(side * blk.j_T / mu0 * W)
    P, Q = np.vstack(pts), np.concatenate(dq)
    dF = Q[:, None] * field(P)
    return dF.sum(axis=0), np.cross(P, dF).sum(axis=0)


# --------------------------------------------------------------------------- 3D: coaxial rings of flat blocks
@dataclass(frozen=True)
class RingPair:
    """Two coaxial rings of flat blocks with alternating radial polarization (block i at 2*pi*i/N, sign (-1)^i).

    Radii are block-centre radii. The flat back face is at r - t/2 and the magnet face is at
    r + t/2 for the inner ring, r - t/2 for the outer ring. Both rings are centred on z = 0.
    """
    npole: int
    r_i_m: float
    t_i_m: float
    w_i_m: float
    l_i_m: float
    j_i_T: float
    r_o_m: float
    t_o_m: float
    w_o_m: float
    l_o_m: float
    j_o_T: float

    @property
    def half_pitch(self) -> float:
        """Half a pole pitch [rad]: where the engine evaluates pull-out (sin(n pi/2) in every tau_n)."""
        return math.pi / self.npole

    def with_length(self, length_m: float) -> "RingPair":
        return replace(self, l_i_m=length_m, l_o_m=length_m)


def ring_pair_from_engine(inp, res) -> RingPair:
    """Build the 3D ring pair exactly as the engine's geometry fields describe it.

    Inner block: radial span [inner_face_radius - t_i, inner_face_radius] (Calculator!C54, C20).
    Outer block: radial span [outer_face_apothem, outer_face_apothem + t_o] (Calculator!C56, C30).
    Width, axial length and Br at the operating temperature come from Calculator!C18-C20,
    C28-C30, C69 and C70. See the module's independence note.
    """
    if inp.coupling.faceted != 1:
        raise ValueError("the 3D reference models flat blocks only (coupling.faceted = 1)")
    n = inp.coupling.npole
    if n < 2 or n % 2:
        raise ValueError(f"alternating rings need an even pole count >= 2, got {n}")
    m = res.model
    return RingPair(npole=n,
                    r_i_m=(m.inner_face_radius_mm - m.inner_thickness_mm / 2) * MM, t_i_m=m.inner_thickness_mm * MM,
                    w_i_m=m.inner_width_mm * MM, l_i_m=m.inner_length_mm * MM, j_i_T=m.br_inner_T_op,
                    r_o_m=(m.outer_face_apothem_mm + m.outer_thickness_mm / 2) * MM, t_o_m=m.outer_thickness_mm * MM,
                    w_o_m=m.outer_width_mm * MM, l_o_m=m.outer_length_mm * MM, j_o_T=m.br_outer_T_op)


def ring_blocks(npole: int, r_m: float, t_m: float, w_m: float, l_m: float, j_T: float, phase: float) -> list[Block]:
    """One ring: N flat blocks, block i centred at angle phase + 2*pi*i/N, polarization (-1)^i * j_T radially."""
    out = []
    for i in range(npole):
        ang = phase + 2 * math.pi * i / npole
        out.append(Block(r_m * math.cos(ang), r_m * math.sin(ang), ang, t_m, w_m, l_m, j_T * (-1) ** i))
    return out


def ring_torque_3d(pair: RingPair, theta: float, panel_m: float = 1.0 * MM, order: int = 4,
                   all_blocks: bool = False) -> float:
    """Restoring torque [N*m] on the outer ring when it is rotated by ``theta`` [rad] from the aligned position.

    The value is positive for 0 < theta < 2*pi/N, where the torque pulls the outer ring back
    to theta = 0. With ``all_blocks=False`` only outer block 0 is integrated and the result is
    multiplied by N. Rotating the whole assembly by 2*pi/N maps inner block i to i+1 and outer
    block i to i+1 with both polarities flipped, so every outer block feels the same torque.
    The sanity tests check this against ``all_blocks=True``.
    """
    inner = ring_blocks(pair.npole, pair.r_i_m, pair.t_i_m, pair.w_i_m, pair.l_i_m, pair.j_i_T, 0.0)
    outer = ring_blocks(pair.npole, pair.r_o_m, pair.t_o_m, pair.w_o_m, pair.l_o_m, pair.j_o_T, theta)
    targets = outer if all_blocks else outer[:1]
    _, t = torque_on_blocks(targets, magpylib_field(inner), panel_m, order)
    return -float(t[2]) * (1 if all_blocks else pair.npole)


def ring_pullout_3d(pair: RingPair, panel_m: float = 1.0 * MM, order: int = 4, n_scan: int = 9) -> tuple[float, float]:
    """Pull-out torque [N*m] and the angle where it occurs [rad]: the max of ``ring_torque_3d`` over one pole pitch.

    A coarse scan of ``n_scan`` interior angles is followed by a bounded Brent refinement to
    1e-4 of a pole pitch.
    """
    pitch = 2 * math.pi / pair.npole
    grid = np.linspace(0.0, pitch, n_scan + 2)
    vals = [ring_torque_3d(pair, th, panel_m, order) for th in grid[1:-1]]
    i = int(np.argmax(vals)) + 1
    opt = minimize_scalar(lambda th: -ring_torque_3d(pair, th, panel_m, order), bounds=(grid[i - 1], grid[i + 1]),
                          method="bounded", options={"xatol": 1e-4 * pitch})
    return -float(opt.fun), float(opt.x)


def torque_per_length_3d(pair: RingPair, l1_m: float, l2_m: float, theta: float, panel_m: float = 1.0 * MM,
                         order: int = 4) -> float:
    """Long-length (2D) torque per unit length [N*m/m] of the ring pair's cross-section at angle ``theta``.

    For lengths well beyond the end-fringing zone the torque is T(L) = a*L - b, where b is the
    constant end deficit. The slope (T(l2) - T(l1)) / (l2 - l1) is therefore the 2D value a,
    with the end deficit removed exactly.
    """
    t1 = ring_torque_3d(pair.with_length(l1_m), theta, panel_m, order)
    t2 = ring_torque_3d(pair.with_length(l2_m), theta, panel_m, order)
    return (t2 - t1) / (l2_m - l1_m)


def end_factor_3d(pair: RingPair, length_m: float, per_length: float, theta: float, panel_m: float = 1.0 * MM,
                  order: int = 4) -> float:
    """3D end-effect factor at ``length_m``: torque at that length / (2D torque per length * length), same angle."""
    return ring_torque_3d(pair.with_length(length_m), theta, panel_m, order) / (per_length * length_m)


# --------------------------------------------------------------------------- prototype calibration
@dataclass(frozen=True)
class PrototypeGeometry:
    """Prototype ring geometry [mm] derived from the Calibration sheet's description."""
    poles: float
    face_radius_mm: float        # inner block magnet face (flat centre)
    corner_radius_mm: float      # inner block outer corner
    corner_gap_mm: float         # inner block corner to the outer block face
    flat_gap_mm: float           # flat centre of the inner face to the flat centre of the outer face
    outer_face_apothem_mm: float
    gap_radius_mm: float
    pole_pitch_mm: float
    fill_inner: float            # block width / pole pitch at the inner block's mid-thickness radius
    fill_outer: float            # the same at the outer block's mid-thickness radius


def prototype_geometry(c) -> PrototypeGeometry:
    """Geometry from a CalibrationInputs-like object (two equal rings of identical flat blocks).

    The inner block's corner sits at hypot(face radius, width/2). The reported spacing is
    measured either at the flat centres (gap_definition 1) or at the inner corners (0). The
    flat-centre gap follows by adding the corner overhang.
    """
    poles = c.total_magnets / 2
    r_face = c.apothem_mm + c.magnet_thickness_mm
    r_corner = math.hypot(r_face, c.magnet_width_mm / 2)
    g = c.spacing_mm if c.gap_definition == 1 else c.spacing_mm + (r_corner - r_face)
    a_o = r_face + g
    r_g = r_face + g / 2
    fill_i = min(1.0, c.magnet_width_mm * poles / (2 * math.pi * (c.apothem_mm + c.magnet_thickness_mm / 2)))
    fill_o = min(1.0, c.magnet_width_mm * poles / (2 * math.pi * (a_o + c.magnet_thickness_mm / 2)))
    return PrototypeGeometry(poles, r_face, r_corner, g - (r_corner - r_face), g, a_o, r_g, 2 * math.pi * r_g / poles,
                             fill_i, fill_o)


def prototype_br(c) -> float:
    """Prototype remanence [T] at the assumed test temperature (Calibration!C16, C21, C22)."""
    return planar.br_at(c.br_T, c.alpha_br_per_C, c.test_temp_C)


def prototype_calibration(c, harmonics: Iterable[int] = (1, 3, 5)) -> dict:
    """One-point correction from a CalibrationInputs-like object.

    Model: the free-space planar shear stress at a displacement of half a pole pitch, times the
    gap area and lever arm (2 pi R_g^2 L), the end factor 1 - c_end * pitch / L, and the
    original calibration factor. Correction: measured / model, applied on top of the original
    factor. Returns the intermediate values, keyed by the engine's result-field names where one
    exists (``tau_n_Pa`` maps each harmonic n to its shear stress).
    """
    geo = prototype_geometry(c)
    br = prototype_br(c)
    t_m, pitch_m = c.magnet_thickness_mm * MM, geo.pole_pitch_mm * MM
    tau_n = {n: planar.planar_shear_stress(br, br, geo.fill_inner, geo.fill_outer, pitch_m, t_m, t_m,
                                           geo.flat_gap_mm * MM, pitch_m / 2, c.mu0, False, (n,))
             for n in harmonics}
    tau = sum(tau_n.values())
    t2d = planar.torque_on_cylinder(tau, geo.gap_radius_mm * MM, c.magnet_length_mm * MM)
    f_end = planar.end_factor(c.c_end, geo.pole_pitch_mm, c.magnet_length_mm)
    model = t2d * f_end * c.f_cal_original
    ratio = c.measured_torque_Nm / model
    return {"geometry": geo, "br_test_T": br, "tau_n_Pa": tau_n, "tau_Pa": tau, "torque_2d_Nm": t2d, "f_end": f_end,
            "model_torque_Nm": model, "model_error": model / c.measured_torque_Nm - 1,
            "measured_over_model": ratio, "f_cal_updated": c.f_cal_original * ratio}


def ring_pair_from_prototype(c) -> RingPair:
    """3D ring pair of the free-space prototype described by a CalibrationInputs-like object."""
    geo = prototype_geometry(c)
    if geo.poles != int(geo.poles) or int(geo.poles) % 2:
        raise ValueError(f"prototype needs an even whole number of poles per ring, got {geo.poles}")
    br = prototype_br(c)
    t, w, length = c.magnet_thickness_mm * MM, c.magnet_width_mm * MM, c.magnet_length_mm * MM
    return RingPair(npole=int(geo.poles), r_i_m=(c.apothem_mm + c.magnet_thickness_mm / 2) * MM, t_i_m=t, w_i_m=w,
                    l_i_m=length, j_i_T=br, r_o_m=(geo.outer_face_apothem_mm + c.magnet_thickness_mm / 2) * MM,
                    t_o_m=t, w_o_m=w, l_o_m=length, j_o_T=br)
