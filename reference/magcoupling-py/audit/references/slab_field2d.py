"""2D magnetostatic reference: alternating magnet layers between two infinitely permeable plates.

Shared slab model of the audit (created in Task 4; Task 5 appends its force and energy functions).
Written from scratch; nothing here imports ``magcoupling``. The square-wave harmonic amplitudes come from
``audit.references.planar.square_wave_harmonic``, the audit's single implementation.

Geometry (unrolled coupling, x tangential, y radial, units mm and T)
    Steel plates (mu -> infinity) at y = 0 and y = h. Each magnet layer occupies
    y in [y_bottom, y_bottom + thickness]; its blocks have width fill * pitch, alternate in
    polarity every pole pitch, and the block centred at x = shift is magnetized +y (Br > 0).
    Magnets have recoil permeability 1, so the whole slab is one linear medium.

Method (magnetic scalar potential with surface charges)
    A layer magnetized M_y(x) carries surface charge sigma = -M_y on its bottom face and
    +M_y on its top face. With mu0*sigma_n cos(k (x - s)) on a sheet at y = ys, the potential
    that vanishes on both plates (H tangential = 0 at mu -> infinity) is
        mu0 phi = a_n cos(k(x - s)) sinh(k y<) sinh(k (h - y>)) / (k sinh(k h)),
    where y< = min(y, ys), y> = max(y, ys), k = n pi / pitch and, for the alternating square wave,
    a_n = Br * 4/(n pi) * sin(n pi fill / 2) (odd n). B = -mu0 grad(phi) in air.

Flux into the plates (``iron_face_coefficients``)
    On a plate face B = mu0 (H + M). The H part is the limit of the sheet potentials at the face;
    a sheet lying on the face itself contributes nothing there (its potential vanishes identically,
    because sinh(k * 0) = 0), and the M part is the square-wave magnetization of a layer that
    touches the plate. Evaluating ``harmonic_coefficients`` does not give this: it is defined for
    air points off the magnet faces, and inside a magnet it returns mu0 H, not B.

This is the standard slotless-machine / linear-coupling field solution; see
Furlani, *Permanent Magnet and Electromechanical Devices* (2001), ch. 4 and 8, and
Zhu & Howe, IEEE Trans. Magn. 29 (1993) 124-135 for the same separation-of-variables approach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audit.references.planar import square_wave_harmonic


@dataclass(frozen=True)
class MagnetLayer:
    y_bottom_mm: float
    thickness_mm: float
    br_T: float          # signed: > 0 means the block centred at x = shift points +y
    fill: float          # block width / pole pitch, 0 < fill <= 1
    shift_mm: float = 0.0


def _sheets(layers: list[MagnetLayer]):
    for lay in layers:
        yield lay.y_bottom_mm, -lay.br_T, lay.fill, lay.shift_mm
        yield lay.y_bottom_mm + lay.thickness_mm, lay.br_T, lay.fill, lay.shift_mm


def _square_wave(br_T: float, n: np.ndarray, fill: float) -> np.ndarray:
    """a_n = Br * 4/(n pi) * sin(n pi fill / 2) for each odd order in ``n``: the audit's single implementation,
    ``planar.square_wave_harmonic``, evaluated order by order."""
    return np.array([square_wave_harmonic(br_T, int(order), fill) for order in n])


def _ratio(k: np.ndarray, p: float, q: float, h: float, cosh_q: bool) -> np.ndarray:
    """sinh(k p) * (cosh or sinh)(k q) / sinh(k h), overflow-free for 0 <= p, q and p + q <= h."""
    sign = 1.0 if cosh_q else -1.0
    return (np.exp(k * (p + q - h)) * (1.0 - np.exp(-2.0 * k * p)) * (1.0 + sign * np.exp(-2.0 * k * q))
            / (2.0 * (1.0 - np.exp(-2.0 * k * h))))


def harmonic_coefficients(y_mm: float, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                          n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n, A, B, C, D) with B_y(x) = sum A cos(kx) + B sin(kx) and B_x(x) = sum C sin(kx) + D cos(kx) at height y.

    ``y_mm`` must not coincide with a charge sheet (a magnet face).
    """
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * math.pi / pitch_mm
    A = np.zeros_like(n); B = np.zeros_like(n); C = np.zeros_like(n); D = np.zeros_like(n)
    for ys, br, fill, shift in _sheets(layers):
        if abs(y_mm - ys) < 1e-12:
            raise ValueError(f"y = {y_mm} mm lies on a magnet face")
        a = _square_wave(br, n, fill)
        if y_mm < ys:
            p = -a * _ratio(k, h_mm - ys, y_mm, h_mm, True)     # B_y amplitude
            qx = a * _ratio(k, y_mm, h_mm - ys, h_mm, False)    # B_x amplitude
        else:
            p = a * _ratio(k, ys, h_mm - y_mm, h_mm, True)
            qx = a * _ratio(k, ys, h_mm - y_mm, h_mm, False)
        c, s = np.cos(k * shift), np.sin(k * shift)
        A += p * c; B += p * s; C += qx * c; D -= qx * s
    return n, A, B, C, D


def iron_face_coefficients(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                           n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, A, B) with B_y(x) = sum A cos(kx) + B sin(kx) [T] on the face of the plate at y = 0 (``surface``
    'inner') or y = h ('outer'): the flux density that enters the steel (see the module docstring)."""
    if surface not in ("inner", "outer"):
        raise ValueError(f"surface must be 'inner' or 'outer', not {surface!r}")
    y_face = 0.0 if surface == "inner" else h_mm
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * math.pi / pitch_mm
    A = np.zeros_like(n); B = np.zeros_like(n)
    for ys, br, fill, shift in _sheets(layers):
        if abs(ys - y_face) < 1e-12:
            continue                                            # a sheet on this plate adds no field here
        a = _square_wave(br, n, fill)
        if surface == "outer":                                  # face above the sheet: y = h
            p = a * _ratio(k, ys, 0.0, h_mm, True)
        else:                                                   # face below the sheet: y = 0
            p = -a * _ratio(k, h_mm - ys, 0.0, h_mm, True)
        A += p * np.cos(k * shift); B += p * np.sin(k * shift)
    for lay in layers:                                          # mu0 M of a layer touching the plate
        touching = lay.y_bottom_mm if surface == "inner" else lay.y_bottom_mm + lay.thickness_mm
        if abs(touching - y_face) < 1e-12:
            a = _square_wave(lay.br_T, n, lay.fill)
            A += a * np.cos(k * lay.shift_mm); B += a * np.sin(k * lay.shift_mm)
    return n, A, B
