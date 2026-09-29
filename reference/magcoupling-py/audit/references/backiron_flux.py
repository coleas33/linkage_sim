"""Back-iron flux by flux conservation (M1 audit, Task 4). Imports nothing from ``magcoupling``.

Two independent routes to the flux the return steel carries between poles:

1. Magnetic-circuit route. With ideal iron and mu_r = 1 magnets, the series circuit of two
   magnets and a gap gives B_g = (Br_i t_i + Br_o t_o) / (t_i + t_o + g)
   [E. P. Furlani, *Permanent Magnet and Electromechanical Devices*, Academic Press 2001,
   sec. 3.3; D. Hanselman, *Brushless Permanent Magnet Motor Design*, 2nd ed. 2006, ch. 2].
   For a sinusoidal gap field of peak B over a pole pitch tau, one pole carries
   integral_0^tau B sin(pi x / tau) dx = 2 B tau / pi per unit axial length and half of it turns
   each way in the back iron, so the back iron needs t = B tau / (pi B_design) [Hanselman ch. 4].

2. Field route (``backiron_flux_per_depth``): the exact 2D field of the workbook's steel-backed
   slab from the shared ``slab_field2d`` model (ideal iron at both faces, square-wave
   magnetization, all odd harmonics). The back-iron section at the inter-pole line carries the
   flux that enters the plate between the pole centre, where the back-iron flux is zero by
   symmetry, and that line.
"""
from __future__ import annotations

import math

import numpy as np

from audit.references.slab_field2d import MagnetLayer, iron_face_coefficients


def flat_circuit_flux_density(br_i: float, br_o: float, t_i: float, t_o: float, gap: float) -> float:
    """Gap flux density [T] of the ideal-iron series circuit of two magnets and a gap (mu_r = 1).
    Ampere's law around a current-free loop, H_i t_i + H_g g + H_o t_o = 0, with B continuous
    and B = mu0 (H + M) in each magnet, gives B = (Br_i t_i + Br_o t_o) / (t_i + t_o + gap)."""
    return (br_i * t_i + br_o * t_o) / (t_i + t_o + gap)


def sinusoidal_backiron_thickness(b_peak: float, pole_pitch: float, b_design: float) -> float:
    """Back-iron thickness [unit of pole_pitch] carrying half the flux of one pole of a sinusoidal
    gap field of peak ``b_peak``: (2 B tau / pi) / 2 / B_design."""
    flux_per_pole = 2 * b_peak * pole_pitch / math.pi
    return flux_per_pole / 2 / b_design


def coupling_slab(br_i: float, br_o: float, fill_i: float, fill_o: float, t_i: float, t_o: float,
                  gap: float) -> tuple[list[MagnetLayer], float]:
    """The steel-backed coupling slab: inner magnet on the inner plate (y = 0), the gap, outer magnet
    on the outer plate (y = h). Both are magnetized +y at x = 0: aligned rings, the no-load and
    maximum-flux position. Returns (layers, h)."""
    layers = [MagnetLayer(0.0, t_i, br_i, fill_i), MagnetLayer(t_i + gap, t_o, br_o, fill_o)]
    return layers, t_i + gap + t_o


def _unrolled_fill(width: float, pitch: float) -> float:
    """Fill of a flat block of real width ``width`` on a slab of pole pitch ``pitch``. A block wider than the
    pitch would overlap its neighbours once unrolled, which a slab layer (0 < fill <= 1) cannot represent."""
    if not 0.0 < width <= pitch:
        raise ValueError(f"a {width} mm block cannot be unrolled onto a {pitch} mm pole pitch "
                         "(need 0 < width <= pitch)")
    return width / pitch


def unrolled_coupling_slab(g, br_i: float, br_o: float, w_i: float, w_o: float, t_i: float,
                           t_o: float) -> tuple[list[MagnetLayer], float]:
    """The coupling slab of a design's two rings of flat blocks, both unrolled at the geometry's gap-radius pole
    pitch ``g.pole_pitch`` with its face gap ``g.face_gap``. Each block keeps its real width, fill = w / pitch.

    Not the arc-length fill at each block's mid-thickness radius (``g.fill_inner``/``g.fill_outer``, the
    workbook's C66/C67): at the gap-radius pitch that fill makes a block w*R_g/r_mid wide, so the inner blocks
    come out too wide and the outer ones too narrow (7.586 and 5.460 mm for the default 6.35 mm block)."""
    return coupling_slab(br_i, br_o, _unrolled_fill(w_i, g.pole_pitch), _unrolled_fill(w_o, g.pole_pitch), t_i, t_o,
                         g.face_gap)


def backiron_flux_per_depth(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                            n_max: int = 4001) -> float:
    """Flux per unit axial length [T*mm] that the 'inner' or 'outer' plate carries across the
    inter-pole line x = tau/2, for aligned layers (every shift zero, so x = 0 is a pole centre):
    integral_0^(tau/2) sum A cos(kx) dx = sum A sin(k tau/2) / k. Terms fall as 1/n^2, so
    n_max = 4001 truncates below 2e-4 relative."""
    if any(lay.shift_mm != 0.0 for lay in layers):
        raise ValueError("backiron_flux_per_depth needs aligned layers (every shift_mm == 0)")
    n, A, _ = iron_face_coefficients(surface, layers, h_mm, pitch_mm, n_max)
    k = n * math.pi / pitch_mm
    return float(np.sum(A * np.sin(k * pitch_mm / 2) / k))
