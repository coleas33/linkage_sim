"""Planar (unrolled) harmonic model of two facing magnet arrays: the textbook building blocks (M1).

Shared by every torque task so the planar formulas exist once. Nothing here uses the engine.

* ``square_wave_harmonic``: amplitude of harmonic n of an alternating square-wave magnetization
  +/-B that fills the fraction ``fill`` of each pole pitch. The Fourier series of that pulse train
  has only odd harmonics, B_n = 4 B / (n pi) sin(n pi fill / 2); even harmonics vanish by half-wave
  symmetry (any Fourier-series text, e.g. Kreyszig, Advanced Engineering Mathematics, ch. 11).
* ``planar_s_factor``: geometry factor S of one harmonic (wave number k) between two planar arrays
  of thickness t_i and t_o separated by the gap g, normalised so the shear-stress amplitude is
  B_i B_o / (2 mu0) * S. Free space: (1 - e^-k t_i)(1 - e^-k t_o) e^-k g / 2 (a magnetic charge
  sheet s cos kx gives H = s/2 e^-k|y| on both sides). Infinitely permeable planes at both array
  backs: sinh(k t_i) sinh(k t_o) / sinh(k (t_i + t_o + g)) (Green's function between two
  equipotential planes). ``test_torque2d_reference_sanity.py`` checks both against the exact
  cylindrical solution in ``field2d`` as R_gap grows, and against the thick-magnet limit e^-k g / 2.

SI units: k in 1/m, lengths in m.
"""
from __future__ import annotations

import math


def square_wave_harmonic(br_T: float, n: int, fill: float) -> float:
    """Amplitude [T] of harmonic n (n >= 1) of an alternating +/-br_T square wave with fill fraction ``fill``."""
    if n < 1:
        raise ValueError(f"harmonic order must be >= 1, got {n}")
    if n % 2 == 0:
        return 0.0
    return 4 * br_T / (n * math.pi) * math.sin(n * math.pi * fill / 2)


def planar_s_factor(k: float, t_i: float, t_o: float, g: float, backed: bool) -> float:
    """Planar geometry factor S for wave number k [1/m], thicknesses t_i, t_o and gap g [m].
    ``backed`` puts infinitely permeable planes at both array backs; otherwise free space."""
    if backed:
        return math.sinh(k * t_i) * math.sinh(k * t_o) / math.sinh(k * (t_i + t_o + g))
    return (1 - math.exp(-k * t_i)) * (1 - math.exp(-k * t_o)) * math.exp(-k * g) / 2
