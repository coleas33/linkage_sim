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



# =========================================================================== Task 3 additions
# The planar (2D slab) torque chain and Br(T), appended by Task 3 on top of Task 2's building blocks
# above. ``square_wave_harmonic`` and ``planar_s_factor`` are Task 2's and are reused here, not
# redefined. The names and signatures below match the copies drafted in Task 7 (sweep_ref: ``br_at``,
# ``square_wave_harmonic``, ``iron_backed_factor``, ``free_space_factor``, ``torque_on_cylinder``,
# ``end_factor``), so later tasks import them instead of keeping their own copies.
#
# * ``wave_number``: k_n = n pi / pole pitch (harmonic n repeats every 2 * pitch / n).
# * ``free_space_factor`` / ``iron_backed_factor``: the two branches of Task 2's ``planar_s_factor``
#   under the names the torque-chain checks use (free space; infinitely permeable backings behind both
#   arrays). Sources: Furlani, *Permanent Magnet and Electromechanical Devices*, Academic Press, 2001
#   (2D fields of magnetized arrays and couplings); Hague, *The Principles of Electromagnetism Applied
#   to Electrical Machines*, 1929 (image solutions between permeable boundaries). The Task 3 sanity
#   tests check the free-space factor against 3D arrays, and both factors against their thick-array
#   and image-series limits.
# * ``harmonic_shear_stress`` / ``planar_shear_stress``: the shear stress between the two arrays
#   displaced by delta, tau_n = B_in B_on / (2 mu0) * S_n * sin(k_n delta). It is positive when it
#   opposes the displacement.
# * ``torque_on_cylinder``: a uniform shear stress tau on a cylinder of radius r and length L gives a
#   force tau 2 pi r L acting at lever arm r.
# * ``end_factor``: the engine's documented EMPIRICAL end-effect form 1 - c_end * pitch / L (README,
#   model step 4). It restates the same algebra and is not physics; the physical end effect is
#   ``torque_ref.end_factor_3d``.
# * ``br_at``: linear reversible remanence Br(T) = Br20 (1 + alpha (T - 20)).
#
# Units are SI (m, T, Pa, N*m) unless a name ends in _mm.
from typing import Iterable  # noqa: E402  (appended section; the module header belongs to Task 2)


def br_at(br20_T: float, alpha_per_C: float, temp_C: float) -> float:
    """Linear reversible remanence [T]: Br(T) = Br20 * (1 + alpha * (T - 20))."""
    return br20_T * (1.0 + alpha_per_C * (temp_C - 20.0))


def wave_number(n: int, pole_pitch_m: float) -> float:
    """Wave number [1/m] of harmonic n for a pole pitch in metres: k_n = n * pi / pitch."""
    return n * math.pi / pole_pitch_m


def iron_backed_factor(k: float, t_i: float, t_o: float, g: float) -> float:
    """S_n with infinitely permeable backings behind both arrays (k in 1/m, lengths in m): Task 2's steel form."""
    return planar_s_factor(k, t_i, t_o, g, backed=True)


def free_space_factor(k: float, t_i: float, t_o: float, g: float) -> float:
    """S_n for two arrays in free space (k in 1/m, lengths in m): Task 2's free-space form."""
    return planar_s_factor(k, t_i, t_o, g, backed=False)


def harmonic_shear_stress(b_i_n: float, b_o_n: float, s_n: float, k: float, delta_m: float, mu0: float) -> float:
    """Shear stress [Pa] of one harmonic: B_in * B_on / (2 mu0) * S_n * sin(k * delta)."""
    return b_i_n * b_o_n / (2.0 * mu0) * s_n * math.sin(k * delta_m)


def planar_shear_stress(br_i: float, br_o: float, fill_i: float, fill_o: float, pole_pitch_m: float,
                        t_i_m: float, t_o_m: float, g_m: float, delta_m: float, mu0: float, backed: bool,
                        harmonics: Iterable[int]) -> float:
    """Shear stress [Pa] between two planar alternating arrays displaced by ``delta_m``, summed over ``harmonics``.

    ``backed`` selects infinitely permeable backings behind both arrays; otherwise free space.
    """
    tau = 0.0
    for n in harmonics:
        k = wave_number(n, pole_pitch_m)
        if backed:
            s = iron_backed_factor(k, t_i_m, t_o_m, g_m)
        else:
            s = free_space_factor(k, t_i_m, t_o_m, g_m)
        tau += harmonic_shear_stress(square_wave_harmonic(br_i, n, fill_i), square_wave_harmonic(br_o, n, fill_o),
                                     s, k, delta_m, mu0)
    return tau


def torque_on_cylinder(tau_Pa: float, r_m: float, length_m: float) -> float:
    """Torque [N*m] of a uniform shear stress on a cylinder: tau * 2 pi r L * r."""
    return tau_Pa * 2.0 * math.pi * r_m ** 2 * length_m


def end_factor(c_end: float, pole_pitch_mm: float, length_mm: float) -> float:
    """The engine's documented empirical end-effect form, 1 - c_end * pitch / L (dimensionless)."""
    return 1.0 - c_end * pole_pitch_mm / length_mm
