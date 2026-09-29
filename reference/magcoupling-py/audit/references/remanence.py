"""Remanence versus temperature and the torque law that follows from it (M1 audit; shared by Tasks 4-7).

Imports nothing from ``magcoupling``.

- Br(T) = Br20 * (1 + alpha * (T - 20 degC)), with alpha the reversible temperature coefficient of Br
  in 1/degC referenced to 20 degC, the datasheet convention for sintered NdFeB (IEC 60404-8-1; e.g.
  Arnold Magnetic Technologies N42SH: alpha(Br) = -0.12 %/degC over 20-150 degC).
- The pull-out torque of a PM-PM coupling is bilinear in the two rings' magnetizations (each
  harmonic's shear stress is B_in * B_on * S_n / (2 mu0)), so with one alpha for both rings the
  torque scales as Br(T)^2.
"""
from __future__ import annotations

# One implementation of Br(T): planar.br_at. Re-exported here so ``audit.references.remanence.br_at`` stays
# importable for its consumers.
from audit.references.planar import br_at


def br_ratio(alpha_per_C: float, temp_C: float, ref_C: float = 20.0) -> float:
    """Br(temp_C) / Br(ref_C)."""
    return br_at(1.0, alpha_per_C, temp_C) / br_at(1.0, alpha_per_C, ref_C)


def torque_at(torque_ref_Nm: float, alpha_per_C: float, ref_C: float, temp_C: float) -> float:
    """Torque moved from ``ref_C`` to ``temp_C`` with torque proportional to Br_inner * Br_outer = Br^2."""
    return torque_ref_Nm * br_ratio(alpha_per_C, temp_C, ref_C) ** 2
