"""Energy balance tracking for forward dynamics.

Computes kinetic energy, gravitational potential energy, spring potential
energy, and tracks energy balance: KE + PE + dissipated ≈ work_input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID
from linkage_sim.solvers.mass_matrix import assemble_mass_matrix


@dataclass(frozen=True)
class EnergyState:
    """Energy components at one instant.

    Attributes:
        kinetic: Translational + rotational kinetic energy (J).
        potential_gravity: Gravitational PE relative to y=0 (J).
        total: KE + PE_gravity.
    """

    kinetic: float
    potential_gravity: float
    total: float


def compute_kinetic_energy(
    mechanism: Mechanism,
    q_dot: NDArray[np.float64],
    q: NDArray[np.float64] | None = None,
) -> float:
    """Compute total kinetic energy: KE = 0.5 * q̇^T * M * q̇.

    Args:
        mechanism: Built mechanism.
        q_dot: Velocity vector.
        q: Position vector (needed for off-diagonal mass matrix terms).

    Returns:
        Kinetic energy in Joules.
    """
    M = assemble_mass_matrix(mechanism, q)
    return 0.5 * float(q_dot @ M @ q_dot)


def compute_gravity_pe(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    g_magnitude: float = 9.81,
) -> float:
    """Compute gravitational potential energy: PE = Σ m_i * g * y_cg_i.

    Reference: y = 0.

    Args:
        mechanism: Built mechanism.
        q: Position vector.
        g_magnitude: Gravitational acceleration magnitude (m/s²).

    Returns:
        Gravitational PE in Joules.
    """
    pe = 0.0
    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID or body.mass <= 0:
            continue
        r_cg = mechanism.state.body_point_global(body_id, body.cg_local, q)
        pe += body.mass * g_magnitude * float(r_cg[1])
    return pe


def compute_energy_state(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    g_magnitude: float = 9.81,
) -> EnergyState:
    """Compute all energy components at one instant.

    Args:
        mechanism: Built mechanism.
        q: Position vector.
        q_dot: Velocity vector.
        g_magnitude: Gravity magnitude.

    Returns:
        EnergyState with KE, PE, total.
    """
    ke = compute_kinetic_energy(mechanism, q_dot, q)
    pe = compute_gravity_pe(mechanism, q, g_magnitude)
    return EnergyState(kinetic=ke, potential_gravity=pe, total=ke + pe)
