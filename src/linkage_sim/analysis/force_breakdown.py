"""Force element contribution breakdown.

Evaluates each force element individually across a sweep to determine
which dominates: inertia, gravity, springs, friction, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.protocol import ForceElement


@dataclass(frozen=True)
class ForceContribution:
    """One force element's Q contribution at a configuration.

    Attributes:
        element_id: Force element identifier.
        Q: Generalized force vector from this element (n_coords,).
        Q_norm: ||Q|| — overall magnitude.
    """

    element_id: str
    Q: NDArray[np.float64]
    Q_norm: float


def evaluate_contributions(
    state: State,
    force_elements: list[ForceElement],
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    t: float,
) -> list[ForceContribution]:
    """Evaluate each force element separately at one configuration.

    Args:
        state: Mechanism state.
        force_elements: List of force elements.
        q: Position vector.
        q_dot: Velocity vector.
        t: Time.

    Returns:
        List of ForceContribution, one per element.
    """
    results: list[ForceContribution] = []
    for fe in force_elements:
        Q = fe.evaluate(state, q, q_dot, t)
        results.append(ForceContribution(
            element_id=fe.id,
            Q=Q,
            Q_norm=float(np.linalg.norm(Q)),
        ))
    return results


def inertia_contribution(
    M_q_ddot: NDArray[np.float64],
) -> ForceContribution:
    """Create a contribution entry for inertial forces.

    Args:
        M_q_ddot: The M * q̈ vector from inverse dynamics.

    Returns:
        ForceContribution representing inertial loads.
    """
    return ForceContribution(
        element_id="inertia",
        Q=M_q_ddot,
        Q_norm=float(np.linalg.norm(M_q_ddot)),
    )
