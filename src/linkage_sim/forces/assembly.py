"""Generalized force vector Q assembly.

Iterates all force elements attached to a mechanism and sums their
contributions into the global Q vector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.protocol import ForceElement


def assemble_Q(
    state: State,
    force_elements: list[ForceElement],
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Assemble the generalized force vector from all force elements.

    Args:
        state: Mechanism state (coordinate bookkeeping).
        force_elements: List of force elements to evaluate.
        q: Generalized coordinate vector (n_coords,).
        q_dot: Generalized velocity vector (n_coords,).
        t: Current time.

    Returns:
        Total generalized force vector Q (n_coords,).
    """
    Q = np.zeros(state.n_coords)

    for fe in force_elements:
        Q += fe.evaluate(state, q, q_dot, t)

    return Q
