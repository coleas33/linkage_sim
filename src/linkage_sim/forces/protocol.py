"""ForceElement protocol — interface for all force-producing elements.

Every force element produces a contribution to the generalized force
vector Q. The mechanism assembler iterates all force elements and sums
their contributions.

Generalized force formulation:
    For a force F_global applied at local point s on body i:
        Q[x_idx]     += F_global[0]
        Q[y_idx]     += F_global[1]
        Q[theta_idx] += (B(θ) @ s) · F_global

    where B = dA/dθ is the rotation matrix derivative.

    For a pure torque τ on body i:
        Q[theta_idx] += τ
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State


class ForceElement(Protocol):
    """Interface that all force elements must implement."""

    @property
    def id(self) -> str:
        """Unique identifier for this force element."""
        ...

    def evaluate(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Compute the generalized force contribution Q.

        Args:
            state: Mechanism state (coordinate bookkeeping).
            q: Generalized coordinate vector (n_coords,).
            q_dot: Generalized velocity vector (n_coords,).
            t: Current time.

        Returns:
            Generalized force vector (n_coords,). Only the entries
            corresponding to affected bodies are non-zero.
        """
        ...
