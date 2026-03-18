"""Gravity force element.

Applies gravitational force F = mass * g_vector at each body's CG.
Uses the generalized force helper to compute Q contributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body
from linkage_sim.core.state import GROUND_ID, State
from linkage_sim.forces.helpers import point_force_to_Q


@dataclass(frozen=True)
class Gravity:
    """Uniform gravitational field applied to all bodies.

    Attributes:
        g_vector: Gravity acceleration vector [gx, gy] in m/s².
            Standard downward gravity: [0, -9.81].
        bodies: Dict of all bodies in the mechanism. Ground and
            zero-mass bodies are automatically skipped.
        _id: Unique identifier for this force element.
    """

    g_vector: NDArray[np.float64]
    bodies: dict[str, Body]
    _id: str = "gravity"

    @property
    def id(self) -> str:
        """Unique identifier for this force element."""
        return self._id

    def evaluate(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Compute gravity generalized forces for all bodies.

        For each body with mass > 0:
            F_gravity = mass * g_vector, applied at body CG.

        Args:
            state: Mechanism state (coordinate bookkeeping).
            q: Generalized coordinate vector (n_coords,).
            q_dot: Generalized velocity vector (n_coords,) — unused.
            t: Current time — unused (gravity is constant).

        Returns:
            Generalized force vector Q (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        for body_id, body in self.bodies.items():
            if body_id == GROUND_ID:
                continue
            if body.mass <= 0.0:
                continue

            force_global = body.mass * self.g_vector
            Q += point_force_to_Q(state, body_id, body.cg_local, force_global, q)

        return Q
