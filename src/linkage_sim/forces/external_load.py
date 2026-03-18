"""External load force element.

Applies a user-defined force (and optionally torque) to a body point.
The force can be constant or a function of position, velocity, and time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q, point_force_to_Q


@dataclass(frozen=True)
class ExternalLoad:
    """External force and/or torque applied to a body point.

    The force can be constant or time/state-dependent via a callable.

    Attributes:
        body_id: Body the load acts on.
        local_point: Application point in body-local coordinates.
        force_func: Callable(q, q_dot, t) -> [Fx, Fy] in global frame.
            For constant loads, use a lambda: lambda q, qd, t: np.array([Fx, Fy]).
        torque_func: Optional callable(q, q_dot, t) -> scalar torque (N·m).
            None means no torque component.
        _id: Unique identifier.
    """

    body_id: str
    local_point: NDArray[np.float64]
    force_func: Callable[
        [NDArray[np.float64], NDArray[np.float64], float],
        NDArray[np.float64],
    ]
    torque_func: Callable[
        [NDArray[np.float64], NDArray[np.float64], float],
        float,
    ] | None = None
    _id: str = "external_load"

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
        """Compute external load generalized forces.

        Args:
            state: Mechanism state.
            q: Generalized coordinate vector.
            q_dot: Generalized velocity vector.
            t: Current time.

        Returns:
            Generalized force vector Q (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        force = self.force_func(q, q_dot, t)
        Q += point_force_to_Q(state, self.body_id, self.local_point, force, q)

        if self.torque_func is not None:
            torque = self.torque_func(q, q_dot, t)
            Q += body_torque_to_Q(state, self.body_id, torque)

        return Q
