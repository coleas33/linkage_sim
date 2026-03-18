"""Linear actuator force element.

Applies a constant force (or force from a lookup table) along the line
between two body points, with an optional speed limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import point_force_to_Q


@dataclass(frozen=True)
class LinearActuator:
    """Linear actuator between two body points.

    Applies a constant force along the actuator line. If speed_limit
    is set, the force ramps to zero as the extension rate approaches
    the speed limit.

    Attributes:
        body_i_id: First body ID.
        point_i_local: Attachment on body_i.
        body_j_id: Second body ID.
        point_j_local: Attachment on body_j.
        force: Actuator force in N (positive = extension/push apart).
        speed_limit: Maximum extension rate in m/s. 0 = no limit.
        _id: Unique identifier.
    """

    body_i_id: str
    point_i_local: NDArray[np.float64]
    body_j_id: str
    point_j_local: NDArray[np.float64]
    force: float
    speed_limit: float = 0.0
    _id: str = "linear_actuator"

    @property
    def id(self) -> str:
        return self._id

    def evaluate(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Compute actuator generalized forces.

        Args:
            state: Mechanism state.
            q: Position vector.
            q_dot: Velocity vector.
            t: Time.

        Returns:
            Q vector (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        r_i = state.body_point_global(self.body_i_id, self.point_i_local, q)
        r_j = state.body_point_global(self.body_j_id, self.point_j_local, q)

        d = r_j - r_i
        length = float(np.linalg.norm(d))
        if length < 1e-15:
            return Q

        n_hat = d / length

        actual_force = self.force

        # Speed limiting
        if self.speed_limit > 0:
            v_i = state.body_point_velocity(
                self.body_i_id, self.point_i_local, q, q_dot
            )
            v_j = state.body_point_velocity(
                self.body_j_id, self.point_j_local, q, q_dot
            )
            v_rel = v_j - v_i
            v_along = float(np.dot(v_rel, n_hat))

            # Ramp force to zero as speed approaches limit
            speed_ratio = abs(v_along) / self.speed_limit
            if speed_ratio >= 1.0:
                actual_force = 0.0
            else:
                actual_force = self.force * (1.0 - speed_ratio)

        # Positive force = push apart (extension)
        F_on_j = actual_force * n_hat
        F_on_i = -actual_force * n_hat

        Q += point_force_to_Q(state, self.body_i_id, self.point_i_local, F_on_i, q)
        Q += point_force_to_Q(state, self.body_j_id, self.point_j_local, F_on_j, q)

        return Q
