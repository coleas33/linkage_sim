"""Torsion spring force element at revolute joints.

Produces a torque proportional to the relative angle between two bodies:
    τ = -k_t * (θ_j - θ_i - θ_free) + preload

Applied as equal and opposite torques on the two connected bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q


@dataclass(frozen=True)
class TorsionSpring:
    """Torsion spring at a revolute joint between two bodies.

    The spring resists relative rotation between body_i and body_j.
    Positive torque is CCW.

    Attributes:
        body_i_id: First body ID.
        body_j_id: Second body ID.
        stiffness: Torsion spring constant k_t in N·m/rad.
        free_angle: Natural (unloaded) relative angle θ_j - θ_i in radians.
        preload: Torque at free angle in N·m (positive = CCW on body_j).
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    stiffness: float
    free_angle: float = 0.0
    preload: float = 0.0
    _id: str = "torsion_spring"

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
        """Compute torsion spring generalized forces.

        τ = k_t * (θ_j - θ_i - θ_free) + preload

        The torque τ is applied as:
            -τ on body_j (resists relative rotation)
            +τ on body_i (reaction)

        For ground connections, only the moving body receives torque.

        Args:
            state: Mechanism state.
            q: Generalized coordinate vector.
            q_dot: Velocity vector (unused for torsion spring).
            t: Current time (unused).

        Returns:
            Generalized force vector Q (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        # Get angles
        theta_i = 0.0
        if not state.is_ground(self.body_i_id):
            idx_i = state.get_index(self.body_i_id)
            theta_i = float(q[idx_i.theta_idx])

        theta_j = 0.0
        if not state.is_ground(self.body_j_id):
            idx_j = state.get_index(self.body_j_id)
            theta_j = float(q[idx_j.theta_idx])

        # Relative angle and torque
        relative_angle = theta_j - theta_i - self.free_angle
        torque = self.stiffness * relative_angle + self.preload

        # Apply torque: -τ on body_j, +τ on body_i (restorative)
        Q += body_torque_to_Q(state, self.body_j_id, -torque)
        Q += body_torque_to_Q(state, self.body_i_id, torque)

        return Q
