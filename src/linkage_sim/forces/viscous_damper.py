"""Viscous damper force elements.

Translational damper: force proportional to rate of change of distance
between two body points.
    F = -c * d(length)/dt

Rotary damper: torque proportional to relative angular velocity.
    τ = -c * (ω_j - ω_i)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q, point_force_to_Q


@dataclass(frozen=True)
class TranslationalDamper:
    """Viscous damper between two body points.

    The damping force acts along the line between the two points,
    opposing the rate of change of their distance.

    F = -c * v_rel_along_line

    where v_rel_along_line is the component of relative velocity
    along the line connecting the two points.

    Attributes:
        body_i_id: First body ID.
        point_i_local: Attachment on body_i in local coordinates.
        body_j_id: Second body ID.
        point_j_local: Attachment on body_j in local coordinates.
        damping: Damping coefficient c in N·s/m.
        _id: Unique identifier.
    """

    body_i_id: str
    point_i_local: NDArray[np.float64]
    body_j_id: str
    point_j_local: NDArray[np.float64]
    damping: float
    _id: str = "translational_damper"

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
        """Compute damper generalized forces.

        Args:
            state: Mechanism state.
            q: Position vector.
            q_dot: Velocity vector.
            t: Time.

        Returns:
            Q vector (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        # Global positions
        r_i = state.body_point_global(self.body_i_id, self.point_i_local, q)
        r_j = state.body_point_global(self.body_j_id, self.point_j_local, q)

        d = r_j - r_i
        length = float(np.linalg.norm(d))
        if length < 1e-15:
            return Q

        n_hat = d / length

        # Global velocities of attachment points
        v_i = state.body_point_velocity(self.body_i_id, self.point_i_local, q, q_dot)
        v_j = state.body_point_velocity(self.body_j_id, self.point_j_local, q, q_dot)

        # Relative velocity along the line
        v_rel = v_j - v_i
        v_along = float(np.dot(v_rel, n_hat))

        # Damping force magnitude (opposes extension rate)
        force_magnitude = -self.damping * v_along

        # Force on body_j directed along n_hat (from i toward j)
        F_on_j = force_magnitude * n_hat
        F_on_i = -force_magnitude * n_hat

        Q += point_force_to_Q(state, self.body_i_id, self.point_i_local, F_on_i, q)
        Q += point_force_to_Q(state, self.body_j_id, self.point_j_local, F_on_j, q)

        return Q


@dataclass(frozen=True)
class RotaryDamper:
    """Viscous rotary damper at a revolute joint.

    τ = -c * (ω_j - ω_i)

    Applied as equal and opposite torques on the connected bodies.

    Attributes:
        body_i_id: First body ID.
        body_j_id: Second body ID.
        damping: Rotary damping coefficient c in N·m·s/rad.
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    damping: float
    _id: str = "rotary_damper"

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
        """Compute rotary damper generalized forces.

        Args:
            state: Mechanism state.
            q: Position vector.
            q_dot: Velocity vector.
            t: Time.

        Returns:
            Q vector (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        omega_i = 0.0
        if not state.is_ground(self.body_i_id):
            idx_i = state.get_index(self.body_i_id)
            omega_i = float(q_dot[idx_i.theta_idx])

        omega_j = 0.0
        if not state.is_ground(self.body_j_id):
            idx_j = state.get_index(self.body_j_id)
            omega_j = float(q_dot[idx_j.theta_idx])

        omega_rel = omega_j - omega_i
        torque = -self.damping * omega_rel

        Q += body_torque_to_Q(state, self.body_j_id, torque)
        Q += body_torque_to_Q(state, self.body_i_id, -torque)

        return Q
