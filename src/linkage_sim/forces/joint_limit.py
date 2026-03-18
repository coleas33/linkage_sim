"""Joint limit force element using penalty method.

Applies a penalty spring force when a revolute joint's relative angle
exceeds specified limits. Optionally includes a restitution-based
damping term to model energy loss on impact.

Penalty approach:
    τ = -k_penalty * penetration - c_damping * penetration_rate

where penetration is the amount by which the angle exceeds the limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q


@dataclass(frozen=True)
class JointLimit:
    """Penalty-based joint limit at a revolute joint.

    Applies restoring torque when θ_rel = θ_j - θ_i goes outside
    [angle_min, angle_max].

    Attributes:
        body_i_id: First body ID.
        body_j_id: Second body ID.
        angle_min: Minimum allowed relative angle (rad).
        angle_max: Maximum allowed relative angle (rad).
        stiffness: Penalty spring stiffness (N·m/rad).
        damping: Penalty damping coefficient (N·m·s/rad).
        restitution: Coefficient of restitution (0=perfectly inelastic,
            1=perfectly elastic). Controls damping scaling.
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    angle_min: float
    angle_max: float
    stiffness: float
    damping: float = 0.0
    restitution: float = 0.5
    _id: str = "joint_limit"

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
        """Compute penalty torque for joint limit violation.

        When θ_rel < angle_min: penetration = angle_min - θ_rel (positive)
        When θ_rel > angle_max: penetration = θ_rel - angle_max (positive)

        Penalty torque pushes the angle back within limits.

        Args:
            state: Mechanism state.
            q: Position vector.
            q_dot: Velocity vector.
            t: Time.

        Returns:
            Q vector (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        # Get relative angle
        theta_i = 0.0
        omega_i = 0.0
        if not state.is_ground(self.body_i_id):
            idx_i = state.get_index(self.body_i_id)
            theta_i = float(q[idx_i.theta_idx])
            omega_i = float(q_dot[idx_i.theta_idx])

        theta_j = 0.0
        omega_j = 0.0
        if not state.is_ground(self.body_j_id):
            idx_j = state.get_index(self.body_j_id)
            theta_j = float(q[idx_j.theta_idx])
            omega_j = float(q_dot[idx_j.theta_idx])

        theta_rel = theta_j - theta_i
        omega_rel = omega_j - omega_i

        torque = 0.0

        if theta_rel < self.angle_min:
            # Below minimum — push CCW (positive torque on j)
            penetration = self.angle_min - theta_rel
            # Only apply damping when moving into the stop
            damp_factor = self.damping if omega_rel < 0 else self.damping * self.restitution
            torque = self.stiffness * penetration - damp_factor * omega_rel

        elif theta_rel > self.angle_max:
            # Above maximum — push CW (negative torque on j)
            penetration = theta_rel - self.angle_max
            # Only apply damping when moving into the stop
            damp_factor = self.damping if omega_rel > 0 else self.damping * self.restitution
            torque = -(self.stiffness * penetration + damp_factor * omega_rel)

        if torque == 0.0:
            return Q

        Q += body_torque_to_Q(state, self.body_j_id, torque)
        Q += body_torque_to_Q(state, self.body_i_id, -torque)

        return Q
