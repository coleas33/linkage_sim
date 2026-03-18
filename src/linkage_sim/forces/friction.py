"""Coulomb friction force element at revolute joints.

Uses velocity-regularized tanh model to approximate Coulomb friction:
    τ_friction = -μ * R * |F_n| * tanh(ω_rel / v_threshold)

where:
    μ = friction coefficient
    R = effective pin radius
    F_n = normal (reaction) force magnitude at the joint
    ω_rel = relative angular velocity between connected bodies
    v_threshold = velocity threshold for regularization

The tanh regularization avoids the discontinuity at zero velocity,
making the friction force smooth and suitable for static analysis
(where ω_rel = 0 gives zero friction) and future dynamic simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q


@dataclass(frozen=True)
class CoulombFriction:
    """Regularized Coulomb friction at a revolute joint.

    Attributes:
        body_i_id: First body ID.
        body_j_id: Second body ID.
        friction_coeff: Coulomb friction coefficient μ.
        pin_radius: Effective pin radius R in meters.
        normal_force: Normal force magnitude |F_n| in Newtons.
            For static analysis, this is typically provided as a constant
            estimate. For dynamic analysis, it would come from reaction forces.
        v_threshold: Velocity regularization threshold in rad/s.
            Smaller = sharper transition. Default 0.01 rad/s.
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    friction_coeff: float
    pin_radius: float
    normal_force: float
    v_threshold: float = 0.01
    _id: str = "coulomb_friction"

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
        """Compute friction torque using tanh-regularized Coulomb model.

        τ = -μ * R * |F_n| * tanh(ω_rel / v_threshold)

        Applied as equal and opposite torques on body_i and body_j.

        Args:
            state: Mechanism state.
            q: Generalized coordinate vector.
            q_dot: Generalized velocity vector.
            t: Current time.

        Returns:
            Generalized force vector Q (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        # Get relative angular velocity
        omega_i = 0.0
        if not state.is_ground(self.body_i_id):
            idx_i = state.get_index(self.body_i_id)
            omega_i = float(q_dot[idx_i.theta_idx])

        omega_j = 0.0
        if not state.is_ground(self.body_j_id):
            idx_j = state.get_index(self.body_j_id)
            omega_j = float(q_dot[idx_j.theta_idx])

        omega_rel = omega_j - omega_i

        # Regularized friction torque
        max_torque = self.friction_coeff * self.pin_radius * self.normal_force
        torque = -max_torque * float(np.tanh(omega_rel / self.v_threshold))

        # Apply torque on body_j, reaction on body_i
        Q += body_torque_to_Q(state, self.body_j_id, torque)
        Q += body_torque_to_Q(state, self.body_i_id, -torque)

        return Q
