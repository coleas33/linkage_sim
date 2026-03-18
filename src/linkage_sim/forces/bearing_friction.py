"""Bearing friction model.

Extends basic Coulomb friction with three components:
1. Constant drag torque (seal friction, etc.)
2. Viscous drag (proportional to speed)
3. Optional radial-load-dependent Coulomb friction

    τ = -(T_drag + c_viscous * |ω_rel| + μ * R * |F_n|) * sign(ω_rel)

Uses tanh regularization for smooth behavior near zero velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q


@dataclass(frozen=True)
class BearingFriction:
    """Multi-component bearing friction at a revolute joint.

    Attributes:
        body_i_id: First body ID.
        body_j_id: Second body ID.
        constant_drag: Constant drag torque in N·m (always present).
        viscous_coeff: Viscous drag coefficient in N·m·s/rad.
        coulomb_coeff: Coulomb friction coefficient μ.
        pin_radius: Effective pin radius for Coulomb term in meters.
        radial_load: Radial load for Coulomb term in N. Set to 0 to disable.
        v_threshold: Velocity regularization threshold in rad/s.
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    constant_drag: float = 0.0
    viscous_coeff: float = 0.0
    coulomb_coeff: float = 0.0
    pin_radius: float = 0.0
    radial_load: float = 0.0
    v_threshold: float = 0.01
    _id: str = "bearing_friction"

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
        """Compute bearing friction torque.

        τ = -(T_drag + c_vis * |ω| + μ * R * F_n) * tanh(ω / v_thresh)

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

        # Direction factor via tanh regularization
        direction = float(np.tanh(omega_rel / self.v_threshold))

        # Total friction magnitude
        coulomb_torque = self.coulomb_coeff * self.pin_radius * self.radial_load
        viscous_torque = self.viscous_coeff * abs(omega_rel)
        total_magnitude = self.constant_drag + viscous_torque + coulomb_torque

        torque = -total_magnitude * direction

        Q += body_torque_to_Q(state, self.body_j_id, torque)
        Q += body_torque_to_Q(state, self.body_i_id, -torque)

        return Q
