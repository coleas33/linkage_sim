"""Motor force element with linear T-ω droop.

Models a DC motor with a linear torque-speed characteristic:
    T = T_stall * (1 - ω/ω_no_load)

The motor applies torque between two bodies at a revolute joint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import body_torque_to_Q


@dataclass(frozen=True)
class Motor:
    """DC motor with linear T-ω droop at a revolute joint.

    Torque-speed relationship:
        T = T_stall * (1 - |ω_rel| / ω_no_load) * sign(ω_command)

    When |ω_rel| > ω_no_load, the motor cannot produce torque in
    the commanded direction (overspeed).

    Attributes:
        body_i_id: First body ID (typically ground).
        body_j_id: Second body ID (driven body).
        stall_torque: Maximum torque at zero speed (N·m).
        no_load_speed: Speed at zero torque (rad/s).
        direction: +1.0 for CCW, -1.0 for CW drive direction.
        _id: Unique identifier.
    """

    body_i_id: str
    body_j_id: str
    stall_torque: float
    no_load_speed: float
    direction: float = 1.0
    _id: str = "motor"

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
        """Compute motor torque from T-ω characteristic.

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

        # Speed in the commanded direction
        speed_in_dir = omega_rel * self.direction

        if self.no_load_speed <= 0:
            return Q

        # Linear droop: T = T_stall * (1 - speed/ω_no_load)
        torque_fraction = 1.0 - speed_in_dir / self.no_load_speed

        # Clamp: motor can't produce negative torque (in commanded direction)
        # and can't exceed stall torque
        torque_fraction = float(np.clip(torque_fraction, 0.0, 1.0))
        torque = self.stall_torque * torque_fraction * self.direction

        Q += body_torque_to_Q(state, self.body_j_id, torque)
        Q += body_torque_to_Q(state, self.body_i_id, -torque)

        return Q
