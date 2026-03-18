"""Gas spring force element.

Models a gas spring with pressure-based force and velocity-dependent
damping. The force-stroke relationship follows:

    F = P0 * A * (L0 / L)^gamma + c * dL/dt

where:
    P0 = initial gas pressure
    A = piston area
    L0 = initial gas column length
    L = current gas column length (= current actuator length - rod length)
    gamma = polytropic exponent (1.0 = isothermal, 1.4 = adiabatic)
    c = velocity-dependent damping coefficient
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import point_force_to_Q


@dataclass(frozen=True)
class GasSpring:
    """Gas spring between two body points.

    Attributes:
        body_i_id: First body ID.
        point_i_local: Attachment on body_i.
        body_j_id: Second body ID.
        point_j_local: Attachment on body_j.
        initial_force: Force at the extended (nominal) length in N.
        extended_length: Nominal extended length in m.
        stroke: Maximum stroke (compression) in m.
        damping: Velocity-dependent damping coefficient in N·s/m.
        polytropic_exp: Polytropic exponent (1.0=isothermal, 1.4=adiabatic).
        _id: Unique identifier.
    """

    body_i_id: str
    point_i_local: NDArray[np.float64]
    body_j_id: str
    point_j_local: NDArray[np.float64]
    initial_force: float
    extended_length: float
    stroke: float
    damping: float = 0.0
    polytropic_exp: float = 1.0
    _id: str = "gas_spring"

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
        """Compute gas spring forces.

        The gas spring always pushes apart (extension force). As it
        compresses, the force increases due to gas compression.

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
        current_length = float(np.linalg.norm(d))
        if current_length < 1e-15:
            return Q

        n_hat = d / current_length

        # Compression from extended position
        compression = self.extended_length - current_length
        compression = max(compression, 0.0)  # can't extend beyond nominal
        compression = min(compression, self.stroke)  # can't compress beyond stroke

        # Gas force: F = F0 * (L0 / L)^n where L0 = stroke, L = stroke - compression
        gas_column = self.stroke - compression
        if gas_column < 1e-10:
            gas_column = 1e-10  # prevent division by zero

        force_ratio = (self.stroke / gas_column) ** self.polytropic_exp
        gas_force = self.initial_force * force_ratio

        # Velocity-dependent damping
        damping_force = 0.0
        if self.damping > 0:
            v_i = state.body_point_velocity(
                self.body_i_id, self.point_i_local, q, q_dot
            )
            v_j = state.body_point_velocity(
                self.body_j_id, self.point_j_local, q, q_dot
            )
            v_rel = v_j - v_i
            v_along = float(np.dot(v_rel, n_hat))
            damping_force = -self.damping * v_along

        total_force = gas_force + damping_force

        # Gas spring pushes apart (positive = extension)
        F_on_j = total_force * n_hat
        F_on_i = -total_force * n_hat

        Q += point_force_to_Q(state, self.body_i_id, self.point_i_local, F_on_i, q)
        Q += point_force_to_Q(state, self.body_j_id, self.point_j_local, F_on_j, q)

        return Q
