"""Driver constraints: prescribed motion as constraint equations.

A driver adds one constraint equation to the system, prescribing
a kinematic variable as a function of time. The associated Lagrange
multiplier gives the required actuator effort.

See NUMERICAL_FORMULATION.md § "Driver Constraints" for derivations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import GROUND_ID, State


@dataclass
class RevoluteDriver:
    """Revolute driver: prescribes relative angle between two bodies.

    Constraint (1 equation):
        Φ = θⱼ - θᵢ - f(t) = 0

    Where θ_relative = θⱼ - θᵢ is the relative angle and f(t) is the
    prescribed motion profile.

    The Lagrange multiplier λ is the required input torque (N·m).
    """

    _id: str
    _body_i_id: str
    _body_j_id: str
    _f: Callable[[float], float]
    _f_dot: Callable[[float], float]
    _f_ddot: Callable[[float], float]

    @property
    def id(self) -> str:
        return self._id

    @property
    def n_equations(self) -> int:
        return 1

    @property
    def dof_removed(self) -> int:
        return 1

    @property
    def body_i_id(self) -> str:
        return self._body_i_id

    @property
    def body_j_id(self) -> str:
        return self._body_j_id

    def constraint(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """Φ = θⱼ - θᵢ - f(t)"""
        theta_i = state.get_angle(self._body_i_id, q)
        theta_j = state.get_angle(self._body_j_id, q)
        return np.array([theta_j - theta_i - self._f(t)])

    def jacobian(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂q: -1 at θᵢ, +1 at θⱼ, 0 elsewhere."""
        n = state.n_coords
        jac = np.zeros((1, n))

        if not state.is_ground(self._body_i_id):
            idx_i = state.get_index(self._body_i_id)
            jac[0, idx_i.theta_idx] = -1.0

        if not state.is_ground(self._body_j_id):
            idx_j = state.get_index(self._body_j_id)
            jac[0, idx_j.theta_idx] = 1.0

        return jac

    def gamma(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Acceleration RHS: γ = f̈(t).

        Since Φ = θⱼ - θᵢ - f(t), all q-dependent terms are linear:
            - (Φ_q * q̇)_q * q̇ = 0  (Jacobian is constant in q)
            - Φ_qt = 0  (Jacobian doesn't depend on t)
            - Φ_tt = -f̈(t)
        So γ = -Φ_tt = f̈(t).
        """
        return np.array([self._f_ddot(t)])


def make_revolute_driver(
    driver_id: str,
    body_i_id: str,
    body_j_id: str,
    f: Callable[[float], float],
    f_dot: Callable[[float], float],
    f_ddot: Callable[[float], float],
) -> RevoluteDriver:
    """Create a revolute driver that prescribes relative angle vs. time.

    For a common case where body_i is ground, the driver prescribes
    the absolute angle of body_j: θⱼ = f(t).

    Args:
        driver_id: Unique identifier.
        body_i_id: Reference body (often "ground").
        body_j_id: Driven body.
        f: Position function f(t) -> prescribed angle (rad).
        f_dot: Velocity function f'(t) -> angular velocity (rad/s).
        f_ddot: Acceleration function f''(t) -> angular acceleration (rad/s²).
    """
    return RevoluteDriver(
        _id=driver_id,
        _body_i_id=body_i_id,
        _body_j_id=body_j_id,
        _f=f,
        _f_dot=f_dot,
        _f_ddot=f_ddot,
    )


def constant_speed_driver(
    driver_id: str,
    body_i_id: str,
    body_j_id: str,
    omega: float,
    theta_0: float = 0.0,
) -> RevoluteDriver:
    """Create a revolute driver with constant angular velocity.

    f(t) = theta_0 + omega * t
    f'(t) = omega
    f''(t) = 0

    Args:
        driver_id: Unique identifier.
        body_i_id: Reference body (often "ground").
        body_j_id: Driven body.
        omega: Angular velocity (rad/s).
        theta_0: Initial angle at t=0 (rad).
    """
    return make_revolute_driver(
        driver_id=driver_id,
        body_i_id=body_i_id,
        body_j_id=body_j_id,
        f=lambda t: theta_0 + omega * t,
        f_dot=lambda t: omega,
        f_ddot=lambda t: 0.0,
    )
