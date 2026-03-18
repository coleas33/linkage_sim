"""Event detection framework for forward dynamics.

Provides event functions that can be passed to SciPy's solve_ivp to
detect zero-crossings: joint limit contact, direction reversals, and
mode switches.

Each event function has the signature: f(t, y) -> float
The integrator detects when f crosses zero.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State


def make_angle_limit_event(
    state: State,
    body_i_id: str,
    body_j_id: str,
    limit_angle: float,
    direction: int = 0,
) -> Callable[[float, NDArray[np.float64]], float]:
    """Create an event function that triggers when relative angle reaches a limit.

    Args:
        state: Mechanism state.
        body_i_id: First body.
        body_j_id: Second body.
        limit_angle: The angle threshold.
        direction: -1 = detect only falling crossings, +1 = only rising, 0 = both.

    Returns:
        Event function f(t, y) -> float. Zero when θ_rel = limit_angle.
    """
    n = state.n_coords

    def event(t: float, y: NDArray[np.float64]) -> float:
        q = y[:n]
        theta_i = 0.0
        if not state.is_ground(body_i_id):
            idx_i = state.get_index(body_i_id)
            theta_i = float(q[idx_i.theta_idx])
        theta_j = 0.0
        if not state.is_ground(body_j_id):
            idx_j = state.get_index(body_j_id)
            theta_j = float(q[idx_j.theta_idx])
        return theta_j - theta_i - limit_angle

    event.terminal = False  # type: ignore[attr-defined]
    event.direction = direction  # type: ignore[attr-defined]
    return event


def make_velocity_reversal_event(
    state: State,
    body_id: str,
    coord: str = "theta",
) -> Callable[[float, NDArray[np.float64]], float]:
    """Create an event for velocity zero-crossing (direction reversal).

    Args:
        state: Mechanism state.
        body_id: Body to monitor.
        coord: 'x', 'y', or 'theta'.

    Returns:
        Event function f(t, y) -> float. Zero when velocity crosses zero.
    """
    n = state.n_coords
    idx = state.get_index(body_id)
    if coord == "x":
        vel_idx = idx.x_idx
    elif coord == "y":
        vel_idx = idx.y_idx
    else:
        vel_idx = idx.theta_idx

    def event(t: float, y: NDArray[np.float64]) -> float:
        return float(y[n + vel_idx])

    event.terminal = False  # type: ignore[attr-defined]
    event.direction = 0  # type: ignore[attr-defined]
    return event


def make_terminal_angle_event(
    state: State,
    body_i_id: str,
    body_j_id: str,
    limit_angle: float,
    direction: int = 0,
) -> Callable[[float, NDArray[np.float64]], float]:
    """Like make_angle_limit_event but terminates integration on trigger.

    Args:
        state: Mechanism state.
        body_i_id: First body.
        body_j_id: Second body.
        limit_angle: The angle threshold.
        direction: Crossing direction.

    Returns:
        Terminal event function.
    """
    event = make_angle_limit_event(state, body_i_id, body_j_id, limit_angle, direction)
    event.terminal = True  # type: ignore[attr-defined]
    return event
