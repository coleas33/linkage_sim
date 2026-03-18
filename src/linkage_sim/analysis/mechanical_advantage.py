"""Mechanical advantage computation.

The instantaneous mechanical advantage (MA) of a mechanism is the ratio
of output velocity to input velocity. For a 1-DOF mechanism:

    MA = dθ_output/dθ_input = ω_output / ω_input

For translational output (e.g., slider):
    MA = dx_output/dθ_input = v_output / ω_input  (units: m/rad)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State


@dataclass(frozen=True)
class MechanicalAdvantageResult:
    """Result of mechanical advantage computation.

    Attributes:
        ma: Instantaneous mechanical advantage (dimensionless for
            angular/angular, m/rad for translational/angular).
        input_velocity: Input coordinate velocity.
        output_velocity: Output coordinate velocity.
    """

    ma: float
    input_velocity: float
    output_velocity: float


def mechanical_advantage(
    state: State,
    q_dot: NDArray[np.float64],
    input_body_id: str,
    output_body_id: str,
    input_coord: str = "theta",
    output_coord: str = "theta",
) -> MechanicalAdvantageResult:
    """Compute instantaneous mechanical advantage from velocity solution.

    Args:
        state: Mechanism state.
        q_dot: Velocity solution vector.
        input_body_id: Input (driven) body ID.
        output_body_id: Output body ID.
        input_coord: Which coordinate of input body ('x', 'y', 'theta').
        output_coord: Which coordinate of output body ('x', 'y', 'theta').

    Returns:
        MechanicalAdvantageResult.

    Raises:
        ValueError: If body is ground or invalid coordinate.
    """
    v_in = _get_velocity(state, q_dot, input_body_id, input_coord)
    v_out = _get_velocity(state, q_dot, output_body_id, output_coord)

    if abs(v_in) < 1e-15:
        ma = float("inf") if abs(v_out) > 1e-15 else float("nan")
    else:
        ma = v_out / v_in

    return MechanicalAdvantageResult(
        ma=ma,
        input_velocity=v_in,
        output_velocity=v_out,
    )


def _get_velocity(
    state: State,
    q_dot: NDArray[np.float64],
    body_id: str,
    coord: str,
) -> float:
    """Extract a velocity component from q_dot."""
    if state.is_ground(body_id):
        return 0.0

    idx = state.get_index(body_id)
    if coord == "x":
        return float(q_dot[idx.x_idx])
    elif coord == "y":
        return float(q_dot[idx.y_idx])
    elif coord == "theta":
        return float(q_dot[idx.theta_idx])
    else:
        raise ValueError(f"Invalid coordinate '{coord}'. Use 'x', 'y', or 'theta'.")
