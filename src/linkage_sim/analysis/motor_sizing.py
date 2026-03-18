"""Motor sizing assistant.

Given required (ω, T) operating points from inverse dynamics and a motor
T-ω envelope, determines feasibility and margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MotorSizingResult:
    """Result of motor sizing analysis.

    Attributes:
        operating_speeds: Required angular speeds (rad/s) at each step.
        operating_torques: Required torques (N·m) at each step.
        stall_torque: Motor stall torque.
        no_load_speed: Motor no-load speed.
        feasible: Array of booleans — True if the operating point is
            within the motor envelope.
        all_feasible: True if every operating point is feasible.
        worst_margin: Smallest margin (as fraction of available torque).
            Negative means infeasible.
        worst_angle_idx: Index of the worst-case operating point.
    """

    operating_speeds: NDArray[np.float64]
    operating_torques: NDArray[np.float64]
    stall_torque: float
    no_load_speed: float
    feasible: NDArray[np.bool_]
    all_feasible: bool
    worst_margin: float
    worst_angle_idx: int


def check_motor_sizing(
    operating_speeds: NDArray[np.float64],
    operating_torques: NDArray[np.float64],
    stall_torque: float,
    no_load_speed: float,
) -> MotorSizingResult:
    """Check if a motor can deliver the required operating points.

    The motor envelope is a linear T-ω droop:
        T_available = T_stall * (1 - |ω| / ω_no_load)

    An operating point is feasible if |T_required| <= T_available.

    Args:
        operating_speeds: Required speeds at each step (rad/s).
        operating_torques: Required torques at each step (N·m).
        stall_torque: Motor stall torque (N·m).
        no_load_speed: Motor no-load speed (rad/s).

    Returns:
        MotorSizingResult with feasibility analysis.
    """
    n = len(operating_speeds)
    feasible = np.zeros(n, dtype=np.bool_)
    margins = np.zeros(n)

    for i in range(n):
        speed = abs(float(operating_speeds[i]))
        torque_required = abs(float(operating_torques[i]))

        if no_load_speed > 0 and speed < no_load_speed:
            torque_available = stall_torque * (1.0 - speed / no_load_speed)
        else:
            torque_available = 0.0

        if torque_available > 1e-15:
            margins[i] = (torque_available - torque_required) / torque_available
        elif torque_required < 1e-15:
            margins[i] = 1.0  # no torque needed, any motor works
        else:
            margins[i] = -1.0  # need torque but none available

        feasible[i] = margins[i] >= 0.0

    worst_idx = int(np.argmin(margins))

    return MotorSizingResult(
        operating_speeds=operating_speeds,
        operating_torques=operating_torques,
        stall_torque=stall_torque,
        no_load_speed=no_load_speed,
        feasible=feasible,
        all_feasible=bool(np.all(feasible)),
        worst_margin=float(margins[worst_idx]),
        worst_angle_idx=worst_idx,
    )
