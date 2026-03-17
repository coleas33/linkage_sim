"""Coupler point evaluation: position, velocity, and acceleration.

Computes the global-frame position, velocity, and acceleration of
named coupler points on mechanism bodies, using the kinematic solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import State


@dataclass(frozen=True)
class CouplerPointResult:
    """Position, velocity, and acceleration of a coupler point.

    All values are in the global frame.
    """

    body_id: str
    point_name: str
    position: NDArray[np.float64]  # (2,)
    velocity: NDArray[np.float64]  # (2,)
    acceleration: NDArray[np.float64]  # (2,)


def eval_coupler_point(
    state: State,
    body_id: str,
    point_local: NDArray[np.float64],
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    q_ddot: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate global position, velocity, and acceleration of a body point.

    Given a point in body-local coordinates, returns its global position,
    velocity, and acceleration using the kinematic solution.

    Args:
        state: State vector mapping.
        body_id: Body on which the point lies.
        point_local: Local coordinates of the point on the body (2,).
        q: Position vector.
        q_dot: Velocity vector.
        q_ddot: Acceleration vector.

    Returns:
        (position, velocity, acceleration) each as (2,) arrays.
    """
    # Position: r + A * s
    pos = state.body_point_global(body_id, point_local, q)

    if state.is_ground(body_id):
        vel = np.zeros(2)
        accel = np.zeros(2)
        return pos, vel, accel

    idx = state.get_index(body_id)
    theta = state.get_angle(body_id, q)
    theta_dot = q_dot[idx.theta_idx]
    theta_ddot = q_ddot[idx.theta_idx]

    # Velocity: ṙ + Ȧ * s = ṙ + B * s * θ̇
    # where B = dA/dθ
    B_s = state.body_point_global_derivative(body_id, point_local, q)
    x_dot = q_dot[idx.x_idx]
    y_dot = q_dot[idx.y_idx]
    vel = np.array([
        x_dot + B_s[0] * theta_dot,
        y_dot + B_s[1] * theta_dot,
    ])

    # Acceleration: r̈ + B * s * θ̈ + A * s * θ̇²
    # (second term from d(B*s*θ̇)/dt = B*s*θ̈ + dB/dθ*s*θ̇² = B*s*θ̈ - A*s*θ̇²)
    # Wait, let me be precise:
    # d²/dt²(A*s) = B*s*θ̈ + (dB/dθ)*s*θ̇² = B*s*θ̈ - A*s*θ̇²
    # Wait: dB/dθ = -A, so:
    # d²/dt²(A*s) = B*s*θ̈ - A*s*θ̇²
    # But actually, dA/dθ = B (the derivative matrix), and dB/dθ = -A
    # So: d/dt(A*s) = B*s*θ̇
    # d²/dt²(A*s) = dB/dθ*s*θ̇² + B*s*θ̈ = -A*s*θ̇² + B*s*θ̈
    # Total point acceleration: r̈ + B*s*θ̈ - A*s*θ̇²
    A = state.rotation_matrix(theta)
    A_s = A @ point_local

    x_ddot = q_ddot[idx.x_idx]
    y_ddot = q_ddot[idx.y_idx]
    accel = np.array([
        x_ddot + B_s[0] * theta_ddot - A_s[0] * theta_dot**2,
        y_ddot + B_s[1] * theta_ddot - A_s[1] * theta_dot**2,
    ])

    return pos, vel, accel


def eval_all_coupler_points(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    q_ddot: NDArray[np.float64],
) -> list[CouplerPointResult]:
    """Evaluate all coupler points across all bodies.

    Returns a list of CouplerPointResult for every coupler point
    defined on every body in the mechanism.
    """
    results: list[CouplerPointResult] = []

    for body_id, body in mechanism.bodies.items():
        for point_name, point_local in body.coupler_points.items():
            pos, vel, accel = eval_coupler_point(
                mechanism.state, body_id, point_local, q, q_dot, q_ddot
            )
            results.append(CouplerPointResult(
                body_id=body_id,
                point_name=point_name,
                position=pos,
                velocity=vel,
                acceleration=accel,
            ))

    return results
