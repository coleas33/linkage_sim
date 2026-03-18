"""Generalized force helper utilities.

Convert physical forces and torques into contributions to the
generalized force vector Q. These are the building blocks used
by all force element implementations.

Formulation:
    Virtual work of force F at point P on body i:
        δW = F · δr_P = F · (δr_i + B_i @ s_i · δθ_i)

    So the generalized force entries are:
        Q_xi     = F_x
        Q_yi     = F_y
        Q_θi     = (B_i @ s_i) · F

    For a pure torque τ on body i:
        Q_θi     = τ
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body
from linkage_sim.core.state import GROUND_ID, State


def point_force_to_Q(
    state: State,
    body_id: str,
    local_point: NDArray[np.float64],
    force_global: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a point force to generalized forces.

    A force F_global applied at a point s_local on body_id produces:
        Q[x_idx]     += F[0]
        Q[y_idx]     += F[1]
        Q[theta_idx] += (B(θ) @ s_local) · F

    Forces on ground are ignored (ground has no generalized coordinates).

    Args:
        state: Mechanism state.
        body_id: Body the force acts on.
        local_point: Point of application in body-local coordinates.
        force_global: Force vector [Fx, Fy] in global coordinates (N).
        q: Generalized coordinate vector.

    Returns:
        Q vector (n_coords,) with the force contribution.
    """
    Q = np.zeros(state.n_coords)

    if state.is_ground(body_id):
        return Q

    idx = state.get_index(body_id)
    Q[idx.x_idx] = force_global[0]
    Q[idx.y_idx] = force_global[1]

    # Moment arm: B(θ) @ s_local
    B_s = state.body_point_global_derivative(body_id, local_point, q)
    Q[idx.theta_idx] = float(np.dot(B_s, force_global))

    return Q


def body_torque_to_Q(
    state: State,
    body_id: str,
    torque: float,
) -> NDArray[np.float64]:
    """Convert a pure torque on a body to generalized forces.

    A torque τ on body_id produces:
        Q[theta_idx] += τ

    Torques on ground are ignored.

    Args:
        state: Mechanism state.
        body_id: Body the torque acts on.
        torque: Pure torque in N·m (positive = CCW).

    Returns:
        Q vector (n_coords,) with the torque contribution.
    """
    Q = np.zeros(state.n_coords)

    if state.is_ground(body_id):
        return Q

    idx = state.get_index(body_id)
    Q[idx.theta_idx] = torque

    return Q


def gravity_to_Q(
    state: State,
    bodies: dict[str, Body],
    q: NDArray[np.float64],
    g_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute gravity generalized forces for all bodies.

    Applies F = mass * g_vector at each body's CG.

    Args:
        state: Mechanism state.
        bodies: Dict of all bodies (including ground).
        q: Generalized coordinate vector.
        g_vector: Gravity acceleration vector [gx, gy] in m/s².
            Standard: [0, -9.81] for downward gravity.

    Returns:
        Q vector (n_coords,) with gravity contributions for all bodies.
    """
    Q = np.zeros(state.n_coords)

    for body_id, body in bodies.items():
        if body_id == GROUND_ID:
            continue
        if body.mass <= 0.0:
            continue

        force_global = body.mass * g_vector
        Q += point_force_to_Q(state, body_id, body.cg_local, force_global, q)

    return Q
