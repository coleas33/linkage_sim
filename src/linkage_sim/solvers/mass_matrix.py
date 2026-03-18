"""Mass matrix assembly.

Builds the mass matrix M from body mass properties. When the body
coordinate origin does not coincide with the CG, the mass matrix
has off-diagonal coupling terms and depends on configuration q.

For body i with coordinates [x_i, y_i, θ_i] at the body origin and
CG at local position s_cg:

    M_i = [[m,     0,     m*Bs_x],
           [0,     m,     m*Bs_y],
           [m*Bs_x, m*Bs_y, Izz_cg + m*|s_cg|²]]

where Bs = B(θ) @ s_cg = dA/dθ @ s_cg (the velocity Jacobian of the CG).

When s_cg = [0, 0] (origin at CG), this reduces to a diagonal matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


def assemble_mass_matrix(
    mechanism: Mechanism,
    q: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Assemble the mass matrix M.

    Args:
        mechanism: Built mechanism.
        q: Configuration vector. Required when any body has CG offset
            from its coordinate origin. If None, assumes CG at origin
            for all bodies (diagonal mass matrix).

    Returns:
        M: (n_coords, n_coords) symmetric positive semi-definite matrix.
    """
    n = mechanism.state.n_coords
    M = np.zeros((n, n))

    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue

        idx = mechanism.state.get_index(body_id)
        m = body.mass
        s_cg = body.cg_local

        # Diagonal terms
        M[idx.x_idx, idx.x_idx] = m
        M[idx.y_idx, idx.y_idx] = m

        # M_θθ = Izz_cg + m * |s_cg|^2 (parallel axis theorem)
        s_cg_sq = float(np.dot(s_cg, s_cg))
        M[idx.theta_idx, idx.theta_idx] = body.Izz_cg + m * s_cg_sq

        # Off-diagonal coupling: m * B(θ) @ s_cg
        if m > 0 and s_cg_sq > 0 and q is not None:
            theta = float(q[idx.theta_idx])
            # B(θ) @ s_cg = [-sinθ*sx - cosθ*sy, cosθ*sx - sinθ*sy]
            Bs_x = -np.sin(theta) * s_cg[0] - np.cos(theta) * s_cg[1]
            Bs_y = np.cos(theta) * s_cg[0] - np.sin(theta) * s_cg[1]

            M[idx.x_idx, idx.theta_idx] = m * Bs_x
            M[idx.theta_idx, idx.x_idx] = m * Bs_x
            M[idx.y_idx, idx.theta_idx] = m * Bs_y
            M[idx.theta_idx, idx.y_idx] = m * Bs_y

    return M
