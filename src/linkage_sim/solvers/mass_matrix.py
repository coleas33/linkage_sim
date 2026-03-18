"""Mass matrix assembly.

Builds the block-diagonal mass matrix M from body mass properties.
Each moving body contributes a 3x3 block:

    M_i = [[m, 0, 0],
           [0, m, 0],
           [0, 0, Izz]]

Ground is excluded (has no generalized coordinates).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


def assemble_mass_matrix(mechanism: Mechanism) -> NDArray[np.float64]:
    """Assemble the block-diagonal mass matrix M.

    Args:
        mechanism: Built mechanism.

    Returns:
        M: (n_coords, n_coords) symmetric positive semi-definite matrix.
            Block-diagonal with 3x3 blocks per moving body.
    """
    n = mechanism.state.n_coords
    M = np.zeros((n, n))

    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue

        idx = mechanism.state.get_index(body_id)
        m = body.mass
        Izz = body.Izz_cg

        M[idx.x_idx, idx.x_idx] = m
        M[idx.y_idx, idx.y_idx] = m
        M[idx.theta_idx, idx.theta_idx] = Izz

    return M
