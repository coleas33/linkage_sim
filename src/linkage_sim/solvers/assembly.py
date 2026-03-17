"""Global constraint system assembly from a Mechanism.

Assembles the global Φ, Φ_q, and γ vectors/matrices by stacking
contributions from all joints in the mechanism.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism


def assemble_constraints(
    mechanism: Mechanism, q: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Assemble global constraint residual vector Φ(q, t).

    Returns:
        Φ: (m,) vector where m = total constraint equations.
    """
    m = mechanism.n_constraints
    phi = np.zeros(m)

    row = 0
    for joint in mechanism.joints:
        n_eq = joint.n_equations
        phi[row : row + n_eq] = joint.constraint(mechanism.state, q, t)
        row += n_eq

    return phi


def assemble_jacobian(
    mechanism: Mechanism, q: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    """Assemble global constraint Jacobian Φ_q(q, t).

    Returns:
        Φ_q: (m, n_coords) matrix.
    """
    m = mechanism.n_constraints
    n = mechanism.state.n_coords
    phi_q = np.zeros((m, n))

    row = 0
    for joint in mechanism.joints:
        n_eq = joint.n_equations
        phi_q[row : row + n_eq, :] = joint.jacobian(mechanism.state, q, t)
        row += n_eq

    return phi_q


def assemble_gamma(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Assemble global acceleration RHS vector γ(q, q̇, t).

    Used in: Φ_q * q̈ = γ

    Returns:
        γ: (m,) vector.
    """
    m = mechanism.n_constraints
    gamma = np.zeros(m)

    row = 0
    for joint in mechanism.joints:
        n_eq = joint.n_equations
        gamma[row : row + n_eq] = joint.gamma(mechanism.state, q, q_dot, t)
        row += n_eq

    return gamma
