"""Inverse dynamics solver.

Given a prescribed motion (q, q_dot, q_ddot) and applied forces Q,
solve for the constraint forces (Lagrange multipliers) that enforce
the constraints while accounting for inertial loads:

    Φ_q^T * λ = Q - M * q̈

This extends static analysis to include acceleration effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.solvers.assembly import assemble_jacobian
from linkage_sim.solvers.mass_matrix import assemble_mass_matrix


@dataclass(frozen=True)
class InverseDynamicsResult:
    """Result of an inverse dynamics solve.

    Attributes:
        lambdas: Lagrange multiplier vector (m,).
        Q: Assembled applied generalized force vector (n_coords,).
        M_q_ddot: Inertial force vector M * q̈ (n_coords,).
        residual_norm: ||Φ_q^T * λ - (Q - M*q̈)||.
        condition_number: Condition number of Φ_q.
    """

    lambdas: NDArray[np.float64]
    Q: NDArray[np.float64]
    M_q_ddot: NDArray[np.float64]
    residual_norm: float
    condition_number: float


def solve_inverse_dynamics(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    q_ddot: NDArray[np.float64],
    force_elements: list[ForceElement],
    t: float = 0.0,
) -> InverseDynamicsResult:
    """Solve inverse dynamics for constraint forces.

    Φ_q^T * λ = Q - M * q̈

    The RHS now includes inertial loads (M * q̈). The multiplier for
    the driver constraint gives the required input torque including
    inertial effects.

    Args:
        mechanism: Built mechanism.
        q: Position vector.
        q_dot: Velocity vector.
        q_ddot: Acceleration vector.
        force_elements: Applied force elements.
        t: Time.

    Returns:
        InverseDynamicsResult.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before inverse dynamics.")

    phi_q = assemble_jacobian(mechanism, q, t)
    phi_q_T = phi_q.T

    Q = assemble_Q(mechanism.state, force_elements, q, q_dot, t)
    M = assemble_mass_matrix(mechanism, q)
    M_q_ddot = M @ q_ddot

    # RHS = Q - M * q̈
    rhs = -(Q - M_q_ddot)

    # Conditioning
    sv = np.linalg.svd(phi_q, compute_uv=False)
    if sv.size > 0 and sv[-1] > 0:
        condition = float(sv[0] / sv[-1])
    else:
        condition = float("inf")

    # Solve
    lstsq_result = np.linalg.lstsq(phi_q_T, rhs, rcond=None)
    lambdas: NDArray[np.float64] = np.asarray(lstsq_result[0], dtype=np.float64)

    # Residual
    residual = phi_q_T @ lambdas - rhs
    residual_norm = float(np.linalg.norm(residual))

    return InverseDynamicsResult(
        lambdas=lambdas,
        Q=Q,
        M_q_ddot=M_q_ddot,
        residual_norm=residual_norm,
        condition_number=condition,
    )
