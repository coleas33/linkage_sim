"""Static force solver.

Solves for Lagrange multipliers λ from the static equilibrium equation:

    Φ_q^T * λ = -Q

where Q is the assembled generalized force vector and Φ_q is the
constraint Jacobian. The multipliers λ represent the constraint forces
(joint reactions and driver effort).

For a well-posed system (DOF = 0 with driver), the Jacobian is square
and full-rank. For overconstrained systems, uses minimum-norm
pseudoinverse solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.solvers.assembly import assemble_jacobian


@dataclass(frozen=True)
class StaticSolveResult:
    """Result of a static force solve.

    Attributes:
        lambdas: Lagrange multiplier vector (m,). Each entry corresponds
            to a constraint equation row. For driver constraints, the
            multiplier is the required input effort (torque or force).
        Q: Assembled generalized force vector (n_coords,).
        residual_norm: ||Φ_q^T * λ + Q|| — should be near zero for
            a successful solve.
        is_overconstrained: True if pseudoinverse was used.
        condition_number: Condition number of Φ_q^T (or Φ_q).
    """

    lambdas: NDArray[np.float64]
    Q: NDArray[np.float64]
    residual_norm: float
    is_overconstrained: bool
    condition_number: float


def solve_statics(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    force_elements: list[ForceElement],
    t: float = 0.0,
) -> StaticSolveResult:
    """Solve static equilibrium for constraint forces.

    Given a kinematically-determined configuration q and applied forces Q,
    solve Φ_q^T * λ = -Q for the Lagrange multipliers λ.

    The multipliers represent:
    - For joint constraints: reaction forces at that joint
    - For driver constraints: required input torque (revolute) or force (prismatic)

    Args:
        mechanism: A built Mechanism instance.
        q: Generalized coordinate vector at the solved position.
        force_elements: List of force elements producing applied loads.
        t: Current time.

    Returns:
        StaticSolveResult with multipliers, forces, and diagnostics.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before static solve.")

    # Assemble Φ_q and Q
    phi_q = assemble_jacobian(mechanism, q, t)
    q_dot = np.zeros(mechanism.state.n_coords)
    Q = assemble_Q(mechanism.state, force_elements, q, q_dot, t)

    # Φ_q^T is (n_coords × m). We solve Φ_q^T * λ = -Q
    phi_q_T = phi_q.T
    m = phi_q.shape[0]
    n = phi_q.shape[1]

    # Check conditioning
    sv = np.linalg.svd(phi_q, compute_uv=False)
    if sv.size > 0 and sv[0] > 0:
        rank_tol = 1e-10 * sv[0]
        rank = int(np.sum(sv > rank_tol))
        sigma_min = sv[min(rank - 1, sv.size - 1)] if rank > 0 else 0.0
        condition = float(sv[0] / sigma_min) if sigma_min > 0 else float("inf")
    else:
        rank = 0
        condition = float("inf")

    is_overconstrained = rank < m

    # Solve using lstsq (handles overconstrained via pseudoinverse)
    lstsq_result = np.linalg.lstsq(phi_q_T, -Q, rcond=None)
    lambdas: NDArray[np.float64] = np.asarray(lstsq_result[0], dtype=np.float64)

    # Compute residual
    residual = phi_q_T @ lambdas + Q
    residual_norm = float(np.linalg.norm(residual))

    return StaticSolveResult(
        lambdas=lambdas,
        Q=Q,
        residual_norm=residual_norm,
        is_overconstrained=is_overconstrained,
        condition_number=condition,
    )
