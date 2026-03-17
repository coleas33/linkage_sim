"""Kinematic solvers: position, velocity, and acceleration.

Position: Newton-Raphson on Φ(q, t) = 0
Velocity: linear solve Φ_q * q̇ = -Φ_t
Acceleration: linear solve Φ_q * q̈ = γ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import (
    assemble_constraints,
    assemble_gamma,
    assemble_jacobian,
    assemble_phi_t,
)


@dataclass(frozen=True)
class PositionSolveResult:
    """Result of a kinematic position solve.

    Attributes:
        q: Converged generalized coordinate vector (or last iterate if failed).
        converged: True if the solver converged within tolerance.
        iterations: Number of Newton-Raphson iterations performed.
        residual_norm: Final ‖Φ(q, t)‖ at the returned q.
    """

    q: NDArray[np.float64]
    converged: bool
    iterations: int
    residual_norm: float


def solve_position(
    mechanism: Mechanism,
    q0: NDArray[np.float64],
    t: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> PositionSolveResult:
    """Solve Φ(q, t) = 0 using Newton-Raphson.

    At each iteration:
        Φ_q * Δq = -Φ(q, t)
        q ← q + Δq

    Convergence when ‖Φ(q, t)‖ < tol.

    Args:
        mechanism: A built Mechanism instance.
        q0: Initial guess for generalized coordinates.
        t: Time at which to evaluate constraints.
        tol: Convergence tolerance on ‖Φ‖.
        max_iter: Maximum number of iterations.

    Returns:
        PositionSolveResult with converged q or last iterate.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before solving.")

    q = q0.copy()

    for iteration in range(1, max_iter + 1):
        phi = assemble_constraints(mechanism, q, t)
        residual_norm = float(np.linalg.norm(phi))

        if residual_norm < tol:
            return PositionSolveResult(
                q=q,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
            )

        phi_q = assemble_jacobian(mechanism, q, t)
        delta_q = np.linalg.lstsq(phi_q, -phi, rcond=None)[0]
        q = q + delta_q

    # Final residual check after last iteration
    phi = assemble_constraints(mechanism, q, t)
    residual_norm = float(np.linalg.norm(phi))

    return PositionSolveResult(
        q=q,
        converged=(residual_norm < tol),
        iterations=max_iter,
        residual_norm=residual_norm,
    )


def solve_velocity(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    t: float = 0.0,
) -> NDArray[np.float64]:
    """Solve for velocity: Φ_q * q̇ = -Φ_t.

    This is a single linear solve (no iteration needed).
    Φ_t is nonzero only for driver constraints.

    Args:
        mechanism: A built Mechanism instance.
        q: Converged position vector from solve_position.
        t: Time at which to evaluate.

    Returns:
        q_dot: Generalized velocity vector.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before solving.")

    phi_q = assemble_jacobian(mechanism, q, t)
    phi_t = assemble_phi_t(mechanism, q, t)
    rhs = -phi_t

    q_dot = np.asarray(np.linalg.lstsq(phi_q, rhs, rcond=None)[0], dtype=np.float64)
    return q_dot


def solve_acceleration(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    q_dot: NDArray[np.float64],
    t: float = 0.0,
) -> NDArray[np.float64]:
    """Solve for acceleration: Φ_q * q̈ = γ.

    This is a single linear solve (no iteration needed).
    γ accounts for centripetal, Coriolis-like, and driver terms.

    Args:
        mechanism: A built Mechanism instance.
        q: Converged position vector.
        q_dot: Velocity vector from solve_velocity.
        t: Time at which to evaluate.

    Returns:
        q_ddot: Generalized acceleration vector.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before solving.")

    phi_q = assemble_jacobian(mechanism, q, t)
    gamma = assemble_gamma(mechanism, q, q_dot, t)

    q_ddot = np.asarray(np.linalg.lstsq(phi_q, gamma, rcond=None)[0], dtype=np.float64)
    return q_ddot
