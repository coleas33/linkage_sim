"""Position sweep: solve mechanism across a range of input values.

Uses continuation — each solve uses the previous converged solution
as the initial guess for the next step.

The sweep passes input values through the time parameter t, so the
driver constraint f(t) maps them to prescribed motion. For a simple
angle sweep, use f(t) = t (identity driver) so t = angle directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.kinematics import PositionSolveResult, solve_position


@dataclass(frozen=True)
class SweepResult:
    """Result of a position sweep.

    Attributes:
        input_values: Input values at which the solve was attempted.
        solutions: List of q vectors at each converged step (None if failed).
        results: List of PositionSolveResult for each step.
        n_converged: Number of steps that converged.
        n_failed: Number of steps that failed to converge.
    """

    input_values: NDArray[np.float64]
    solutions: list[NDArray[np.float64] | None]
    results: list[PositionSolveResult]
    n_converged: int
    n_failed: int


def position_sweep(
    mechanism: Mechanism,
    q0: NDArray[np.float64],
    input_values: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> SweepResult:
    """Solve the mechanism at each input value using continuation.

    Each input value is passed as the time parameter t to the constraint
    system. The driver constraint f(t) maps t to the prescribed motion.

    For an angle sweep of a revolute driver, use f(t) = t (identity)
    so that input_values are directly the crank angles in radians.

    Args:
        mechanism: A built Mechanism with driver constraint(s).
        q0: Initial guess for the first solve step.
        input_values: Array of input values (passed as t to constraints).
        tol: Newton-Raphson convergence tolerance.
        max_iter: Maximum NR iterations per step.

    Returns:
        SweepResult with solutions at each input value.
    """
    solutions: list[NDArray[np.float64] | None] = []
    results: list[PositionSolveResult] = []
    n_converged = 0
    n_failed = 0

    q_current = q0.copy()

    for val in input_values:
        result = solve_position(
            mechanism, q_current, t=float(val), tol=tol, max_iter=max_iter
        )
        results.append(result)

        if result.converged:
            solutions.append(result.q.copy())
            q_current = result.q.copy()
            n_converged += 1
        else:
            solutions.append(None)
            n_failed += 1

    return SweepResult(
        input_values=input_values,
        solutions=solutions,
        results=results,
        n_converged=n_converged,
        n_failed=n_failed,
    )
