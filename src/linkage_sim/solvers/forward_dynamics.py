"""Forward dynamics integrator for constrained multibody systems.

Solves the index-3 DAE:
    M * q̈ + Φ_q^T * λ = Q
    Φ(q, t) = 0

Using Baumgarte stabilization to control constraint drift:
    Φ̈ + 2α * Φ̇ + β² * Φ = 0

This converts the DAE into an ODE system that can be integrated with
standard stiff ODE solvers (BDF/Radau via SciPy solve_ivp).

State vector for integration: y = [q, q̇] (2 * n_coords)
At each time step:
1. Assemble M, Φ_q, Q, γ
2. Solve augmented system for [q̈, λ]
3. Return ẏ = [q̇, q̈]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.solvers.assembly import (
    assemble_constraints,
    assemble_gamma,
    assemble_jacobian,
    assemble_phi_t,
)
from linkage_sim.solvers.mass_matrix import assemble_mass_matrix


@dataclass(frozen=True)
class ForwardDynamicsConfig:
    """Configuration for the forward dynamics integrator.

    Attributes:
        alpha: Baumgarte velocity stabilization parameter.
        beta: Baumgarte position stabilization parameter.
        method: SciPy integration method ('Radau', 'BDF', 'RK45').
        rtol: Relative tolerance for the integrator.
        atol: Absolute tolerance for the integrator.
        max_step: Maximum step size.
        project_interval: Apply constraint projection every N steps.
            0 = no projection.
        project_tol: Tolerance for constraint projection Newton-Raphson.
        max_project_iter: Max iterations for constraint projection.
    """

    alpha: float = 5.0
    beta: float = 5.0
    method: str = "Radau"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 0.01
    project_interval: int = 0
    project_tol: float = 1e-10
    max_project_iter: int = 10


@dataclass
class ForwardDynamicsResult:
    """Result of a forward dynamics simulation.

    Attributes:
        t: Time array (N,).
        q: Position history (N, n_coords).
        q_dot: Velocity history (N, n_coords).
        constraint_drift: ||Φ(q, t)|| at each time step (N,).
        success: True if integration completed without error.
        message: Status message from the integrator.
    """

    t: NDArray[np.float64]
    q: NDArray[np.float64]
    q_dot: NDArray[np.float64]
    constraint_drift: NDArray[np.float64]
    success: bool
    message: str


def simulate(
    mechanism: Mechanism,
    q0: NDArray[np.float64],
    q_dot0: NDArray[np.float64],
    t_span: tuple[float, float],
    force_elements: list[ForceElement],
    config: ForwardDynamicsConfig | None = None,
    t_eval: NDArray[np.float64] | None = None,
) -> ForwardDynamicsResult:
    """Run a forward dynamics simulation.

    Args:
        mechanism: Built mechanism (no driver constraints — the system
            must have DOF > 0 for free motion).
        q0: Initial position satisfying Φ(q0, t0) ≈ 0.
        q_dot0: Initial velocity satisfying Φ_q * q̇0 ≈ -Φ_t.
        t_span: (t_start, t_end) time interval.
        force_elements: Applied force elements.
        config: Integration parameters. Uses defaults if None.
        t_eval: Optional array of times at which to store solution.

    Returns:
        ForwardDynamicsResult with time histories.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built.")

    if config is None:
        config = ForwardDynamicsConfig()

    n = mechanism.state.n_coords
    m = mechanism.n_constraints
    # M is assembled inside rhs() since it depends on q

    alpha = config.alpha
    beta = config.beta

    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute dy/dt = [q_dot, q_ddot]."""
        q = y[:n]
        qd = y[n:]

        M = assemble_mass_matrix(mechanism, q)
        phi_q = assemble_jacobian(mechanism, q, t)
        phi = assemble_constraints(mechanism, q, t)
        phi_t = assemble_phi_t(mechanism, q, t)
        gamma = assemble_gamma(mechanism, q, qd, t)
        Q = assemble_Q(mechanism.state, force_elements, q, qd, t)

        # Baumgarte-stabilized acceleration RHS:
        # γ_stab = γ - 2α(Φ_q*q̇ + Φ_t) - β²*Φ
        phi_dot = phi_q @ qd + phi_t
        gamma_stab = gamma - 2.0 * alpha * phi_dot - beta**2 * phi

        # Build augmented system:
        # [M, Φ_q^T] [q̈]   [Q        ]
        # [Φ_q,  0  ] [λ ] = [γ_stab   ]
        A = np.zeros((n + m, n + m))
        A[:n, :n] = M
        A[:n, n:] = phi_q.T
        A[n:, :n] = phi_q

        b = np.zeros(n + m)
        b[:n] = Q
        b[n:] = gamma_stab

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular system — use lstsq as fallback
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        q_ddot = x[:n]

        dy = np.zeros(2 * n)
        dy[:n] = qd
        dy[n:] = q_ddot
        return dy

    # Pack initial state
    y0 = np.concatenate([q0, q_dot0])

    # Integrate
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        t_eval=t_eval,
        dense_output=False,
    )

    # Unpack results
    t_out = np.asarray(sol.t, dtype=np.float64)
    q_out = np.asarray(sol.y[:n, :].T, dtype=np.float64)
    qd_out = np.asarray(sol.y[n:, :].T, dtype=np.float64)

    # Compute constraint drift at each output time
    drift = np.zeros(len(t_out))
    for i in range(len(t_out)):
        phi = assemble_constraints(mechanism, q_out[i], t_out[i])
        drift[i] = float(np.linalg.norm(phi))

    # Apply constraint projection if configured
    if config.project_interval > 0:
        for i in range(len(t_out)):
            if i % config.project_interval == 0 and i > 0:
                q_proj = _project_constraints(
                    mechanism, q_out[i], t_out[i],
                    config.project_tol, config.max_project_iter,
                )
                q_out[i] = q_proj
                phi = assemble_constraints(mechanism, q_proj, t_out[i])
                drift[i] = float(np.linalg.norm(phi))

    return ForwardDynamicsResult(
        t=t_out,
        q=q_out,
        q_dot=qd_out,
        constraint_drift=drift,
        success=bool(sol.success),
        message=str(sol.message),
    )


def _project_constraints(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    t: float,
    tol: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """Newton-Raphson constraint projection."""
    q_proj = q.copy()
    for _ in range(max_iter):
        phi = assemble_constraints(mechanism, q_proj, t)
        if float(np.linalg.norm(phi)) < tol:
            break
        phi_q = assemble_jacobian(mechanism, q_proj, t)
        try:
            dq = np.linalg.lstsq(phi_q, -phi, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        q_proj = q_proj + dq
    return q_proj
