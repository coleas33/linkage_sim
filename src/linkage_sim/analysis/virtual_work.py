"""Virtual work cross-check for static force analysis.

For a 1-DOF mechanism, the required input torque can be computed
independently via the virtual work principle:

    τ_input * δθ_input = -Σ(Q_i * δq_i)

where δq is the virtual displacement from the velocity solution.
This provides a cross-check against the Lagrange multiplier result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.solvers.assembly import assemble_jacobian, assemble_phi_t
from linkage_sim.solvers.kinematics import solve_velocity


@dataclass(frozen=True)
class VirtualWorkResult:
    """Result of virtual work input torque computation.

    Attributes:
        input_torque: Required input torque from virtual work (N·m).
        Q: Assembled generalized force vector.
        q_dot: Velocity solution used for virtual displacements.
        delta_theta_input: Virtual displacement of the input coordinate.
    """

    input_torque: float
    Q: NDArray[np.float64]
    q_dot: NDArray[np.float64]
    delta_theta_input: float


def virtual_work_input_torque(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    force_elements: list[ForceElement],
    input_body_id: str,
    t: float = 0.0,
) -> VirtualWorkResult:
    """Compute required input torque via virtual work principle.

    For a 1-DOF mechanism with a revolute driver, the velocity solution
    gives the ratio of all coordinate velocities to the input velocity.
    The virtual work equation is:

        τ_input = -Σ(Q_applied_i * (q_dot_i / ω_input))

    where Q_applied excludes the driver reaction (which is what we solve for).

    Args:
        mechanism: Built mechanism with a driver.
        q: Solved configuration.
        force_elements: Applied force elements (gravity, springs, etc.).
        input_body_id: The driven body whose θ is the input coordinate.
        t: Time.

    Returns:
        VirtualWorkResult with computed input torque.

    Raises:
        ValueError: If input body not found or is ground.
    """
    if mechanism.state.is_ground(input_body_id):
        raise ValueError("Input body cannot be ground.")

    # Solve velocity to get virtual displacement ratios
    q_dot = solve_velocity(mechanism, q, t)

    # Get input angular velocity
    idx = mechanism.state.get_index(input_body_id)
    omega_input = float(q_dot[idx.theta_idx])

    if abs(omega_input) < 1e-15:
        # At singularity or no motion — cannot compute virtual work
        return VirtualWorkResult(
            input_torque=float("nan"),
            Q=np.zeros(mechanism.state.n_coords),
            q_dot=q_dot,
            delta_theta_input=0.0,
        )

    # Assemble applied forces (excluding driver reactions)
    q_dot_zero = np.zeros(mechanism.state.n_coords)
    Q = assemble_Q(mechanism.state, force_elements, q, q_dot_zero, t)

    # Virtual work: τ_input * δθ_input + Σ(Q_i * δq_i) = 0
    # δq_i = q_dot_i * dt, δθ_input = ω_input * dt
    # So: τ_input = -Σ(Q_i * q_dot_i) / ω_input
    work_rate = float(np.dot(Q, q_dot))
    input_torque = -work_rate / omega_input

    return VirtualWorkResult(
        input_torque=input_torque,
        Q=Q,
        q_dot=q_dot,
        delta_theta_input=omega_input,
    )
