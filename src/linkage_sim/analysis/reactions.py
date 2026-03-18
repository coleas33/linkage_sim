"""Joint reaction and driver effort extraction from Lagrange multipliers.

Maps the raw λ vector back to individual joints and drivers, providing
reaction forces in multiple output formats: global Fx/Fy, resultant
magnitude, body-local coordinates, and radial/tangential decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.constraints import Constraint
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import State
from linkage_sim.solvers.statics import StaticSolveResult


@dataclass(frozen=True)
class JointReaction:
    """Reaction forces at a single joint.

    For a revolute joint (2 equations): Fx, Fy in global frame.
    For a prismatic joint (2 equations): normal force + rotation reaction.
    For a fixed joint (3 equations): Fx, Fy, Mz.
    For a driver (1 equation): scalar effort (torque or force).

    Attributes:
        joint_id: Joint identifier.
        body_i_id: First body.
        body_j_id: Second body.
        lambdas: Raw multiplier values for this joint's constraint rows.
        n_equations: Number of constraint equations.
        force_global: Reaction force in global frame [Fx, Fy] (N).
            For drivers, this is [0, 0].
        moment: Reaction moment (N·m). Nonzero only for fixed joints.
        resultant: Magnitude of the reaction force (N).
        effort: For drivers: required input torque (N·m) or force (N).
            For joints: 0.0.
    """

    joint_id: str
    body_i_id: str
    body_j_id: str
    lambdas: NDArray[np.float64]
    n_equations: int
    force_global: NDArray[np.float64]
    moment: float
    resultant: float
    effort: float


def extract_reactions(
    mechanism: Mechanism,
    static_result: StaticSolveResult,
    q: NDArray[np.float64],
    t: float = 0.0,
) -> list[JointReaction]:
    """Extract individual joint reactions from the static solve result.

    Maps rows of the λ vector back to each joint/driver constraint.
    For revolute joints (2 eq), the two multipliers are the global
    Fx, Fy reaction. For drivers (1 eq), the multiplier is the effort.

    Args:
        mechanism: Built mechanism.
        static_result: Result from solve_statics().
        q: Configuration at which statics were solved.
        t: Time.

    Returns:
        List of JointReaction, one per joint/driver, in mechanism order.
    """
    reactions: list[JointReaction] = []
    row = 0

    for joint in mechanism.joints:
        n_eq = joint.n_equations
        lam = static_result.lambdas[row : row + n_eq]

        # Determine force, moment, effort based on constraint type
        if n_eq == 2:
            # Revolute or prismatic: 2 multipliers → reaction force
            force_global = np.array([float(lam[0]), float(lam[1])])
            moment = 0.0
            effort = 0.0
            resultant = float(np.linalg.norm(force_global))
        elif n_eq == 3:
            # Fixed joint: Fx, Fy, Mz
            force_global = np.array([float(lam[0]), float(lam[1])])
            moment = float(lam[2])
            effort = 0.0
            resultant = float(np.linalg.norm(force_global))
        elif n_eq == 1:
            # Driver: scalar effort
            force_global = np.zeros(2)
            moment = 0.0
            effort = float(lam[0])
            resultant = 0.0
        else:
            # Unknown constraint type: store raw lambdas
            force_global = np.zeros(2)
            moment = 0.0
            effort = 0.0
            resultant = 0.0

        reactions.append(JointReaction(
            joint_id=joint.id,
            body_i_id=joint.body_i_id,
            body_j_id=joint.body_j_id,
            lambdas=lam.copy(),
            n_equations=n_eq,
            force_global=force_global,
            moment=moment,
            resultant=resultant,
            effort=effort,
        ))
        row += n_eq

    return reactions


def get_driver_reactions(
    reactions: list[JointReaction],
) -> list[JointReaction]:
    """Filter reactions to only driver constraints (n_equations == 1)."""
    return [r for r in reactions if r.n_equations == 1]


def get_joint_reactions(
    reactions: list[JointReaction],
) -> list[JointReaction]:
    """Filter reactions to only joint constraints (n_equations > 1)."""
    return [r for r in reactions if r.n_equations > 1]


def reaction_to_local(
    state: State,
    reaction: JointReaction,
    q: NDArray[np.float64],
    body_id: str,
) -> NDArray[np.float64]:
    """Transform a reaction force from global to body-local frame.

    Args:
        state: Mechanism state.
        reaction: Joint reaction with global force.
        q: Configuration vector.
        body_id: Body whose local frame to use.

    Returns:
        [Fx_local, Fy_local] in the body's local coordinate frame.
    """
    if state.is_ground(body_id):
        return reaction.force_global.copy()

    idx = state.get_index(body_id)
    theta = float(q[idx.theta_idx])

    # A^T rotates from global to local
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    fx_g = reaction.force_global[0]
    fy_g = reaction.force_global[1]

    fx_local = cos_t * fx_g + sin_t * fy_g
    fy_local = -sin_t * fx_g + cos_t * fy_g

    return np.array([fx_local, fy_local])
