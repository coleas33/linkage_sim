"""Mechanism validation: topology checks, connectivity, and DOF analysis.

Layer 1 (topology) and Layer 2 (constraint analysis) validations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID
from linkage_sim.solvers.assembly import assemble_jacobian


@dataclass(frozen=True)
class GrublerResult:
    """Result of a Grübler DOF calculation.

    Attributes:
        n_moving_bodies: Number of non-ground bodies.
        total_dof_removed: Sum of DOF removed by all joints.
        dof: Grübler mobility count (M = 3*n - Σ dof_removed).
        expected_dof: The expected DOF for comparison (default 1).
        is_warning: True if computed DOF != expected DOF.
    """

    n_moving_bodies: int
    total_dof_removed: int
    dof: int
    expected_dof: int
    is_warning: bool


def grubler_dof(mechanism: Mechanism, expected_dof: int = 1) -> GrublerResult:
    """Compute Grübler DOF count for a mechanism.

    Formula: M = 3 * n_moving_bodies - Σ(DOF removed by each joint)

    This is an *informational sanity check*, not authoritative. Grübler
    can be wrong for mechanisms with redundant constraints or special
    geometric conditions. See VALIDATION.md for details.

    Args:
        mechanism: A built Mechanism instance.
        expected_dof: Expected DOF for comparison (default 1 for
            single-input mechanisms).

    Returns:
        GrublerResult with the computed mobility and warning flag.

    Raises:
        RuntimeError: If the mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before computing DOF.")

    n = mechanism.state.n_moving_bodies
    total_removed = sum(j.dof_removed for j in mechanism.joints)
    dof = 3 * n - total_removed

    return GrublerResult(
        n_moving_bodies=n,
        total_dof_removed=total_removed,
        dof=dof,
        expected_dof=expected_dof,
        is_warning=(dof != expected_dof),
    )


@dataclass(frozen=True)
class JacobianRankResult:
    """Result of a Jacobian rank analysis at a specific configuration.

    Attributes:
        constraint_rank: Numerical rank of Φ_q.
        n_constraints: Total number of constraint equations (rows of Φ_q).
        n_coords: Total number of generalized coordinates (columns of Φ_q).
        instantaneous_mobility: n_coords - constraint_rank.
        singular_values: All singular values of Φ_q (descending order).
        condition_number: σ_max / σ_min (inf if rank-deficient).
        has_redundant_constraints: True if rank < n_constraints.
        grubler_agrees: True if instantaneous mobility == Grübler DOF.
    """

    constraint_rank: int
    n_constraints: int
    n_coords: int
    instantaneous_mobility: int
    singular_values: NDArray[np.float64]
    condition_number: float
    has_redundant_constraints: bool
    grubler_agrees: bool


def jacobian_rank_analysis(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    t: float = 0.0,
    rank_tol: float | None = None,
) -> JacobianRankResult:
    """Analyze the Jacobian rank at configuration q.

    Assembles Φ_q and computes its SVD to determine the numerical rank,
    instantaneous mobility, condition number, and whether constraints
    are redundant.

    This is the authoritative DOF measure at the given configuration,
    unlike Grübler which is purely topological.

    Args:
        mechanism: A built Mechanism instance.
        q: Generalized coordinate vector (n_coords,).
        t: Time (default 0.0).
        rank_tol: Tolerance for treating singular values as zero.
            Default: 1e-10 * max(singular_values).

    Returns:
        JacobianRankResult with rank, mobility, conditioning info.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before rank analysis.")

    phi_q = assemble_jacobian(mechanism, q, t)
    singular_values: NDArray[np.float64] = np.asarray(
        np.linalg.svd(phi_q, compute_uv=False), dtype=np.float64
    )

    # Determine rank tolerance
    if rank_tol is None:
        if singular_values.size > 0 and singular_values[0] > 0:
            rank_tol = 1e-10 * singular_values[0]
        else:
            rank_tol = 1e-10

    constraint_rank = int(np.sum(singular_values > rank_tol))

    n_constraints = phi_q.shape[0]
    n_coords = phi_q.shape[1]
    instantaneous_mobility = n_coords - constraint_rank

    # Condition number
    if constraint_rank == 0:
        condition_number = float("inf")
    else:
        sigma_min = singular_values[constraint_rank - 1]
        if sigma_min > 0:
            condition_number = float(singular_values[0] / sigma_min)
        else:
            condition_number = float("inf")

    has_redundant = constraint_rank < n_constraints

    # Compare with Grübler
    grubler = grubler_dof(mechanism, expected_dof=instantaneous_mobility)

    return JacobianRankResult(
        constraint_rank=constraint_rank,
        n_constraints=n_constraints,
        n_coords=n_coords,
        instantaneous_mobility=instantaneous_mobility,
        singular_values=singular_values,
        condition_number=condition_number,
        has_redundant_constraints=has_redundant,
        grubler_agrees=not grubler.is_warning,
    )


@dataclass(frozen=True)
class ConnectivityResult:
    """Result of a graph connectivity check.

    Attributes:
        is_connected: True if all bodies are reachable from ground.
        reachable_bodies: Set of body IDs reachable from ground.
        disconnected_bodies: Set of body IDs NOT reachable from ground.
        n_components: Number of connected components (1 = fully connected).
    """

    is_connected: bool
    reachable_bodies: frozenset[str]
    disconnected_bodies: frozenset[str]
    n_components: int


def check_connectivity(mechanism: Mechanism) -> ConnectivityResult:
    """Check that all bodies are reachable from ground via joint connections.

    Builds an undirected adjacency graph from joint body pairs and performs
    BFS from the ground node. Bodies not reached are reported as disconnected.

    Args:
        mechanism: A built Mechanism instance.

    Returns:
        ConnectivityResult with reachability info.

    Raises:
        RuntimeError: If mechanism has not been built.
    """
    if not mechanism._built:
        raise RuntimeError("Mechanism must be built before connectivity check.")

    all_body_ids = set(mechanism.bodies.keys())

    # Build adjacency list
    adjacency: dict[str, set[str]] = {bid: set() for bid in all_body_ids}
    for joint in mechanism.joints:
        bi = joint.body_i_id
        bj = joint.body_j_id
        if bi in adjacency and bj in adjacency:
            adjacency[bi].add(bj)
            adjacency[bj].add(bi)

    # BFS from ground
    visited: set[str] = set()
    queue: deque[str] = deque()

    if GROUND_ID in adjacency:
        queue.append(GROUND_ID)
        visited.add(GROUND_ID)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    disconnected = all_body_ids - visited

    # Count total connected components
    remaining = set(all_body_ids)
    n_components = 0
    while remaining:
        n_components += 1
        start = next(iter(remaining))
        component_queue: deque[str] = deque([start])
        while component_queue:
            current = component_queue.popleft()
            if current in remaining:
                remaining.discard(current)
                for neighbor in adjacency[current]:
                    if neighbor in remaining:
                        component_queue.append(neighbor)

    return ConnectivityResult(
        is_connected=len(disconnected) == 0,
        reachable_bodies=frozenset(visited),
        disconnected_bodies=frozenset(disconnected),
        n_components=n_components,
    )
