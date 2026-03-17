"""Mechanism validation: topology checks and DOF analysis.

Layer 1 validations that can run instantly on any mechanism definition.
"""

from __future__ import annotations

from dataclasses import dataclass

from linkage_sim.core.mechanism import Mechanism


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
