"""Crank selection analysis: recommend which link to drive for maximum rotation.

For 4-bar mechanisms, uses Grashof classification to determine the optimal
crank analytically. For general mechanisms (6-bar, etc.), uses numerical
probing to estimate the valid angular range of each candidate driven link.

The user can always override the recommendation by specifying their own
revolute driver -- these functions are advisory, not prescriptive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkage_sim.analysis.grashof import GrashofType, check_grashof
from linkage_sim.core.constraints import RevoluteJoint
from linkage_sim.core.drivers import RevoluteDriver
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


@dataclass(frozen=True)
class CrankRecommendation:
    """Recommendation for which link to drive in a mechanism.

    Attributes:
        body_id: ID of the recommended crank body.
        estimated_range_deg: Estimated angular range in degrees (360 = full rotation).
        full_rotation: True if the link can complete a full revolution.
        reason: Human-readable explanation of why this recommendation was made.
    """

    body_id: str
    estimated_range_deg: float
    full_rotation: bool
    reason: str


def detect_fourbar_topology(
    mechanism: Mechanism,
) -> dict[str, object] | None:
    """Detect whether a mechanism has 4-bar topology and extract its parameters.

    A valid 4-bar has:
    - Exactly 3 moving bodies (plus ground)
    - Exactly 4 RevoluteJoint instances (not counting RevoluteDriver)
    - Exactly 2 of those joints connect to ground

    Args:
        mechanism: A Mechanism instance (built or unbuilt, with or without driver).

    Returns:
        A dict with topology info if the mechanism is a 4-bar, or None otherwise.
        Keys: ground_length, ground_adjacent (dict body_id->link_length),
              coupler_id, coupler_length, driven_body_id (or None if no driver).
    """
    bodies = mechanism.bodies
    moving_body_ids = [bid for bid in bodies if bid != GROUND_ID]

    # Must have exactly 3 moving bodies
    if len(moving_body_ids) != 3:
        return None

    # Count RevoluteJoint instances (not RevoluteDriver)
    revolute_joints: list[RevoluteJoint] = [
        j for j in mechanism.joints if isinstance(j, RevoluteJoint)
    ]
    if len(revolute_joints) != 4:
        return None

    # Find joints that connect to ground
    ground_joints: list[RevoluteJoint] = [
        j
        for j in revolute_joints
        if j.body_i_id == GROUND_ID or j.body_j_id == GROUND_ID
    ]
    if len(ground_joints) != 2:
        return None

    # Identify ground-adjacent bodies (connected to ground via revolute joint)
    ground_adjacent: dict[str, float] = {}
    ground_attachment_points: list[str] = []

    for gj in ground_joints:
        if gj.body_i_id == GROUND_ID:
            adjacent_body_id = gj.body_j_id
            ground_pt_name = gj._point_i_name
        else:
            adjacent_body_id = gj.body_i_id
            ground_pt_name = gj._point_j_name

        # Compute link length from attachment points
        body = bodies[adjacent_body_id]
        pts = list(body.attachment_points.values())
        link_length = float(np.linalg.norm(pts[1] - pts[0]))
        ground_adjacent[adjacent_body_id] = link_length
        ground_attachment_points.append(ground_pt_name)

    # Compute ground length from first two ground attachment points
    ground_body = bodies[GROUND_ID]
    ground_pts = list(ground_body.attachment_points.values())
    ground_length = float(np.linalg.norm(ground_pts[1] - ground_pts[0]))

    # Identify coupler (the moving body not connected to ground)
    coupler_ids = [bid for bid in moving_body_ids if bid not in ground_adjacent]
    assert len(coupler_ids) == 1
    coupler_id = coupler_ids[0]

    coupler_body = bodies[coupler_id]
    coupler_pts = list(coupler_body.attachment_points.values())
    coupler_length = float(np.linalg.norm(coupler_pts[1] - coupler_pts[0]))

    # Detect driven body from RevoluteDriver
    driven_body_id: str | None = None
    for j in mechanism.joints:
        if isinstance(j, RevoluteDriver):
            # The driven body is the non-ground body in the driver
            if j.body_i_id == GROUND_ID:
                driven_body_id = j.body_j_id
            elif j.body_j_id == GROUND_ID:
                driven_body_id = j.body_i_id
            else:
                # Driver between two moving bodies; pick body_j as driven
                driven_body_id = j.body_j_id
            break

    return {
        "ground_length": ground_length,
        "ground_adjacent": ground_adjacent,
        "coupler_id": coupler_id,
        "coupler_length": coupler_length,
        "driven_body_id": driven_body_id,
    }


def _estimate_driven_range_analytical(
    ground_len: float,
    crank_len: float,
    coupler_len: float,
    rocker_len: float,
) -> float:
    """Estimate the angular range of the crank that produces valid assembly.

    Scans 3600 evenly spaced crank angles and counts how many produce a
    valid diagonal distance d_tip that satisfies the triangle inequality
    with the coupler and rocker.

    Args:
        ground_len: Distance between ground pivots.
        crank_len: Length of the candidate crank link.
        coupler_len: Length of the coupler link.
        rocker_len: Length of the rocker link.

    Returns:
        Estimated angular range in degrees (0 to 360).
    """
    n_samples = 3600
    thetas = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)

    # Distance from ground pivot to crank tip
    d_tip = np.sqrt(
        crank_len**2
        + ground_len**2
        - 2.0 * crank_len * ground_len * np.cos(thetas)
    )

    # Triangle inequality for coupler + rocker to close the loop
    lower = np.abs(coupler_len - rocker_len)
    upper = coupler_len + rocker_len

    valid_count = int(np.sum((d_tip >= lower) & (d_tip <= upper)))
    return (valid_count / n_samples) * 360.0


def recommend_crank_fourbar(mechanism: Mechanism) -> list[CrankRecommendation]:
    """Recommend which ground-adjacent link to drive for maximum rotation.

    Uses Grashof classification to determine the optimal crank for 4-bar
    mechanisms. Returns recommendations ranked best-first (full rotation
    candidates first, then by range descending).

    Args:
        mechanism: A Mechanism instance (built or unbuilt).

    Returns:
        List of CrankRecommendation sorted best-first.
        Empty list if the mechanism is not a 4-bar.
    """
    topo = detect_fourbar_topology(mechanism)
    if topo is None:
        return []

    ground_length: float = topo["ground_length"]  # type: ignore[assignment]
    ground_adjacent: dict[str, float] = topo["ground_adjacent"]  # type: ignore[assignment]
    coupler_length: float = topo["coupler_length"]  # type: ignore[assignment]

    recommendations: list[CrankRecommendation] = []

    for body_id, link_length in ground_adjacent.items():
        # Determine which is "crank" and which is "rocker" for this trial
        other_ids = [bid for bid in ground_adjacent if bid != body_id]
        assert len(other_ids) == 1
        other_length = ground_adjacent[other_ids[0]]

        # Call check_grashof with this body as the "crank" position
        result = check_grashof(
            ground_length=ground_length,
            crank_length=link_length,
            coupler_length=coupler_length,
            rocker_length=other_length,
        )

        if result.classification == GrashofType.GRASHOF_CRANK_ROCKER:
            # Full rotation only if shortest link is the crank
            if result.shortest_is == "crank":
                recommendations.append(
                    CrankRecommendation(
                        body_id=body_id,
                        estimated_range_deg=360.0,
                        full_rotation=True,
                        reason=(
                            f"Grashof crank-rocker: '{body_id}' is the shortest "
                            f"link ({link_length}) and can fully rotate."
                        ),
                    )
                )
            else:
                # Crank-rocker but this link is the rocker side
                est_range = _estimate_driven_range_analytical(
                    ground_length, link_length, coupler_length, other_length
                )
                recommendations.append(
                    CrankRecommendation(
                        body_id=body_id,
                        estimated_range_deg=est_range,
                        full_rotation=False,
                        reason=(
                            f"Grashof crank-rocker: '{body_id}' is not the "
                            f"shortest link; limited to ~{est_range:.1f} deg."
                        ),
                    )
                )
        elif result.classification == GrashofType.GRASHOF_DOUBLE_CRANK:
            recommendations.append(
                CrankRecommendation(
                    body_id=body_id,
                    estimated_range_deg=360.0,
                    full_rotation=True,
                    reason=(
                        f"Grashof double-crank: ground is shortest link, "
                        f"'{body_id}' can fully rotate."
                    ),
                )
            )
        elif result.classification == GrashofType.CHANGE_POINT:
            recommendations.append(
                CrankRecommendation(
                    body_id=body_id,
                    estimated_range_deg=360.0,
                    full_rotation=True,
                    reason=(
                        f"Change-point: '{body_id}' can fully rotate "
                        f"(may lock at dead points)."
                    ),
                )
            )
        else:
            # GRASHOF_DOUBLE_ROCKER or NON_GRASHOF: limited range
            est_range = _estimate_driven_range_analytical(
                ground_length, link_length, coupler_length, other_length
            )
            recommendations.append(
                CrankRecommendation(
                    body_id=body_id,
                    estimated_range_deg=est_range,
                    full_rotation=False,
                    reason=(
                        f"'{body_id}' cannot fully rotate; "
                        f"estimated range ~{est_range:.1f} deg."
                    ),
                )
            )

    # Sort: full_rotation first, then by range descending
    recommendations.sort(
        key=lambda r: (not r.full_rotation, -r.estimated_range_deg)
    )

    return recommendations
