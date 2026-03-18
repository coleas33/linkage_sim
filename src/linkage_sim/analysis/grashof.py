"""Grashof condition check for 4-bar linkages.

Classifies a 4-bar linkage based on the Grashof criterion:
    S + L <= P + Q

where S = shortest link, L = longest link, P and Q are the other two.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GrashofType(Enum):
    """Classification of a 4-bar linkage by Grashof condition."""

    GRASHOF_CRANK_ROCKER = "crank_rocker"
    GRASHOF_DOUBLE_CRANK = "double_crank"
    GRASHOF_DOUBLE_ROCKER = "double_rocker"
    CHANGE_POINT = "change_point"
    NON_GRASHOF = "non_grashof"


@dataclass(frozen=True)
class GrashofResult:
    """Result of Grashof condition analysis.

    Attributes:
        link_lengths: Ordered as [ground, crank, coupler, rocker].
        shortest: Length of shortest link.
        longest: Length of longest link.
        grashof_sum: S + L.
        other_sum: P + Q.
        is_grashof: True if S + L <= P + Q.
        is_change_point: True if S + L == P + Q (within tolerance).
        classification: GrashofType enum.
        shortest_is: Which link is shortest ('ground', 'crank', 'coupler', 'rocker').
    """

    link_lengths: tuple[float, float, float, float]
    shortest: float
    longest: float
    grashof_sum: float
    other_sum: float
    is_grashof: bool
    is_change_point: bool
    classification: GrashofType
    shortest_is: str


def check_grashof(
    ground_length: float,
    crank_length: float,
    coupler_length: float,
    rocker_length: float,
    tol: float = 1e-10,
) -> GrashofResult:
    """Classify a 4-bar linkage by the Grashof condition.

    The four link lengths are: ground (fixed frame distance between pivots),
    crank (input), coupler (connecting), rocker (output).

    Classification rules:
    - If S + L > P + Q: NON_GRASHOF (no link can fully rotate)
    - If S + L == P + Q: CHANGE_POINT (special case, can lock)
    - If S + L < P + Q:
        - Shortest is ground → DOUBLE_CRANK
        - Shortest is crank or rocker → CRANK_ROCKER
        - Shortest is coupler → DOUBLE_ROCKER (Grashof type)

    Args:
        ground_length: Distance between ground pivots.
        crank_length: Input link length.
        coupler_length: Connecting link length.
        rocker_length: Output link length.
        tol: Tolerance for change-point detection.

    Returns:
        GrashofResult with classification.
    """
    links = {
        "ground": ground_length,
        "crank": crank_length,
        "coupler": coupler_length,
        "rocker": rocker_length,
    }

    lengths = sorted(links.values())
    s = lengths[0]
    l_val = lengths[3]
    p = lengths[1]
    q_val = lengths[2]

    grashof_sum = s + l_val
    other_sum = p + q_val

    # Find which link is shortest
    shortest_name = min(links, key=links.get)  # type: ignore[arg-type]

    is_change_point = abs(grashof_sum - other_sum) < tol
    is_grashof = grashof_sum <= other_sum + tol

    if is_change_point:
        classification = GrashofType.CHANGE_POINT
    elif not is_grashof:
        classification = GrashofType.NON_GRASHOF
    else:
        # Grashof: classify by which link is shortest
        if shortest_name == "ground":
            classification = GrashofType.GRASHOF_DOUBLE_CRANK
        elif shortest_name in ("crank", "rocker"):
            classification = GrashofType.GRASHOF_CRANK_ROCKER
        else:  # coupler
            classification = GrashofType.GRASHOF_DOUBLE_ROCKER

    return GrashofResult(
        link_lengths=(ground_length, crank_length, coupler_length, rocker_length),
        shortest=s,
        longest=l_val,
        grashof_sum=grashof_sum,
        other_sum=other_sum,
        is_grashof=is_grashof,
        is_change_point=is_change_point,
        classification=classification,
        shortest_is=shortest_name,
    )
