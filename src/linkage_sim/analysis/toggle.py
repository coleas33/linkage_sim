"""Toggle position and dead point detection.

Detects singular (toggle/dead-point) configurations by monitoring the
smallest singular value of the constraint Jacobian. Near a toggle:
    σ_min → 0, κ(Φ_q) → ∞

This module provides sweep-based detection: compute σ_min across a range
of input angles and identify where it drops below a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_jacobian


@dataclass(frozen=True)
class ToggleDetectionResult:
    """Result of toggle/dead-point detection at one configuration.

    Attributes:
        sigma_min: Smallest singular value of Φ_q.
        condition_number: κ(Φ_q) = σ_max / σ_min.
        is_near_toggle: True if σ_min < threshold.
    """

    sigma_min: float
    condition_number: float
    is_near_toggle: bool


def detect_toggle(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    t: float = 0.0,
    threshold: float = 1e-3,
) -> ToggleDetectionResult:
    """Check if the current configuration is near a toggle/dead point.

    Args:
        mechanism: Built mechanism.
        q: Configuration vector.
        t: Time.
        threshold: σ_min below this is flagged as near-toggle.

    Returns:
        ToggleDetectionResult.
    """
    phi_q = assemble_jacobian(mechanism, q, t)
    sv = np.linalg.svd(phi_q, compute_uv=False)

    if sv.size == 0:
        return ToggleDetectionResult(
            sigma_min=0.0,
            condition_number=float("inf"),
            is_near_toggle=True,
        )

    sigma_min = float(sv[-1])
    sigma_max = float(sv[0])
    condition = sigma_max / sigma_min if sigma_min > 0 else float("inf")

    return ToggleDetectionResult(
        sigma_min=sigma_min,
        condition_number=condition,
        is_near_toggle=sigma_min < threshold,
    )


@dataclass(frozen=True)
class ToggleSweepPoint:
    """One point in a toggle detection sweep.

    Attributes:
        angle: Input angle in radians.
        sigma_min: Smallest singular value at this angle.
        condition_number: Condition number at this angle.
        is_near_toggle: True if near toggle.
    """

    angle: float
    sigma_min: float
    condition_number: float
    is_near_toggle: bool
