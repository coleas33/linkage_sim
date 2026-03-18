"""Transmission angle computation for linkage mechanisms.

The transmission angle μ is the angle between the coupler and the output
link at their connecting joint. It measures how effectively force is
transmitted through the mechanism — μ = 90° is ideal, μ near 0° or 180°
indicates poor force transmission (near toggle/singularity).

For a 4-bar with links a (crank), b (coupler), c (rocker), d (ground):
    cos μ = (b² + c² - a² - d² + 2ad·cos θ) / (2bc)

Also computes transmission angle from body poses for general mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State


@dataclass(frozen=True)
class TransmissionAngleResult:
    """Transmission angle at a configuration.

    Attributes:
        angle_rad: Transmission angle in radians (0, π).
        angle_deg: Transmission angle in degrees (0, 180).
        deviation_from_ideal: |μ - 90°| in degrees.
        is_poor: True if deviation > 50° (μ < 40° or μ > 140°).
    """

    angle_rad: float
    angle_deg: float
    deviation_from_ideal: float
    is_poor: bool


def transmission_angle_fourbar(
    a: float,
    b: float,
    c: float,
    d: float,
    theta: float,
) -> TransmissionAngleResult:
    """Compute transmission angle for a 4-bar linkage analytically.

    Uses the closed-form formula:
        cos μ = (b² + c² - a² - d² + 2ad·cos θ) / (2bc)

    Args:
        a: Crank (input) length.
        b: Coupler length.
        c: Rocker (output) length.
        d: Ground (frame) length.
        theta: Crank angle in radians.

    Returns:
        TransmissionAngleResult.
    """
    cos_mu = (b**2 + c**2 - a**2 - d**2 + 2 * a * d * np.cos(theta)) / (2 * b * c)
    # Clamp to [-1, 1] for numerical safety
    cos_mu = float(np.clip(cos_mu, -1.0, 1.0))
    mu = float(np.arccos(cos_mu))

    deg = float(np.degrees(mu))
    deviation = abs(deg - 90.0)

    return TransmissionAngleResult(
        angle_rad=mu,
        angle_deg=deg,
        deviation_from_ideal=deviation,
        is_poor=deviation > 50.0,
    )


def transmission_angle_from_poses(
    state: State,
    q: NDArray[np.float64],
    coupler_id: str,
    output_id: str,
    coupler_point_name: str,
    output_point_name: str,
    joint_point_coupler: str,
    joint_point_output: str,
) -> TransmissionAngleResult:
    """Compute transmission angle from body poses for any mechanism.

    The transmission angle is the angle at the joint between the coupler
    and output link, measured between the two link directions.

    This works by getting the global positions of:
    - The joint connecting coupler and output
    - The "other end" of the coupler (back toward input)
    - The "other end" of the output (back toward ground)

    Then computing the angle between the two vectors at the joint.

    Args:
        state: Mechanism state.
        q: Configuration vector.
        coupler_id: Coupler body ID.
        output_id: Output (rocker) body ID.
        coupler_point_name: Attachment point on coupler at the input side.
        output_point_name: Attachment point on output at the ground side.
        joint_point_coupler: Attachment point on coupler at the coupler-output joint.
        joint_point_output: Attachment point on output at the coupler-output joint.

    Returns:
        TransmissionAngleResult.
    """
    from linkage_sim.core.mechanism import Mechanism

    # Get body objects to look up attachment point coordinates
    # We need the local coordinates from the state's registered bodies
    # For now, compute from global positions

    # We need the attachment point local coordinates. The state doesn't store
    # bodies directly, but we can get positions from the state.

    # This function works with global positions directly.
    # The caller provides attachment point names; we need local coords.
    # For a cleaner API, accept local coordinates directly.

    # Actually, let's compute the angle between the two link vectors
    # at the shared joint point.
    raise NotImplementedError(
        "Use transmission_angle_fourbar() for 4-bar mechanisms. "
        "General pose-based transmission angle requires body attachment "
        "point lookup which will be added in a follow-up."
    )
