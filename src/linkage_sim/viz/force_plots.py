"""Force-related plotting utilities.

Plots for input torque vs angle, joint reactions vs angle,
transmission angle vs angle, and spring/damper state vs angle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import matplotlib lazily to avoid hard dependency in non-GUI contexts
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_input_torque(
    angles_deg: NDArray[np.float64],
    torques: NDArray[np.float64],
    title: str = "Required Input Torque vs. Crank Angle",
) -> Any:
    """Plot input torque across a crank angle sweep.

    Args:
        angles_deg: Crank angles in degrees.
        torques: Corresponding input torques in N·m.
        title: Plot title.

    Returns:
        matplotlib Figure, or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_deg, torques, "b-", linewidth=1.5)
    ax.set_xlabel("Crank Angle (deg)")
    ax.set_ylabel("Input Torque (N·m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_joint_reactions(
    angles_deg: NDArray[np.float64],
    reactions: dict[str, NDArray[np.float64]],
    title: str = "Joint Reaction Magnitudes vs. Crank Angle",
) -> Any:
    """Plot joint reaction force magnitudes across a sweep.

    Args:
        angles_deg: Crank angles in degrees.
        reactions: Dict mapping joint_id to reaction magnitude arrays.
        title: Plot title.

    Returns:
        matplotlib Figure, or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for joint_id, magnitudes in reactions.items():
        ax.plot(angles_deg, magnitudes, linewidth=1.5, label=joint_id)
    ax.set_xlabel("Crank Angle (deg)")
    ax.set_ylabel("Reaction Force (N)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_transmission_angle(
    angles_deg: NDArray[np.float64],
    mu_deg: NDArray[np.float64],
    title: str = "Transmission Angle vs. Crank Angle",
    poor_threshold: float = 40.0,
) -> Any:
    """Plot transmission angle with poor-geometry warning band.

    Args:
        angles_deg: Crank angles in degrees.
        mu_deg: Transmission angles in degrees.
        title: Plot title.
        poor_threshold: Angles below this are flagged poor.

    Returns:
        matplotlib Figure, or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_deg, mu_deg, "b-", linewidth=1.5)
    ax.axhline(y=90, color="g", linewidth=0.5, linestyle="--", label="Ideal (90°)")
    ax.axhline(y=poor_threshold, color="r", linewidth=0.5, linestyle="--",
               label=f"Poor (<{poor_threshold}°)")
    ax.axhline(y=180 - poor_threshold, color="r", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Crank Angle (deg)")
    ax.set_ylabel("Transmission Angle (deg)")
    ax.set_title(title)
    ax.set_ylim(0, 180)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
