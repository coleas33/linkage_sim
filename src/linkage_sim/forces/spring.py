"""Linear spring force element.

A spring connecting two points on two bodies, producing a force along the
line between the points: F = -k * (length - free_length) + preload.

Supports tension-only, compression-only, or both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State
from linkage_sim.forces.helpers import point_force_to_Q


class SpringMode(Enum):
    """Spring operating mode."""

    BOTH = "both"                  # resists both tension and compression
    TENSION_ONLY = "tension_only"  # only pulls (goes slack in compression)
    COMPRESSION_ONLY = "compression_only"  # only pushes (disconnects in tension)


@dataclass(frozen=True)
class LinearSpring:
    """Linear spring between two body points.

    Attributes:
        body_i_id: First body ID.
        point_i_local: Attachment point on body_i in local coordinates.
        body_j_id: Second body ID.
        point_j_local: Attachment point on body_j in local coordinates.
        stiffness: Spring constant k in N/m.
        free_length: Natural (unstretched) length in meters.
        preload: Initial force at free length in N (positive = tension).
        mode: Operating mode (both, tension_only, compression_only).
        _id: Unique identifier.
    """

    body_i_id: str
    point_i_local: NDArray[np.float64]
    body_j_id: str
    point_j_local: NDArray[np.float64]
    stiffness: float
    free_length: float
    preload: float = 0.0
    mode: SpringMode = SpringMode.BOTH
    _id: str = "spring"

    @property
    def id(self) -> str:
        """Unique identifier for this force element."""
        return self._id

    def evaluate(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Compute spring generalized forces.

        The spring force acts along the line between the two attachment
        points. Positive force = tension (pulling points together).

        F_spring = k * (length - free_length) + preload

        The force on body_j is directed from j toward i (tension pulls j
        toward i). The force on body_i is the equal-opposite reaction.

        Args:
            state: Mechanism state.
            q: Generalized coordinate vector.
            q_dot: Generalized velocity vector (unused for linear spring).
            t: Current time (unused).

        Returns:
            Generalized force vector Q (n_coords,).
        """
        Q = np.zeros(state.n_coords)

        # Global positions of attachment points
        r_i = state.body_point_global(self.body_i_id, self.point_i_local, q)
        r_j = state.body_point_global(self.body_j_id, self.point_j_local, q)

        # Vector from i to j and current length
        d = r_j - r_i
        length = float(np.linalg.norm(d))

        if length < 1e-15:
            # Zero-length spring: no direction defined, no force
            return Q

        # Unit vector from i toward j
        n_hat = d / length

        # Spring force magnitude (positive = tension)
        extension = length - self.free_length
        force_magnitude = self.stiffness * extension + self.preload

        # Apply mode filter
        if self.mode == SpringMode.TENSION_ONLY and force_magnitude < 0.0:
            return Q  # Spring goes slack
        if self.mode == SpringMode.COMPRESSION_ONLY and force_magnitude > 0.0:
            return Q  # Spring disconnects

        # Force on body_j: directed from j toward i (tension pulls j toward i)
        F_on_j = -force_magnitude * n_hat
        # Force on body_i: equal and opposite
        F_on_i = force_magnitude * n_hat

        Q += point_force_to_Q(state, self.body_i_id, self.point_i_local, F_on_i, q)
        Q += point_force_to_Q(state, self.body_j_id, self.point_j_local, F_on_j, q)

        return Q
