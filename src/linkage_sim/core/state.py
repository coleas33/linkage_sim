"""Generalized coordinate vector q and body-to-index mapping.

Each moving body contributes 3 coordinates to q: (x, y, θ).
Ground is excluded from q — its pose is fixed at (0, 0, 0).

This module is the single source of truth for coordinate bookkeeping.
All code that needs body poses should go through State, not raw index math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BodyIndex:
    """Index mapping for one body's coordinates in the state vector q."""

    body_id: str
    q_start: int  # index of x in q

    @property
    def x_idx(self) -> int:
        return self.q_start

    @property
    def y_idx(self) -> int:
        return self.q_start + 1

    @property
    def theta_idx(self) -> int:
        return self.q_start + 2


GROUND_ID = "ground"


@dataclass
class State:
    """Generalized coordinate state for a mechanism.

    Manages the mapping between body IDs and their positions in q.
    Provides accessors so the rest of the codebase never does raw
    index arithmetic on q.
    """

    _body_indices: dict[str, BodyIndex] = field(default_factory=dict)
    _n_moving_bodies: int = 0

    @property
    def n_coords(self) -> int:
        """Total number of generalized coordinates."""
        return 3 * self._n_moving_bodies

    @property
    def n_moving_bodies(self) -> int:
        return self._n_moving_bodies

    @property
    def body_ids(self) -> list[str]:
        """Ordered list of moving body IDs (same order as in q)."""
        return sorted(
            self._body_indices.keys(),
            key=lambda bid: self._body_indices[bid].q_start,
        )

    def register_body(self, body_id: str) -> BodyIndex:
        """Register a moving body and assign it a coordinate block in q.

        Bodies must be registered before any solve. Ground is never registered.

        Raises:
            ValueError: if body_id is ground or already registered.
        """
        if body_id == GROUND_ID:
            raise ValueError("Ground body must not be registered in the state vector.")
        if body_id in self._body_indices:
            raise ValueError(f"Body '{body_id}' is already registered.")

        idx = BodyIndex(body_id=body_id, q_start=3 * self._n_moving_bodies)
        self._body_indices[body_id] = idx
        self._n_moving_bodies += 1
        return idx

    def get_index(self, body_id: str) -> BodyIndex:
        """Get the coordinate index mapping for a body.

        Raises:
            KeyError: if body_id is not registered.
            ValueError: if body_id is ground (ground has no state indices).
        """
        if body_id == GROUND_ID:
            raise ValueError(
                "Ground body has no state indices — its pose is fixed at (0, 0, 0)."
            )
        return self._body_indices[body_id]

    def is_ground(self, body_id: str) -> bool:
        return body_id == GROUND_ID

    def get_pose(
        self, body_id: str, q: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Get (x, y, θ) for a body from the state vector.

        For ground, returns (0, 0, 0).
        """
        if body_id == GROUND_ID:
            return (0.0, 0.0, 0.0)
        idx = self.get_index(body_id)
        return (float(q[idx.x_idx]), float(q[idx.y_idx]), float(q[idx.theta_idx]))

    def get_position(
        self, body_id: str, q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get [x, y] position vector for a body from q."""
        if body_id == GROUND_ID:
            return np.zeros(2)
        idx = self.get_index(body_id)
        return np.array([q[idx.x_idx], q[idx.y_idx]])

    def get_angle(self, body_id: str, q: NDArray[np.float64]) -> float:
        """Get θ for a body from q."""
        if body_id == GROUND_ID:
            return 0.0
        idx = self.get_index(body_id)
        return float(q[idx.theta_idx])

    def make_q(self) -> NDArray[np.float64]:
        """Create a zero-initialized state vector of the correct size."""
        return np.zeros(self.n_coords)

    def set_pose(
        self,
        body_id: str,
        q: NDArray[np.float64],
        x: float,
        y: float,
        theta: float,
    ) -> None:
        """Set (x, y, θ) for a body in the state vector.

        Raises:
            ValueError: if body_id is ground.
        """
        if body_id == GROUND_ID:
            raise ValueError("Cannot set pose for ground body.")
        idx = self.get_index(body_id)
        q[idx.x_idx] = x
        q[idx.y_idx] = y
        q[idx.theta_idx] = theta

    def rotation_matrix(self, theta: float) -> NDArray[np.float64]:
        """2x2 rotation matrix A(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def rotation_matrix_derivative(self, theta: float) -> NDArray[np.float64]:
        """2x2 derivative B(θ) = dA/dθ = [[-sin θ, -cos θ], [cos θ, -sin θ]]."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[-s, -c], [c, -s]])

    def body_point_global(
        self,
        body_id: str,
        local_point: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transform a point from body-local to global coordinates.

        r_global = r_body + A(θ) * s_local

        For ground, the local point IS the global point (no transformation).
        """
        if body_id == GROUND_ID:
            return local_point.copy()
        pos = self.get_position(body_id, q)
        theta = self.get_angle(body_id, q)
        A = self.rotation_matrix(theta)
        return pos + A @ local_point

    def body_point_global_derivative(
        self,
        body_id: str,
        local_point: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Partial derivative of global point position w.r.t. θ.

        d(A * s)/dθ = B(θ) * s_local

        Returns a 2-element vector. For ground, returns zeros.
        """
        if body_id == GROUND_ID:
            return np.zeros(2)
        theta = self.get_angle(body_id, q)
        B = self.rotation_matrix_derivative(theta)
        return B @ local_point
