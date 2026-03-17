"""Rigid body data structures.

Bodies are first-class rigid objects with multiple named attachment points.
A binary bar is a body with two attachment points. A ternary plate is a body
with three. They are the same object type.

All coordinates are in the body's local frame. All units are SI (meters, kg).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import GROUND_ID


@dataclass
class Body:
    """A rigid body in the mechanism.

    Attributes:
        id: Unique identifier.
        attachment_points: Named points in body-local coordinates (meters)
            where joints and force elements can connect.
        mass: Body mass in kg. Zero for ground.
        cg_local: Center of gravity in body-local coordinates (meters).
        Izz_cg: Moment of inertia about z-axis through CG (kg*m^2).
        coupler_points: Named points tracked for output (path tracing,
            velocity/acceleration) but not used for connections.
        render_shape: Optional polygon outline for GUI rendering.
            Does not affect analysis.
    """

    id: str
    attachment_points: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    mass: float = 0.0
    cg_local: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2)
    )
    Izz_cg: float = 0.0
    coupler_points: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    render_shape: list[NDArray[np.float64]] | None = None

    def get_attachment_point(self, name: str) -> NDArray[np.float64]:
        """Get an attachment point's local coordinates.

        Raises:
            KeyError: if the named point doesn't exist on this body.
        """
        if name not in self.attachment_points:
            raise KeyError(
                f"Attachment point '{name}' not found on body '{self.id}'. "
                f"Available: {list(self.attachment_points.keys())}"
            )
        return self.attachment_points[name]

    def add_attachment_point(self, name: str, x: float, y: float) -> None:
        """Add a named attachment point in body-local coordinates (meters)."""
        if name in self.attachment_points:
            raise ValueError(
                f"Attachment point '{name}' already exists on body '{self.id}'."
            )
        self.attachment_points[name] = np.array([x, y])

    def add_coupler_point(self, name: str, x: float, y: float) -> None:
        """Add a named coupler point for output tracking."""
        if name in self.coupler_points:
            raise ValueError(
                f"Coupler point '{name}' already exists on body '{self.id}'."
            )
        self.coupler_points[name] = np.array([x, y])


def make_ground(**attachment_points: tuple[float, float]) -> Body:
    """Create the ground body with named fixed pivot locations.

    Ground is fixed at the global origin with zero mass.
    Attachment points are in global coordinates (since ground doesn't move).

    Usage:
        ground = make_ground(O2=(0.0, 0.0), O4=(0.038, 0.0))
    """
    pts = {name: np.array(coords) for name, coords in attachment_points.items()}
    return Body(
        id=GROUND_ID,
        attachment_points=pts,
        mass=0.0,
        cg_local=np.zeros(2),
        Izz_cg=0.0,
    )


def make_bar(
    body_id: str,
    p1_name: str,
    p2_name: str,
    length: float,
    mass: float = 0.0,
    Izz_cg: float = 0.0,
) -> Body:
    """Create a binary bar (two attachment points along x-axis).

    The bar's local frame origin is at p1, with p2 at (length, 0).
    CG is at the midpoint by default.

    Args:
        body_id: Unique identifier.
        p1_name: Name of the first attachment point (at local origin).
        p2_name: Name of the second attachment point (at (length, 0)).
        length: Distance between attachment points in meters.
        mass: Body mass in kg.
        Izz_cg: Moment of inertia about CG in kg*m^2.
    """
    return Body(
        id=body_id,
        attachment_points={
            p1_name: np.array([0.0, 0.0]),
            p2_name: np.array([length, 0.0]),
        },
        mass=mass,
        cg_local=np.array([length / 2.0, 0.0]),
        Izz_cg=Izz_cg,
    )
