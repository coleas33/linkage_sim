"""Minimal Matplotlib viewer for mechanism visualization.

Renders bodies as line segments between attachment points,
joints as circles, ground pivots as triangles, and optional
coupler point traces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


def plot_mechanism(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    ax: Any | None = None,
    show_joints: bool = True,
    show_ground: bool = True,
    body_color: str = "steelblue",
    joint_color: str = "red",
    ground_color: str = "black",
    linewidth: float = 2.0,
    joint_size: float = 6.0,
) -> Any:
    """Plot the mechanism at configuration q.

    Draws each body as line segments connecting its attachment points
    (in order of definition). Joints are marked with circles. Ground
    pivots are marked with triangles.

    Args:
        mechanism: A built Mechanism.
        q: Generalized coordinate vector.
        ax: Matplotlib Axes. If None, creates a new figure.
        show_joints: Draw joint markers.
        show_ground: Draw ground markers.
        body_color: Color for body line segments.
        joint_color: Color for joint markers.
        ground_color: Color for ground markers.
        linewidth: Line width for bodies.
        joint_size: Marker size for joints.

    Returns:
        The Matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    state = mechanism.state

    # Draw each body as lines between attachment points
    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue

        pts_global = []
        for pt_local in body.attachment_points.values():
            pt_g = state.body_point_global(body_id, pt_local, q)
            pts_global.append(pt_g)

        if len(pts_global) >= 2:
            xs = [p[0] for p in pts_global]
            ys = [p[1] for p in pts_global]
            ax.plot(xs, ys, "-o", color=body_color, linewidth=linewidth,
                    markersize=3, zorder=2)

    # Draw joint locations
    if show_joints:
        for joint in mechanism.joints:
            # Get the global position of the joint (body_i's point)
            body_i = mechanism.bodies[joint.body_i_id]

            from linkage_sim.core.constraints import (
                FixedJoint,
                PrismaticJoint,
                RevoluteJoint,
            )

            if isinstance(joint, (RevoluteJoint, FixedJoint, PrismaticJoint)):
                pt_local = body_i.get_attachment_point(joint._point_i_name)
                pt_global = state.body_point_global(joint.body_i_id, pt_local, q)
                if isinstance(joint, RevoluteJoint):
                    marker = "o"
                elif isinstance(joint, PrismaticJoint):
                    marker = "D"
                else:
                    marker = "s"
                ax.plot(
                    pt_global[0], pt_global[1],
                    marker, color=joint_color, markersize=joint_size,
                    zorder=3,
                )
            # Drivers don't have spatial positions to plot

    # Draw ground pivot markers
    if show_ground:
        ground = mechanism.bodies.get(GROUND_ID)
        if ground is not None:
            for pt_name, pt_local in ground.attachment_points.items():
                pt_global = state.body_point_global(GROUND_ID, pt_local, q)
                ax.plot(
                    pt_global[0], pt_global[1],
                    "^", color=ground_color, markersize=8, zorder=4,
                )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Mechanism Configuration")

    return ax


def plot_coupler_trace(
    mechanism: Mechanism,
    solutions: list[NDArray[np.float64] | None],
    body_id: str,
    point_name: str,
    ax: Any | None = None,
    color: str = "green",
    linewidth: float = 1.5,
) -> Any:
    """Plot the trace of a coupler point across a sweep.

    Args:
        mechanism: A built Mechanism.
        solutions: List of q vectors (None entries are skipped).
        body_id: Body on which the coupler point lives.
        point_name: Name of the coupler point.
        ax: Matplotlib Axes. If None, creates a new figure.
        color: Trace color.
        linewidth: Trace line width.

    Returns:
        The Matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    body = mechanism.bodies[body_id]
    pt_local = body.coupler_points[point_name]

    xs = []
    ys = []
    for q in solutions:
        if q is not None:
            pt_g = mechanism.state.body_point_global(body_id, pt_local, q)
            xs.append(pt_g[0])
            ys.append(pt_g[1])

    ax.plot(xs, ys, "-", color=color, linewidth=linewidth, label=f"{body_id}.{point_name}")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax
