"""Mechanism animation: frame-by-frame rendering of a position sweep.

Uses matplotlib.animation.FuncAnimation to animate the mechanism
moving through a sequence of solved configurations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


def animate_mechanism(
    mechanism: Mechanism,
    solutions: list[NDArray[np.float64] | None],
    interval: int = 50,
    figsize: tuple[float, float] = (8, 6),
    body_color: str = "steelblue",
    joint_color: str = "red",
    ground_color: str = "black",
    coupler_trace_body: str | None = None,
    coupler_trace_point: str | None = None,
    coupler_trace_color: str = "green",
) -> Any:
    """Create a FuncAnimation of the mechanism through solved positions.

    Args:
        mechanism: A built Mechanism.
        solutions: List of q vectors from a position sweep.
            None entries are skipped (frame not drawn).
        interval: Milliseconds between frames.
        figsize: Figure size.
        body_color: Color for body line segments.
        joint_color: Color for joint markers.
        ground_color: Color for ground markers.
        coupler_trace_body: If set, draw the trace of this coupler point.
        coupler_trace_point: Name of the coupler point to trace.
        coupler_trace_color: Color for the coupler trace.

    Returns:
        matplotlib.animation.FuncAnimation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    state = mechanism.state

    # Compute bounds from all solutions
    all_x: list[float] = []
    all_y: list[float] = []
    for q in solutions:
        if q is None:
            continue
        for body_id, body in mechanism.bodies.items():
            for pt_local in body.attachment_points.values():
                pt_g = state.body_point_global(body_id, pt_local, q)
                all_x.append(float(pt_g[0]))
                all_y.append(float(pt_g[1]))

    if all_x and all_y:
        margin = 0.1 * max(max(all_x) - min(all_x), max(all_y) - min(all_y), 0.01)
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Pre-compute coupler trace
    trace_xs: list[float] = []
    trace_ys: list[float] = []

    # Filter valid frames
    valid_frames = [(i, q) for i, q in enumerate(solutions) if q is not None]

    def update(frame_idx: int) -> list[Any]:
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if all_x and all_y:
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        idx, q = valid_frames[frame_idx]
        ax.set_title(f"Frame {idx + 1}/{len(solutions)}")

        artists: list[Any] = []

        # Draw bodies
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
                (line,) = ax.plot(xs, ys, "-o", color=body_color,
                                  linewidth=2, markersize=3, zorder=2)
                artists.append(line)

        # Draw joints
        from linkage_sim.core.constraints import (
            FixedJoint,
            PrismaticJoint,
            RevoluteJoint,
        )

        for joint in mechanism.joints:
            if isinstance(joint, (RevoluteJoint, FixedJoint, PrismaticJoint)):
                body_i = mechanism.bodies[joint.body_i_id]
                pt_local = body_i.get_attachment_point(joint._point_i_name)
                pt_global = state.body_point_global(joint.body_i_id, pt_local, q)
                if isinstance(joint, RevoluteJoint):
                    marker = "o"
                elif isinstance(joint, PrismaticJoint):
                    marker = "D"
                else:
                    marker = "s"
                (pt,) = ax.plot(pt_global[0], pt_global[1], marker,
                                color=joint_color, markersize=6, zorder=3)
                artists.append(pt)

        # Draw ground
        ground = mechanism.bodies.get(GROUND_ID)
        if ground is not None:
            for pt_local in ground.attachment_points.values():
                pt_global = state.body_point_global(GROUND_ID, pt_local, q)
                (pt,) = ax.plot(pt_global[0], pt_global[1], "^",
                                color=ground_color, markersize=8, zorder=4)
                artists.append(pt)

        # Coupler trace
        if coupler_trace_body and coupler_trace_point:
            body = mechanism.bodies[coupler_trace_body]
            pt_local = body.coupler_points[coupler_trace_point]
            pt_g = state.body_point_global(coupler_trace_body, pt_local, q)
            trace_xs.append(float(pt_g[0]))
            trace_ys.append(float(pt_g[1]))
            (trace_line,) = ax.plot(
                trace_xs, trace_ys, "-", color=coupler_trace_color,
                linewidth=1.5, zorder=1,
            )
            artists.append(trace_line)

        return artists

    anim = FuncAnimation(
        fig, update, frames=len(valid_frames), interval=interval, blit=False,
    )

    return anim
