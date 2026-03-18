"""Interactive Matplotlib viewer for debugging/testing the planar linkage simulator.

This is NOT the production GUI (that will be Rust/egui). This is a thin
debugging and discovery tool to visually verify solver behavior before
the Rust port.

Usage:
    from linkage_sim.viz.interactive_viewer import launch_interactive
    launch_interactive(mechanism, force_elements=[gravity], ...)

Features:
    - Mechanism view panel (left) with bodies, joints, ground markers, coupler trace
    - Angle slider to sweep input angle 0-360 degrees
    - Overlay plots panel (right) with input torque, joint reactions,
      transmission angle (if 4-bar), and vertical position indicator
    - Toggle checkboxes for show/hide bodies, joints, forces, coupler path
    - Info text panel with current angle, driver torque, transmission angle,
      max reaction force
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from linkage_sim.analysis.reactions import (
    JointReaction,
    extract_reactions,
    get_driver_reactions,
    get_joint_reactions,
)
from linkage_sim.analysis.transmission import transmission_angle_fourbar
from linkage_sim.core.constraints import (
    FixedJoint,
    PrismaticJoint,
    RevoluteJoint,
)
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.solvers.kinematics import solve_position, solve_velocity
from linkage_sim.solvers.statics import StaticSolveResult, solve_statics
from linkage_sim.solvers.sweep import position_sweep


# ---------------------------------------------------------------------------
# Pre-computation data container
# ---------------------------------------------------------------------------

@dataclass
class SweepData:
    """Pre-computed sweep results for all angles.

    Stores position solutions, velocity solutions, static results,
    reactions, and derived quantities for every angle in the sweep.
    """

    angles_deg: NDArray[np.float64]
    angles_rad: NDArray[np.float64]
    n_steps: int

    # Per-step results (None where solve failed)
    solutions: list[NDArray[np.float64] | None]
    velocities: list[NDArray[np.float64] | None]
    static_results: list[StaticSolveResult | None]
    reactions: list[list[JointReaction] | None]

    # Derived arrays (NaN where solve failed)
    driver_torques: NDArray[np.float64]
    joint_reaction_mags: dict[str, NDArray[np.float64]]
    transmission_angles_deg: NDArray[np.float64] | None

    # Coupler trace coordinates (filtered to valid solutions only)
    coupler_trace_x: NDArray[np.float64] | None = None
    coupler_trace_y: NDArray[np.float64] | None = None


def _detect_fourbar_link_lengths(
    mechanism: Mechanism,
) -> tuple[float, float, float, float] | None:
    """Detect if the mechanism is a 4-bar and return (a, b, c, d) link lengths.

    Returns None if the mechanism is not a standard 4-bar.
    A standard 4-bar has exactly 3 moving bodies connected by 4 revolute joints
    and 1 revolute driver, with the topology: ground-crank-coupler-rocker-ground.
    """
    moving_ids = [bid for bid in mechanism.bodies if bid != GROUND_ID]
    if len(moving_ids) != 3:
        return None

    revolute_joints = [
        j for j in mechanism.joints if isinstance(j, RevoluteJoint)
    ]
    if len(revolute_joints) != 4:
        return None

    # Find ground-connected joints (one end is ground)
    ground_joints = [
        j for j in revolute_joints
        if j.body_i_id == GROUND_ID or j.body_j_id == GROUND_ID
    ]
    if len(ground_joints) != 2:
        return None

    # Identify crank (ground joint whose non-ground body has the driver)
    from linkage_sim.core.drivers import RevoluteDriver
    drivers = [j for j in mechanism.joints if isinstance(j, RevoluteDriver)]
    if len(drivers) != 1:
        return None

    driver = drivers[0]
    driver_body = driver.body_j_id if driver.body_i_id == GROUND_ID else driver.body_i_id

    # Crank is the body connected to ground that is also driven
    crank_id = None
    rocker_id = None
    for gj in ground_joints:
        other = gj.body_j_id if gj.body_i_id == GROUND_ID else gj.body_i_id
        if other == driver_body:
            crank_id = other
        else:
            rocker_id = other

    if crank_id is None or rocker_id is None:
        return None

    # Coupler is the remaining body
    coupler_candidates = [bid for bid in moving_ids if bid not in (crank_id, rocker_id)]
    if len(coupler_candidates) != 1:
        return None
    coupler_id = coupler_candidates[0]

    # Compute link lengths from attachment points
    crank_body = mechanism.bodies[crank_id]
    coupler_body = mechanism.bodies[coupler_id]
    rocker_body = mechanism.bodies[rocker_id]

    pts_crank = list(crank_body.attachment_points.values())
    pts_coupler = list(coupler_body.attachment_points.values())
    pts_rocker = list(rocker_body.attachment_points.values())

    if len(pts_crank) < 2 or len(pts_coupler) < 2 or len(pts_rocker) < 2:
        return None

    a = float(np.linalg.norm(pts_crank[1] - pts_crank[0]))  # crank length
    b = float(np.linalg.norm(pts_coupler[1] - pts_coupler[0]))  # coupler length
    c = float(np.linalg.norm(pts_rocker[1] - pts_rocker[0]))  # rocker length

    # Ground length: distance between the two ground pivot points
    ground = mechanism.bodies[GROUND_ID]
    ground_pts = list(ground.attachment_points.values())
    if len(ground_pts) < 2:
        return None
    d = float(np.linalg.norm(ground_pts[1] - ground_pts[0]))

    return (a, b, c, d)


def _find_coupler_body_and_point(
    mechanism: Mechanism,
) -> tuple[str, str] | None:
    """Find a body with coupler points for tracing.

    Returns (body_id, point_name) or None if no coupler points exist.
    """
    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue
        if body.coupler_points:
            first_name = next(iter(body.coupler_points))
            return (body_id, first_name)
    return None


def precompute_sweep(
    mechanism: Mechanism,
    force_elements: list[ForceElement],
    n_steps: int = 360,
    coupler_body_id: str | None = None,
    coupler_point_name: str | None = None,
) -> SweepData:
    """Pre-compute a full 0-360 degree sweep of the mechanism.

    Solves position, velocity, and statics at each angle. Also computes
    transmission angle if the mechanism is a 4-bar.

    Args:
        mechanism: A built Mechanism with a revolute driver.
        force_elements: Force elements for static analysis.
        n_steps: Number of angular steps (default 360 for 1-degree resolution).
        coupler_body_id: Body ID for coupler trace (auto-detected if None).
        coupler_point_name: Coupler point name (auto-detected if None).

    Returns:
        SweepData with all pre-computed results.
    """
    angles_deg = np.linspace(0, 360, n_steps, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # --- Position sweep using continuation ---
    # Build an initial guess for the first angle
    q0 = mechanism.state.make_q()
    first_angle = angles_rad[0]

    # Set a reasonable initial guess: bodies at origin with correct angles
    for body_id in mechanism.state.body_ids:
        mechanism.state.set_pose(body_id, q0, 0.0, 0.0, 0.0)

    # Use position sweep for robust continuation
    sweep = position_sweep(mechanism, q0, angles_rad)
    solutions = sweep.solutions

    # --- Velocity + statics at each converged position ---
    velocities: list[NDArray[np.float64] | None] = []
    static_results: list[StaticSolveResult | None] = []
    all_reactions: list[list[JointReaction] | None] = []
    driver_torques = np.full(n_steps, np.nan)

    # Collect joint IDs for reaction tracking
    joint_ids_for_reactions: list[str] = []
    for joint in mechanism.joints:
        if isinstance(joint, (RevoluteJoint, FixedJoint, PrismaticJoint)):
            joint_ids_for_reactions.append(joint.id)

    joint_reaction_mags: dict[str, NDArray[np.float64]] = {
        jid: np.full(n_steps, np.nan) for jid in joint_ids_for_reactions
    }

    for i, q in enumerate(solutions):
        if q is None:
            velocities.append(None)
            static_results.append(None)
            all_reactions.append(None)
            continue

        t = float(angles_rad[i])

        # Velocity
        q_dot = solve_velocity(mechanism, q, t=t)
        velocities.append(q_dot)

        # Statics
        if force_elements:
            sr = solve_statics(mechanism, q, force_elements, t=t)
            static_results.append(sr)

            rxns = extract_reactions(mechanism, sr, q, t=t)
            all_reactions.append(rxns)

            # Extract driver torque
            driver_rxns = get_driver_reactions(rxns)
            if driver_rxns:
                driver_torques[i] = driver_rxns[0].effort

            # Extract joint reaction magnitudes
            joint_rxns = get_joint_reactions(rxns)
            for jr in joint_rxns:
                if jr.joint_id in joint_reaction_mags:
                    joint_reaction_mags[jr.joint_id][i] = jr.resultant
        else:
            static_results.append(None)
            all_reactions.append(None)

    # --- Transmission angle (4-bar only) ---
    fourbar_lengths = _detect_fourbar_link_lengths(mechanism)
    transmission_angles_deg: NDArray[np.float64] | None = None
    if fourbar_lengths is not None:
        a, b, c, d = fourbar_lengths
        transmission_angles_deg = np.full(n_steps, np.nan)
        for i in range(n_steps):
            if solutions[i] is not None:
                result = transmission_angle_fourbar(a, b, c, d, angles_rad[i])
                transmission_angles_deg[i] = result.angle_deg

    # --- Coupler trace ---
    coupler_trace_x: NDArray[np.float64] | None = None
    coupler_trace_y: NDArray[np.float64] | None = None

    # Auto-detect coupler if not specified
    if coupler_body_id is None or coupler_point_name is None:
        detected = _find_coupler_body_and_point(mechanism)
        if detected is not None:
            coupler_body_id, coupler_point_name = detected

    if coupler_body_id is not None and coupler_point_name is not None:
        body = mechanism.bodies[coupler_body_id]
        pt_local = body.coupler_points[coupler_point_name]
        xs = []
        ys = []
        for q in solutions:
            if q is not None:
                pt_g = mechanism.state.body_point_global(coupler_body_id, pt_local, q)
                xs.append(float(pt_g[0]))
                ys.append(float(pt_g[1]))
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        coupler_trace_x = np.array(xs)
        coupler_trace_y = np.array(ys)

    return SweepData(
        angles_deg=np.asarray(angles_deg, dtype=np.float64),
        angles_rad=angles_rad,
        n_steps=n_steps,
        solutions=solutions,
        velocities=velocities,
        static_results=static_results,
        reactions=all_reactions,
        driver_torques=driver_torques,
        joint_reaction_mags=joint_reaction_mags,
        transmission_angles_deg=transmission_angles_deg,
        coupler_trace_x=coupler_trace_x,
        coupler_trace_y=coupler_trace_y,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_bodies(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    ax: Any,
    color: str = "steelblue",
    linewidth: float = 2.0,
) -> list[Any]:
    """Draw bodies as line segments between attachment points."""
    artists = []
    state = mechanism.state
    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue
        pts_global = []
        for pt_local in body.attachment_points.values():
            pt_g = state.body_point_global(body_id, pt_local, q)
            pts_global.append(pt_g)
        if len(pts_global) >= 2:
            xs = [float(p[0]) for p in pts_global]
            ys = [float(p[1]) for p in pts_global]
            (line,) = ax.plot(
                xs, ys, "-o", color=color, linewidth=linewidth,
                markersize=3, zorder=2,
            )
            artists.append(line)
    return artists


def _draw_joints(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    ax: Any,
    color: str = "red",
    size: float = 6.0,
) -> list[Any]:
    """Draw joint markers at joint locations."""
    artists = []
    state = mechanism.state
    for joint in mechanism.joints:
        body_i = mechanism.bodies[joint.body_i_id]
        if isinstance(joint, (RevoluteJoint, FixedJoint, PrismaticJoint)):
            pt_local = body_i.get_attachment_point(joint._point_i_name)
            pt_global = state.body_point_global(joint.body_i_id, pt_local, q)
            if isinstance(joint, RevoluteJoint):
                marker = "o"
            elif isinstance(joint, PrismaticJoint):
                marker = "D"
            else:
                marker = "s"
            (pt,) = ax.plot(
                pt_global[0], pt_global[1], marker,
                color=color, markersize=size, zorder=3,
            )
            artists.append(pt)
    return artists


def _draw_ground(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    ax: Any,
    color: str = "black",
) -> list[Any]:
    """Draw ground pivot markers."""
    artists = []
    state = mechanism.state
    ground = mechanism.bodies.get(GROUND_ID)
    if ground is not None:
        for pt_local in ground.attachment_points.values():
            pt_global = state.body_point_global(GROUND_ID, pt_local, q)
            (pt,) = ax.plot(
                pt_global[0], pt_global[1], "^",
                color=color, markersize=8, zorder=4,
            )
            artists.append(pt)
    return artists


def _draw_force_vectors(
    mechanism: Mechanism,
    q: NDArray[np.float64],
    ax: Any,
    force_scale: float = 0.01,
    color: str = "purple",
) -> list[Any]:
    """Draw gravity force arrows at each body's CG."""
    artists = []
    state = mechanism.state
    g_vector = np.array([0.0, -9.81])

    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue
        if body.mass <= 0.0:
            continue

        cg_global = state.body_point_global(body_id, body.cg_local, q)
        force = body.mass * g_vector * force_scale

        arrow = ax.annotate(
            "",
            xy=(cg_global[0] + force[0], cg_global[1] + force[1]),
            xytext=(cg_global[0], cg_global[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=5,
        )
        artists.append(arrow)
    return artists


# ---------------------------------------------------------------------------
# Bounding box computation
# ---------------------------------------------------------------------------

def _compute_bounds(
    mechanism: Mechanism,
    solutions: list[NDArray[np.float64] | None],
    margin_frac: float = 0.15,
) -> tuple[float, float, float, float]:
    """Compute x/y bounds from all solved configurations."""
    state = mechanism.state
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

    if not all_x:
        return (-1.0, 1.0, -1.0, 1.0)

    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    margin = margin_frac * max(x_range, y_range, 0.01)

    return (
        min(all_x) - margin,
        max(all_x) + margin,
        min(all_y) - margin,
        max(all_y) + margin,
    )


# ---------------------------------------------------------------------------
# Main interactive viewer
# ---------------------------------------------------------------------------

def launch_interactive(
    mechanism: Mechanism,
    force_elements: list[ForceElement] | None = None,
    n_steps: int = 360,
    coupler_body_id: str | None = None,
    coupler_point_name: str | None = None,
    figsize: tuple[float, float] = (16, 9),
    show: bool = True,
) -> tuple[Any, SweepData]:
    """Launch the interactive mechanism viewer.

    Pre-computes a full sweep at initialization, then uses a slider
    to index into the pre-computed data for snappy interaction.

    Args:
        mechanism: A built Mechanism with a revolute driver.
        force_elements: Force elements for static analysis (e.g., gravity).
        n_steps: Number of angular steps in the sweep.
        coupler_body_id: Body for coupler trace (auto-detected if None).
        coupler_point_name: Coupler point name (auto-detected if None).
        figsize: Figure size (width, height) in inches.
        show: If True, call plt.show() (set False for testing).

    Returns:
        Tuple of (figure, sweep_data) for programmatic access.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider

    if force_elements is None:
        force_elements = []

    # --- Pre-compute the full sweep ---
    sweep_data = precompute_sweep(
        mechanism,
        force_elements,
        n_steps=n_steps,
        coupler_body_id=coupler_body_id,
        coupler_point_name=coupler_point_name,
    )

    # --- Compute bounds for the mechanism view ---
    x_min, x_max, y_min, y_max = _compute_bounds(mechanism, sweep_data.solutions)

    # --- Determine how many overlay plots we need ---
    has_torque = bool(
        force_elements and not np.all(np.isnan(sweep_data.driver_torques))
    )
    has_reactions = bool(
        force_elements
        and any(
            not np.all(np.isnan(v))
            for v in sweep_data.joint_reaction_mags.values()
        )
    )
    has_transmission = sweep_data.transmission_angles_deg is not None

    n_overlay = sum([has_torque, has_reactions, has_transmission])
    if n_overlay == 0:
        n_overlay = 1  # at least one placeholder

    # --- Create figure layout ---
    # Left: mechanism view, Right: stacked overlay plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Interactive Mechanism Viewer (Debug)", fontsize=14)

    # GridSpec: 2 columns. Left = mechanism + controls, Right = overlay plots
    # Bottom row: slider + checkboxes
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        n_overlay + 2, 2,
        figure=fig,
        width_ratios=[1.2, 1],
        hspace=0.4,
        wspace=0.3,
        left=0.06, right=0.95, top=0.92, bottom=0.08,
    )

    # Mechanism axes (spans all overlay rows on the left)
    ax_mech = fig.add_subplot(gs[:n_overlay, 0])
    ax_mech.set_aspect("equal")
    ax_mech.set_xlim(x_min, x_max)
    ax_mech.set_ylim(y_min, y_max)
    ax_mech.grid(True, alpha=0.3)
    ax_mech.set_xlabel("x (m)")
    ax_mech.set_ylabel("y (m)")
    ax_mech.set_title("Mechanism Configuration")

    # Overlay plot axes (right column)
    overlay_axes: list[Any] = []
    overlay_vlines: list[Any] = []
    overlay_idx = 0

    if has_torque:
        ax_torque = fig.add_subplot(gs[overlay_idx, 1])
        valid_mask = ~np.isnan(sweep_data.driver_torques)
        ax_torque.plot(
            sweep_data.angles_deg[valid_mask],
            sweep_data.driver_torques[valid_mask],
            "b-", linewidth=1.0,
        )
        ax_torque.set_ylabel("Torque (N*m)")
        ax_torque.set_title("Input Torque", fontsize=10)
        ax_torque.grid(True, alpha=0.3)
        ax_torque.axhline(y=0, color="k", linewidth=0.5)
        vl = ax_torque.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_torque)
        overlay_vlines.append(vl)
        overlay_idx += 1

    if has_reactions:
        ax_react = fig.add_subplot(gs[overlay_idx, 1])
        for jid, mags in sweep_data.joint_reaction_mags.items():
            valid_mask = ~np.isnan(mags)
            if np.any(valid_mask):
                ax_react.plot(
                    sweep_data.angles_deg[valid_mask],
                    mags[valid_mask],
                    linewidth=1.0, label=jid,
                )
        ax_react.set_ylabel("Force (N)")
        ax_react.set_title("Joint Reactions", fontsize=10)
        ax_react.legend(fontsize=7, loc="upper right")
        ax_react.grid(True, alpha=0.3)
        vl = ax_react.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_react)
        overlay_vlines.append(vl)
        overlay_idx += 1

    if has_transmission:
        ax_trans = fig.add_subplot(gs[overlay_idx, 1])
        ta = sweep_data.transmission_angles_deg
        assert ta is not None
        valid_mask = ~np.isnan(ta)
        ax_trans.plot(
            sweep_data.angles_deg[valid_mask], ta[valid_mask],
            "b-", linewidth=1.0,
        )
        ax_trans.axhline(y=90, color="g", linewidth=0.5, linestyle="--", label="Ideal")
        ax_trans.axhline(y=40, color="r", linewidth=0.5, linestyle="--", label="Poor")
        ax_trans.axhline(y=140, color="r", linewidth=0.5, linestyle="--")
        ax_trans.set_ylabel("Angle (deg)")
        ax_trans.set_title("Transmission Angle", fontsize=10)
        ax_trans.set_ylim(0, 180)
        ax_trans.legend(fontsize=7, loc="upper right")
        ax_trans.grid(True, alpha=0.3)
        vl = ax_trans.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_trans)
        overlay_vlines.append(vl)
        overlay_idx += 1

    # If no overlay plots were created, add a placeholder
    if overlay_idx == 0:
        ax_placeholder = fig.add_subplot(gs[0, 1])
        ax_placeholder.text(
            0.5, 0.5, "No force elements\nprovided",
            ha="center", va="center", fontsize=12, color="gray",
        )
        ax_placeholder.set_axis_off()

    # Set x-axis label on the last overlay plot
    if overlay_axes:
        overlay_axes[-1].set_xlabel("Crank Angle (deg)")

    # --- Info text area ---
    ax_info = fig.add_subplot(gs[n_overlay, 0])
    ax_info.set_axis_off()
    info_text = ax_info.text(
        0.02, 0.9, "", fontsize=9, fontfamily="monospace",
        verticalalignment="top", transform=ax_info.transAxes,
    )

    # --- Slider ---
    ax_slider = fig.add_subplot(gs[n_overlay + 1, :])
    slider = Slider(
        ax_slider,
        "Angle (deg)",
        0, 360 - 360 / n_steps,
        valinit=0,
        valstep=360 / n_steps,
    )

    # --- Checkboxes ---
    ax_check = fig.add_subplot(gs[n_overlay, 1])
    ax_check.set_axis_off()

    check_labels = ["Bodies", "Joints", "Forces", "Coupler"]
    check_defaults = [True, True, True, True]
    check = CheckButtons(ax_check, check_labels, check_defaults)

    # Mutable state for checkbox toggles
    visibility = {label: default for label, default in zip(check_labels, check_defaults)}

    # --- Update function ---
    def update(val: Any) -> None:
        angle_deg = slider.val
        step_idx = int(round(angle_deg / (360 / n_steps)))
        step_idx = max(0, min(step_idx, n_steps - 1))

        q = sweep_data.solutions[step_idx]

        # Clear mechanism axes
        ax_mech.clear()
        ax_mech.set_aspect("equal")
        ax_mech.set_xlim(x_min, x_max)
        ax_mech.set_ylim(y_min, y_max)
        ax_mech.grid(True, alpha=0.3)
        ax_mech.set_xlabel("x (m)")
        ax_mech.set_ylabel("y (m)")

        if q is None:
            ax_mech.set_title(
                f"Angle = {angle_deg:.1f} deg  [SOLVE FAILED]", fontsize=11
            )
            info_text.set_text(f"Angle: {angle_deg:.1f} deg\nSOLVE FAILED at this angle")
            fig.canvas.draw_idle()
            return

        ax_mech.set_title(f"Angle = {angle_deg:.1f} deg", fontsize=11)

        # Draw elements based on checkbox state
        if visibility["Bodies"]:
            _draw_bodies(mechanism, q, ax_mech)

        if visibility["Joints"]:
            _draw_joints(mechanism, q, ax_mech)

        # Always draw ground markers
        _draw_ground(mechanism, q, ax_mech)

        if visibility["Forces"]:
            _draw_force_vectors(mechanism, q, ax_mech)

        if visibility["Coupler"] and sweep_data.coupler_trace_x is not None and sweep_data.coupler_trace_y is not None:
            # Draw full coupler trace up to current step
            ctx = sweep_data.coupler_trace_x
            cty = sweep_data.coupler_trace_y
            trace_x = ctx[: step_idx + 1]
            trace_y = cty[: step_idx + 1]
            valid = ~np.isnan(trace_x)
            if np.any(valid):
                ax_mech.plot(
                    trace_x[valid], trace_y[valid],
                    "-", color="green", linewidth=1.5, alpha=0.6, zorder=1,
                )
            # Draw current coupler point
            cx = ctx[step_idx]
            cy = cty[step_idx]
            if not np.isnan(cx):
                ax_mech.plot(cx, cy, "o", color="green", markersize=5, zorder=5)

        # Update vertical lines on overlay plots
        for vl in overlay_vlines:
            vl.set_xdata([angle_deg, angle_deg])

        # Update info text
        torque_str = "N/A"
        if not np.isnan(sweep_data.driver_torques[step_idx]):
            torque_str = f"{sweep_data.driver_torques[step_idx]:.4f} N*m"

        trans_str = "N/A"
        if (sweep_data.transmission_angles_deg is not None
                and not np.isnan(sweep_data.transmission_angles_deg[step_idx])):
            trans_str = f"{sweep_data.transmission_angles_deg[step_idx]:.2f} deg"

        max_react = 0.0
        max_react_name = "N/A"
        for jid, mags in sweep_data.joint_reaction_mags.items():
            val_at_step = mags[step_idx]
            if not np.isnan(val_at_step) and val_at_step > max_react:
                max_react = val_at_step
                max_react_name = jid

        react_str = "N/A"
        if max_react > 0:
            react_str = f"{max_react:.4f} N ({max_react_name})"

        info_text.set_text(
            f"Angle:       {angle_deg:.1f} deg\n"
            f"Driver Torq: {torque_str}\n"
            f"Trans Angle: {trans_str}\n"
            f"Max Reaction:{react_str}"
        )

        fig.canvas.draw_idle()

    def on_check_clicked(label: str | None) -> None:
        if label is not None:
            visibility[label] = not visibility[label]
        update(None)

    slider.on_changed(update)
    check.on_clicked(on_check_clicked)

    # Initial draw
    update(0)

    if show:
        plt.show()

    return fig, sweep_data
