#!/usr/bin/env python3
"""Export animated GIFs for all mechanism viewer scripts.

Usage:
    python scripts/export_gifs.py              # export all
    python scripts/export_gifs.py chebyshev    # export one by name substring
    python scripts/export_gifs.py --list       # list available mechanisms

Each GIF includes the mechanism animation (left) with coupler trace,
plus overlay plots (right) showing input torque, joint reactions, and
transmission angle (4-bar only). A vertical red line tracks the current
crank angle across all plots.

Output directory: output/gifs/
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import numpy as np
from matplotlib.gridspec import GridSpec

from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID
from linkage_sim.forces.protocol import ForceElement
from linkage_sim.viz.interactive_viewer import (
    SweepData,
    _compute_bounds,
    _draw_bodies,
    _draw_ground,
    _draw_joints,
    precompute_sweep,
)


# ---------------------------------------------------------------------------
# Registry: all mechanisms and their build functions
# ---------------------------------------------------------------------------

def _load_mechanism(script_name: str) -> tuple[str, Mechanism, list[Any], Any]:
    """Load a mechanism from a viewer script.

    Returns (display_name, mechanism, force_elements, q0).
    """
    ns: dict[str, Any] = {}
    path = os.path.join("scripts", f"{script_name}.py")
    code = open(path).read().replace("if __name__", "if False")
    exec(code, ns)  # noqa: S102

    build_fn = None
    for name, obj in ns.items():
        if callable(obj) and name.startswith("build_"):
            build_fn = obj
            break

    if build_fn is None:
        raise RuntimeError(f"No build_* function in {path}")

    result = build_fn()
    mech = result[0]
    forces = result[1]
    q0 = result[2] if len(result) >= 3 else None
    return script_name, mech, forces, q0


SCRIPT_NAMES = [
    "view_crank_rocker",
    "view_double_crank",
    "view_double_rocker",
    "view_parallelogram",
    "view_slider_crank",
    "view_chebyshev",
    "view_sixbar",
    "view_sixbar_A1",
    "view_sixbar_A2",
    "view_sixbar_B2",
    "view_sixbar_B3",
]


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def export_gif(
    name: str,
    mechanism: Mechanism,
    force_elements: list[ForceElement],
    sweep: SweepData,
    output_dir: str = "output/gifs",
    n_frames: int = 180,
    fps: int = 30,
    figsize: tuple[float, float] = (14, 7),
) -> str:
    """Create and save an animated GIF with mechanism + overlay plots.

    Returns the output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    os.makedirs(output_dir, exist_ok=True)

    # --- Bounds ---
    x_min, x_max, y_min, y_max = _compute_bounds(mechanism, sweep.solutions)

    # --- Determine overlay plots ---
    has_torque = bool(
        force_elements and not np.all(np.isnan(sweep.driver_torques))
    )
    has_reactions = bool(
        force_elements
        and any(
            not np.all(np.isnan(v))
            for v in sweep.joint_reaction_mags.values()
        )
    )
    has_transmission = sweep.transmission_angles_deg is not None
    n_overlay = max(1, sum([has_torque, has_reactions, has_transmission]))

    # --- Figure layout ---
    fig = plt.figure(figsize=figsize)
    display_name = name.replace("view_", "").replace("_", " ").title()
    fig.suptitle(display_name, fontsize=14, fontweight="bold")

    gs = GridSpec(
        n_overlay, 2, figure=fig,
        width_ratios=[1.3, 1],
        hspace=0.45, wspace=0.30,
        left=0.06, right=0.95, top=0.90, bottom=0.08,
    )

    # Mechanism axes (spans all rows on left)
    ax_mech = fig.add_subplot(gs[:, 0])
    ax_mech.set_aspect("equal")
    ax_mech.set_xlim(x_min, x_max)
    ax_mech.set_ylim(y_min, y_max)
    ax_mech.grid(True, alpha=0.3)
    ax_mech.set_xlabel("x (m)")
    ax_mech.set_ylabel("y (m)")

    # --- Overlay plots (right column) ---
    overlay_axes: list[Any] = []
    overlay_vlines: list[Any] = []
    overlay_idx = 0

    if has_torque:
        ax_t = fig.add_subplot(gs[overlay_idx, 1])
        valid = ~np.isnan(sweep.driver_torques)
        ax_t.plot(sweep.angles_deg[valid], sweep.driver_torques[valid],
                  "b-", linewidth=0.8)
        ax_t.set_ylabel("Torque (N*m)", fontsize=8)
        ax_t.set_title("Input Torque", fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_t.axhline(y=0, color="k", linewidth=0.5)
        ax_t.tick_params(labelsize=7)
        vl = ax_t.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_t)
        overlay_vlines.append(vl)
        overlay_idx += 1

    if has_reactions:
        ax_r = fig.add_subplot(gs[overlay_idx, 1])
        for jid, mags in sweep.joint_reaction_mags.items():
            valid = ~np.isnan(mags)
            if np.any(valid):
                ax_r.plot(sweep.angles_deg[valid], mags[valid],
                          linewidth=0.8, label=jid)
        ax_r.set_ylabel("Force (N)", fontsize=8)
        ax_r.set_title("Joint Reactions", fontsize=9)
        ax_r.legend(fontsize=6, loc="upper right")
        ax_r.grid(True, alpha=0.3)
        ax_r.tick_params(labelsize=7)
        vl = ax_r.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_r)
        overlay_vlines.append(vl)
        overlay_idx += 1

    if has_transmission:
        ax_tr = fig.add_subplot(gs[overlay_idx, 1])
        ta = sweep.transmission_angles_deg
        assert ta is not None
        valid = ~np.isnan(ta)
        ax_tr.plot(sweep.angles_deg[valid], ta[valid], "b-", linewidth=0.8)
        ax_tr.axhline(y=90, color="g", linewidth=0.5, linestyle="--")
        ax_tr.axhline(y=40, color="r", linewidth=0.5, linestyle="--")
        ax_tr.axhline(y=140, color="r", linewidth=0.5, linestyle="--")
        ax_tr.set_ylabel("Angle (deg)", fontsize=8)
        ax_tr.set_title("Transmission Angle", fontsize=9)
        ax_tr.set_ylim(0, 180)
        ax_tr.grid(True, alpha=0.3)
        ax_tr.tick_params(labelsize=7)
        vl = ax_tr.axvline(x=0, color="r", linewidth=1, linestyle="--")
        overlay_axes.append(ax_tr)
        overlay_vlines.append(vl)
        overlay_idx += 1

    if overlay_idx == 0:
        ax_ph = fig.add_subplot(gs[0, 1])
        ax_ph.text(0.5, 0.5, "No force data", ha="center", va="center",
                   fontsize=11, color="gray")
        ax_ph.set_axis_off()

    if overlay_axes:
        overlay_axes[-1].set_xlabel("Crank Angle (deg)", fontsize=8)

    # --- Downsample frames ---
    total_steps = sweep.n_steps
    frame_indices = np.linspace(0, total_steps - 1, n_frames,
                                dtype=int, endpoint=False)

    # --- Animation update ---
    def update(frame_num: int) -> list[Any]:
        step_idx = int(frame_indices[frame_num])
        q = sweep.solutions[step_idx]
        angle_deg = float(sweep.angles_deg[step_idx])

        ax_mech.clear()
        ax_mech.set_aspect("equal")
        ax_mech.set_xlim(x_min, x_max)
        ax_mech.set_ylim(y_min, y_max)
        ax_mech.grid(True, alpha=0.3)
        ax_mech.set_xlabel("x (m)")
        ax_mech.set_ylabel("y (m)")

        if q is None:
            ax_mech.set_title(f"{angle_deg:.0f} deg  [FAILED]", fontsize=10)
        else:
            ax_mech.set_title(f"{angle_deg:.0f} deg", fontsize=10)
            _draw_bodies(mechanism, q, ax_mech)
            _draw_joints(mechanism, q, ax_mech)
            _draw_ground(mechanism, q, ax_mech)

            # Coupler trace up to current step
            if sweep.coupler_traces:
                trace = sweep.coupler_traces[0]
                ctx = trace.x[:step_idx + 1]
                cty = trace.y[:step_idx + 1]
                valid = ~np.isnan(ctx)
                if np.any(valid):
                    ax_mech.plot(ctx[valid], cty[valid], "-",
                                color="green", linewidth=1.2, alpha=0.5,
                                zorder=1)
                cx = trace.x[step_idx]
                cy = trace.y[step_idx]
                if not np.isnan(cx):
                    ax_mech.plot(cx, cy, "o", color="green",
                                markersize=4, zorder=5)

        # Move vertical indicators on overlay plots
        for vl in overlay_vlines:
            vl.set_xdata([angle_deg, angle_deg])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)

    out_path = os.path.join(output_dir, f"{name}.gif")
    print(f"  Saving {out_path} ({n_frames} frames, {fps} fps)...", end=" ",
          flush=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print("done.")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "--list" in sys.argv:
        print("Available mechanisms:")
        for s in SCRIPT_NAMES:
            print(f"  {s}")
        return

    # Filter by name substring if provided
    filter_arg = None
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            filter_arg = arg
            break

    scripts_to_run = SCRIPT_NAMES
    if filter_arg:
        scripts_to_run = [s for s in SCRIPT_NAMES if filter_arg in s]
        if not scripts_to_run:
            print(f"No mechanism matching '{filter_arg}'. Use --list to see options.")
            return

    print(f"Exporting {len(scripts_to_run)} mechanism GIF(s)...")
    print()

    for script_name in scripts_to_run:
        print(f"[{script_name}]")
        print("  Building mechanism...", end=" ", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, mech, forces, q0 = _load_mechanism(script_name)
            except Exception as e:
                print(f"BUILD FAILED: {e}")
                continue
            print("done.")

            print("  Computing sweep...", end=" ", flush=True)
            sweep = precompute_sweep(mech, forces, n_steps=360, q0=q0)
            n_ok = sum(1 for s in sweep.solutions if s is not None)
            print(f"done ({n_ok}/360 converged).")

            export_gif(script_name, mech, forces, sweep)
        print()

    print("All done. GIFs saved to output/gifs/")


if __name__ == "__main__":
    main()
