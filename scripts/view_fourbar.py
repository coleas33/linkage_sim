#!/usr/bin/env python3
"""Launch the interactive viewer for the benchmark 4-bar with gravity.

Usage:
    python scripts/view_fourbar.py

This builds the standard benchmark 4-bar (a=1, b=3, c=2, d=4) with
gravity applied to all links, then opens the interactive viewer.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_fourbar_with_gravity() -> tuple[Mechanism, list]:
    """Build the benchmark 4-bar and gravity force element.

    Returns:
        (mechanism, [gravity_element])
    """
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0, mass=0.5, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=3.0, mass=1.5, Izz_cg=0.1)
    coupler.add_coupler_point("P", 1.5, 0.5)
    rocker = make_bar("rocker", "D", "C", length=2.0, mass=1.0, Izz_cg=0.05)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()

    gravity = Gravity(
        g_vector=np.array([0.0, -9.81]),
        bodies=mech.bodies,
    )

    return mech, [gravity]


def main() -> None:
    """Build mechanism and launch the interactive viewer."""
    mech, force_elements = build_fourbar_with_gravity()
    print("Launching interactive 4-bar viewer...")
    print("  Links: a=1, b=3, c=2, d=4  (boundary Grashof crank-rocker)")
    print("  Gravity: [0, -9.81] m/s^2 applied to all links")
    print("  Coupler point P at (1.5, 0.5) on coupler body")
    print()
    print("Controls:")
    print("  - Drag the slider to sweep crank angle 0-360 degrees")
    print("  - Use checkboxes to toggle Bodies / Joints / Forces / Coupler")
    print("  - Right panels show torque, reactions, and transmission angle")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
