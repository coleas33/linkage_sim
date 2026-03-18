#!/usr/bin/env python3
"""Launch the interactive viewer for a sample 4-bar linkage.

Usage:
    python scripts/view_fourbar.py

This builds a 4-bar linkage with a coupler point and opens the interactive
viewer with gravity applied.

Mechanism:
    Ground (d) = 4.0    (distance between fixed pivots O2 and O4)
    Crank  (a) = 2.0    (input link, pivots at O2)
    Coupler(b) = 5.0    (connecting rod)
    Rocker (c) = 2.0    (output link, pivots at O4)

    Coupler point P at the midpoint of the coupler (local coords 2.5, 0.0).

    Grashof classification: non-Grashof (triple rocker).
    S=2, L=5, P=2, Q=4  ->  S + L = 7 > P + Q = 6.
    The crank cannot make a full rotation; the mechanism operates over a
    limited angular range. The viewer will show the valid portion of the
    sweep and leave failed steps blank.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_fourbar_with_gravity() -> tuple[Mechanism, list]:
    """Build a sample 4-bar linkage with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    d, a, b, c = 4.0, 2.0, 5.0, 2.0

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.5, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=b, mass=1.5, Izz_cg=0.15)
    coupler.add_coupler_point("P", b / 2, 0.0)
    rocker = make_bar("rocker", "D", "C", length=c, mass=0.5, Izz_cg=0.01)

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

    print("Launching interactive 4-bar linkage viewer...")
    print("  d=4, a=2, b=5, c=2  (non-Grashof triple rocker)")
    print("  Coupler point P at midpoint of coupler (2.5, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2 applied to all links")
    print()
    print("Controls:")
    print("  - Drag the slider to sweep crank angle 0-360 degrees")
    print("  - Use checkboxes to toggle Bodies / Joints / Forces / Coupler")
    print("  - Right panels show torque, reactions, and transmission angle")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
