#!/usr/bin/env python3
"""Launch the interactive viewer for a Grashof crank-rocker mechanism.

Usage:
    python scripts/view_crank_rocker.py

A crank-rocker is a Grashof 4-bar where the shortest link is the input
crank, which rotates 360 degrees. The output rocker oscillates through
a limited arc. Common applications include car wiper systems and dump
truck bed lifts.

Mechanism:
    Ground (d) = 4.0    (fixed frame, distance between pivots O2 and O4)
    Crank  (a) = 2.0    (input, shortest link — rotates 360 deg)
    Coupler(b) = 4.0    (connecting rod)
    Rocker (c) = 3.0    (output, oscillates)
    Coupler point P at (2.0, 0.0) — traces a coupler curve

    Grashof check: S + L = 2 + 4 = 6,  P + Q = 3 + 4 = 7
    S + L < P + Q  →  Grashof  ✓
    Shortest link (a=2) is the crank  →  crank-rocker  ✓

Validation sources:
    - Norton, R.L., "Design of Machinery" (6th ed.), Ch. 2, Table 2-1:
      Grashof condition and mechanism type classification.
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery",
      Ch. 1: Four-bar classification by Grashof criterion.
    - METU OpenCourseWare, ME 301 Theory of Machines:
      Crank-rocker design examples and link-length requirements.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_crank_rocker_with_gravity() -> tuple[Mechanism, list]:
    """Build a Grashof crank-rocker with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    d, a, b, c = 4.0, 2.0, 4.0, 3.0

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.5, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=b, mass=1.0, Izz_cg=0.08)
    coupler.add_coupler_point("P", 2.0, 0.0)
    rocker = make_bar("rocker", "D", "C", length=c, mass=0.8, Izz_cg=0.05)

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
    mech, force_elements = build_crank_rocker_with_gravity()

    print("Launching interactive Grashof crank-rocker viewer...")
    print("  Dimensions: d=4, a=2, b=4, c=3")
    print("  Grashof: S+L=6 < P+Q=7  →  crank-rocker (crank rotates 360°)")
    print("  Coupler point P at (2.0, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  Applications: car wiper systems, dump truck lifts")
    print("  Norton, 'Design of Machinery' Ch. 2; Waldron & Kinzel Ch. 1")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
