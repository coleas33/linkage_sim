#!/usr/bin/env python3
"""Launch the interactive viewer for a non-Grashof double-rocker mechanism.

Usage:
    python scripts/view_double_rocker.py

A double-rocker (also called a triple-rocker or non-Grashof 4-bar) is a
mechanism where NO link can rotate a full 360 degrees. Both the input and
output links oscillate through limited arcs. The coupler is the only link
that can (in some configurations) make full rotation.

Mechanism:
    Ground (d) = 5.0    (fixed frame)
    Crank  (a) = 3.0    (input, oscillates — cannot rotate 360 deg)
    Coupler(b) = 4.0    (connecting rod)
    Rocker (c) = 7.0    (output, oscillates)
    Coupler point P at (2.0, 0.0)

    Grashof check: S + L = 3 + 7 = 10,  P + Q = 4 + 5 = 9
    S + L > P + Q  →  non-Grashof  ✓
    No link can make full rotation  →  double-rocker  ✓

    The input crank can only sweep a limited angular range before reaching
    a toggle (dead) point. The viewer will show the valid portion of the
    sweep and leave failed steps blank.

Validation sources:
    - Norton, R.L., "Design of Machinery" (6th ed.), Ch. 2, Table 2-1:
      When S + L > P + Q, the mechanism is non-Grashof (triple rocker).
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery",
      Ch. 1: Non-Grashof classification and toggle position analysis.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_double_rocker_with_gravity() -> tuple[Mechanism, list]:
    """Build a non-Grashof double-rocker with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    d, a, b, c = 5.0, 3.0, 4.0, 7.0

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.8, Izz_cg=0.04)
    coupler = make_bar("coupler", "B", "C", length=b, mass=1.0, Izz_cg=0.08)
    coupler.add_coupler_point("P", 2.0, 0.0)
    rocker = make_bar("rocker", "D", "C", length=c, mass=1.5, Izz_cg=0.15)

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
    mech, force_elements = build_double_rocker_with_gravity()

    print("Launching interactive non-Grashof double-rocker viewer...")
    print("  Dimensions: d=5, a=3, b=4, c=7")
    print("  Grashof: S+L=10 > P+Q=9  →  non-Grashof (double-rocker)")
    print("  Neither crank nor rocker can rotate 360°")
    print("  Viewer shows valid portion of sweep; blanks at toggle points")
    print("  Coupler point P at (2.0, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  Norton, 'Design of Machinery' Ch. 2; Waldron & Kinzel Ch. 1")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
