#!/usr/bin/env python3
"""Launch the interactive viewer for a double-crank (drag-link) mechanism.

Usage:
    python scripts/view_double_crank.py

A double-crank (drag-link) is a Grashof 4-bar where the shortest link
is the fixed ground link. Both the input and output links rotate a full
360 degrees, but with a non-uniform velocity ratio. The output link
"drags" through its rotation, speeding up and slowing down relative
to the input.

Mechanism:
    Ground (d) = 2.0    (fixed frame, SHORTEST link)
    Crank  (a) = 4.0    (input, rotates 360 deg)
    Coupler(b) = 3.5    (connecting rod)
    Rocker (c) = 3.0    (output, also rotates 360 deg)
    Coupler point P at (1.75, 0.0) — traces a coupler curve

    Grashof check: S + L = 2 + 4 = 6,  P + Q = 3 + 3.5 = 6.5
    S + L < P + Q  →  Grashof  ✓
    Shortest link (d=2) is ground  →  double-crank (drag-link)  ✓

Validation sources:
    - Norton, R.L., "Design of Machinery" (6th ed.), Ch. 2, Table 2-1:
      When the ground link is shortest, both grounded links are cranks.
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery",
      Ch. 1: Drag-link mechanism analysis.
    - Scribd / engineering references: Double-crank classification and
      velocity ratio analysis.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_double_crank_with_gravity() -> tuple[Mechanism, list]:
    """Build a double-crank (drag-link) mechanism with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    d, a, b, c = 2.0, 4.0, 3.5, 3.0

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=1.0, Izz_cg=0.08)
    coupler = make_bar("coupler", "B", "C", length=b, mass=0.9, Izz_cg=0.06)
    coupler.add_coupler_point("P", 1.75, 0.0)
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
    mech, force_elements = build_double_crank_with_gravity()

    print("Launching interactive double-crank (drag-link) viewer...")
    print("  Dimensions: d=2 (ground, shortest), a=4, b=3.5, c=3")
    print("  Grashof: S+L=6 < P+Q=6.5  →  double-crank")
    print("  Both crank and rocker rotate 360° with non-uniform velocity ratio")
    print("  Coupler point P at (1.75, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  Norton, 'Design of Machinery' Ch. 2; Waldron & Kinzel Ch. 1")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
