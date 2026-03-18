#!/usr/bin/env python3
"""Launch the interactive viewer for a parallelogram linkage.

Usage:
    python scripts/view_parallelogram.py

A parallelogram linkage is a 4-bar where opposite links are equal in
length (ground = coupler, crank = rocker). The coupler maintains a
constant orientation (it translates without rotating), making it useful
for parallel motion applications such as:
    - Drafting machines and pantographs
    - Windshield wiper linkages (parallel type)
    - Locomotive wheel couplings

Mechanism:
    Ground (d) = 4.0    (fixed frame)
    Crank  (a) = 2.0    (input, rotates 360 deg)
    Coupler(b) = 4.0    (connecting rod — same length as ground)
    Rocker (c) = 2.0    (output — same length as crank)
    Coupler point P at (2.0, 0.0) — traces a circle (pure translation)

    Opposite links equal: a = c = 2, b = d = 4
    Grashof check: S + L = 2 + 4 = 6,  P + Q = 2 + 4 = 6
    S + L = P + Q  →  change-point (boundary Grashof)

    The coupler maintains constant orientation (theta_coupler = const),
    so the coupler point P traces a perfect circle centered at the
    midpoint of the ground link.

    WARNING: At theta = 0 and theta = pi, the mechanism reaches a change
    point (all links collinear) where it can flip between parallelogram
    and antiparallelogram (crossed) configurations. The solver may have
    difficulty at these exact angles.

Validation sources:
    - Norton, R.L., "Design of Machinery" (6th ed.), Ch. 2: Change-point
      mechanisms and special-case Grashof linkages.
    - Wikipedia, "Four-bar linkage": Parallelogram and antiparallelogram
      configurations.
    - Key property to verify: coupler angular velocity = 0 (pure translation).
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_parallelogram_with_gravity() -> tuple[Mechanism, list]:
    """Build a parallelogram linkage with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    d, a, b, c = 4.0, 2.0, 4.0, 2.0

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.5, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=b, mass=1.0, Izz_cg=0.08)
    coupler.add_coupler_point("P", 2.0, 0.0)
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
    mech, force_elements = build_parallelogram_with_gravity()

    print("Launching interactive parallelogram linkage viewer...")
    print("  Dimensions: d=4, a=2, b=4, c=2 (opposite sides equal)")
    print("  Change-point: S+L=6 = P+Q=6  (boundary Grashof)")
    print("  Coupler maintains constant orientation (pure translation)")
    print("  Coupler point P at (2.0, 0.0) — traces a circle")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  WARNING: Singular at theta=0,180 (change points)")
    print("  Applications: drafting machines, locomotive wheel couplings")
    print("  Norton, 'Design of Machinery' Ch. 2")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
