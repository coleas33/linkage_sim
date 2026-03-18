#!/usr/bin/env python3
"""Launch the interactive viewer for Chebyshev's straight-line linkage.

Usage:
    python scripts/view_chebyshev.py

This builds Chebyshev's Lambda Mechanism -- a 4-bar linkage whose coupler
midpoint traces an approximate straight line -- then opens the interactive
viewer with gravity applied.

Mechanism (Chebyshev Lambda):
    Ground         = 4a    (horizontal distance between fixed pivots O2 and O4)
    Crank (input)  = 2a    (short driven link at O2, makes full rotation)
    Coupler        = 5a    (long connecting rod; midpoint P traces straight line)
    Rocker         = 5a    (long follower link at O4)
    Unit length a = 1.0

    Link lengths: ground : crank : coupler : rocker = 4 : 2 : 5 : 5

    Coupler point P at the midpoint of the coupler (local coords 2.5, 0.0).
    This point traces a curve whose central portion closely approximates
    a straight horizontal line.

    Grashof classification: Grashof crank-rocker.
    S=2 (crank), L=5 (coupler or rocker), P=5, Q=4
    S + L = 7 < P + Q = 9.
    The shortest link is the crank (adjacent to ground), so the crank
    can make a full 360-degree rotation.

    Closure check: crank tip to O4 distance ranges from 2 (at 0 deg) to
    6 (at 180 deg). Since |coupler - rocker| = 0 <= 2 and 6 <= 10 =
    coupler + rocker, the mechanism closes at all crank angles.

Validation:
    The key property to verify is that the coupler midpoint P traces a path
    with very small vertical (y) deviation over the central portion of travel.

Validation sources:
    - Chebyshev, P.L. (1854), "Theorie des mecanismes connus sous le nom
      de parallelogrammes" -- original derivation of optimal link proportions.
    - Norton, R.L., "Design of Machinery" (6th ed.), Ch. 3, Section 3.7:
      Approximate straight-line mechanisms.
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery",
      Ch. 1: Survey of straight-line mechanisms.
    - Kempe, A.B. (1877), "How to Draw a Straight Line" -- historical context.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_chebyshev_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    """Build Chebyshev's Lambda Mechanism with gravity.

    Chebyshev proportions: ground=4a, crank=2a, coupler=rocker=5a (a=1).
    The short crank (2a) is the driven input; full 360-degree rotation.
    Coupler midpoint traces the approximate straight line.

    Returns:
        (mechanism, [gravity_element], q0_initial_guess)
    """
    # Chebyshev proportions with unit length a = 1
    ground_len = 4.0   # distance between O2 and O4
    crank_len = 2.0     # short input link (a + a = 2a)
    coupler_len = 5.0   # long connecting rod
    rocker_len = 5.0    # long follower

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(ground_len, 0.0))
    crank = make_bar("crank", "A", "B", length=crank_len, mass=0.2, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=coupler_len, mass=0.5, Izz_cg=0.1)
    coupler.add_coupler_point("P", coupler_len / 2, 0.0)  # midpoint at (2.5a, 0)
    rocker = make_bar("rocker", "D", "C", length=rocker_len, mass=0.5, Izz_cg=0.1)

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

    # --- Geometric initial guess at crank angle 0° ---
    # Crank-rocker closes at all angles; compute upper assembly at theta=0.
    theta0 = 0.0

    # Crank tip B at theta=0
    bx = crank_len  # (2, 0)
    by = 0.0

    # Find C by intersecting coupler circle (r=5 from B) and rocker circle
    # (r=5 from O4). At theta=0, B=(2,0), O4=(4,0), |B-O4|=2.
    # Circles: (x-2)^2+y^2=25, (x-4)^2+y^2=25 => x=3, y=+-sqrt(24)
    cx = 3.0
    cy = np.sqrt(24.0)  # upper assembly mode

    theta0_coupler = np.arctan2(cy - by, cx - bx)  # B toward C
    theta0_rocker = np.arctan2(cy, cx - ground_len)  # O4 toward C

    q0 = mech.state.make_q()
    mech.state.set_pose("crank", q0, 0.0, 0.0, theta0)
    mech.state.set_pose("coupler", q0, bx, by, float(theta0_coupler))
    mech.state.set_pose("rocker", q0, ground_len, 0.0, float(theta0_rocker))

    gravity = Gravity(
        g_vector=np.array([0.0, -9.81]),
        bodies=mech.bodies,
    )

    return mech, [gravity], q0


def main() -> None:
    """Build mechanism and launch the interactive viewer."""
    mech, force_elements, q0 = build_chebyshev_with_gravity()

    print("Launching interactive Chebyshev straight-line linkage viewer...")
    print("  Chebyshev's Lambda Mechanism")
    print("  Link lengths: ground:crank:coupler:rocker = 4:2:5:5")
    print("  Crank (2a) is the driven input — full 360-degree rotation")
    print("  Coupler point P at midpoint of coupler (2.5, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("Validation:")
    print("  The coupler midpoint should trace an approximate straight")
    print("  horizontal line over the central portion of its travel.")
    print("  Chebyshev (1854), Norton Ch. 3, Waldron & Kinzel Ch. 1")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360, q0=q0)


if __name__ == "__main__":
    main()
