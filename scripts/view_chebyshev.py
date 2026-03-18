#!/usr/bin/env python3
"""Launch the interactive viewer for Chebyshev's straight-line linkage.

Usage:
    python scripts/view_chebyshev.py

This builds Chebyshev's Lambda Mechanism -- a 4-bar linkage whose coupler
midpoint traces an approximate straight line -- then opens the interactive
viewer with gravity applied.

Mechanism (Chebyshev Lambda):
    Ground (link 1) = 4a    (horizontal distance between fixed pivots O2 and O4)
    Crank  (link 2) = 5a    (left grounded link, pivots at O2)
    Coupler(link 3) = 2a    (a + a; links 2 and 4 connect to each end)
    Rocker (link 4) = 5a    (right grounded link, pivots at O4)
    Unit length a = 1.0

    Proportions: ground : crank : coupler : rocker = 4 : 5 : 2 : 5

    Coupler point P at the midpoint of the coupler (local coords 1.0, 0.0).
    This point traces an approximate straight horizontal line over the
    central portion of its travel.

    Grashof classification: Grashof double-rocker.
    S=2 (coupler), L=5 (crank or rocker), P=5, Q=4
    S + L = 7 < P + Q = 9.
    The shortest link is the coupler (opposite ground), so neither grounded
    link can make a full rotation. The mechanism operates over a limited
    angular range (~65 degrees per branch). The viewer will show the valid
    portion of the sweep and leave failed steps blank.

    Valid crank range (upper branch): arccos(0.8) to arccos(-0.2) ≈ 37° to 102°.
    Toggle positions occur at the range boundaries where B, C, O4 are collinear.

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

    Chebyshev proportions: ground=4a, crank=rocker=5a, coupler=2a (a=1).

    Returns:
        (mechanism, [gravity_element], q0_initial_guess)
    """
    # Chebyshev proportions with unit length a = 1
    ground_len = 4.0   # link 1
    crank_len = 5.0     # link 2
    coupler_len = 2.0   # link 3 (= a + a)
    rocker_len = 5.0    # link 4

    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(ground_len, 0.0))
    crank = make_bar("crank", "A", "B", length=crank_len, mass=0.5, Izz_cg=0.1)
    coupler = make_bar("coupler", "B", "C", length=coupler_len, mass=0.2, Izz_cg=0.01)
    coupler.add_coupler_point("P", coupler_len / 2, 0.0)  # midpoint at (a, 0)
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

    # --- Geometric initial guess at crank angle 70° ---
    # Valid crank range for this double-rocker: ~[37°, 102°] (upper branch).
    # 70° is well within the range, giving a robust starting configuration.
    theta0 = np.radians(70.0)

    # Crank tip B (= coupler body origin) in global coords
    bx = crank_len * np.cos(theta0)
    by = crank_len * np.sin(theta0)

    # Find C by solving the coupler-rocker triangle B-C-O4.
    # Law of cosines gives angle beta at B, then C follows.
    o4x = ground_len
    dist_bo4 = np.hypot(bx - o4x, by)

    cos_beta = (coupler_len**2 + dist_bo4**2 - rocker_len**2) / (
        2 * coupler_len * dist_bo4
    )
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))

    # Direction from B toward O4
    angle_b_to_o4 = np.arctan2(-by, o4x - bx)

    # Coupler angle: B toward C, taking "upper" assembly mode (+beta)
    theta0_coupler = angle_b_to_o4 + beta

    # Rocker angle: O4 toward C
    cx = bx + coupler_len * np.cos(theta0_coupler)
    cy = by + coupler_len * np.sin(theta0_coupler)
    theta0_rocker = np.arctan2(cy, cx - o4x)

    q0 = mech.state.make_q()
    mech.state.set_pose("crank", q0, 0.0, 0.0, float(theta0))
    mech.state.set_pose("coupler", q0, float(bx), float(by), float(theta0_coupler))
    mech.state.set_pose("rocker", q0, o4x, 0.0, float(theta0_rocker))

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
    print("  Proportions: ground:crank:coupler:rocker = 4:5:2:5")
    print("  Coupler point P at midpoint of coupler (1.0, 0.0)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  Grashof double-rocker: crank sweeps ~65 deg per branch.")
    print("  Steps outside the valid range show as SOLVE FAILED.")
    print()
    print("Validation:")
    print("  The coupler midpoint should trace an approximate straight")
    print("  horizontal line over the central portion of its travel.")
    print("  Chebyshev (1854), Norton Ch. 3, Waldron & Kinzel Ch. 1")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360, q0=q0)


if __name__ == "__main__":
    main()
