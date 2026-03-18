#!/usr/bin/env python3
"""Launch the interactive viewer for a slider-crank (RRRP) mechanism.

Usage:
    python scripts/view_slider_crank.py

A slider-crank mechanism converts rotary motion to linear motion (or
vice versa). It consists of three revolute joints (R) and one prismatic
joint (P), giving the kinematic notation RRRP.

Applications:
    - Internal combustion engines (piston-crank)
    - Reciprocating pumps and compressors
    - Toggle clamps and stamping presses

Mechanism:
    Crank  (r) = 1.0    (input, rotates 360 deg at O2)
    ConRod (l) = 3.0    (connecting rod)
    Slider        —     (output, translates along x-axis)
    Coupler point P at (1.5, 0.0) on the connecting rod

    DOF: 3*3 - (3*2 + 1*2 + 1*1) = 9 - 9 = 0 (with driver)

    Analytical position (Norton Ch. 4):
        x_slider = r*cos(theta) + sqrt(l^2 - r^2*sin^2(theta))

    Analytical velocity (Norton Ch. 6):
        v_slider = -r*sin(theta) * (1 + r*cos(theta) / sqrt(l^2 - r^2*sin^2(theta)))

Validation sources:
    - Norton, R.L., "Design of Machinery" (6th ed.):
      Ch. 4 (position), Ch. 6 (velocity), Ch. 7 (acceleration).
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery",
      Ch. 4: Slider-crank analysis.
    - ScienceDirect: Slider-crank mechanism design and optimization.
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import launch_interactive


def build_slider_crank_with_gravity() -> tuple[Mechanism, list]:
    """Build a slider-crank (RRRP) mechanism with gravity.

    Returns:
        (mechanism, [gravity_element])
    """
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0, mass=0.5, Izz_cg=0.01)
    conrod = make_bar("conrod", "B", "C", length=3.0, mass=1.0, Izz_cg=0.08)
    conrod.add_coupler_point("P", 1.5, 0.0)
    slider = Body(
        id="slider",
        attachment_points={"pin": np.array([0.0, 0.0])},
        mass=2.0,
    )

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(conrod)
    mech.add_body(slider)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "pin")
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "pin",
        axis_local_i=np.array([1.0, 0.0]),
        delta_theta_0=0.0,
    )
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
    mech, force_elements = build_slider_crank_with_gravity()

    print("Launching interactive slider-crank (RRRP) viewer...")
    print("  Crank r=1.0, Connecting rod l=3.0, Slider on x-axis")
    print("  Coupler point P at (1.5, 0.0) on connecting rod")
    print("  Gravity: [0, -9.81] m/s^2")
    print()
    print("  Analytical position: x = r*cos(θ) + √(l² - r²*sin²(θ))")
    print("  Applications: engines, pumps, presses")
    print("  Norton, 'Design of Machinery' Ch. 4, 6, 7")
    print()

    launch_interactive(mech, force_elements=force_elements, n_steps=360)


if __name__ == "__main__":
    main()
