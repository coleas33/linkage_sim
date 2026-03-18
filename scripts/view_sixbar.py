#!/usr/bin/env python3
"""Launch the interactive viewer for a 6-bar mechanism (type B1) with gravity.

Usage:
    python scripts/view_sixbar.py

6-bar mechanism with ternary coupler. The two ternary links (ground and
coupler) are NOT adjacent -- they share neighbors (crank, rocker4) but
have no direct joint between them. Ground is a ternary link (3 pivots).

Graph definition (type B1 -- Chain B, ternary ground):
    Links (6): ground(T), crank(B), ternary(T), rocker4(B), link5(B), output6(B)
    Joints (7):
        J1: ground-crank      J2: crank-ternary    J3: ternary-rocker4
        J4: ground-rocker4    J5: ternary-link5    J6: link5-output6
        J7: ground-output6
    Ternary links: ground (deg 3), ternary (deg 3)
    Ternaries adjacent: NO (no direct ground-ternary joint)
    Ground type: ternary
    Driver: ground-crank (J1)

    Note: The conventional name (Watt I vs Stephenson I) depends on which
    textbook convention is used for the adjacency criterion. The graph
    definition above is unambiguous.

Dimensions (Grashof-compliant for full crank rotation):
    Ground: O2=(0,0), O4=(2.5,0.5), O6=(3.5,0)
    Crank: length=1.5, pivot at O2
    Ternary coupler: P1, P2=(3,0), P3=(1.5,1) in local coords
    Rocker4: length=2.5, pivot at O6
    Link5: length=2.0
    Output6: length=2.0, pivot at O4

    Loop 1 (O2-crank-ternary_P2-rocker4-O6): 3.5, 1.5, 3.0, 2.5
        S+L=1.5+3.5=5.0 <= P+Q=3.0+2.5=5.5  GRASHOF
    Loop 2 (ternary_P3-link5-output6-O4): effective 4-bar with moving ground

    DOF: 3*5 - 2*7 - 1 = 0  (Grubler, with driver)

Validation sources:
    - Grubler DOF: M = 3(n-1) - 2*j1 - j2 = 3(5) - 2(7) - 1 = 0
    - Norton, "Design of Machinery" (6th ed.), Ch. 2
    - Waldron & Kinzel, "Kinematics, Dynamics, and Design of Machinery"
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.viz.interactive_viewer import launch_interactive


def make_ternary_link(
    body_id: str,
    p1_name: str,
    p2_name: str,
    p3_name: str,
    p2_local: tuple[float, float],
    p3_local: tuple[float, float],
    mass: float = 0.0,
    Izz_cg: float = 0.0,
) -> Body:
    """Create a ternary body with 3 attachment points.

    P1 is at the local origin (0,0). P2 and P3 are specified in local coords.
    """
    body = Body(id=body_id)
    body.add_attachment_point(p1_name, 0.0, 0.0)
    body.add_attachment_point(p2_name, p2_local[0], p2_local[1])
    body.add_attachment_point(p3_name, p3_local[0], p3_local[1])
    cg_x = (0.0 + p2_local[0] + p3_local[0]) / 3.0
    cg_y = (0.0 + p2_local[1] + p3_local[1]) / 3.0
    body.cg_local = np.array([cg_x, cg_y])
    body.mass = mass
    body.Izz_cg = Izz_cg
    return body


def build_sixbar_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    """Build a Watt I 6-bar with gravity and return an initial guess.

    Returns:
        (mechanism, [gravity_element], q0_initial_guess)
    """
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(2.5, 0.5), O6=(3.5, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5, mass=0.5, Izz_cg=0.01)
    ternary = make_ternary_link(
        "ternary", "P1", "P2", "P3",
        p2_local=(3.0, 0.0),
        p3_local=(1.5, 1.0),
        mass=2.0,
        Izz_cg=0.15,
    )
    ternary.add_coupler_point("CP", 1.5, 0.0)
    rocker4 = make_bar("rocker4", "R4A", "R4B", length=2.5, mass=1.0, Izz_cg=0.05)
    link5 = make_bar("link5", "L5A", "L5B", length=2.5, mass=0.8, Izz_cg=0.03)
    output6 = make_bar("output6", "R6A", "R6B", length=2.5, mass=0.8, Izz_cg=0.03)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(ternary)
    mech.add_body(rocker4)
    mech.add_body(link5)
    mech.add_body(output6)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "ternary", "P1")
    mech.add_revolute_joint("J3", "ternary", "P2", "rocker4", "R4B")
    mech.add_revolute_joint("J4", "ground", "O6", "rocker4", "R4A")
    mech.add_revolute_joint("J5", "ternary", "P3", "link5", "L5A")
    mech.add_revolute_joint("J6", "link5", "L5B", "output6", "R6B")
    mech.add_revolute_joint("J7", "ground", "O4", "output6", "R6A")

    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )

    mech.build()

    gravity = Gravity(
        g_vector=np.array([0.0, -9.81]),
        bodies=mech.bodies,
    )

    # Build a converged initial guess for the sweep (which starts at t=0).
    # We solve at a small angle first (where the geometry guess is reliable),
    # then step back toward 0 using continuation.
    q_guess = _initial_guess(mech, 0.3)
    result = solve_position(mech, q_guess, t=0.3)
    if not result.converged:
        raise RuntimeError("Could not find initial configuration for 6-bar")
    # Step from 0.3 back to 0.0 via continuation
    from linkage_sim.solvers.sweep import position_sweep
    step_back = position_sweep(mech, result.q, np.linspace(0.3, 0.0, 10))
    q0 = step_back.solutions[-1] if step_back.solutions[-1] is not None else result.q

    return mech, [gravity], q0


def _initial_guess(mech: Mechanism, crank_angle: float) -> np.ndarray:
    """Compute a geometric initial guess for the Newton-Raphson solver."""
    q = mech.state.make_q()

    # Crank
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)

    # Crank tip B
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)

    # Ternary: P1 at B, initial theta ~ 0
    theta_tern = 0.15
    mech.state.set_pose("ternary", q, bx, by, theta_tern)

    # Ternary P2 global (approximate)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st

    # Rocker4: R4A at O6=(3.5,0), R4B should be near P2
    dx = p2_gx - 3.5
    dy = p2_gy
    theta_r4 = np.arctan2(dy, dx)
    mech.state.set_pose("rocker4", q, 3.5, 0.0, theta_r4)

    # Ternary P3 global (approximate)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct

    # Link5: L5A at P3, pointing toward O4 region
    dx5 = 2.5 - p3_gx
    dy5 = 0.5 - p3_gy
    theta_l5 = np.arctan2(dy5, dx5)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)

    # Output6: R6A at O4=(2.5,0.5), R6B should be near link5 L5B
    l5b_gx = p3_gx + 2.5 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.5 * np.sin(theta_l5)
    dx6 = l5b_gx - 2.5
    dy6 = l5b_gy - 0.5
    theta_o6 = np.arctan2(dy6, dx6)
    mech.state.set_pose("output6", q, 2.5, 0.5, theta_o6)

    return q


def main() -> None:
    """Build mechanism and launch the interactive viewer."""
    mech, force_elements, q0 = build_sixbar_with_gravity()

    print("Launching interactive Watt I 6-bar viewer...")
    print("  Topology: Watt I with ternary coupler")
    print("  Ground pivots: O2=(0,0), O4=(2.5,0.5), O6=(3.5,0)")
    print("  Links: crank=1.5, ternary (3 pts), rocker4=2.5, link5=2.5, output6=2.5")
    print("  Gravity: [0, -9.81] m/s^2 applied to all links")
    print("  Coupler point CP at (1.5, 0.0) on ternary body")
    print()
    print("Validation:")
    print("  Grübler DOF = 3(5) - 2(7) - 1 = 0 (fully constrained with driver)")
    print("  Norton, 'Design of Machinery' Ch. 2 (Watt I topology)")
    print("  Waldron & Kinzel, 'Kinematics, Dynamics, and Design of Machinery' Ch. 1-2")
    print()
    print("Controls:")
    print("  - Drag the slider to sweep crank angle")
    print("  - Use checkboxes to toggle Bodies / Joints / Forces / Coupler")
    print("  - Right panels show torque, reactions")
    print()

    launch_interactive(
        mech,
        force_elements=force_elements,
        n_steps=180,
        coupler_body_id="ternary",
        coupler_point_name="CP",
        q0=q0,
    )


if __name__ == "__main__":
    main()
