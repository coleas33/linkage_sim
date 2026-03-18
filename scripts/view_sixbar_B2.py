#!/usr/bin/env python3
"""Launch the interactive viewer for a 6-bar mechanism (type B2) with gravity.

Usage:
    python scripts/view_sixbar_B2.py

6-bar mechanism where the two ternary links are NOT adjacent (no direct
joint). Ground is a binary link that is a shared neighbor of BOTH ternary
links -- i.e., ground has one joint with T1 and one joint with T2.

Graph definition (type B2 -- Chain B, shared-binary ground):
    Chain: B (ternaries NOT adjacent)
    Links (6): ground(B), T1(T), B2(B), T2(T), B3(B), B4(B)
    Joints (7):
        J1: ground-T1     J2: ground-T2     J3: T1-B2
        J4: T1-B3         J5: T2-B2         J6: T2-B4
        J7: B3-B4
    Ternary links: T1 (deg 3: J1,J3,J4), T2 (deg 3: J2,J5,J6)
    Ternaries adjacent: NO (no direct T1-T2 joint)
    Ground type: binary (deg 2: J1,J2)
    Ground neighbors: T1, T2 (ground connects to BOTH ternaries)
    Driver: ground-T1 (J1) -- drive ternary T1 relative to ground

    Note: The driven input T1 is a ternary body with 3 attachment points.
    This is valid -- the driver prescribes T1's angle relative to ground.

Dimensions:
    Ground pivots: O2=(0,0), O4=(5,0)
    T1 (ternary, driven): P1=(0,0), P2=(2.5,0), P3=(1.2,1.0) at O2
    B2 (binary): length=2.5
    T2 (ternary): Q1=(0,0), Q2=(2.5,0), Q3=(1.2,-1.0) at O4
    B3 (binary): length=3.0
    B4 (binary): length=3.0

    DOF: 3*5 - 2*7 - 1 = 0 (Grubler, with driver)
"""

from __future__ import annotations

import numpy as np

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.sweep import position_sweep
from linkage_sim.viz.interactive_viewer import launch_interactive


def _make_ternary(
    body_id: str, p1: str, p2: str, p3: str,
    p2_local: tuple[float, float], p3_local: tuple[float, float],
    mass: float = 0.0, Izz_cg: float = 0.0,
) -> Body:
    body = Body(id=body_id)
    body.add_attachment_point(p1, 0.0, 0.0)
    body.add_attachment_point(p2, p2_local[0], p2_local[1])
    body.add_attachment_point(p3, p3_local[0], p3_local[1])
    body.cg_local = np.array([
        (p2_local[0] + p3_local[0]) / 3.0,
        (p2_local[1] + p3_local[1]) / 3.0,
    ])
    body.mass = mass
    body.Izz_cg = Izz_cg
    return body


def build_B2_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 0.0))

    t1 = _make_ternary("t1", "P1", "P2", "P3",
                        p2_local=(1.2, 0.0), p3_local=(0.6, 0.5),
                        mass=0.8, Izz_cg=0.04)
    t1.add_coupler_point("CP", 0.6, 0.25)

    b2 = make_bar("b2", "B2A", "B2B", length=1.5, mass=0.5, Izz_cg=0.01)

    t2 = _make_ternary("t2", "Q1", "Q2", "Q3",
                        p2_local=(1.2, 0.0), p3_local=(0.6, -0.5),
                        mass=0.8, Izz_cg=0.04)

    b3 = make_bar("b3", "B3A", "B3B", length=2.0, mass=0.6, Izz_cg=0.02)
    b4 = make_bar("b4", "B4A", "B4B", length=2.0, mass=0.6, Izz_cg=0.02)

    mech.add_body(ground)
    mech.add_body(t1)
    mech.add_body(b2)
    mech.add_body(t2)
    mech.add_body(b3)
    mech.add_body(b4)

    # Joints per graph definition
    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1")     # ground-T1
    mech.add_revolute_joint("J2", "ground", "O4", "t2", "Q1")     # ground-T2
    mech.add_revolute_joint("J3", "t1", "P2", "b2", "B2A")        # T1-B2
    mech.add_revolute_joint("J4", "t1", "P3", "b3", "B3A")        # T1-B3
    mech.add_revolute_joint("J5", "t2", "Q2", "b2", "B2B")        # T2-B2
    mech.add_revolute_joint("J6", "t2", "Q3", "b4", "B4A")        # T2-B4
    mech.add_revolute_joint("J7", "b3", "B3B", "b4", "B4B")       # B3-B4

    mech.add_revolute_driver(
        "D1", "ground", "t1",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()

    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    q0 = _find_initial(mech)
    return mech, [gravity], q0


def _find_initial(mech: Mechanism) -> np.ndarray:
    angle = np.radians(5)
    q = mech.state.make_q()

    ct1, st1 = np.cos(angle), np.sin(angle)
    mech.state.set_pose("t1", q, 0.0, 0.0, angle)
    p2x, p2y = 1.2 * ct1, 1.2 * st1
    p3x, p3y = 0.6 * ct1 - 0.5 * st1, 0.6 * st1 + 0.5 * ct1

    theta_t2 = -2.0
    mech.state.set_pose("t2", q, 3.0, 0.0, theta_t2)
    ct2, st2 = np.cos(theta_t2), np.sin(theta_t2)
    q2x, q2y = 3.0 + 1.2 * ct2, 1.2 * st2
    q3x, q3y = 3.0 + 0.6 * ct2 + 0.5 * st2, 0.6 * st2 - 0.5 * ct2

    theta_b2 = np.arctan2(q2y - p2y, q2x - p2x)
    mech.state.set_pose("b2", q, p2x, p2y, theta_b2)

    theta_b3 = np.arctan2(q3y - p3y, q3x - p3x)
    mech.state.set_pose("b3", q, p3x, p3y, theta_b3)

    b3ex = p3x + 2.0 * np.cos(theta_b3)
    b3ey = p3y + 2.0 * np.sin(theta_b3)
    theta_b4 = np.arctan2(b3ey - q3y, b3ex - q3x)
    mech.state.set_pose("b4", q, q3x, q3y, theta_b4)

    result = solve_position(mech, q, t=angle)
    if not result.converged:
        raise RuntimeError("B2 initial config did not converge")
    step_back = position_sweep(mech, result.q, np.linspace(angle, 0.0, 15))
    return step_back.solutions[-1] if step_back.solutions[-1] is not None else result.q


def main() -> None:
    mech, force_elements, q0 = build_B2_with_gravity()

    print("Launching 6-bar type B2 viewer (Chain B, shared-binary ground)...")
    print("  Ternaries adjacent: NO")
    print("  Ground: binary (O2, O4), adjacent to BOTH ternaries")
    print("  Input: ternary T1 (driven at O2)")
    print("  Gravity: [0, -9.81] m/s^2")
    print()

    launch_interactive(
        mech, force_elements=force_elements, n_steps=360,
        coupler_body_id="t1", coupler_point_name="CP", q0=q0,
    )


if __name__ == "__main__":
    main()
