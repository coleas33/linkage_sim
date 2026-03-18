#!/usr/bin/env python3
"""Launch the interactive viewer for a 6-bar mechanism (type A1) with gravity.

Usage:
    python scripts/view_sixbar_A1.py

6-bar mechanism where the two ternary links ARE adjacent (share a direct
revolute joint). Ground is a binary link. Both ternary links are moving
bodies. The two 4-bar sub-loops share the ternary-ternary joint edge.

Graph definition (type A1 -- Chain A, binary ground):
    Chain: A (ternaries adjacent)
    Links (6): ground(B), T1(T), B2(B), T2(T), B3(B), B4(B)
    Joints (7):
        J1: ground-T1     J2: T1-B2       J3: T1-T2
        J4: T2-B3         J5: T2-B4       J6: B2-B3
        J7: B4-ground
    Ternary links: T1 (deg 3: J1,J2,J3), T2 (deg 3: J3,J4,J5)
    Ternaries adjacent: YES (joint J3: T1-T2)
    Ground type: binary (deg 2: J1,J7)
    Driver: ground-T1 (J1)
    Loops:
        Loop 1: ground - T1 - T2 - B4 - ground  (4-bar)
        Loop 2: T1 - B2 - B3 - T2 - T1          (4-bar)

    Note: Conventional name depends on textbook -- may be called
    "Watt I" or "Stephenson I" depending on the adjacency convention used.

Dimensions:
    Ground pivots: O2=(0,0), O4=(5,0)
    T1 (ternary): P1=(0,0), P2=(3,0), P3=(1.5,1.0)
    B2 (binary): length=2.5
    T2 (ternary): Q1=(0,0), Q2=(2.5,0), Q3=(1.0,1.0)
    B3 (binary): length=2.5
    B4 (binary): length=2.0

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


def build_A1_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    mech = Mechanism()

    # Ground: binary, 2 pivots
    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 0.0))

    # T1: ternary, driven by ground at P1
    t1 = _make_ternary("t1", "P1", "P2", "P3",
                        p2_local=(1.5, 0.0), p3_local=(0.8, 0.6),
                        mass=1.0, Izz_cg=0.06)
    t1.add_coupler_point("CP", 0.75, 0.3)

    # B2: binary link
    b2 = make_bar("b2", "B2A", "B2B", length=2.0, mass=0.6, Izz_cg=0.02)

    # T2: ternary, adjacent to T1
    t2 = _make_ternary("t2", "Q1", "Q2", "Q3",
                        p2_local=(1.5, 0.0), p3_local=(0.8, 0.6),
                        mass=1.0, Izz_cg=0.06)

    # B3: binary link
    b3 = make_bar("b3", "B3A", "B3B", length=2.0, mass=0.6, Izz_cg=0.02)

    # B4: binary link
    b4 = make_bar("b4", "B4A", "B4B", length=2.0, mass=0.5, Izz_cg=0.01)

    mech.add_body(ground)
    mech.add_body(t1)
    mech.add_body(b2)
    mech.add_body(t2)
    mech.add_body(b3)
    mech.add_body(b4)

    # Joints per graph definition
    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1")     # ground-T1
    mech.add_revolute_joint("J2", "t1", "P2", "b2", "B2A")        # T1-B2
    mech.add_revolute_joint("J3", "t1", "P3", "t2", "Q1")         # T1-T2 (adjacent!)
    mech.add_revolute_joint("J4", "t2", "Q2", "b3", "B3A")        # T2-B3
    mech.add_revolute_joint("J5", "t2", "Q3", "b4", "B4A")        # T2-B4
    mech.add_revolute_joint("J6", "b2", "B2B", "b3", "B3B")       # B2-B3
    mech.add_revolute_joint("J7", "b4", "B4B", "ground", "O4")    # B4-ground

    mech.add_revolute_driver(
        "D1", "ground", "t1",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()

    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    q0 = _find_initial(mech)
    return mech, [gravity], q0


def _find_initial(mech: Mechanism) -> np.ndarray:
    angle = np.radians(10)
    q = mech.state.make_q()

    ct, st = np.cos(angle), np.sin(angle)
    mech.state.set_pose("t1", q, 0.0, 0.0, angle)
    p2x, p2y = 1.5 * ct, 1.5 * st
    p3x, p3y = 0.8 * ct - 0.6 * st, 0.8 * st + 0.6 * ct

    theta_t2 = angle - 1.5
    mech.state.set_pose("t2", q, p3x, p3y, theta_t2)
    ct2, st2 = np.cos(theta_t2), np.sin(theta_t2)
    q2x = p3x + 1.5 * ct2
    q2y = p3y + 1.5 * st2
    q3x = p3x + 0.8 * ct2 - 0.6 * st2
    q3y = p3y + 0.8 * st2 + 0.6 * ct2

    theta_b2 = np.arctan2(q2y - p2y, q2x - p2x)
    mech.state.set_pose("b2", q, p2x, p2y, theta_b2)
    b2ex = p2x + 2.0 * np.cos(theta_b2)
    b2ey = p2y + 2.0 * np.sin(theta_b2)
    theta_b3 = np.arctan2(b2ey - q2y, b2ex - q2x)
    mech.state.set_pose("b3", q, q2x, q2y, theta_b3)

    theta_b4 = np.arctan2(0.0 - q3y, 3.0 - q3x)
    mech.state.set_pose("b4", q, q3x, q3y, theta_b4)

    result = solve_position(mech, q, t=angle)
    if not result.converged:
        raise RuntimeError("A1 initial config did not converge")
    step_back = position_sweep(mech, result.q, np.linspace(angle, 0.0, 15))
    return step_back.solutions[-1] if step_back.solutions[-1] is not None else result.q


def main() -> None:
    mech, force_elements, q0 = build_A1_with_gravity()

    print("Launching 6-bar type A1 viewer (Chain A, binary ground)...")
    print("  Ternaries adjacent: YES (T1-T2 share joint J3)")
    print("  Ground: binary (O2, O4)")
    print("  Two 4-bar sub-loops sharing the T1-T2 edge")
    print("  Gravity: [0, -9.81] m/s^2")
    print()

    launch_interactive(
        mech, force_elements=force_elements, n_steps=360,
        coupler_body_id="t1", coupler_point_name="CP", q0=q0,
    )


if __name__ == "__main__":
    main()
