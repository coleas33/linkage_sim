#!/usr/bin/env python3
"""Launch the interactive viewer for a 6-bar mechanism (type A2) with gravity.

Usage:
    python scripts/view_sixbar_A2.py

6-bar mechanism where the two ternary links ARE adjacent (share a direct
revolute joint). Ground IS one of the ternary links (3 pivot points).
The other ternary is a moving body directly connected to ground.

Graph definition (type A2 -- Chain A, ternary ground):
    Chain: A (ternaries adjacent)
    Links (6): ground(T), B1(B), B2(B), T2(T), B3(B), B4(B)
    Joints (7):
        J1: ground-B1     J2: ground-B2     J3: ground-T2
        J4: T2-B3         J5: T2-B4         J6: B1-B4
        J7: B2-B3
    Ternary links: ground (deg 3: J1,J2,J3), T2 (deg 3: J3,J4,J5)
    Ternaries adjacent: YES (joint J3: ground-T2)
    Ground type: ternary (deg 3)
    Driver: ground-B1 (J1) -- drive binary crank B1
    Loops:
        Loop 1: ground - B1 - B4 - T2 - ground  (4-bar)
        Loop 2: ground - B2 - B3 - T2 - ground  (4-bar)

Dimensions (Grashof-compliant for full crank rotation):
    Ground pivots: O2=(0,0), O4=(4.5,0), O6=(2.8,0)
    B1 (crank, driven): length=1.0, at O2
    B2 (binary): length=1.5, at O4
    T2 (ternary): Q1=(0,0), Q2=(2.5,0), Q3=(1.5,1.0)
    B3 (binary): length=2.5
    B4 (binary): length=2.5

    Loop 1 (O2-B1-B4-T2_Q3-T2_Q1-O6):
        ground=2.8, B1=1.0, B4=2.5, T2_arm(Q1-Q3)=1.803
        S+L=1.0+2.8=3.8 <= P+Q=2.5+1.803=4.303  GRASHOF
    Loop 2 (O4-B2-B3-T2_Q2-T2_Q1-O6):
        ground=1.7, B2=1.5, B3=2.5, T2_arm(Q1-Q2)=2.5
        S+L=1.5+2.5=4.0 <= P+Q=1.7+2.5=4.2  GRASHOF

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


def build_A2_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    mech = Mechanism()

    # Ground: ternary, 3 pivots
    ground = make_ground(O2=(0.0, 0.0), O4=(4.5, 0.0), O6=(2.8, 0.0))

    # B1: binary crank at O2 (driven)
    b1 = make_bar("b1", "B1A", "B1B", length=1.0, mass=0.3, Izz_cg=0.005)

    # B2: binary link at O4
    b2 = make_bar("b2", "B2A", "B2B", length=1.5, mass=0.4, Izz_cg=0.01)

    # T2: ternary, adjacent to ground via J3
    t2 = _make_ternary("t2", "Q1", "Q2", "Q3",
                        p2_local=(2.5, 0.0), p3_local=(1.5, 1.0),
                        mass=1.5, Izz_cg=0.12)
    t2.add_coupler_point("CP", 1.25, 0.5)

    # B3: binary link
    b3 = make_bar("b3", "B3A", "B3B", length=2.5, mass=0.6, Izz_cg=0.03)

    # B4: binary link
    b4 = make_bar("b4", "B4A", "B4B", length=2.5, mass=0.6, Izz_cg=0.03)

    mech.add_body(ground)
    mech.add_body(b1)
    mech.add_body(b2)
    mech.add_body(t2)
    mech.add_body(b3)
    mech.add_body(b4)

    # Joints per graph definition
    mech.add_revolute_joint("J1", "ground", "O2", "b1", "B1A")    # ground-B1
    mech.add_revolute_joint("J2", "ground", "O4", "b2", "B2A")    # ground-B2
    mech.add_revolute_joint("J3", "ground", "O6", "t2", "Q1")     # ground-T2 (adjacent!)
    mech.add_revolute_joint("J4", "t2", "Q2", "b3", "B3A")        # T2-B3
    mech.add_revolute_joint("J5", "t2", "Q3", "b4", "B4A")        # T2-B4
    mech.add_revolute_joint("J6", "b1", "B1B", "b4", "B4B")       # B1-B4
    mech.add_revolute_joint("J7", "b2", "B2B", "b3", "B3B")       # B2-B3

    mech.add_revolute_driver(
        "D1", "ground", "b1",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()

    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    q0 = _find_initial(mech)
    return mech, [gravity], q0


def _find_initial(mech: Mechanism) -> np.ndarray:
    angle = np.radians(5)
    q = mech.state.make_q()

    # B1 crank at O2
    mech.state.set_pose("b1", q, 0.0, 0.0, angle)
    b1x, b1y = 1.0 * np.cos(angle), 1.0 * np.sin(angle)

    # T2: Q1 at O6=(2.8,0)
    theta_t2 = -0.5
    mech.state.set_pose("t2", q, 2.8, 0.0, theta_t2)
    ct2, st2 = np.cos(theta_t2), np.sin(theta_t2)

    # T2 Q3 global -> B4 connects to B1
    q3x = 2.8 + 1.5 * ct2 - 1.0 * st2
    q3y = 1.5 * st2 + 1.0 * ct2
    theta_b4 = np.arctan2(b1y - q3y, b1x - q3x)
    mech.state.set_pose("b4", q, q3x, q3y, theta_b4)

    # T2 Q2 global -> B3 connects to B2
    q2x = 2.8 + 2.5 * ct2
    q2y = 2.5 * st2
    theta_b2 = np.arctan2(q2y, q2x - 4.5)
    mech.state.set_pose("b2", q, 4.5, 0.0, theta_b2)
    b2x = 4.5 + 1.5 * np.cos(theta_b2)
    b2y = 1.5 * np.sin(theta_b2)
    theta_b3 = np.arctan2(b2y - q2y, b2x - q2x)
    mech.state.set_pose("b3", q, q2x, q2y, theta_b3)

    result = solve_position(mech, q, t=angle)
    if not result.converged:
        raise RuntimeError("A2 initial config did not converge")
    step_back = position_sweep(mech, result.q, np.linspace(angle, 0.0, 15))
    return step_back.solutions[-1] if step_back.solutions[-1] is not None else result.q


def main() -> None:
    mech, force_elements, q0 = build_A2_with_gravity()

    print("Launching 6-bar type A2 viewer (Chain A, ternary ground)...")
    print("  Ternaries adjacent: YES (ground-T2 share joint J3)")
    print("  Ground: ternary (O2, O4, O6)")
    print("  Two 4-bar sub-loops through ground and T2")
    print("  Gravity: [0, -9.81] m/s^2")
    print()

    launch_interactive(
        mech, force_elements=force_elements, n_steps=360,
        coupler_body_id="t2", coupler_point_name="CP", q0=q0,
    )


if __name__ == "__main__":
    main()
