#!/usr/bin/env python3
"""Launch the interactive viewer for a 6-bar mechanism (type B3) with gravity.

Usage:
    python scripts/view_sixbar_B3.py

6-bar mechanism where the two ternary links are NOT adjacent. Ground is
a binary link that is an exclusive neighbor of only ONE ternary (T1).
Ground connects to T1 and to B4, but NOT directly to T2.

Graph definition (type B3 -- Chain B, exclusive-binary ground):
    Chain: B (ternaries NOT adjacent)
    Links (6): ground(B), T1(T), B1(B), T2(T), B2(B), B4(B)
    Joints (7):
        J1: ground-T1     J2: ground-B4     J3: T1-B1
        J4: T1-B2         J5: T2-B1         J6: T2-B2
        J7: T2-B4
    Ternary links: T1 (deg 3: J1,J3,J4), T2 (deg 3: J5,J6,J7)
    Ternaries adjacent: NO (no direct T1-T2 joint)
    Ground type: binary (deg 2: J1,J2)
    Ground neighbors: T1, B4 (ground adjacent to ONE ternary only)
    Driver: ground-T1 (J1) -- drive ternary T1 relative to ground

Dimensions (Grashof-compliant for full crank rotation):
    Ground pivots: O2=(0,0), O4=(2.5,0)
    T1 (ternary, driven): P1=(0,0), P2=(1.0,0), P3=(0.5,0.5) at O2
    B1 (binary): length=2.0
    T2 (ternary): Q1=(0,0), Q2=(1.5,0), Q3=(0.8,-0.6)
    B2 (binary): length=2.0
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


def build_B3_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(2.5, 0.0))

    t1 = _make_ternary("t1", "P1", "P2", "P3",
                        p2_local=(1.0, 0.0), p3_local=(0.5, 0.5),
                        mass=1.2, Izz_cg=0.08)
    t1.add_coupler_point("CP", 0.5, 0.25)

    b1 = make_bar("b1", "B1A", "B1B", length=2.0, mass=0.6, Izz_cg=0.02)

    t2 = _make_ternary("t2", "Q1", "Q2", "Q3",
                        p2_local=(1.5, 0.0), p3_local=(0.8, -0.6),
                        mass=1.2, Izz_cg=0.08)

    b2 = make_bar("b2", "B2A", "B2B", length=2.0, mass=0.6, Izz_cg=0.02)
    b4 = make_bar("b4", "B4A", "B4B", length=2.0, mass=0.6, Izz_cg=0.02)

    mech.add_body(ground)
    mech.add_body(t1)
    mech.add_body(b1)
    mech.add_body(t2)
    mech.add_body(b2)
    mech.add_body(b4)

    # Joints per graph definition
    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1")     # ground-T1
    mech.add_revolute_joint("J2", "ground", "O4", "b4", "B4B")    # ground-B4
    mech.add_revolute_joint("J3", "t1", "P2", "b1", "B1A")        # T1-B1
    mech.add_revolute_joint("J4", "t1", "P3", "b2", "B2A")        # T1-B2
    mech.add_revolute_joint("J5", "t2", "Q1", "b1", "B1B")        # T2-B1
    mech.add_revolute_joint("J6", "t2", "Q2", "b2", "B2B")        # T2-B2
    mech.add_revolute_joint("J7", "t2", "Q3", "b4", "B4A")        # T2-B4

    mech.add_revolute_driver(
        "D1", "ground", "t1",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )

    mech.add_trace_point("TP_t2", "t2", 0.5, -0.25)

    mech.build()

    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    q0 = _find_initial(mech)
    return mech, [gravity], q0


def _find_initial(mech: Mechanism) -> np.ndarray:
    angle = 0.3
    q = mech.state.make_q()

    # T1 at O2, driven
    mech.state.set_pose("t1", q, 0.0, 0.0, angle)
    ct1, st1 = np.cos(angle), np.sin(angle)
    p2x, p2y = 1.0 * ct1, 1.0 * st1
    p3x, p3y = 0.5 * ct1 - 0.5 * st1, 0.5 * st1 + 0.5 * ct1

    # B1: from T1 P2, pointing toward the middle
    theta_b1 = angle + 0.8
    mech.state.set_pose("b1", q, p2x, p2y, theta_b1)
    b1ex = p2x + 2.0 * np.cos(theta_b1)
    b1ey = p2y + 2.0 * np.sin(theta_b1)

    # T2: Q1 at B1 end
    theta_t2 = theta_b1 + 0.5
    mech.state.set_pose("t2", q, b1ex, b1ey, theta_t2)
    ct2, st2 = np.cos(theta_t2), np.sin(theta_t2)
    q2x = b1ex + 1.5 * ct2
    q2y = b1ey + 1.5 * st2
    q3x = b1ex + 0.8 * ct2 - (-0.6) * st2
    q3y = b1ey + 0.8 * st2 + (-0.6) * ct2

    # B2: from T1 P3 toward T2 Q2
    theta_b2 = np.arctan2(q2y - p3y, q2x - p3x)
    mech.state.set_pose("b2", q, p3x, p3y, theta_b2)

    # B4: from T2 Q3 toward O4=(2.5,0)
    theta_b4 = np.arctan2(0.0 - q3y, 2.5 - q3x)
    mech.state.set_pose("b4", q, q3x, q3y, theta_b4)

    result = solve_position(mech, q, t=angle)
    if not result.converged:
        raise RuntimeError("B3 initial config did not converge")
    step_back = position_sweep(mech, result.q, np.linspace(angle, 0.0, 15))
    return step_back.solutions[-1] if step_back.solutions[-1] is not None else result.q


def main() -> None:
    mech, force_elements, q0 = build_B3_with_gravity()

    print("Launching 6-bar type B3 viewer (Chain B, exclusive-binary ground)...")
    print("  Ternaries adjacent: NO")
    print("  Ground: binary (O2, O4), adjacent to T1 only (not T2)")
    print("  Input: ternary T1 (driven at O2)")
    print("  T2 connects to ground only through B4")
    print("  Gravity: [0, -9.81] m/s^2")
    print()

    launch_interactive(
        mech, force_elements=force_elements, n_steps=360,
        q0=q0,
    )


if __name__ == "__main__":
    main()
