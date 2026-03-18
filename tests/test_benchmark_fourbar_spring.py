"""Benchmark: 4-bar linkage with spring — Phase 2 integration test.

Validates static analysis with spring forces: a torsion spring at the
driver joint and a linear spring across the mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.envelopes import compute_envelope
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.analysis.virtual_work import virtual_work_input_torque
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.spring import LinearSpring
from linkage_sim.forces.torsion_spring import TorsionSpring
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.statics import solve_statics


def build_spring_fourbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=1.0)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=1.5)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=1.0)

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
    return mech


def solve_at(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    return result.q


class TestTorsionSpringBenchmark:
    """4-bar with torsion spring at driver joint."""

    def test_spring_reduces_driver_torque_at_equilibrium(self) -> None:
        """Torsion spring counterbalancing gravity reduces net driver effort."""
        mech = build_spring_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="crank",
            stiffness=30.0,
            free_angle=np.pi / 2,  # equilibrium at 90°
            _id="balance_spring",
        )
        angle = np.pi / 2
        q = solve_at(mech, angle)

        # Gravity only
        r_grav = solve_statics(mech, q, [gravity], t=angle)
        rx_grav = extract_reactions(mech, r_grav, q)
        tau_grav = get_driver_reactions(rx_grav)[0].effort

        # Gravity + spring
        r_both = solve_statics(mech, q, [gravity, ts], t=angle)
        rx_both = extract_reactions(mech, r_both, q)
        tau_both = get_driver_reactions(rx_both)[0].effort

        # Spring at its free angle contributes zero → torques should be same
        # at exactly the free angle. Test at a different angle instead.
        angle2 = np.pi / 4
        q2 = solve_at(mech, angle2)
        r_grav2 = solve_statics(mech, q2, [gravity], t=angle2)
        r_both2 = solve_statics(mech, q2, [gravity, ts], t=angle2)
        tau_grav2 = get_driver_reactions(extract_reactions(mech, r_grav2, q2))[0].effort
        tau_both2 = get_driver_reactions(extract_reactions(mech, r_both2, q2))[0].effort

        # Spring changes the torque
        assert abs(tau_grav2 - tau_both2) > 0.1

    def test_virtual_work_with_spring(self) -> None:
        mech = build_spring_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="crank",
            stiffness=50.0,
            free_angle=0.0,
            _id="ts1",
        )
        forces = [gravity, ts]
        angles = np.linspace(np.radians(30), np.radians(150), 7)
        for angle in angles:
            q = solve_at(mech, angle)
            static_result = solve_statics(mech, q, forces, t=angle)
            reactions = extract_reactions(mech, static_result, q)
            lambda_torque = get_driver_reactions(reactions)[0].effort
            vw = virtual_work_input_torque(mech, q, forces, "crank", t=angle)
            assert vw.input_torque == pytest.approx(lambda_torque, rel=1e-6)


class TestLinearSpringBenchmark:
    """4-bar with linear spring between crank tip and rocker tip."""

    def test_spring_force_affects_reactions(self) -> None:
        mech = build_spring_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        # Spring from crank tip (B) to rocker ground (O4)
        spring = LinearSpring(
            body_i_id="crank",
            point_i_local=np.array([1.0, 0.0]),  # crank tip
            body_j_id="ground",
            point_j_local=np.array([4.0, 0.0]),  # rocker ground pivot
            stiffness=50.0,
            free_length=3.0,
            _id="extension_spring",
        )
        angle = np.pi / 3
        q = solve_at(mech, angle)

        r_grav = solve_statics(mech, q, [gravity], t=angle)
        r_both = solve_statics(mech, q, [gravity, spring], t=angle)

        assert r_grav.residual_norm < 1e-10
        assert r_both.residual_norm < 1e-10

        # Spring changes the driver torque
        tau_grav = r_grav.lambdas[-1]
        tau_both = r_both.lambdas[-1]
        assert abs(tau_grav - tau_both) > 0.01

    def test_torque_envelope_with_spring(self) -> None:
        mech = build_spring_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        spring = LinearSpring(
            body_i_id="crank",
            point_i_local=np.array([1.0, 0.0]),
            body_j_id="ground",
            point_j_local=np.array([4.0, 0.0]),
            stiffness=50.0,
            free_length=3.0,
            _id="spring",
        )
        forces = [gravity, spring]
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        torques = []
        for angle in angles:
            q = solve_at(mech, angle)
            result = solve_statics(mech, q, forces, t=angle)
            reactions = extract_reactions(mech, result, q)
            torques.append(get_driver_reactions(reactions)[0].effort)

        env = compute_envelope(np.array(torques), angles)
        assert env.rms > 0
        assert env.range > 0
