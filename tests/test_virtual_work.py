"""Tests for virtual work cross-check of input torque."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.torsion_spring import TorsionSpring
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.analysis.virtual_work import virtual_work_input_torque
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.statics import solve_statics


def build_fourbar(mass: float = 0.0) -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0, mass=mass)
    coupler = make_bar("coupler", "B", "C", length=3.0, mass=mass)
    rocker = make_bar("rocker", "D", "C", length=2.0, mass=mass)

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


class TestVirtualWorkCrossCheck:
    """Virtual work torque must match Lagrange multiplier torque."""

    def test_gravity_single_angle(self) -> None:
        """Virtual work matches statics at one angle with gravity."""
        mech = build_fourbar(mass=2.0)
        angle = np.pi / 4
        q = solve_at(mech, angle)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        # Statics result
        static_result = solve_statics(mech, q, [gravity], t=angle)
        reactions = extract_reactions(mech, static_result, q)
        drivers = get_driver_reactions(reactions)
        lambda_torque = drivers[0].effort

        # Virtual work result
        vw_result = virtual_work_input_torque(
            mech, q, [gravity], "crank", t=angle
        )

        assert np.isfinite(vw_result.input_torque)
        assert vw_result.input_torque == pytest.approx(lambda_torque, rel=1e-6)

    def test_gravity_sweep(self) -> None:
        """Virtual work matches statics across multiple angles."""
        mech = build_fourbar(mass=1.5)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        angles = np.linspace(np.radians(30), np.radians(150), 13)
        for angle in angles:
            q = solve_at(mech, angle)

            static_result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, static_result, q)
            lambda_torque = get_driver_reactions(reactions)[0].effort

            vw_result = virtual_work_input_torque(
                mech, q, [gravity], "crank", t=angle
            )

            assert vw_result.input_torque == pytest.approx(
                lambda_torque, rel=1e-6
            ), f"Mismatch at {np.degrees(angle):.0f} deg"

    def test_torsion_spring(self) -> None:
        """Virtual work matches statics with a torsion spring."""
        mech = build_fourbar(mass=0.0)
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="crank",
            stiffness=100.0,
            free_angle=0.0,
            _id="ts1",
        )
        angle = np.pi / 6
        q = solve_at(mech, angle)

        static_result = solve_statics(mech, q, [ts], t=angle)
        reactions = extract_reactions(mech, static_result, q)
        lambda_torque = get_driver_reactions(reactions)[0].effort

        vw_result = virtual_work_input_torque(
            mech, q, [ts], "crank", t=angle
        )

        assert vw_result.input_torque == pytest.approx(lambda_torque, rel=1e-6)

    def test_combined_forces(self) -> None:
        """Virtual work matches statics with gravity + spring."""
        mech = build_fourbar(mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="crank",
            stiffness=50.0,
            free_angle=np.pi / 4,
            _id="ts1",
        )
        angle = np.pi / 3
        q = solve_at(mech, angle)
        forces = [gravity, ts]

        static_result = solve_statics(mech, q, forces, t=angle)
        reactions = extract_reactions(mech, static_result, q)
        lambda_torque = get_driver_reactions(reactions)[0].effort

        vw_result = virtual_work_input_torque(
            mech, q, forces, "crank", t=angle
        )

        assert vw_result.input_torque == pytest.approx(lambda_torque, rel=1e-6)

    def test_zero_force_zero_torque(self) -> None:
        """No forces → zero torque from both methods."""
        mech = build_fourbar(mass=0.0)
        angle = np.pi / 4
        q = solve_at(mech, angle)

        vw_result = virtual_work_input_torque(
            mech, q, [], "crank", t=angle
        )

        assert vw_result.input_torque == pytest.approx(0.0, abs=1e-10)
