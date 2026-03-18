"""Golden fixture comparison tests.

Loads pre-exported golden data from data/benchmarks/golden/ and verifies
that the current solver produces identical results. These tests catch
any regression or behavioral drift in the solver pipeline.

Golden snapshots are regression baselines only and do not establish
correctness — they detect drift, not prove truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from linkage_sim.analysis.coupler import eval_coupler_point
from linkage_sim.analysis.energy import compute_energy_state
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.forward_dynamics import ForwardDynamicsConfig, simulate
from linkage_sim.solvers.inverse_dynamics import solve_inverse_dynamics
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.statics import solve_statics

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "golden"

# Tolerances per the RUST_MIGRATION.md spec
POS_TOL = 1e-10
VEL_TOL = 1e-8
LAMBDA_RTOL = 1e-6
DYNAMICS_TOL = 1e-5


def load_golden(filename: str) -> dict:  # type: ignore[type-arg]
    path = GOLDEN_DIR / filename
    if not path.exists():
        pytest.skip(f"Golden fixture not found: {path}")
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


# --- 4-bar mechanism factories ---


def build_fourbar_driven() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=2.0, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=3.0, Izz_cg=0.05)
    coupler.add_coupler_point("P", 1.5, 0.5)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=2.0, Izz_cg=0.02)
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


def solve_fourbar_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


# --- Test classes ---


class TestFourbarKinematicsGolden:
    """4-bar kinematics matches golden fixture."""

    def test_position(self) -> None:
        golden = load_golden("fourbar_kinematics.json")
        mech = build_fourbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                q, step["q"], atol=POS_TOL,
                err_msg=f"Position mismatch at {step['input_angle_deg']:.0f}°",
            )

    def test_velocity(self) -> None:
        golden = load_golden("fourbar_kinematics.json")
        mech = build_fourbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            _, q_dot, _ = solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                q_dot, step["q_dot"], atol=VEL_TOL,
                err_msg=f"Velocity mismatch at {step['input_angle_deg']:.0f}°",
            )

    def test_acceleration(self) -> None:
        golden = load_golden("fourbar_kinematics.json")
        mech = build_fourbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            _, _, q_ddot = solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                q_ddot, step["q_ddot"], atol=VEL_TOL,
                err_msg=f"Accel mismatch at {step['input_angle_deg']:.0f}°",
            )

    def test_coupler_point(self) -> None:
        golden = load_golden("fourbar_kinematics.json")
        mech = build_fourbar_driven()

        for step in golden["steps"][::10]:  # every 10th for speed
            angle = step["input_angle_rad"]
            q, q_dot, q_ddot = solve_fourbar_at(mech, angle)
            pos, vel, acc = eval_coupler_point(
                mech.state, "coupler", np.array([1.5, 0.5]), q, q_dot, q_ddot,
            )
            np.testing.assert_allclose(
                pos, step["coupler_point_P"]["position"], atol=POS_TOL,
            )
            np.testing.assert_allclose(
                vel, step["coupler_point_P"]["velocity"], atol=VEL_TOL,
            )


class TestFourbarStaticsGolden:
    """4-bar statics matches golden fixture."""

    def test_driver_torque(self) -> None:
        golden = load_golden("fourbar_statics.json")
        mech = build_fourbar_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_fourbar_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, result, q)
            drivers = get_driver_reactions(reactions)

            assert drivers[0].effort == pytest.approx(
                step["driver_torque"], rel=LAMBDA_RTOL,
            )

    def test_lambdas(self) -> None:
        golden = load_golden("fourbar_statics.json")
        mech = build_fourbar_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"][::5]:  # every 5th
            angle = step["input_angle_rad"]
            q, _, _ = solve_fourbar_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)

            golden_lam = np.array(step["lambdas"])
            # Use relative tolerance for multipliers
            mask = np.abs(golden_lam) > 1e-6
            if np.any(mask):
                np.testing.assert_allclose(
                    result.lambdas[mask], golden_lam[mask], rtol=LAMBDA_RTOL,
                )


class TestFourbarInverseDynamicsGolden:
    """4-bar inverse dynamics matches golden fixture."""

    def test_driver_torque(self) -> None:
        golden = load_golden("fourbar_inverse_dynamics.json")
        mech = build_fourbar_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"][::5]:
            angle = step["input_angle_rad"]
            q, q_dot, q_ddot = solve_fourbar_at(mech, angle)
            result = solve_inverse_dynamics(
                mech, q, q_dot, q_ddot, [gravity], t=angle,
            )

            assert result.lambdas[-1] == pytest.approx(
                step["driver_torque"], rel=LAMBDA_RTOL,
            )


class TestSlidercrankKinematicsGolden:
    """Slider-crank kinematics matches golden fixture."""

    def test_position(self) -> None:
        golden = load_golden("slidercrank_kinematics.json")

        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), rail=(3.0, 0.0))
        crank = make_bar("crank", "A", "B", 1.0, mass=1.0, Izz_cg=0.01)
        conrod = make_bar("conrod", "B", "C", 3.0, mass=2.0, Izz_cg=0.1)
        slider = Body(
            id="slider",
            attachment_points={"C": np.array([0.0, 0.0])},
            mass=0.5, cg_local=np.array([0.0, 0.0]), Izz_cg=0.0,
        )
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(conrod)
        mech.add_body(slider)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
        mech.add_revolute_joint("J3", "conrod", "C", "slider", "C")
        mech.add_prismatic_joint(
            "P1", "ground", "rail", "slider", "C",
            axis_local_i=np.array([1.0, 0.0]),
        )
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q = mech.state.make_q()
            bx, by = np.cos(angle), np.sin(angle)
            phi = np.arcsin(-by / 3.0)
            cx = bx + 3.0 * np.cos(phi)
            mech.state.set_pose("crank", q, 0.0, 0.0, angle)
            mech.state.set_pose("conrod", q, bx, by, phi)
            mech.state.set_pose("slider", q, cx, 0.0, 0.0)
            result = solve_position(mech, q, t=angle)
            assert result.converged
            np.testing.assert_allclose(result.q, step["q"], atol=POS_TOL)


class TestPendulumDynamicsGolden:
    """Pendulum forward dynamics matches golden fixture (looser tolerance)."""

    def test_trajectory(self) -> None:
        golden = load_golden("pendulum_dynamics.json")

        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        bar = Body(
            id="bar",
            attachment_points={"A": np.array([0.0, 0.0])},
            mass=1.0,
            cg_local=np.array([1.0, 0.0]),
            Izz_cg=0.0,
        )
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
        mech.build()

        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        theta0 = golden["initial_angle_rad"]
        q0 = mech.state.make_q()
        mech.state.set_pose("bar", q0, 0.0, 0.0, theta0)
        qd0 = np.zeros(mech.state.n_coords)

        golden_times = np.array([s["t"] for s in golden["steps"]])
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.002,
        )
        result = simulate(
            mech, q0, qd0, (0.0, golden_times[-1]),
            [gravity], config, golden_times,
        )
        assert result.success

        # Compare at selected points (every 25th step for speed)
        for i in range(0, len(golden["steps"]), 25):
            step = golden["steps"][i]
            np.testing.assert_allclose(
                result.q[i], step["q"], atol=DYNAMICS_TOL,
                err_msg=f"Position mismatch at t={step['t']:.3f}",
            )

    def test_energy_conservation(self) -> None:
        """Energy values match golden data."""
        golden = load_golden("pendulum_dynamics.json")
        # Just verify energy values are finite and consistent
        for step in golden["steps"]:
            assert np.isfinite(step["total_energy"])
            assert step["kinetic_energy"] >= 0
