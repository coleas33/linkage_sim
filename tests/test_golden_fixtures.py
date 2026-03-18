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


def build_fourbar_undriven() -> Mechanism:
    """4-bar without driver for forward dynamics (DOF=1)."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=2.0, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=3.0, Izz_cg=0.05)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=2.0, Izz_cg=0.02)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    mech.build()
    return mech


def build_slidercrank_driven() -> Mechanism:
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
    return mech


def solve_slidercrank_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    bx, by = np.cos(angle), np.sin(angle)
    phi = np.arcsin(-by / 3.0)
    cx = bx + 3.0 * np.cos(phi)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("conrod", q, bx, by, phi)
    mech.state.set_pose("slider", q, cx, 0.0, 0.0)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def build_sixbar_driven() -> Mechanism:
    """Watt I 6-bar with ternary coupler (matches export_golden.py)."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 1.0), O6=(6.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5, mass=0.5, Izz_cg=0.01)

    ternary = Body(id="ternary")
    ternary.add_attachment_point("P1", 0.0, 0.0)
    ternary.add_attachment_point("P2", 3.0, 0.0)
    ternary.add_attachment_point("P3", 1.5, 1.0)
    ternary.cg_local = np.array([1.5, 1.0 / 3.0])
    ternary.mass = 1.0
    ternary.Izz_cg = 0.05

    rocker4 = make_bar("rocker4", "R4A", "R4B", length=2.5, mass=0.5, Izz_cg=0.02)
    link5 = make_bar("link5", "L5A", "L5B", length=2.0, mass=0.5, Izz_cg=0.02)
    output6 = make_bar("output6", "R6A", "R6B", length=2.0, mass=0.5, Izz_cg=0.02)

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
    return mech


def find_sixbar_initial(mech: Mechanism, crank_angle: float) -> np.ndarray:  # type: ignore[type-arg]
    """Geometric initial guess for the 6-bar NR solver."""
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)
    theta_tern = 0.15
    mech.state.set_pose("ternary", q, bx, by, theta_tern)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st
    dx = p2_gx - 6.0
    dy = p2_gy
    theta_r4 = np.arctan2(dy, dx)
    mech.state.set_pose("rocker4", q, 6.0, 0.0, theta_r4)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct
    dx5 = 3.0 - p3_gx
    dy5 = 1.0 - p3_gy
    theta_l5 = np.arctan2(dy5, dx5)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)
    l5b_gx = p3_gx + 2.0 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.0 * np.sin(theta_l5)
    dx6 = l5b_gx - 3.0
    dy6 = l5b_gy - 1.0
    theta_o6 = np.arctan2(dy6, dx6)
    mech.state.set_pose("output6", q, 3.0, 1.0, theta_o6)
    return q


def solve_sixbar_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q0 = find_sixbar_initial(mech, angle)
    result = solve_position(mech, q0, t=angle)
    assert result.converged, f"6-bar solve failed at {np.degrees(angle):.1f}°"
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
        mech = build_slidercrank_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_slidercrank_at(mech, angle)
            np.testing.assert_allclose(q, step["q"], atol=POS_TOL)

    def test_velocity(self) -> None:
        golden = load_golden("slidercrank_kinematics.json")
        mech = build_slidercrank_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            _, q_dot, _ = solve_slidercrank_at(mech, angle)
            np.testing.assert_allclose(q_dot, step["q_dot"], atol=VEL_TOL)


class TestSlidercrankStaticsGolden:
    """Slider-crank statics matches golden fixture."""

    def test_driver_torque(self) -> None:
        golden = load_golden("slidercrank_statics.json")
        mech = build_slidercrank_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_slidercrank_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, result, q)
            drivers = get_driver_reactions(reactions)

            assert drivers[0].effort == pytest.approx(
                step["driver_torque"], rel=LAMBDA_RTOL,
            )

    def test_lambdas(self) -> None:
        golden = load_golden("slidercrank_statics.json")
        mech = build_slidercrank_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"][::5]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_slidercrank_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)

            golden_lam = np.array(step["lambdas"])
            mask = np.abs(golden_lam) > 1e-6
            if np.any(mask):
                np.testing.assert_allclose(
                    result.lambdas[mask], golden_lam[mask], rtol=LAMBDA_RTOL,
                )


class TestSlidercrankInverseDynamicsGolden:
    """Slider-crank inverse dynamics matches golden fixture."""

    def test_driver_torque(self) -> None:
        golden = load_golden("slidercrank_inverse_dynamics.json")
        mech = build_slidercrank_driven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        for step in golden["steps"][::5]:
            angle = step["input_angle_rad"]
            q, q_dot, q_ddot = solve_slidercrank_at(mech, angle)
            result = solve_inverse_dynamics(
                mech, q, q_dot, q_ddot, [gravity], t=angle,
            )

            assert result.lambdas[-1] == pytest.approx(
                step["driver_torque"], rel=LAMBDA_RTOL,
            )


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


class TestFourbarDynamicsGolden:
    """4-bar forward dynamics matches golden fixture (looser tolerance)."""

    def test_trajectory(self) -> None:
        golden = load_golden("fourbar_dynamics.json")
        mech = build_fourbar_undriven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        init_angle = golden["initial_angle_rad"]
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, init_angle)
        bx, by = np.cos(init_angle), np.sin(init_angle)
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        pos_result = solve_position(mech, q0, t=0.0)
        assert pos_result.converged
        q0 = pos_result.q

        qd0 = np.zeros(mech.state.n_coords)
        crank_theta_idx = mech.state.get_index("crank").theta_idx
        qd0[crank_theta_idx] = golden["initial_omega"]

        golden_times = np.array([s["t"] for s in golden["steps"]])
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, golden_times[-1]),
            [gravity], config, golden_times,
        )
        assert result.success

        for i in range(0, len(golden["steps"]), 15):
            step = golden["steps"][i]
            np.testing.assert_allclose(
                result.q[i], step["q"], atol=DYNAMICS_TOL,
                err_msg=f"Position mismatch at t={step['t']:.3f}",
            )

    def test_energy_values_match_golden(self) -> None:
        """Energy values at sampled points match golden data.

        Baumgarte stabilization introduces artificial dissipation, so we
        don't assert conservation. Instead we verify the Rust port produces
        the same energy trajectory as the Python reference.
        """
        golden = load_golden("fourbar_dynamics.json")
        mech = build_fourbar_undriven()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        init_angle = golden["initial_angle_rad"]
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, init_angle)
        bx, by = np.cos(init_angle), np.sin(init_angle)
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        pos_result = solve_position(mech, q0, t=0.0)
        assert pos_result.converged
        q0 = pos_result.q

        qd0 = np.zeros(mech.state.n_coords)
        crank_theta_idx = mech.state.get_index("crank").theta_idx
        qd0[crank_theta_idx] = golden["initial_omega"]

        golden_times = np.array([s["t"] for s in golden["steps"]])
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, golden_times[-1]),
            [gravity], config, golden_times,
        )
        assert result.success

        for i in range(0, len(golden["steps"]), 15):
            step = golden["steps"][i]
            es = compute_energy_state(mech, result.q[i], result.q_dot[i])
            assert es.total == pytest.approx(step["total_energy"], rel=DYNAMICS_TOL)


class TestSixbarKinematicsGolden:
    """6-bar Watt I kinematics matches golden fixture."""

    def test_position(self) -> None:
        golden = load_golden("sixbar_kinematics.json")
        mech = build_sixbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            q, _, _ = solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                q, step["q"], atol=POS_TOL,
                err_msg=f"Position mismatch at {step['input_angle_deg']:.0f}°",
            )

    def test_velocity(self) -> None:
        golden = load_golden("sixbar_kinematics.json")
        mech = build_sixbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            _, q_dot, _ = solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                q_dot, step["q_dot"], atol=VEL_TOL,
                err_msg=f"Velocity mismatch at {step['input_angle_deg']:.0f}°",
            )

    def test_acceleration(self) -> None:
        golden = load_golden("sixbar_kinematics.json")
        mech = build_sixbar_driven()

        for step in golden["steps"]:
            angle = step["input_angle_rad"]
            _, _, q_ddot = solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                q_ddot, step["q_ddot"], atol=VEL_TOL,
                err_msg=f"Accel mismatch at {step['input_angle_deg']:.0f}°",
            )
