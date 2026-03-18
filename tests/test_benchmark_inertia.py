"""Benchmark: 4-bar with inertia — Phase 3 integration test.

Tests inverse dynamics pipeline including:
- Mass matrix assembly
- Inverse dynamics solve
- Comparison with statics (zero mass limit)
- Motor sizing feasibility check
- Force element contribution breakdown
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.envelopes import compute_envelope
from linkage_sim.analysis.force_breakdown import evaluate_contributions, inertia_contribution
from linkage_sim.analysis.motor_sizing import check_motor_sizing
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.inverse_dynamics import solve_inverse_dynamics
from linkage_sim.solvers.kinematics import solve_acceleration, solve_position, solve_velocity
from linkage_sim.solvers.statics import solve_statics


def build_fourbar(mass: float, Izz: float) -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=mass, Izz_cg=Izz)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=mass * 1.5, Izz_cg=Izz * 3)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=mass, Izz_cg=Izz)
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


def solve_full(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_s = result.q
    q_dot = solve_velocity(mech, q_s, t=angle)
    q_ddot = solve_acceleration(mech, q_s, q_dot, t=angle)
    return q_s, q_dot, q_ddot


class TestInverseWithGravity:
    """Inverse dynamics with gravity + inertia."""

    def test_sweep_residuals(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.05)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        angles = np.linspace(np.radians(30), np.radians(150), 13)
        for angle in angles:
            q, qd, qdd = solve_full(mech, angle)
            r = solve_inverse_dynamics(mech, q, qd, qdd, [gravity], t=angle)
            assert r.residual_norm < 1e-10

    def test_torque_includes_inertia(self) -> None:
        """Inverse dynamics torque differs from statics due to inertia."""
        mech = build_fourbar(mass=5.0, Izz=0.5)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        angle = np.pi / 4
        q, qd, qdd = solve_full(mech, angle)

        static_r = solve_statics(mech, q, [gravity], t=angle)
        id_r = solve_inverse_dynamics(mech, q, qd, qdd, [gravity], t=angle)

        # At ω=1 rad/s with moderate mass, inertia effect is small but nonzero
        assert abs(static_r.lambdas[-1] - id_r.lambdas[-1]) > 0.001

    def test_torque_envelope(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.05)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        torques = []
        for angle in angles:
            q, qd, qdd = solve_full(mech, angle)
            r = solve_inverse_dynamics(mech, q, qd, qdd, [gravity], t=angle)
            torques.append(r.lambdas[-1])

        env = compute_envelope(np.array(torques), angles)
        assert env.peak > 0
        assert env.rms > 0


class TestMotorSizing:
    """Motor sizing assistant."""

    def test_adequate_motor(self) -> None:
        """Large motor should be feasible."""
        speeds = np.array([1.0, 2.0, 3.0])
        torques = np.array([5.0, 3.0, 1.0])
        r = check_motor_sizing(speeds, torques, stall_torque=20.0, no_load_speed=10.0)
        assert r.all_feasible
        assert r.worst_margin > 0

    def test_inadequate_motor(self) -> None:
        """Small motor should fail."""
        speeds = np.array([1.0, 2.0, 3.0])
        torques = np.array([5.0, 3.0, 1.0])
        r = check_motor_sizing(speeds, torques, stall_torque=2.0, no_load_speed=5.0)
        assert not r.all_feasible
        assert r.worst_margin < 0

    def test_overspeed_infeasible(self) -> None:
        """Operating above no-load speed is infeasible."""
        speeds = np.array([0.0, 50.0, 120.0])
        torques = np.array([1.0, 1.0, 1.0])
        r = check_motor_sizing(speeds, torques, stall_torque=10.0, no_load_speed=100.0)
        assert not r.feasible[2]  # 120 > 100 = overspeed

    def test_zero_torque_always_feasible(self) -> None:
        speeds = np.array([50.0])
        torques = np.array([0.0])
        r = check_motor_sizing(speeds, torques, stall_torque=10.0, no_load_speed=100.0)
        assert r.all_feasible


class TestForceBreakdown:
    """Force element contribution breakdown."""

    def test_contributions_sum_to_total(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.05)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        q, qd, _ = solve_full(mech, np.pi / 4)

        from linkage_sim.forces.assembly import assemble_Q
        Q_total = assemble_Q(mech.state, [gravity], q, qd, 0.0)

        contributions = evaluate_contributions(mech.state, [gravity], q, qd, 0.0)
        Q_sum = sum(c.Q for c in contributions)
        np.testing.assert_allclose(Q_sum, Q_total)

    def test_inertia_contribution(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.05)
        q, qd, qdd = solve_full(mech, np.pi / 4)
        r = solve_inverse_dynamics(mech, q, qd, qdd, [], t=np.pi / 4)

        ic = inertia_contribution(r.M_q_ddot)
        assert ic.element_id == "inertia"
        assert ic.Q_norm > 0


class TestBenchmarkSliderCrankMotor:
    """Slider-crank with motor T-ω check."""

    def test_motor_sizing_on_slider_crank(self) -> None:
        """Build slider-crank, run inverse dynamics, check motor."""
        from linkage_sim.core.bodies import Body
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), rail=(3.0, 0.0))
        crank = make_bar("crank", "A", "B", 1.0, mass=1.0, Izz_cg=0.01)
        conrod = make_bar("conrod", "B", "C", 3.0, mass=2.0, Izz_cg=0.1)
        slider = Body(id="slider", attachment_points={"C": np.array([0.0, 0.0])},
                       mass=0.5, cg_local=np.array([0.0, 0.0]), Izz_cg=0.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(conrod)
        mech.add_body(slider)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
        mech.add_revolute_joint("J3", "conrod", "C", "slider", "C")
        mech.add_prismatic_joint("P1", "ground", "rail", "slider", "C",
                                  axis_local_i=np.array([1.0, 0.0]))
        mech.add_revolute_driver("D1", "ground", "crank",
                                  f=lambda t: t, f_dot=lambda t: 1.0,
                                  f_ddot=lambda t: 0.0)
        mech.build()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        angles = np.linspace(np.radians(15), np.radians(345), 24)
        speeds = []
        torques = []
        for angle in angles:
            q = mech.state.make_q()
            bx, by = np.cos(angle), np.sin(angle)
            phi = np.arcsin(-by / 3.0)
            cx = bx + 3.0 * np.cos(phi)
            mech.state.set_pose("crank", q, 0.0, 0.0, angle)
            mech.state.set_pose("conrod", q, bx, by, phi)
            mech.state.set_pose("slider", q, cx, 0.0, 0.0)
            pr = solve_position(mech, q, t=angle)
            assert pr.converged
            qd = solve_velocity(mech, pr.q, t=angle)
            qdd = solve_acceleration(mech, pr.q, qd, t=angle)
            r = solve_inverse_dynamics(mech, pr.q, qd, qdd, [gravity], t=angle)
            assert r.residual_norm < 1e-9

            crank_idx = mech.state.get_index("crank")
            speeds.append(qd[crank_idx.theta_idx])
            torques.append(r.lambdas[-1])

        # Check motor sizing
        result = check_motor_sizing(
            np.array(speeds), np.array(torques),
            stall_torque=50.0, no_load_speed=10.0,
        )
        assert result.all_feasible  # 50 N·m motor should handle light slider-crank
