"""Tests for coupler point evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.coupler import (
    CouplerPointResult,
    eval_all_coupler_points,
    eval_coupler_point,
)
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import State
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)


class TestEvalCouplerPoint:
    def test_stationary_point(self) -> None:
        """Point on a stationary body at origin."""
        state = State()
        state.register_body("body")
        q = state.make_q()  # at origin, θ=0
        q_dot = state.make_q()
        q_ddot = state.make_q()

        point_local = np.array([0.5, 0.3])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        np.testing.assert_array_almost_equal(pos, [0.5, 0.3])
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0])
        np.testing.assert_array_almost_equal(accel, [0.0, 0.0])

    def test_rotating_point_position(self) -> None:
        """Point on body at θ=π/2."""
        state = State()
        state.register_body("body")
        q = state.make_q()
        state.set_pose("body", q, 0.0, 0.0, np.pi / 2)
        q_dot = state.make_q()
        q_ddot = state.make_q()

        point_local = np.array([1.0, 0.0])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        np.testing.assert_array_almost_equal(pos, [0.0, 1.0])

    def test_translating_point_velocity(self) -> None:
        """Point on body translating at constant velocity."""
        state = State()
        state.register_body("body")
        q = state.make_q()
        q_dot = state.make_q()
        q_dot[0] = 2.0  # ẋ = 2
        q_dot[1] = 3.0  # ẏ = 3
        q_ddot = state.make_q()

        point_local = np.array([0.5, 0.0])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        np.testing.assert_array_almost_equal(vel, [2.0, 3.0])

    def test_rotating_point_velocity(self) -> None:
        """Point on body rotating at constant angular velocity."""
        state = State()
        state.register_body("body")
        q = state.make_q()
        # Body at origin, θ=0
        q_dot = state.make_q()
        q_dot[2] = 5.0  # θ̇ = 5
        q_ddot = state.make_q()

        point_local = np.array([1.0, 0.0])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        # v = ṙ + B*s*θ̇ = [0,0] + [-sin(0), cos(0)]*(1)*5 = [0, 5]
        np.testing.assert_array_almost_equal(vel, [0.0, 5.0])

    def test_centripetal_acceleration(self) -> None:
        """Pure rotation: acceleration should be centripetal = -ω²*r."""
        state = State()
        state.register_body("body")
        q = state.make_q()
        omega = 3.0
        q_dot = state.make_q()
        q_dot[2] = omega  # θ̇ = ω
        q_ddot = state.make_q()  # θ̈ = 0

        point_local = np.array([1.0, 0.0])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        # accel = -A*s*ω² = -[1,0]*9 = [-9, 0]
        np.testing.assert_array_almost_equal(accel, [-9.0, 0.0])

    def test_ground_point(self) -> None:
        """Point on ground should have zero velocity and acceleration."""
        state = State()
        state.register_body("body")
        q = state.make_q()
        q_dot = state.make_q()
        q_ddot = state.make_q()

        point_local = np.array([1.0, 2.0])
        pos, vel, accel = eval_coupler_point(
            state, "ground", point_local, q, q_dot, q_ddot
        )
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0])
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0])
        np.testing.assert_array_almost_equal(accel, [0.0, 0.0])

    def test_velocity_finite_difference(self) -> None:
        """Velocity should match FD of position for rotating body."""
        state = State()
        state.register_body("body")

        dt = 1e-7
        theta = 0.5
        omega = 2.0

        q = state.make_q()
        state.set_pose("body", q, 0.0, 0.0, theta)
        q_plus = state.make_q()
        state.set_pose("body", q_plus, 0.0, 0.0, theta + omega * dt)

        q_dot = state.make_q()
        q_dot[2] = omega
        q_ddot = state.make_q()

        point_local = np.array([0.8, 0.3])
        pos, vel, accel = eval_coupler_point(
            state, "body", point_local, q, q_dot, q_ddot
        )
        pos_plus = state.body_point_global("body", point_local, q_plus)
        vel_fd = (pos_plus - pos) / dt

        np.testing.assert_array_almost_equal(vel, vel_fd, decimal=5)


class TestEvalAllCouplerPoints:
    def test_collects_all_coupler_points(self) -> None:
        """Should return results for all defined coupler points."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        crank.add_coupler_point("mid", 0.5, 0.0)
        crank.add_coupler_point("tip", 1.0, 0.2)

        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
        mech.add_constant_speed_driver("D1", "ground", "crank", omega=1.0)
        mech.build()

        q = mech.state.make_q()
        result = solve_position(mech, q, t=0.0)
        q_dot = solve_velocity(mech, result.q, t=0.0)
        q_ddot = solve_acceleration(mech, result.q, q_dot, t=0.0)

        coupler_results = eval_all_coupler_points(
            mech, result.q, q_dot, q_ddot
        )
        assert len(coupler_results) == 2
        names = {r.point_name for r in coupler_results}
        assert names == {"mid", "tip"}

    def test_empty_when_no_coupler_points(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
        mech.add_constant_speed_driver("D1", "ground", "crank", omega=1.0)
        mech.build()

        q = mech.state.make_q()
        q_dot = mech.state.make_q()
        q_ddot = mech.state.make_q()

        coupler_results = eval_all_coupler_points(mech, q, q_dot, q_ddot)
        assert len(coupler_results) == 0

    def test_result_is_frozen(self) -> None:
        result = CouplerPointResult(
            body_id="test",
            point_name="pt",
            position=np.zeros(2),
            velocity=np.zeros(2),
            acceleration=np.zeros(2),
        )
        with pytest.raises(AttributeError):
            result.body_id = "other"  # type: ignore[misc]

    def test_fourbar_coupler_curve(self) -> None:
        """Coupler point on a driven 4-bar traces a smooth curve."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        coupler = make_bar("coupler", "B", "C", length=3.0)
        coupler.add_coupler_point("P", 1.5, 0.5)
        rocker = make_bar("rocker", "D", "C", length=2.0)

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
            f=lambda t: t,
            f_dot=lambda t: 1.0,
            f_ddot=lambda t: 0.0,
        )
        mech.build()

        # Solve at angle = π/4
        angle = np.pi / 4
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        mech.state.set_pose("coupler", q, np.cos(angle), np.sin(angle), 0.0)
        mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)

        result = solve_position(mech, q, t=angle)
        assert result.converged

        q_dot = solve_velocity(mech, result.q, t=angle)
        q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)

        coupler_results = eval_all_coupler_points(
            mech, result.q, q_dot, q_ddot
        )
        assert len(coupler_results) == 1
        pt = coupler_results[0]
        assert pt.body_id == "coupler"
        assert pt.point_name == "P"
        assert pt.position.shape == (2,)
        assert pt.velocity.shape == (2,)
        assert pt.acceleration.shape == (2,)
        # Velocity should be nonzero (mechanism is moving)
        assert np.linalg.norm(pt.velocity) > 0.01
