"""Tests for viscous damper force elements (translational and rotary)."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.state import State
from linkage_sim.forces.viscous_damper import RotaryDamper, TranslationalDamper


# --- Translational damper ---


class TestTranslationalDamper:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_zero_velocity_zero_force(self) -> None:
        """No relative motion → no damping force."""
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            damping=100.0,
            _id="d1",
        )
        q = np.array([3.0, 0.0, 0.0])
        q_dot = np.zeros(3)
        Q = d.evaluate(self.state, q, q_dot, 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_extending_produces_opposing_force(self) -> None:
        """Body moving away from ground → damper opposes extension."""
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            damping=50.0,
            _id="d1",
        )
        q = np.array([3.0, 0.0, 0.0])  # body at (3,0)
        q_dot = np.array([2.0, 0.0, 0.0])  # moving +x (extending)
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        # Extension rate = 2 m/s, F = -50*2 = -100 N on body (toward ground)
        assert Q[0] == pytest.approx(-100.0)
        assert Q[1] == pytest.approx(0.0)

    def test_compressing_produces_opposing_force(self) -> None:
        """Body moving toward ground → damper opposes compression."""
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            damping=50.0,
            _id="d1",
        )
        q = np.array([3.0, 0.0, 0.0])
        q_dot = np.array([-2.0, 0.0, 0.0])  # moving -x (compressing)
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        assert Q[0] == pytest.approx(100.0)  # pushes away from ground

    def test_transverse_motion_no_force(self) -> None:
        """Motion perpendicular to damper line → no damping force."""
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            damping=100.0,
            _id="d1",
        )
        q = np.array([3.0, 0.0, 0.0])  # line along x-axis
        q_dot = np.array([0.0, 5.0, 0.0])  # moving pure y (transverse)
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        # No component along the line → zero force
        np.testing.assert_allclose(Q[:2], np.zeros(2), atol=1e-12)

    def test_coincident_points_zero_force(self) -> None:
        """Zero-length damper: no direction, no force."""
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            damping=100.0,
            _id="d1",
        )
        q = np.array([0.0, 0.0, 0.0])
        q_dot = np.array([1.0, 0.0, 0.0])
        Q = d.evaluate(self.state, q, q_dot, 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_has_id(self) -> None:
        d = TranslationalDamper(
            body_i_id="ground",
            point_i_local=np.zeros(2),
            body_j_id="b1",
            point_j_local=np.zeros(2),
            damping=10.0,
            _id="damper_1",
        )
        assert d.id == "damper_1"


class TestTranslationalDamperTwoBodies:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")
        self.state.register_body("b2")

    def test_action_reaction(self) -> None:
        """Net translational force sums to zero."""
        d = TranslationalDamper(
            body_i_id="b1",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b2",
            point_j_local=np.array([0.0, 0.0]),
            damping=100.0,
            _id="d1",
        )
        q = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        q_dot = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        Fx_total = Q[0] + Q[3]
        Fy_total = Q[1] + Q[4]
        assert Fx_total == pytest.approx(0.0)
        assert Fy_total == pytest.approx(0.0)


# --- Rotary damper ---


class TestRotaryDamper:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_zero_velocity_zero_torque(self) -> None:
        d = RotaryDamper(
            body_i_id="ground", body_j_id="b1", damping=10.0, _id="rd1"
        )
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = d.evaluate(self.state, q, q_dot, 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_ccw_rotation_produces_cw_torque(self) -> None:
        d = RotaryDamper(
            body_i_id="ground", body_j_id="b1", damping=10.0, _id="rd1"
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 2.0])  # ω = 2 CCW
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        assert Q[2] == pytest.approx(-20.0)  # opposes CCW

    def test_cw_rotation_produces_ccw_torque(self) -> None:
        d = RotaryDamper(
            body_i_id="ground", body_j_id="b1", damping=10.0, _id="rd1"
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, -3.0])
        Q = d.evaluate(self.state, q, q_dot, 0.0)

        assert Q[2] == pytest.approx(30.0)

    def test_no_translational_force(self) -> None:
        d = RotaryDamper(
            body_i_id="ground", body_j_id="b1", damping=10.0, _id="rd1"
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 5.0])
        Q = d.evaluate(self.state, q, q_dot, 0.0)
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(0.0)

    def test_action_reaction_two_bodies(self) -> None:
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        d = RotaryDamper(
            body_i_id="b1", body_j_id="b2", damping=20.0, _id="rd1"
        )
        q = np.zeros(6)
        q_dot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 4.0])
        Q = d.evaluate(state, q, q_dot, 0.0)

        assert Q[2] + Q[5] == pytest.approx(0.0)

    def test_relative_velocity_only(self) -> None:
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        d = RotaryDamper(
            body_i_id="b1", body_j_id="b2", damping=10.0, _id="rd1"
        )
        q = np.zeros(6)
        # Same relative velocity (2 rad/s)
        qd1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        qd2 = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 7.0])
        Q1 = d.evaluate(state, q, qd1, 0.0)
        Q2 = d.evaluate(state, q, qd2, 0.0)
        assert Q1[5] == pytest.approx(Q2[5])


# --- body_point_velocity helper ---


class TestBodyPointVelocity:
    """Test the State.body_point_velocity method."""

    def test_pure_translation(self) -> None:
        state = State()
        state.register_body("b1")
        q = np.array([0.0, 0.0, 0.0])
        q_dot = np.array([3.0, -2.0, 0.0])
        v = state.body_point_velocity("b1", np.array([1.0, 0.0]), q, q_dot)
        np.testing.assert_allclose(v, [3.0, -2.0])

    def test_pure_rotation(self) -> None:
        state = State()
        state.register_body("b1")
        q = np.array([0.0, 0.0, 0.0])  # θ = 0
        q_dot = np.array([0.0, 0.0, 1.0])  # ω = 1
        # Point at (1, 0): B(0)@[1,0] = [0, 1], v = [0,1]*1 = [0, 1]
        v = state.body_point_velocity("b1", np.array([1.0, 0.0]), q, q_dot)
        np.testing.assert_allclose(v, [0.0, 1.0])

    def test_ground_returns_zero(self) -> None:
        state = State()
        state.register_body("b1")
        q = np.array([0.0, 0.0, 0.0])
        q_dot = np.array([1.0, 2.0, 3.0])
        v = state.body_point_velocity("ground", np.array([5.0, 0.0]), q, q_dot)
        np.testing.assert_array_equal(v, np.zeros(2))
