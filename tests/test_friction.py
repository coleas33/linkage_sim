"""Tests for the CoulombFriction force element."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.state import State
from linkage_sim.forces.friction import CoulombFriction


class TestCoulombFrictionBasic:
    """Core friction behavior."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_zero_velocity_zero_friction(self) -> None:
        """At zero relative velocity, tanh(0)=0 → no friction."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=100.0,
            _id="cf1",
        )
        q = np.array([0.0, 0.0, 0.0])
        q_dot = np.zeros(3)
        Q = cf.evaluate(self.state, q, q_dot, 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_positive_velocity_negative_torque(self) -> None:
        """CCW rotation → friction opposes → CW torque on body."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=1000.0,
            v_threshold=0.01,
            _id="cf1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 1.0])  # ω = 1 rad/s CCW
        Q = cf.evaluate(self.state, q, q_dot, 0.0)

        # τ = -μ*R*Fn*tanh(1/0.01) ≈ -0.1*0.01*1000*1 = -1.0
        assert Q[2] < 0  # opposes CCW
        assert Q[2] == pytest.approx(-1.0, abs=0.01)

    def test_negative_velocity_positive_torque(self) -> None:
        """CW rotation → friction opposes → CCW torque."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=1000.0,
            v_threshold=0.01,
            _id="cf1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, -1.0])
        Q = cf.evaluate(self.state, q, q_dot, 0.0)

        assert Q[2] > 0  # opposes CW

    def test_friction_magnitude(self) -> None:
        """At high velocity, friction ≈ μ*R*F_n."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.2,
            pin_radius=0.005,
            normal_force=500.0,
            v_threshold=0.01,
            _id="cf1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 10.0])  # well above threshold
        Q = cf.evaluate(self.state, q, q_dot, 0.0)

        max_torque = 0.2 * 0.005 * 500.0  # = 0.5
        assert abs(Q[2]) == pytest.approx(max_torque, abs=0.001)

    def test_no_translational_forces(self) -> None:
        """Friction at revolute only affects θ, not x/y."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=100.0,
            _id="cf1",
        )
        q = np.array([1.0, 2.0, 0.5])
        q_dot = np.array([0.0, 0.0, 1.0])
        Q = cf.evaluate(self.state, q, q_dot, 0.0)

        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(0.0)

    def test_zero_friction_coeff(self) -> None:
        """Zero friction coefficient → zero torque."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.0,
            pin_radius=0.01,
            normal_force=100.0,
            _id="cf1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 5.0])
        Q = cf.evaluate(self.state, q, q_dot, 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_regularization_smooth(self) -> None:
        """Friction transitions smoothly through zero velocity."""
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=1000.0,
            v_threshold=0.1,
            _id="cf1",
        )
        q = np.zeros(3)
        velocities = np.linspace(-1.0, 1.0, 21)
        torques = []
        for v in velocities:
            q_dot = np.array([0.0, 0.0, v])
            Q = cf.evaluate(self.state, q, q_dot, 0.0)
            torques.append(Q[2])

        torques_arr = np.array(torques)
        # Should be monotonically decreasing (higher ω → more negative torque)
        diffs = np.diff(torques_arr)
        assert np.all(diffs <= 0.0), "Friction should be monotonically opposing"


class TestCoulombFrictionTwoBodies:
    """Friction between two moving bodies."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")
        self.state.register_body("b2")

    def test_action_reaction(self) -> None:
        """Net torque from friction is zero."""
        cf = CoulombFriction(
            body_i_id="b1",
            body_j_id="b2",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=100.0,
            _id="cf1",
        )
        q = np.zeros(6)
        q_dot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 3.0])
        Q = cf.evaluate(self.state, q, q_dot, 0.0)

        # Net torque = 0
        assert Q[2] + Q[5] == pytest.approx(0.0)

    def test_relative_velocity_matters(self) -> None:
        """Only relative velocity determines friction, not absolute."""
        cf = CoulombFriction(
            body_i_id="b1",
            body_j_id="b2",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=100.0,
            _id="cf1",
        )
        q = np.zeros(6)
        # Same relative velocity (2 rad/s) at different absolutes
        q_dot1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        q_dot2 = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 7.0])

        Q1 = cf.evaluate(self.state, q, q_dot1, 0.0)
        Q2 = cf.evaluate(self.state, q, q_dot2, 0.0)

        # Torques on b2 should be equal (same ω_rel)
        assert Q1[5] == pytest.approx(Q2[5])


class TestCoulombFrictionProtocol:
    """ForceElement protocol compliance."""

    def test_has_id(self) -> None:
        cf = CoulombFriction(
            body_i_id="ground",
            body_j_id="b1",
            friction_coeff=0.1,
            pin_radius=0.01,
            normal_force=100.0,
            _id="friction_J1",
        )
        assert cf.id == "friction_J1"
