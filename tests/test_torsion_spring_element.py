"""Tests for the TorsionSpring force element."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.state import State
from linkage_sim.forces.torsion_spring import TorsionSpring


class TestTorsionSpringBasic:
    """Core torsion spring force computation."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_at_free_angle_no_preload(self) -> None:
        """No torque when at free angle with zero preload."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=50.0,
            free_angle=0.0,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, 0.0])  # θ_b1 = 0 = free_angle
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_positive_deflection_from_ground(self) -> None:
        """Body rotated CCW from free angle: spring resists."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            free_angle=0.0,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, 0.5])  # θ = 0.5 rad CCW
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)

        # τ = k * (θ_j - θ_i - θ_free) = 100 * 0.5 = 50
        # Applied as -τ on b1 (resists CCW rotation)
        assert Q[2] == pytest.approx(-50.0)

    def test_negative_deflection_from_ground(self) -> None:
        """Body rotated CW from free angle: spring pushes CCW."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            free_angle=0.0,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, -0.3])  # θ = -0.3 rad CW
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)

        # τ = 100 * (-0.3) = -30, applied as -(-30) = +30 on b1
        assert Q[2] == pytest.approx(30.0)

    def test_nonzero_free_angle(self) -> None:
        """Free angle offset: no torque at θ = free_angle."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            free_angle=np.pi / 4,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, np.pi / 4])
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-12)

    def test_preload(self) -> None:
        """Preload adds constant torque even at free angle."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            free_angle=0.0,
            preload=20.0,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, 0.0])  # at free angle
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)

        # τ = k*0 + preload = 20, applied as -20 on b1
        assert Q[2] == pytest.approx(-20.0)

    def test_translational_coords_unaffected(self) -> None:
        """Torsion spring affects only θ coordinates."""
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            free_angle=0.0,
            _id="ts1",
        )
        q = np.array([5.0, 3.0, 1.0])
        Q = ts.evaluate(self.state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(0.0)  # Qx = 0
        assert Q[1] == pytest.approx(0.0)  # Qy = 0
        assert Q[2] != 0.0                  # Qθ ≠ 0


class TestTorsionSpringTwoBodies:
    """Torsion spring between two moving bodies."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")
        self.state.register_body("b2")

    def test_action_reaction_torque(self) -> None:
        """Net torque from torsion spring is zero (action-reaction)."""
        ts = TorsionSpring(
            body_i_id="b1",
            body_j_id="b2",
            stiffness=50.0,
            free_angle=0.0,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.8])
        Q = ts.evaluate(self.state, q, np.zeros(6), 0.0)

        # Net torque should be zero
        assert Q[2] + Q[5] == pytest.approx(0.0)
        # All translational components zero
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[3] == pytest.approx(0.0)
        assert Q[4] == pytest.approx(0.0)

    def test_relative_angle_matters(self) -> None:
        """Only relative angle drives torsion spring, not absolutes."""
        ts = TorsionSpring(
            body_i_id="b1",
            body_j_id="b2",
            stiffness=100.0,
            free_angle=0.0,
            _id="ts1",
        )
        # Same relative angle (0.5) at two different absolute positions
        q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        q2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.5])
        Q1 = ts.evaluate(self.state, q1, np.zeros(6), 0.0)
        Q2 = ts.evaluate(self.state, q2, np.zeros(6), 0.0)

        # Torques should be identical
        assert Q1[2] == pytest.approx(Q2[2])
        assert Q1[5] == pytest.approx(Q2[5])


class TestTorsionSpringVirtualWork:
    """Virtual work consistency for torsion spring."""

    def test_virtual_work_consistency(self) -> None:
        """Q·δq matches -ΔPE for small angular perturbation."""
        state = State()
        state.register_body("b1")

        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=150.0,
            free_angle=0.2,
            _id="ts1",
        )
        q = np.array([0.0, 0.0, 0.7])
        dq = np.array([0.0, 0.0, 1e-7])

        Q = ts.evaluate(state, q, np.zeros(3), 0.0)
        dW_Q = float(np.dot(Q, dq))

        # PE = 0.5 * k * (θ - θ_free)^2
        angle_before = q[2] - 0.2
        angle_after = (q[2] + dq[2]) - 0.2
        PE_before = 0.5 * 150.0 * angle_before**2
        PE_after = 0.5 * 150.0 * angle_after**2
        dW_expected = -(PE_after - PE_before)

        np.testing.assert_allclose(dW_Q, dW_expected, rtol=1e-5)


class TestTorsionSpringProtocol:
    """ForceElement protocol compliance."""

    def test_has_id(self) -> None:
        ts = TorsionSpring(
            body_i_id="ground",
            body_j_id="b1",
            stiffness=100.0,
            _id="ts_joint_A",
        )
        assert ts.id == "ts_joint_A"

    def test_independent_of_velocity(self) -> None:
        """Torsion spring is position-dependent only."""
        state = State()
        state.register_body("b1")
        ts = TorsionSpring(
            body_i_id="ground", body_j_id="b1", stiffness=100.0, _id="ts1"
        )
        q = np.array([0.0, 0.0, 0.5])
        Q1 = ts.evaluate(state, q, np.zeros(3), 0.0)
        Q2 = ts.evaluate(state, q, np.array([1.0, 2.0, 3.0]), 0.0)
        np.testing.assert_array_equal(Q1, Q2)
