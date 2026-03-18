"""Tests for Motor force element."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.state import State
from linkage_sim.forces.motor import Motor


class TestMotor:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_stall_torque_at_zero_speed(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0, _id="m1",
        )
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = m.evaluate(self.state, q, q_dot, 0.0)
        assert Q[2] == pytest.approx(10.0)

    def test_zero_torque_at_no_load_speed(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0, _id="m1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 100.0])
        Q = m.evaluate(self.state, q, q_dot, 0.0)
        assert Q[2] == pytest.approx(0.0)

    def test_half_speed_half_torque(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0, _id="m1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 50.0])
        Q = m.evaluate(self.state, q, q_dot, 0.0)
        assert Q[2] == pytest.approx(5.0)

    def test_overspeed_zero_torque(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0, _id="m1",
        )
        q = np.zeros(3)
        q_dot = np.array([0.0, 0.0, 150.0])
        Q = m.evaluate(self.state, q, q_dot, 0.0)
        assert Q[2] == pytest.approx(0.0)

    def test_reverse_direction(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0,
            direction=-1.0, _id="m1",
        )
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = m.evaluate(self.state, q, q_dot, 0.0)
        assert Q[2] == pytest.approx(-10.0)

    def test_action_reaction(self) -> None:
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        m = Motor(
            body_i_id="b1", body_j_id="b2",
            stall_torque=10.0, no_load_speed=100.0, _id="m1",
        )
        q = np.zeros(6)
        q_dot = np.zeros(6)
        Q = m.evaluate(state, q, q_dot, 0.0)
        assert Q[2] + Q[5] == pytest.approx(0.0)

    def test_has_id(self) -> None:
        m = Motor(
            body_i_id="ground", body_j_id="b1",
            stall_torque=10.0, no_load_speed=100.0, _id="motor_1",
        )
        assert m.id == "motor_1"
