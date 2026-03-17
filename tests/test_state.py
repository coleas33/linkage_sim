"""Tests for state vector and coordinate bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from linkage_sim.core.state import GROUND_ID, BodyIndex, State


class TestBodyIndex:
    def test_index_offsets(self) -> None:
        idx = BodyIndex(body_id="link1", q_start=6)
        assert idx.x_idx == 6
        assert idx.y_idx == 7
        assert idx.theta_idx == 8

    def test_zero_start(self) -> None:
        idx = BodyIndex(body_id="link1", q_start=0)
        assert idx.x_idx == 0
        assert idx.y_idx == 1
        assert idx.theta_idx == 2


class TestStateRegistration:
    def test_register_single_body(self) -> None:
        state = State()
        idx = state.register_body("crank")
        assert idx.q_start == 0
        assert state.n_coords == 3
        assert state.n_moving_bodies == 1

    def test_register_multiple_bodies(self) -> None:
        state = State()
        idx1 = state.register_body("crank")
        idx2 = state.register_body("coupler")
        idx3 = state.register_body("rocker")
        assert idx1.q_start == 0
        assert idx2.q_start == 3
        assert idx3.q_start == 6
        assert state.n_coords == 9
        assert state.n_moving_bodies == 3

    def test_register_ground_raises(self) -> None:
        state = State()
        with pytest.raises(ValueError, match="Ground"):
            state.register_body(GROUND_ID)

    def test_register_duplicate_raises(self) -> None:
        state = State()
        state.register_body("crank")
        with pytest.raises(ValueError, match="already registered"):
            state.register_body("crank")

    def test_get_index_unregistered_raises(self) -> None:
        state = State()
        with pytest.raises(KeyError):
            state.get_index("nonexistent")

    def test_get_index_ground_raises(self) -> None:
        state = State()
        with pytest.raises(ValueError, match="Ground"):
            state.get_index(GROUND_ID)

    def test_body_ids_ordered(self) -> None:
        state = State()
        state.register_body("crank")
        state.register_body("coupler")
        state.register_body("rocker")
        assert state.body_ids == ["crank", "coupler", "rocker"]


class TestStateAccessors:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("crank")
        self.state.register_body("coupler")
        self.q = np.array([1.0, 2.0, 0.5, 3.0, 4.0, 1.2])

    def test_get_pose(self) -> None:
        x, y, theta = self.state.get_pose("crank", self.q)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert theta == pytest.approx(0.5)

    def test_get_pose_second_body(self) -> None:
        x, y, theta = self.state.get_pose("coupler", self.q)
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(4.0)
        assert theta == pytest.approx(1.2)

    def test_get_pose_ground(self) -> None:
        x, y, theta = self.state.get_pose(GROUND_ID, self.q)
        assert x == 0.0
        assert y == 0.0
        assert theta == 0.0

    def test_get_position(self) -> None:
        pos = self.state.get_position("crank", self.q)
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0])

    def test_get_position_ground(self) -> None:
        pos = self.state.get_position(GROUND_ID, self.q)
        np.testing.assert_array_almost_equal(pos, [0.0, 0.0])

    def test_get_angle(self) -> None:
        assert self.state.get_angle("crank", self.q) == pytest.approx(0.5)

    def test_get_angle_ground(self) -> None:
        assert self.state.get_angle(GROUND_ID, self.q) == 0.0

    def test_set_pose(self) -> None:
        q = self.state.make_q()
        self.state.set_pose("coupler", q, 10.0, 20.0, 1.57)
        assert q[3] == pytest.approx(10.0)
        assert q[4] == pytest.approx(20.0)
        assert q[5] == pytest.approx(1.57)

    def test_set_pose_ground_raises(self) -> None:
        q = self.state.make_q()
        with pytest.raises(ValueError, match="ground"):
            self.state.set_pose(GROUND_ID, q, 0.0, 0.0, 0.0)

    def test_is_ground(self) -> None:
        assert self.state.is_ground(GROUND_ID) is True
        assert self.state.is_ground("crank") is False

    def test_make_q_correct_size(self) -> None:
        q = self.state.make_q()
        assert q.shape == (6,)
        np.testing.assert_array_equal(q, np.zeros(6))


class TestTransformations:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("link")

    def test_rotation_matrix_zero(self) -> None:
        A = self.state.rotation_matrix(0.0)
        np.testing.assert_array_almost_equal(A, np.eye(2))

    def test_rotation_matrix_90deg(self) -> None:
        A = self.state.rotation_matrix(np.pi / 2)
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(A, expected)

    def test_rotation_matrix_derivative_zero(self) -> None:
        B = self.state.rotation_matrix_derivative(0.0)
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(B, expected)

    def test_body_point_global_no_rotation(self) -> None:
        q = np.array([1.0, 2.0, 0.0])
        local_pt = np.array([0.5, 0.0])
        global_pt = self.state.body_point_global("link", local_pt, q)
        np.testing.assert_array_almost_equal(global_pt, [1.5, 2.0])

    def test_body_point_global_with_rotation(self) -> None:
        q = np.array([0.0, 0.0, np.pi / 2])
        local_pt = np.array([1.0, 0.0])
        global_pt = self.state.body_point_global("link", local_pt, q)
        np.testing.assert_array_almost_equal(global_pt, [0.0, 1.0])

    def test_body_point_global_ground(self) -> None:
        q = np.array([1.0, 2.0, 0.5])
        local_pt = np.array([3.0, 4.0])
        global_pt = self.state.body_point_global(GROUND_ID, local_pt, q)
        np.testing.assert_array_almost_equal(global_pt, [3.0, 4.0])

    def test_body_point_derivative_ground(self) -> None:
        q = np.array([1.0, 2.0, 0.5])
        local_pt = np.array([3.0, 4.0])
        deriv = self.state.body_point_global_derivative(GROUND_ID, local_pt, q)
        np.testing.assert_array_almost_equal(deriv, [0.0, 0.0])

    @given(
        theta=st.floats(min_value=-2 * np.pi, max_value=2 * np.pi),
        sx=st.floats(min_value=-1.0, max_value=1.0),
        sy=st.floats(min_value=-1.0, max_value=1.0),
    )
    def test_rotation_derivative_matches_finite_difference(
        self, theta: float, sx: float, sy: float
    ) -> None:
        """B(θ) * s should match d(A(θ) * s)/dθ by finite difference."""
        s = np.array([sx, sy])
        B = self.state.rotation_matrix_derivative(theta)
        analytical = B @ s

        eps = 1e-7
        A_plus = self.state.rotation_matrix(theta + eps)
        A_minus = self.state.rotation_matrix(theta - eps)
        numerical = (A_plus @ s - A_minus @ s) / (2 * eps)

        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)
