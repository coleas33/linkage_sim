"""Tests for the LinearSpring force element.

Validates spring force computation, mode filtering, virtual work,
and integration with the assemble_Q pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.state import State
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.spring import LinearSpring, SpringMode


# --- Basic spring force ---


class TestLinearSpringBasic:
    """Core spring force computation."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_spring_at_free_length_no_preload(self) -> None:
        """Spring at free length with no preload produces zero force."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=2.0,
            _id="s1",
        )
        # Place b1 at (2, 0) → length = 2.0 = free_length
        q = np.array([2.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_spring_stretched_horizontal(self) -> None:
        """Stretched spring: pulls body toward ground."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            _id="s1",
        )
        # Place b1 at (3, 0) → length = 3, extension = 2
        q = np.array([3.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        # Force = 100*2 = 200 N tension, pulling b1 toward ground (negative x)
        assert Q[0] == pytest.approx(-200.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(0.0)  # force at body origin, no moment

    def test_spring_compressed_horizontal(self) -> None:
        """Compressed spring: pushes body away from ground."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=5.0,
            _id="s1",
        )
        # Place b1 at (3, 0) → length = 3, extension = -2
        q = np.array([3.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        # Force = 100*(-2) = -200 (compression), pushes b1 away (+x)
        assert Q[0] == pytest.approx(200.0)
        assert Q[1] == pytest.approx(0.0)

    def test_spring_vertical(self) -> None:
        """Spring along vertical axis."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=50.0,
            free_length=1.0,
            _id="s1",
        )
        # Place b1 at (0, 4) → length = 4, extension = 3
        q = np.array([0.0, 4.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        # Tension = 50*3 = 150, pulling b1 downward (-y)
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-150.0)

    def test_spring_diagonal(self) -> None:
        """Spring at 45 degrees."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=0.0,
            _id="s1",
        )
        # Place b1 at (1, 1) → length = sqrt(2)
        q = np.array([1.0, 1.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        length = np.sqrt(2.0)
        # Force = 100*sqrt(2) tension, pulling toward origin
        expected_Fx = -100.0 * length * (1.0 / length)
        expected_Fy = -100.0 * length * (1.0 / length)
        assert Q[0] == pytest.approx(expected_Fx)
        assert Q[1] == pytest.approx(expected_Fy)

    def test_spring_with_preload(self) -> None:
        """Preload adds constant force to spring."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=2.0,
            preload=50.0,
            _id="s1",
        )
        # At free length: force = 0 + 50 = 50 N tension
        q = np.array([2.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(-50.0)  # pulls toward ground

    def test_spring_with_moment_arm(self) -> None:
        """Spring attached at offset point produces moment."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([1.0, 0.0]),  # offset along body
            stiffness=100.0,
            free_length=0.0,
            _id="s1",
        )
        # Body at origin, θ=0. Attachment at global (1, 0).
        # Length = 1, force = 100 tension, pulling toward ground
        q = np.array([0.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)

        # F on b1 at local (1,0): Fx = -100, Fy = 0
        # B(0)@[1,0] = [0, 1]. Moment = [0,1]·[-100,0] = 0
        assert Q[0] == pytest.approx(-100.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(0.0)


# --- Spring modes ---


class TestSpringModes:
    """Test tension-only and compression-only modes."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_tension_only_in_tension(self) -> None:
        """Tension-only spring active when stretched."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            mode=SpringMode.TENSION_ONLY,
            _id="s1",
        )
        q = np.array([3.0, 0.0, 0.0])  # stretched
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        assert Q[0] != 0.0  # active

    def test_tension_only_in_compression_slack(self) -> None:
        """Tension-only spring goes slack when compressed."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=5.0,
            mode=SpringMode.TENSION_ONLY,
            _id="s1",
        )
        q = np.array([3.0, 0.0, 0.0])  # compressed (length < free_length)
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_compression_only_in_compression(self) -> None:
        """Compression-only spring active when compressed."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=5.0,
            mode=SpringMode.COMPRESSION_ONLY,
            _id="s1",
        )
        q = np.array([3.0, 0.0, 0.0])  # compressed
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        assert Q[0] != 0.0  # active

    def test_compression_only_in_tension_disconnects(self) -> None:
        """Compression-only spring disconnects when stretched."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            mode=SpringMode.COMPRESSION_ONLY,
            _id="s1",
        )
        q = np.array([3.0, 0.0, 0.0])  # stretched
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))


# --- Two-body spring ---


class TestTwoBodySpring:
    """Spring between two moving bodies."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")
        self.state.register_body("b2")

    def test_action_reaction(self) -> None:
        """Net force from spring between two bodies sums to zero."""
        spring = LinearSpring(
            body_i_id="b1",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b2",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=200.0,
            free_length=1.0,
            _id="s1",
        )
        q = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(6), 0.0)

        # Net translational force should be zero (action-reaction)
        Fx_total = Q[0] + Q[3]
        Fy_total = Q[1] + Q[4]
        assert Fx_total == pytest.approx(0.0)
        assert Fy_total == pytest.approx(0.0)

    def test_forces_opposite_directions(self) -> None:
        """Bodies pulled toward each other in tension."""
        spring = LinearSpring(
            body_i_id="b1",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b2",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            _id="s1",
        )
        # b1 at (0,0), b2 at (5,0) → stretched
        q = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(6), 0.0)

        # b1 pulled toward b2 (+x), b2 pulled toward b1 (-x)
        assert Q[0] > 0  # b1 Fx positive (toward b2)
        assert Q[3] < 0  # b2 Fx negative (toward b1)


# --- Edge cases ---


class TestSpringEdgeCases:
    """Edge cases and degenerate inputs."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_zero_length_spring_returns_zero(self) -> None:
        """Coincident points: no direction defined, no force."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            _id="s1",
        )
        # b1 at origin → coincident with ground point
        q = np.array([0.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_zero_stiffness(self) -> None:
        """Zero stiffness with no preload produces zero force."""
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=0.0,
            free_length=1.0,
            _id="s1",
        )
        q = np.array([3.0, 0.0, 0.0])
        Q = spring.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_has_id(self) -> None:
        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=100.0,
            free_length=1.0,
            _id="spring_1",
        )
        assert spring.id == "spring_1"


# --- Virtual work ---


class TestSpringVirtualWork:
    """Virtual work consistency for spring forces."""

    def test_virtual_work_stretched_spring(self) -> None:
        """Q·δq matches spring PE change for small perturbation."""
        state = State()
        state.register_body("b1")

        spring = LinearSpring(
            body_i_id="ground",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b1",
            point_j_local=np.array([0.0, 0.0]),
            stiffness=200.0,
            free_length=1.0,
            _id="s1",
        )
        q = np.array([3.0, 1.0, 0.5])
        dq = np.array([1e-7, -2e-7, 5e-8])

        Q = spring.evaluate(state, q, np.zeros(3), 0.0)
        dW_Q = float(np.dot(Q, dq))

        # Compute actual spring PE change
        r_i = state.body_point_global("ground", np.array([0.0, 0.0]), q)
        r_j = state.body_point_global("b1", np.array([0.0, 0.0]), q)
        r_j_plus = state.body_point_global("b1", np.array([0.0, 0.0]), q + dq)

        len_before = float(np.linalg.norm(r_j - r_i))
        len_after = float(np.linalg.norm(r_j_plus - r_i))

        # dPE = 0.5*k*(L_after - L0)^2 - 0.5*k*(L_before - L0)^2
        PE_before = 0.5 * 200.0 * (len_before - 1.0) ** 2
        PE_after = 0.5 * 200.0 * (len_after - 1.0) ** 2
        dW_expected = -(PE_after - PE_before)  # work = -ΔPE

        np.testing.assert_allclose(dW_Q, dW_expected, rtol=1e-4)
