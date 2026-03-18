"""Tests for generalized force helpers and Q assembly."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.state import GROUND_ID, State
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.helpers import (
    body_torque_to_Q,
    gravity_to_Q,
    point_force_to_Q,
)
from linkage_sim.forces.protocol import ForceElement


# --- point_force_to_Q tests ---


class TestPointForceToQ:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body1")

    def test_horizontal_force_at_origin(self) -> None:
        """Force at body origin: no moment arm, Q_θ = 0."""
        q = np.array([1.0, 2.0, 0.0])
        F = np.array([10.0, 0.0])
        Q = point_force_to_Q(self.state, "body1", np.array([0.0, 0.0]), F, q)

        assert Q[0] == pytest.approx(10.0)  # Qx
        assert Q[1] == pytest.approx(0.0)   # Qy
        assert Q[2] == pytest.approx(0.0)   # Qθ (no moment arm)

    def test_vertical_force_at_origin(self) -> None:
        q = np.array([0.0, 0.0, 0.0])
        F = np.array([0.0, -5.0])
        Q = point_force_to_Q(self.state, "body1", np.array([0.0, 0.0]), F, q)

        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-5.0)
        assert Q[2] == pytest.approx(0.0)

    def test_force_at_offset_point_produces_moment(self) -> None:
        """Force at offset point produces torque contribution."""
        # Body at origin, θ=0, point at local (1, 0)
        # B(0) @ [1,0] = [[-sin0, -cos0], [cos0, -sin0]] @ [1,0] = [0, 1]
        # Q_θ = [0, 1] · F
        q = np.array([0.0, 0.0, 0.0])
        s_local = np.array([1.0, 0.0])

        # Vertical force at the tip: Q_θ = [0,1] · [0, -10] = -10
        F = np.array([0.0, -10.0])
        Q = point_force_to_Q(self.state, "body1", s_local, F, q)

        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-10.0)
        assert Q[2] == pytest.approx(-10.0)

    def test_force_at_rotated_body(self) -> None:
        """When body is rotated, moment arm changes accordingly."""
        # Body at origin, θ=π/2, point at local (1, 0) → global (0, 1)
        # B(π/2) @ [1,0] = [[-1, 0], [0, -1]] @ [1,0] = [-1, 0]
        # Q_θ = [-1, 0] · F
        q = np.array([0.0, 0.0, np.pi / 2])
        s_local = np.array([1.0, 0.0])

        F = np.array([5.0, 0.0])
        Q = point_force_to_Q(self.state, "body1", s_local, F, q)

        assert Q[0] == pytest.approx(5.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(-5.0)  # [-1, 0] · [5, 0] = -5

    def test_force_on_ground_returns_zero(self) -> None:
        """Forces on ground produce no generalized forces."""
        q = np.array([0.0, 0.0, 0.0])
        F = np.array([100.0, 200.0])
        Q = point_force_to_Q(self.state, GROUND_ID, np.array([0.0, 0.0]), F, q)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_shape(self) -> None:
        q = np.array([0.0, 0.0, 0.0])
        Q = point_force_to_Q(
            self.state, "body1", np.array([0.0, 0.0]),
            np.array([1.0, 0.0]), q
        )
        assert Q.shape == (3,)

    def test_two_bodies(self) -> None:
        """Force on second body only affects that body's entries."""
        state = State()
        state.register_body("A")
        state.register_body("B")
        q = np.zeros(6)
        F = np.array([1.0, 2.0])

        Q = point_force_to_Q(state, "B", np.array([0.0, 0.0]), F, q)
        # Body A entries (0,1,2) should be zero
        np.testing.assert_array_equal(Q[0:3], [0.0, 0.0, 0.0])
        # Body B entries (3,4,5)
        assert Q[3] == pytest.approx(1.0)
        assert Q[4] == pytest.approx(2.0)

    def test_virtual_work_consistency(self) -> None:
        """Virtual work δW = F · δr_P = Q · δq for arbitrary perturbation."""
        q = np.array([0.5, 0.3, 0.8])
        s_local = np.array([0.2, 0.1])
        F = np.array([3.0, -2.0])

        Q = point_force_to_Q(self.state, "body1", s_local, F, q)

        # Perturb q — use small perturbation so linearized virtual work
        # (Q · δq) matches actual work (F · δr_P) to O(δq²).
        dq = np.array([1e-7, -2e-7, 5e-8])

        # Virtual work via Q (linearized)
        dW_Q = float(np.dot(Q, dq))

        # Virtual work via F · δr_P (finite difference)
        r_P = self.state.body_point_global("body1", s_local, q)
        r_P_plus = self.state.body_point_global("body1", s_local, q + dq)
        dr_P = r_P_plus - r_P
        dW_F = float(np.dot(F, dr_P))

        np.testing.assert_allclose(dW_Q, dW_F, rtol=1e-5)


# --- body_torque_to_Q tests ---


class TestBodyTorqueToQ:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body1")

    def test_positive_torque(self) -> None:
        Q = body_torque_to_Q(self.state, "body1", 5.0)
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(5.0)

    def test_negative_torque(self) -> None:
        Q = body_torque_to_Q(self.state, "body1", -3.0)
        assert Q[2] == pytest.approx(-3.0)

    def test_torque_on_ground_returns_zero(self) -> None:
        Q = body_torque_to_Q(self.state, GROUND_ID, 100.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_shape(self) -> None:
        Q = body_torque_to_Q(self.state, "body1", 1.0)
        assert Q.shape == (3,)

    def test_two_bodies_correct_index(self) -> None:
        state = State()
        state.register_body("A")
        state.register_body("B")
        Q = body_torque_to_Q(state, "B", 7.0)
        np.testing.assert_array_equal(Q[0:3], [0.0, 0.0, 0.0])
        assert Q[5] == pytest.approx(7.0)


# --- gravity_to_Q tests ---


class TestGravityToQ:
    def test_single_body_downward_gravity(self) -> None:
        """Body with mass, gravity = [0, -9.81]."""
        state = State()
        state.register_body("bar")
        bodies: dict[str, Body] = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=1.0, mass=2.0),
        }
        q = np.zeros(3)
        g = np.array([0.0, -9.81])

        Q = gravity_to_Q(state, bodies, q, g)
        # F = 2.0 * [0, -9.81] = [0, -19.62]
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-19.62)
        # CG at (0.5, 0), θ=0: B(0) @ [0.5, 0] = [0, 0.5]
        # Q_θ = [0, 0.5] · [0, -19.62] = -9.81
        assert Q[2] == pytest.approx(-9.81)

    def test_zero_mass_body_no_contribution(self) -> None:
        state = State()
        state.register_body("bar")
        bodies: dict[str, Body] = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=1.0, mass=0.0),
        }
        q = np.zeros(3)
        Q = gravity_to_Q(state, bodies, q, np.array([0.0, -9.81]))
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_ground_excluded(self) -> None:
        """Ground should not contribute to Q even if mass were set."""
        state = State()
        state.register_body("bar")
        ground = make_ground()
        ground.mass = 1000.0  # shouldn't matter
        bodies: dict[str, Body] = {
            GROUND_ID: ground,
            "bar": make_bar("bar", "A", "B", length=1.0, mass=1.0),
        }
        q = np.zeros(3)
        Q = gravity_to_Q(state, bodies, q, np.array([0.0, -9.81]))
        # Only bar's contribution
        assert Q[1] == pytest.approx(-9.81)

    def test_multiple_bodies(self) -> None:
        state = State()
        state.register_body("bar1")
        state.register_body("bar2")
        bodies: dict[str, Body] = {
            GROUND_ID: make_ground(),
            "bar1": make_bar("bar1", "A", "B", length=1.0, mass=1.0),
            "bar2": make_bar("bar2", "C", "D", length=2.0, mass=3.0),
        }
        q = np.zeros(6)
        g = np.array([0.0, -10.0])
        Q = gravity_to_Q(state, bodies, q, g)

        # bar1: Fy = -10, Qθ = [0, 0.5] · [0, -10] = -5
        assert Q[1] == pytest.approx(-10.0)
        assert Q[2] == pytest.approx(-5.0)
        # bar2: Fy = -30, CG at (1.0, 0), Qθ = [0, 1.0] · [0, -30] = -30
        assert Q[4] == pytest.approx(-30.0)
        assert Q[5] == pytest.approx(-30.0)

    def test_rotated_body_gravity(self) -> None:
        """Gravity moment arm changes with body orientation."""
        state = State()
        state.register_body("bar")
        bodies: dict[str, Body] = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=2.0, mass=1.0),
        }
        # θ = π/2: CG at local (1, 0) → B(π/2) @ [1,0] = [-1, 0]
        # F = [0, -9.81], Q_θ = [-1, 0] · [0, -9.81] = 0
        q = np.array([0.0, 0.0, np.pi / 2])
        Q = gravity_to_Q(state, bodies, q, np.array([0.0, -9.81]))
        assert Q[2] == pytest.approx(0.0, abs=1e-10)

    def test_horizontal_gravity(self) -> None:
        """Non-standard gravity direction (e.g., centrifuge)."""
        state = State()
        state.register_body("bar")
        bodies: dict[str, Body] = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=1.0, mass=2.0),
        }
        q = np.zeros(3)
        Q = gravity_to_Q(state, bodies, q, np.array([5.0, 0.0]))
        assert Q[0] == pytest.approx(10.0)
        assert Q[1] == pytest.approx(0.0)


# --- assemble_Q tests ---


class _ConstantForce:
    """Test force element that returns a constant Q."""

    def __init__(self, id: str, Q_value: NDArray[np.float64]) -> None:  # noqa: A002
        self._id = id
        self._Q = Q_value

    @property
    def id(self) -> str:
        return self._id

    def evaluate(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        return self._Q.copy()


class TestAssembleQ:
    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body1")

    def test_no_force_elements(self) -> None:
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = assemble_Q(self.state, [], q, q_dot, 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_single_force_element(self) -> None:
        fe = _ConstantForce("f1", np.array([1.0, 2.0, 3.0]))
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = assemble_Q(self.state, [fe], q, q_dot, 0.0)
        np.testing.assert_array_equal(Q, [1.0, 2.0, 3.0])

    def test_multiple_force_elements_sum(self) -> None:
        fe1 = _ConstantForce("f1", np.array([1.0, 0.0, 0.0]))
        fe2 = _ConstantForce("f2", np.array([0.0, 2.0, 3.0]))
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = assemble_Q(self.state, [fe1, fe2], q, q_dot, 0.0)
        np.testing.assert_array_equal(Q, [1.0, 2.0, 3.0])

    def test_shape(self) -> None:
        q = np.zeros(3)
        q_dot = np.zeros(3)
        Q = assemble_Q(self.state, [], q, q_dot, 0.0)
        assert Q.shape == (3,)
