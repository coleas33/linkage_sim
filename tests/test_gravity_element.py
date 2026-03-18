"""Tests for the Gravity force element.

Validates the Gravity class as a ForceElement implementation, including
integration with mechanism assembly and virtual work consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import State
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.helpers import gravity_to_Q


# --- Gravity element protocol conformance ---


class TestGravityProtocol:
    """Verify Gravity satisfies ForceElement protocol."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=2.0)
        self.bodies = {"ground": make_ground(O=(0.0, 0.0)), "b1": body}
        self.gravity = Gravity(
            g_vector=np.array([0.0, -9.81]),
            bodies=self.bodies,
        )

    def test_has_id(self) -> None:
        assert self.gravity.id == "gravity"

    def test_custom_id(self) -> None:
        g = Gravity(
            g_vector=np.array([0.0, -9.81]),
            bodies=self.bodies,
            _id="gravity_moon",
        )
        assert g.id == "gravity_moon"

    def test_evaluate_returns_correct_shape(self) -> None:
        q = np.array([0.0, 0.0, 0.0])
        q_dot = np.zeros(3)
        Q = self.gravity.evaluate(self.state, q, q_dot, 0.0)
        assert Q.shape == (3,)

    def test_evaluate_matches_helper(self) -> None:
        """Gravity element must produce same result as gravity_to_Q helper."""
        q = np.array([1.0, 2.0, 0.5])
        q_dot = np.zeros(3)
        Q_element = self.gravity.evaluate(self.state, q, q_dot, 0.0)
        Q_helper = gravity_to_Q(
            self.state, self.bodies, q, np.array([0.0, -9.81])
        )
        np.testing.assert_array_equal(Q_element, Q_helper)


# --- Gravity force correctness ---


class TestGravityForce:
    """Verify gravity force values for single and multiple bodies."""

    def test_single_body_downward_gravity(self) -> None:
        """Single body, CG at origin, θ=0: only Qy = -mg."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=3.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        # CG at (0.5, 0) in local frame, θ=0 → CG at (0.5, 0) global
        q = np.array([0.0, 0.0, 0.0])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(0.0)          # Qx = 0 (no horizontal gravity)
        assert Q[1] == pytest.approx(-3.0 * 9.81)  # Qy = -mg
        # Qθ = B(0)@cg_local · F = [0, 1]·[0, -mg] (cg at x=0.5)
        # B(0)@[0.5, 0] = [-sin0*0.5 - cos0*0, cos0*0.5 - sin0*0] = [0, 0.5]
        # Qθ = [0, 0.5] · [0, -29.43] = -14.715
        assert Q[2] == pytest.approx(-3.0 * 9.81 * 0.5)

    def test_single_body_rotated(self) -> None:
        """Rotated body: gravity moment changes with orientation."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 2.0, mass=1.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        theta = np.pi / 3  # 60 degrees
        q = np.array([0.0, 0.0, theta])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)

        # CG at local (1, 0). B(θ)@[1,0] = [-sinθ, cosθ]
        # Qθ = [-sinθ, cosθ] · [0, -9.81] = -9.81*cosθ
        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-9.81)
        assert Q[2] == pytest.approx(-9.81 * np.cos(theta))

    def test_two_bodies(self) -> None:
        """Two bodies: gravity contributions sum."""
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        b1 = make_bar("b1", "A", "B", 1.0, mass=2.0)
        b2 = make_bar("b2", "C", "D", 1.0, mass=3.0)
        bodies = {"ground": make_ground(), "b1": b1, "b2": b2}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.zeros(6)
        Q = g.evaluate(state, q, np.zeros(6), 0.0)

        # Body 1: Qy1 = -2*9.81, Body 2: Qy2 = -3*9.81
        assert Q[1] == pytest.approx(-2.0 * 9.81)
        assert Q[4] == pytest.approx(-3.0 * 9.81)

    def test_zero_mass_body_skipped(self) -> None:
        """Zero-mass body contributes nothing."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=0.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([0.0, 0.0, 0.0])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))

    def test_ground_body_skipped(self) -> None:
        """Ground body contributes nothing even if it had mass."""
        state = State()
        state.register_body("b1")
        ground = make_ground(O=(0.0, 0.0))
        ground.mass = 999.0  # should still be skipped
        bodies = {"ground": ground, "b1": make_bar("b1", "A", "B", 1.0, mass=1.0)}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([0.0, 0.0, 0.0])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)

        # Only b1 contributes
        assert Q[1] == pytest.approx(-9.81)

    def test_non_standard_gravity_direction(self) -> None:
        """Gravity in arbitrary direction (e.g., sideways)."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=1.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([5.0, 0.0]), bodies=bodies)
        q = np.array([0.0, 0.0, 0.0])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(5.0)   # Qx = mg_x
        assert Q[1] == pytest.approx(0.0)   # Qy = 0
        # B(0)@[0.5,0] = [0, 0.5], F = [5, 0] → Qθ = 0*5 + 0.5*0 = 0
        assert Q[2] == pytest.approx(0.0)

    def test_gravity_independent_of_velocity(self) -> None:
        """Gravity does not depend on q_dot."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=2.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([1.0, 2.0, 0.5])

        Q1 = g.evaluate(state, q, np.zeros(3), 0.0)
        Q2 = g.evaluate(state, q, np.array([10.0, -5.0, 3.0]), 0.0)
        np.testing.assert_array_equal(Q1, Q2)

    def test_gravity_independent_of_time(self) -> None:
        """Gravity does not depend on t."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=2.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([1.0, 2.0, 0.5])

        Q1 = g.evaluate(state, q, np.zeros(3), 0.0)
        Q2 = g.evaluate(state, q, np.zeros(3), 100.0)
        np.testing.assert_array_equal(Q1, Q2)


# --- Integration with assemble_Q ---


class TestGravityAssembly:
    """Test Gravity element through the assemble_Q pipeline."""

    def test_single_gravity_element(self) -> None:
        """Gravity via assemble_Q matches direct evaluate."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=2.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([0.0, 0.0, np.pi / 4])
        q_dot = np.zeros(3)

        Q_direct = g.evaluate(state, q, q_dot, 0.0)
        Q_assembled = assemble_Q(state, [g], q, q_dot, 0.0)
        np.testing.assert_array_equal(Q_direct, Q_assembled)

    def test_zero_gravity(self) -> None:
        """Zero gravity vector produces zero Q."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=5.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, 0.0]), bodies=bodies)
        q = np.array([1.0, 2.0, 0.5])
        Q = g.evaluate(state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))


# --- Virtual work consistency ---


class TestGravityVirtualWork:
    """Verify that gravity Q satisfies virtual work principle.

    For gravity: δW = Σ(-m_i * g * δh_cg_i) = Q · δq
    """

    def test_virtual_work_single_body(self) -> None:
        """δW_gravity = -mg·Δh_cg should equal Q·δq."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 2.0, mass=3.0)
        bodies = {"ground": make_ground(), "b1": body}

        g = Gravity(g_vector=np.array([0.0, -9.81]), bodies=bodies)
        q = np.array([1.0, 0.5, np.pi / 6])
        dq = np.array([1e-7, 2e-7, -1e-7])

        Q = g.evaluate(state, q, np.zeros(3), 0.0)
        dW_Q = float(np.dot(Q, dq))

        # Actual work: F · δr_cg for each body
        r_cg = state.body_point_global("b1", body.cg_local, q)
        r_cg_plus = state.body_point_global("b1", body.cg_local, q + dq)
        dr_cg = r_cg_plus - r_cg
        dW_actual = float(np.dot(body.mass * np.array([0.0, -9.81]), dr_cg))

        np.testing.assert_allclose(dW_Q, dW_actual, rtol=1e-5)

    def test_virtual_work_two_bodies(self) -> None:
        """Virtual work consistency with multiple bodies."""
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        b1 = make_bar("b1", "A", "B", 1.0, mass=2.0)
        b2 = make_bar("b2", "C", "D", 1.5, mass=4.0)
        bodies = {"ground": make_ground(), "b1": b1, "b2": b2}

        g_vec = np.array([0.0, -9.81])
        g = Gravity(g_vector=g_vec, bodies=bodies)
        q = np.array([0.0, 0.0, np.pi / 4, 1.0, 0.5, np.pi / 3])
        dq = np.array([1e-7, -1e-7, 5e-8, 2e-7, 1e-7, -5e-8])

        Q = g.evaluate(state, q, np.zeros(6), 0.0)
        dW_Q = float(np.dot(Q, dq))

        # Sum actual work for both bodies
        dW_actual = 0.0
        for bid, body in [("b1", b1), ("b2", b2)]:
            r_cg = state.body_point_global(bid, body.cg_local, q)
            r_cg_plus = state.body_point_global(bid, body.cg_local, q + dq)
            dr_cg = r_cg_plus - r_cg
            dW_actual += float(np.dot(body.mass * g_vec, dr_cg))

        np.testing.assert_allclose(dW_Q, dW_actual, rtol=1e-5)

    def test_virtual_work_potential_energy(self) -> None:
        """δW = -mg·Δh_cg: gravitational PE change matches work."""
        state = State()
        state.register_body("b1")
        body = make_bar("b1", "A", "B", 1.0, mass=2.0)
        bodies = {"ground": make_ground(), "b1": body}

        g_mag = 9.81
        g = Gravity(g_vector=np.array([0.0, -g_mag]), bodies=bodies)
        q = np.array([0.0, 0.0, 0.0])
        dq = np.array([0.0, 1e-7, 0.0])  # pure vertical displacement

        Q = g.evaluate(state, q, np.zeros(3), 0.0)
        dW_Q = float(np.dot(Q, dq))

        # For pure vertical lift: δW = -mg * δy_cg = -mg * dq[1]
        dW_expected = -body.mass * g_mag * dq[1]
        np.testing.assert_allclose(dW_Q, dW_expected, rtol=1e-10)
