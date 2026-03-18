"""Tests for the ExternalLoad force element."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.state import State
from linkage_sim.forces.external_load import ExternalLoad


class TestExternalLoadBasic:
    """Core external load behavior."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("b1")

    def test_constant_force(self) -> None:
        """Constant downward force at body origin."""
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([0.0, -100.0]),
            _id="load1",
        )
        q = np.array([1.0, 2.0, 0.0])
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(-100.0)
        assert Q[2] == pytest.approx(0.0)  # at origin, no moment

    def test_force_at_offset_produces_moment(self) -> None:
        """Force at offset point creates torque."""
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([1.0, 0.0]),
            force_func=lambda q, qd, t: np.array([0.0, -50.0]),
            _id="load1",
        )
        q = np.array([0.0, 0.0, 0.0])  # θ=0
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)

        # B(0)@[1,0] = [0, 1], F = [0, -50] → Qθ = 0*0 + 1*(-50) = -50
        assert Q[2] == pytest.approx(-50.0)

    def test_time_dependent_force(self) -> None:
        """Force that varies with time."""
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([10.0 * t, 0.0]),
            _id="load1",
        )
        q = np.zeros(3)
        Q_t0 = load.evaluate(self.state, q, np.zeros(3), 0.0)
        Q_t5 = load.evaluate(self.state, q, np.zeros(3), 5.0)

        assert Q_t0[0] == pytest.approx(0.0)
        assert Q_t5[0] == pytest.approx(50.0)

    def test_position_dependent_force(self) -> None:
        """Force that depends on configuration."""
        # Force proportional to y-position (like a positional spring)
        def force_fn(
            q: np.ndarray, qd: np.ndarray, t: float  # type: ignore[type-arg]
        ) -> np.ndarray:  # type: ignore[type-arg]
            return np.array([0.0, -100.0 * q[1]])

        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=force_fn,
            _id="load1",
        )
        q = np.array([0.0, 2.0, 0.0])
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)
        assert Q[1] == pytest.approx(-200.0)

    def test_with_torque(self) -> None:
        """External load with both force and torque."""
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([10.0, 0.0]),
            torque_func=lambda q, qd, t: 25.0,
            _id="load1",
        )
        q = np.array([0.0, 0.0, 0.0])
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(10.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(25.0)  # torque only (force at origin)

    def test_torque_only(self) -> None:
        """Zero force with nonzero torque."""
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([0.0, 0.0]),
            torque_func=lambda q, qd, t: -15.0,
            _id="load1",
        )
        q = np.array([0.0, 0.0, 0.5])
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)

        assert Q[0] == pytest.approx(0.0)
        assert Q[1] == pytest.approx(0.0)
        assert Q[2] == pytest.approx(-15.0)

    def test_has_id(self) -> None:
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.zeros(2),
            _id="my_load",
        )
        assert load.id == "my_load"

    def test_ground_body_ignored(self) -> None:
        """Load on ground produces zero Q."""
        load = ExternalLoad(
            body_id="ground",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([100.0, 200.0]),
            _id="load1",
        )
        q = np.array([0.0, 0.0, 0.0])
        Q = load.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_array_equal(Q, np.zeros(3))


class TestExternalLoadVirtualWork:
    """Virtual work consistency."""

    def test_virtual_work(self) -> None:
        """Q·δq matches F·δr_P for small perturbation."""
        state = State()
        state.register_body("b1")

        F_const = np.array([30.0, -20.0])
        load = ExternalLoad(
            body_id="b1",
            local_point=np.array([0.5, 0.2]),
            force_func=lambda q, qd, t: F_const,
            _id="load1",
        )
        q = np.array([1.0, 0.5, 0.8])
        dq = np.array([1e-7, -2e-7, 5e-8])

        Q = load.evaluate(state, q, np.zeros(3), 0.0)
        dW_Q = float(np.dot(Q, dq))

        r_P = state.body_point_global("b1", np.array([0.5, 0.2]), q)
        r_P_plus = state.body_point_global("b1", np.array([0.5, 0.2]), q + dq)
        dW_F = float(np.dot(F_const, r_P_plus - r_P))

        np.testing.assert_allclose(dW_Q, dW_F, rtol=1e-5)
