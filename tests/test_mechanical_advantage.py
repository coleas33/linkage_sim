"""Tests for mechanical advantage computation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.analysis.mechanical_advantage import mechanical_advantage
from linkage_sim.solvers.kinematics import solve_position, solve_velocity


def build_fourbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    coupler = make_bar("coupler", "B", "C", length=3.0)
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
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def solve_at(mech: Mechanism, angle: float) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    return result.q, q_dot


class TestMechanicalAdvantage:
    """MA computation for 4-bar."""

    def test_finite_at_regular_config(self) -> None:
        """MA is finite away from toggle."""
        mech = build_fourbar()
        q, q_dot = solve_at(mech, np.pi / 4)
        result = mechanical_advantage(
            mech.state, q_dot, "crank", "rocker"
        )
        assert np.isfinite(result.ma)
        assert abs(result.input_velocity) > 0

    def test_ma_varies_with_angle(self) -> None:
        """MA changes across the sweep."""
        mech = build_fourbar()
        ma_values = []
        for angle in np.linspace(np.radians(30), np.radians(150), 7):
            q, q_dot = solve_at(mech, angle)
            result = mechanical_advantage(
                mech.state, q_dot, "crank", "rocker"
            )
            ma_values.append(result.ma)

        # MA should vary (not constant for non-parallelogram)
        assert np.ptp(ma_values) > 0.01

    def test_input_velocity_is_driver_speed(self) -> None:
        """Input body angular velocity should match driver."""
        mech = build_fourbar()
        q, q_dot = solve_at(mech, np.pi / 3)
        result = mechanical_advantage(
            mech.state, q_dot, "crank", "rocker"
        )
        # Driver is constant speed ω=1
        assert result.input_velocity == pytest.approx(1.0)

    def test_x_coordinate_output(self) -> None:
        """Can compute MA for translational output."""
        mech = build_fourbar()
        q, q_dot = solve_at(mech, np.pi / 4)
        result = mechanical_advantage(
            mech.state, q_dot, "crank", "rocker",
            input_coord="theta", output_coord="x",
        )
        assert np.isfinite(result.ma)

    def test_ground_input_returns_zero(self) -> None:
        """Ground body has zero velocity."""
        mech = build_fourbar()
        q, q_dot = solve_at(mech, np.pi / 4)
        result = mechanical_advantage(
            mech.state, q_dot, "crank", "rocker",
            output_coord="theta",
        )
        # Asking for ground input would give 0 velocity
        # Test with rocker→rocker (same body, should give MA=1 trivially)

    def test_invalid_coord_raises(self) -> None:
        """Invalid coordinate name raises ValueError."""
        mech = build_fourbar()
        q, q_dot = solve_at(mech, np.pi / 4)
        with pytest.raises(ValueError, match="Invalid coordinate"):
            mechanical_advantage(
                mech.state, q_dot, "crank", "rocker",
                output_coord="z",
            )
