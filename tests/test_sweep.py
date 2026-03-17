"""Tests for position sweep solver."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.sweep import SweepResult, position_sweep


def build_driven_fourbar() -> Mechanism:
    """4-bar with identity driver: f(t) = t, so t = crank angle.

    Ground pivots at O2=(0,0), O4=(4,0).
    Crank=1, Coupler=3, Rocker=2, Ground=4.
    Grashof (s+l=1+4=5 <= p+q=3+2=5), crank-rocker.
    """
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

    # Identity driver: f(t) = t, so t IS the crank angle
    mech.add_revolute_driver(
        "D1",
        "ground",
        "crank",
        f=lambda t: t,
        f_dot=lambda t: 1.0,
        f_ddot=lambda t: 0.0,
    )

    mech.build()
    return mech


def get_valid_q0(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    """Compute a valid initial q for the 4-bar at a given crank angle.

    Uses the analytical two-circle intersection for crank=1, coupler=3,
    rocker=2, ground=4.
    """
    q = mech.state.make_q()

    bx = np.cos(angle)
    by = np.sin(angle)

    dx = 4.0 - bx
    dy = -by
    d2 = dx * dx + dy * dy
    d = np.sqrt(d2)

    a_dist = (d2 + 9.0 - 4.0) / (2.0 * d)
    h = np.sqrt(max(9.0 - a_dist * a_dist, 0.0))

    ex = dx / d
    ey = dy / d
    mx = bx + a_dist * ex
    my = by + a_dist * ey

    cx = mx + h * (-ey)
    cy = my + h * ex

    # Body origins (at p1 of each bar)
    # Crank: origin at A=(0,0)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    # Coupler: origin at B
    coupler_theta = np.arctan2(cy - by, cx - bx)
    mech.state.set_pose("coupler", q, bx, by, coupler_theta)
    # Rocker: origin at D=(4,0)
    rocker_theta = np.arctan2(cy, cx - 4.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, rocker_theta)

    return q


class TestPositionSweep:
    def test_sweep_all_converge(self) -> None:
        """Sweep over 0 to 150° should all converge for this Grashof 4-bar."""
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, np.radians(150), 30)
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)
        assert result.n_converged == 30
        assert result.n_failed == 0
        assert len(result.solutions) == 30

    def test_sweep_constraints_satisfied(self) -> None:
        """Every converged solution should satisfy constraints."""
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, np.pi, 20)
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)

        for i, (sol, angle) in enumerate(zip(result.solutions, angles)):
            assert sol is not None, f"Step {i} did not converge"
            phi = assemble_constraints(mech, sol, float(angle))
            norm = float(np.linalg.norm(phi))
            assert norm < 1e-9, f"Step {i}: residual = {norm}"

    def test_sweep_crank_angle_matches_input(self) -> None:
        """Crank angle in solution should match the input angle."""
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, np.pi / 2, 10)
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)

        for sol, angle in zip(result.solutions, angles):
            assert sol is not None
            crank_theta = mech.state.get_angle("crank", sol)
            assert abs(crank_theta - angle) < 1e-8

    def test_sweep_result_fields(self) -> None:
        mech = build_driven_fourbar()
        angles = np.array([0.5, 1.0, 1.5])
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)
        assert isinstance(result, SweepResult)
        np.testing.assert_array_equal(result.input_values, angles)
        assert len(result.results) == 3
        assert result.n_converged + result.n_failed == 3

    def test_sweep_continuation_from_previous(self) -> None:
        """Continuation should help convergence at larger steps."""
        mech = build_driven_fourbar()
        # Large step size but starting from valid config
        angles = np.linspace(0.1, np.radians(90), 5)
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)
        assert result.n_converged == 5

    def test_sweep_single_step(self) -> None:
        """Sweep with one angle should work."""
        mech = build_driven_fourbar()
        angles = np.array([np.pi / 4])
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)
        assert result.n_converged == 1
        assert len(result.solutions) == 1

    def test_sweep_does_not_modify_q0(self) -> None:
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, 1.0, 5)
        q0 = get_valid_q0(mech, angles[0])
        q0_copy = q0.copy()

        position_sweep(mech, q0, angles)
        np.testing.assert_array_equal(q0, q0_copy)

    def test_sweep_full_rotation_grashof(self) -> None:
        """Grashof 4-bar should handle full crank rotation (with small steps)."""
        mech = build_driven_fourbar()
        # Full rotation with 72 steps (5° each)
        angles = np.linspace(0.01, 2 * np.pi - 0.01, 72)
        q0 = get_valid_q0(mech, angles[0])

        result = position_sweep(mech, q0, angles)
        # Most steps should converge for a Grashof mechanism
        assert result.n_converged >= 70

    def test_result_is_frozen(self) -> None:
        mech = build_driven_fourbar()
        angles = np.array([0.5])
        q0 = get_valid_q0(mech, angles[0])
        result = position_sweep(mech, q0, angles)

        with pytest.raises(AttributeError):
            result.n_converged = 99  # type: ignore[misc]
