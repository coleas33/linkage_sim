"""Tests for mechanism animation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.sweep import position_sweep
from linkage_sim.viz.animation import animate_mechanism


def build_driven_fourbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    coupler = make_bar("coupler", "B", "C", length=3.0)
    coupler.add_coupler_point("P", 1.5, 0.5)
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


def get_sweep_solutions(mech: Mechanism, n: int = 20) -> list[np.ndarray | None]:  # type: ignore[type-arg]
    angles = np.linspace(0.1, np.pi, n)
    q0 = mech.state.make_q()
    mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
    mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
    mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q0, t=angles[0])
    assert result.converged

    sweep = position_sweep(mech, result.q, angles)
    return sweep.solutions


class TestAnimateMechanism:
    def test_returns_animation(self) -> None:
        from matplotlib.animation import FuncAnimation

        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 10)
        anim = animate_mechanism(mech, solutions)
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_frame_count(self) -> None:
        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 15)
        anim = animate_mechanism(mech, solutions)
        # All should converge for this mechanism
        n_valid = sum(1 for s in solutions if s is not None)
        assert anim._save_count == n_valid  # type: ignore[attr-defined]
        plt.close("all")

    def test_with_coupler_trace(self) -> None:
        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 10)
        anim = animate_mechanism(
            mech, solutions,
            coupler_trace_body="coupler",
            coupler_trace_point="P",
        )
        assert anim is not None
        plt.close("all")

    def test_handles_none_solutions(self) -> None:
        """None entries should be skipped."""
        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 10)
        # Insert some Nones
        solutions_with_gaps: list[np.ndarray | None] = [  # type: ignore[type-arg]
            solutions[0], None, solutions[2], None, solutions[4]
        ]
        anim = animate_mechanism(mech, solutions_with_gaps)
        assert anim._save_count == 3  # type: ignore[attr-defined]
        plt.close("all")

    def test_custom_interval(self) -> None:
        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 5)
        anim = animate_mechanism(mech, solutions, interval=100)
        assert anim._interval == 100  # type: ignore[attr-defined]
        plt.close("all")

    def test_single_frame(self) -> None:
        mech = build_driven_fourbar()
        solutions = get_sweep_solutions(mech, 1)
        anim = animate_mechanism(mech, solutions)
        assert anim is not None
        plt.close("all")
