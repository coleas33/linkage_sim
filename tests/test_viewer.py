"""Tests for Matplotlib viewer (non-interactive, backend 'Agg')."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.sweep import position_sweep
from linkage_sim.viz.viewer import plot_coupler_trace, plot_mechanism


def build_driven_fourbar() -> Mechanism:
    """4-bar with identity driver."""
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


def get_solved_q(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("coupler", q, np.cos(angle), np.sin(angle), 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    return result.q


class TestPlotMechanism:
    def test_returns_axes(self) -> None:
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        ax = plot_mechanism(mech, q)
        assert ax is not None
        plt.close("all")

    def test_with_existing_axes(self) -> None:
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        fig, ax = plt.subplots()
        result_ax = plot_mechanism(mech, q, ax=ax)
        assert result_ax is ax
        plt.close("all")

    def test_draws_lines(self) -> None:
        """Should have line objects for bodies."""
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        ax = plot_mechanism(mech, q)
        # At least 3 body lines + joint markers + ground markers
        assert len(ax.lines) > 0
        plt.close("all")

    def test_no_joints_option(self) -> None:
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        ax = plot_mechanism(mech, q, show_joints=False, show_ground=False)
        # Fewer elements when joints/ground are hidden
        n_no_markers = len(ax.lines)

        ax2 = plot_mechanism(mech, q, show_joints=True, show_ground=True)
        n_with_markers = len(ax2.lines)

        assert n_with_markers > n_no_markers
        plt.close("all")

    def test_equal_aspect(self) -> None:
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        ax = plot_mechanism(mech, q)
        # set_aspect("equal") may return "equal" or 1.0 depending on mpl version
        assert ax.get_aspect() in ("equal", 1.0)
        plt.close("all")


class TestPlotCouplerTrace:
    def test_returns_axes(self) -> None:
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, np.pi, 20)
        q0 = get_solved_q(mech, angles[0])
        result = position_sweep(mech, q0, angles)

        ax = plot_coupler_trace(
            mech, result.solutions, "coupler", "P"
        )
        assert ax is not None
        plt.close("all")

    def test_trace_has_correct_point_count(self) -> None:
        mech = build_driven_fourbar()
        angles = np.linspace(0.1, np.pi, 15)
        q0 = get_solved_q(mech, angles[0])
        result = position_sweep(mech, q0, angles)

        ax = plot_coupler_trace(
            mech, result.solutions, "coupler", "P"
        )
        # Should have one line with 15 points (all converged)
        line = ax.lines[0]
        assert len(line.get_xdata()) == result.n_converged
        plt.close("all")

    def test_handles_none_solutions(self) -> None:
        """Should skip None entries gracefully."""
        mech = build_driven_fourbar()
        solutions: list[np.ndarray | None] = [  # type: ignore[type-arg]
            get_solved_q(mech, 0.5),
            None,
            get_solved_q(mech, 1.0),
        ]
        ax = plot_coupler_trace(
            mech, solutions, "coupler", "P"
        )
        line = ax.lines[0]
        assert len(line.get_xdata()) == 2  # skipped the None
        plt.close("all")

    def test_with_existing_axes(self) -> None:
        mech = build_driven_fourbar()
        q = get_solved_q(mech, np.pi / 4)
        fig, ax = plt.subplots()
        result_ax = plot_coupler_trace(
            mech, [q], "coupler", "P", ax=ax
        )
        assert result_ax is ax
        plt.close("all")
