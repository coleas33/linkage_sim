"""Tests for the interactive mechanism viewer.

Tests focus on:
1. Pre-computation produces correct-length arrays.
2. launch_interactive can be called without error (Agg backend, non-blocking).
3. The viewer works for both 4-bar and slider-crank mechanisms.
4. Sweep data integrity (no crashes, correct shapes).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before any pyplot import

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import (
    SweepData,
    _compute_bounds,
    _detect_fourbar_link_lengths,
    _find_coupler_body_and_point,
    launch_interactive,
    precompute_sweep,
)


# ---------------------------------------------------------------------------
# Mechanism builders (mirror the benchmark factories, with mass for statics)
# ---------------------------------------------------------------------------

def build_fourbar() -> tuple[Mechanism, list]:
    """Build the benchmark 4-bar with gravity."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0, mass=0.5, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", length=3.0, mass=1.5, Izz_cg=0.1)
    coupler.add_coupler_point("P", 1.5, 0.5)
    rocker = make_bar("rocker", "D", "C", length=2.0, mass=1.0, Izz_cg=0.05)

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

    gravity = Gravity(
        g_vector=np.array([0.0, -9.81]),
        bodies=mech.bodies,
    )
    return mech, [gravity]


def build_slidercrank() -> tuple[Mechanism, list]:
    """Build the benchmark slider-crank with gravity."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0, mass=0.5, Izz_cg=0.01)
    conrod = make_bar("conrod", "B", "C", length=3.0, mass=1.5, Izz_cg=0.1)
    conrod.add_coupler_point("P", 1.5, 0.3)
    slider = Body(
        id="slider",
        attachment_points={"pin": np.array([0.0, 0.0])},
        mass=0.8,
    )

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(conrod)
    mech.add_body(slider)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "pin")
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "pin",
        axis_local_i=np.array([1.0, 0.0]),
        delta_theta_0=0.0,
    )
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()

    gravity = Gravity(
        g_vector=np.array([0.0, -9.81]),
        bodies=mech.bodies,
    )
    return mech, [gravity]


def build_fourbar_no_forces() -> Mechanism:
    """Build the benchmark 4-bar without mass/forces (kinematic only)."""
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


# ---------------------------------------------------------------------------
# Tests: Pre-computation array lengths
# ---------------------------------------------------------------------------

class TestPrecomputeArrayLengths:
    """Verify that precompute_sweep produces arrays of the expected size."""

    def test_fourbar_sweep_lengths(self) -> None:
        """4-bar: all arrays should have n_steps entries."""
        mech, forces = build_fourbar()
        n_steps = 72
        data = precompute_sweep(mech, forces, n_steps=n_steps)

        assert data.n_steps == n_steps
        assert len(data.angles_deg) == n_steps
        assert len(data.angles_rad) == n_steps
        assert len(data.solutions) == n_steps
        assert len(data.velocities) == n_steps
        assert len(data.static_results) == n_steps
        assert len(data.reactions) == n_steps
        assert len(data.driver_torques) == n_steps

    def test_slidercrank_sweep_lengths(self) -> None:
        """Slider-crank: all arrays should have n_steps entries."""
        mech, forces = build_slidercrank()
        n_steps = 36
        data = precompute_sweep(mech, forces, n_steps=n_steps)

        assert data.n_steps == n_steps
        assert len(data.solutions) == n_steps
        assert len(data.velocities) == n_steps
        assert len(data.driver_torques) == n_steps

    def test_joint_reaction_arrays_correct_length(self) -> None:
        """Each joint's reaction magnitude array should have n_steps entries."""
        mech, forces = build_fourbar()
        n_steps = 36
        data = precompute_sweep(mech, forces, n_steps=n_steps)

        for jid, mags in data.joint_reaction_mags.items():
            assert len(mags) == n_steps, f"Joint {jid} has wrong array length"

    def test_fourbar_transmission_angle_present(self) -> None:
        """4-bar should have transmission angle data."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=36)

        assert data.transmission_angles_deg is not None
        assert len(data.transmission_angles_deg) == 36

    def test_slidercrank_no_transmission_angle(self) -> None:
        """Slider-crank should NOT have transmission angle data."""
        mech, forces = build_slidercrank()
        data = precompute_sweep(mech, forces, n_steps=36)

        assert data.transmission_angles_deg is None

    def test_coupler_trace_present(self) -> None:
        """Coupler trace arrays should be present when coupler points exist."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=36)

        assert data.coupler_trace_x is not None
        assert data.coupler_trace_y is not None
        assert len(data.coupler_trace_x) == 36
        assert len(data.coupler_trace_y) == 36


# ---------------------------------------------------------------------------
# Tests: Pre-computation data integrity
# ---------------------------------------------------------------------------

class TestPrecomputeDataIntegrity:
    """Verify that pre-computed data has reasonable values."""

    def test_fourbar_most_steps_converge(self) -> None:
        """Most sweep steps should converge for the benchmark 4-bar."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=72)

        converged = sum(1 for q in data.solutions if q is not None)
        # Expect at least 90% convergence
        assert converged >= 0.9 * 72, f"Only {converged}/72 steps converged"

    def test_slidercrank_most_steps_converge(self) -> None:
        """Most sweep steps should converge for the slider-crank."""
        mech, forces = build_slidercrank()
        data = precompute_sweep(mech, forces, n_steps=72)

        converged = sum(1 for q in data.solutions if q is not None)
        assert converged >= 0.9 * 72, f"Only {converged}/72 steps converged"

    def test_driver_torques_are_finite_where_converged(self) -> None:
        """Driver torques should be finite at converged steps."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=72)

        for i, q in enumerate(data.solutions):
            if q is not None:
                torque = data.driver_torques[i]
                if not np.isnan(torque):
                    assert np.isfinite(torque), f"Non-finite torque at step {i}"

    def test_transmission_angles_in_valid_range(self) -> None:
        """Transmission angles should be between 0 and 180 degrees."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=72)

        assert data.transmission_angles_deg is not None
        valid = data.transmission_angles_deg[~np.isnan(data.transmission_angles_deg)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 180.0)

    def test_no_force_elements_still_works(self) -> None:
        """Pre-computation should work without force elements (kinematic only)."""
        mech = build_fourbar_no_forces()
        data = precompute_sweep(mech, [], n_steps=36)

        assert len(data.solutions) == 36
        assert np.all(np.isnan(data.driver_torques))
        # Transmission angle should still be computed
        assert data.transmission_angles_deg is not None


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Test internal helper functions."""

    def test_detect_fourbar_link_lengths(self) -> None:
        """Should detect 4-bar and return correct link lengths."""
        mech, _ = build_fourbar()
        result = _detect_fourbar_link_lengths(mech)

        assert result is not None
        a, b, c, d = result
        assert abs(a - 1.0) < 1e-10  # crank
        assert abs(b - 3.0) < 1e-10  # coupler
        assert abs(c - 2.0) < 1e-10  # rocker
        assert abs(d - 4.0) < 1e-10  # ground

    def test_detect_fourbar_returns_none_for_slidercrank(self) -> None:
        """Slider-crank should NOT be detected as a 4-bar."""
        mech, _ = build_slidercrank()
        result = _detect_fourbar_link_lengths(mech)
        assert result is None

    def test_find_coupler_body_and_point_fourbar(self) -> None:
        """Should find the coupler point on the coupler body."""
        mech, _ = build_fourbar()
        result = _find_coupler_body_and_point(mech)

        assert result is not None
        body_id, point_name = result
        assert body_id == "coupler"
        assert point_name == "P"

    def test_find_coupler_body_and_point_slidercrank(self) -> None:
        """Should find the coupler point on the conrod body."""
        mech, _ = build_slidercrank()
        result = _find_coupler_body_and_point(mech)

        assert result is not None
        body_id, point_name = result
        assert body_id == "conrod"
        assert point_name == "P"

    def test_compute_bounds_returns_valid_range(self) -> None:
        """Bounds should be a valid (non-degenerate) bounding box."""
        mech, forces = build_fourbar()
        data = precompute_sweep(mech, forces, n_steps=36)

        x_min, x_max, y_min, y_max = _compute_bounds(mech, data.solutions)
        assert x_max > x_min
        assert y_max > y_min

    def test_compute_bounds_empty_solutions(self) -> None:
        """Bounds should return a default range for empty solutions."""
        mech, _ = build_fourbar()
        x_min, x_max, y_min, y_max = _compute_bounds(mech, [None, None])
        assert x_max > x_min
        assert y_max > y_min


# ---------------------------------------------------------------------------
# Tests: launch_interactive (Agg backend, non-blocking)
# ---------------------------------------------------------------------------

class TestLaunchInteractive:
    """Test that the viewer can be created without errors."""

    def test_fourbar_viewer_creates_figure(self) -> None:
        """launch_interactive should return a figure and sweep data for 4-bar."""
        mech, forces = build_fourbar()
        fig, data = launch_interactive(
            mech, force_elements=forces, n_steps=36, show=False,
        )

        import matplotlib.pyplot as plt
        assert fig is not None
        assert isinstance(data, SweepData)
        assert data.n_steps == 36
        plt.close(fig)

    def test_slidercrank_viewer_creates_figure(self) -> None:
        """launch_interactive should return a figure and sweep data for slider-crank."""
        mech, forces = build_slidercrank()
        fig, data = launch_interactive(
            mech, force_elements=forces, n_steps=36, show=False,
        )

        import matplotlib.pyplot as plt
        assert fig is not None
        assert isinstance(data, SweepData)
        assert data.n_steps == 36
        plt.close(fig)

    def test_fourbar_no_forces_viewer(self) -> None:
        """Viewer should work without force elements (kinematic only mode)."""
        mech = build_fourbar_no_forces()
        fig, data = launch_interactive(
            mech, force_elements=[], n_steps=36, show=False,
        )

        import matplotlib.pyplot as plt
        assert fig is not None
        assert isinstance(data, SweepData)
        plt.close(fig)

    def test_viewer_returns_correct_sweep_data(self) -> None:
        """The returned sweep data should match what precompute_sweep produces."""
        mech, forces = build_fourbar()
        fig, data = launch_interactive(
            mech, force_elements=forces, n_steps=36, show=False,
        )

        import matplotlib.pyplot as plt
        assert len(data.solutions) == 36
        assert len(data.driver_torques) == 36
        assert data.transmission_angles_deg is not None
        assert data.coupler_trace_x is not None
        plt.close(fig)
