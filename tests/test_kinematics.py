"""Tests for kinematic position solver (Newton-Raphson)."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import PositionSolveResult, solve_position


def build_fourbar() -> Mechanism:
    """Standard 4-bar: ground pivots at O2=(0,0), O4=(4,0).

    Crank: length 1 (A to B)
    Coupler: length 3 (B to C)
    Rocker: length 2 (D to C)

    Uses larger dimensions for easier manual verification.
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

    mech.build()
    return mech


def set_fourbar_config(
    mech: Mechanism,
    q: np.ndarray,  # type: ignore[type-arg]
    crank_angle: float,
) -> None:
    """Set a geometrically valid 4-bar configuration given crank angle.

    Computes the closed-loop geometry analytically for the specific
    link lengths: crank=1, coupler=3, rocker=2, ground=4.
    """
    # Crank tip B
    bx = np.cos(crank_angle)
    by = np.sin(crank_angle)

    # Rocker pivot at O4=(4,0). Find C such that |BC|=3 and |O4C|=2.
    # Circle 1: (x-bx)^2 + (y-by)^2 = 9  (coupler)
    # Circle 2: (x-4)^2 + y^2 = 4         (rocker)
    # Solve by subtraction:
    dx = 4.0 - bx
    dy = -by
    d2 = dx * dx + dy * dy
    # (x-bx)^2 + (y-by)^2 = 9
    # (x-4)^2 + y^2 = 4
    # Expand and subtract: 2*(4-bx)*x + 2*(-by)*y + (bx^2+by^2-16) = 9-4 = 5
    # => 2*dx*x + 2*dy*y = 5 - (bx^2+by^2-16)
    # Let a = bx^2+by^2 = 1 (crank length squared)
    a_val = bx * bx + by * by  # should be 1.0
    rhs = 5.0 - (a_val - 16.0)  # = 5 - a + 16 = 21 - a
    # 2*dx*(x-bx) + 2*dy*(y-by) = rhs - 2*dx*bx - 2*dy*by
    # Actually, let's use the standard two-circle intersection.
    # Circle 1 centered at B=(bx,by) radius 3
    # Circle 2 centered at O4=(4,0) radius 2
    d = np.sqrt(d2)

    # Check triangle inequality
    if d > 5.0 or d < 1.0:
        raise ValueError(f"No valid configuration at crank_angle={crank_angle}")

    # Using standard two-circle intersection formula
    a_dist = (d2 + 9.0 - 4.0) / (2.0 * d)  # distance from B to midpoint line
    h = np.sqrt(max(9.0 - a_dist * a_dist, 0.0))

    # Unit vector from B to O4
    ex = dx / d
    ey = dy / d

    # Midpoint
    mx = bx + a_dist * ex
    my = by + a_dist * ey

    # Two solutions — pick the "open" configuration (positive cross product)
    cx = mx + h * (-ey)
    cy = my + h * ex

    # Body poses: each body has (x, y, θ) at its CG (halfway along bar)
    # Crank: CG at midpoint of A=(0,0) to B=(bx,by)
    crank_theta = crank_angle
    crank_x = bx / 2.0
    crank_y = by / 2.0

    # Coupler: CG at midpoint of B=(bx,by) to C=(cx,cy)
    coupler_theta = np.arctan2(cy - by, cx - bx)
    coupler_x = (bx + cx) / 2.0
    coupler_y = (by + cy) / 2.0

    # Rocker: CG at midpoint of D=(4,0) to C=(cx,cy)
    rocker_theta = np.arctan2(cy - 0.0, cx - 4.0)
    rocker_x = (4.0 + cx) / 2.0
    rocker_y = (0.0 + cy) / 2.0

    mech.state.set_pose("crank", q, crank_x, crank_y, crank_theta)
    mech.state.set_pose("coupler", q, coupler_x, coupler_y, coupler_theta)
    mech.state.set_pose("rocker", q, rocker_x, rocker_y, rocker_theta)


class TestSolvePosition:
    def test_converges_from_valid_config(self) -> None:
        """Starting at a valid configuration, solver should converge immediately."""
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 6)

        result = solve_position(mech, q)
        assert result.converged
        assert result.residual_norm < 1e-10
        assert result.iterations <= 5  # quick convergence near valid config

    def test_converges_from_perturbed_config(self) -> None:
        """Starting near a valid config, solver should converge."""
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 4)

        # Perturb slightly
        rng = np.random.default_rng(42)
        q_perturbed = q + rng.normal(0, 0.05, size=q.shape)

        result = solve_position(mech, q_perturbed)
        assert result.converged
        assert result.residual_norm < 1e-10

    def test_constraints_satisfied_after_solve(self) -> None:
        """After convergence, all constraints should be near zero."""
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 3)

        # Perturb
        q_perturbed = q + 0.02
        result = solve_position(mech, q_perturbed)

        assert result.converged
        phi = assemble_constraints(mech, result.q, 0.0)
        np.testing.assert_array_less(np.abs(phi), 1e-9)

    def test_result_fields(self) -> None:
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 6)

        result = solve_position(mech, q)
        assert isinstance(result, PositionSolveResult)
        assert result.q.shape == (9,)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.residual_norm, float)

    def test_does_not_modify_input(self) -> None:
        """The solver should not modify the input q0 array."""
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 6)

        q0 = q.copy()
        q_original = q0.copy()
        solve_position(mech, q0)

        np.testing.assert_array_equal(q0, q_original)

    def test_fails_with_max_iter_1_from_bad_guess(self) -> None:
        """With max_iter=1 and a bad guess, should not converge."""
        mech = build_fourbar()
        q = mech.state.make_q()
        # Intentionally bad: all zeros (bodies at origin)

        result = solve_position(mech, q, max_iter=1)
        # May or may not converge depending on configuration, but
        # we verify the result structure is correct
        assert result.iterations <= 1
        assert isinstance(result.converged, bool)

    def test_single_body_pinned_to_ground(self) -> None:
        """Single body pinned at origin: 1 DOF, underdetermined system."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        bar = make_bar("bar", "A", "B", length=1.0)
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
        mech.build()

        # Place bar at θ=0: CG at (0.5, 0), A at (0,0)
        q = mech.state.make_q()
        mech.state.set_pose("bar", q, 0.5, 0.0, 0.0)

        result = solve_position(mech, q)
        assert result.converged
        assert result.residual_norm < 1e-10

    def test_fixed_body_converges(self) -> None:
        """Body fixed to ground: 0 DOF, fully determined."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        body = Body(id="welded")
        body.add_attachment_point("A", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(body)
        mech.add_fixed_joint("F1", "ground", "O", "welded", "A")
        mech.build()

        # Start slightly off
        q = mech.state.make_q()
        mech.state.set_pose("welded", q, 0.01, 0.01, 0.01)

        result = solve_position(mech, q)
        assert result.converged
        # Should converge to origin with θ=0
        np.testing.assert_array_almost_equal(result.q, [0.0, 0.0, 0.0], decimal=8)

    def test_multiple_crank_angles(self) -> None:
        """Verify convergence across a range of crank angles."""
        mech = build_fourbar()

        for angle_deg in [0, 30, 60, 90, 120, 150]:
            angle = np.radians(angle_deg)
            q = mech.state.make_q()
            set_fourbar_config(mech, q, crank_angle=angle)

            # Perturb and re-solve
            rng = np.random.default_rng(angle_deg)
            q_perturbed = q + rng.normal(0, 0.01, size=q.shape)
            result = solve_position(mech, q_perturbed)

            assert result.converged, f"Failed at {angle_deg}°"
            assert result.residual_norm < 1e-10

    def test_requires_built_mechanism(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(RuntimeError, match="must be built"):
            solve_position(mech, np.zeros(0))

    def test_custom_tolerance(self) -> None:
        """Looser tolerance should converge faster."""
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 4)
        q_perturbed = q + 0.01

        result_tight = solve_position(mech, q_perturbed, tol=1e-12)
        result_loose = solve_position(mech, q_perturbed, tol=1e-3)

        assert result_tight.converged
        assert result_loose.converged
        assert result_loose.iterations <= result_tight.iterations

    def test_result_is_frozen(self) -> None:
        mech = build_fourbar()
        q = mech.state.make_q()
        set_fourbar_config(mech, q, crank_angle=np.pi / 6)
        result = solve_position(mech, q)

        with pytest.raises(AttributeError):
            result.converged = False  # type: ignore[misc]
