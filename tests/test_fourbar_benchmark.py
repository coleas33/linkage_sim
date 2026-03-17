"""4-bar benchmark test suite.

Validates the complete kinematic pipeline (position, velocity, acceleration,
coupler point, DOF, sweep) against analytically derived reference values
for a well-defined Grashof crank-rocker 4-bar linkage.

Mechanism: ground=4, crank=1, coupler=3, rocker=2.
Ground pivots: O2=(0,0), O4=(4,0).
Grashof condition: s+l=1+4=5 = p+q=3+2=5 (boundary Grashof).
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.coupler import eval_coupler_point
from linkage_sim.analysis.validation import grubler_dof, jacobian_rank_analysis
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep


# --- Mechanism factory ---

def build_benchmark_fourbar() -> Mechanism:
    """Build the benchmark 4-bar with identity driver f(t) = t."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    coupler = make_bar("coupler", "B", "C", length=3.0)
    coupler.add_coupler_point("P", 1.5, 0.5)  # midpoint offset
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


def solve_at_angle(
    mech: Mechanism, angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Solve position, velocity, acceleration at a given crank angle."""
    q = mech.state.make_q()
    # Initial guess
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)

    pos_result = solve_position(mech, q, t=angle)
    assert pos_result.converged, f"Position solve failed at angle={np.degrees(angle):.1f}°"

    q_dot = solve_velocity(mech, pos_result.q, t=angle)
    q_ddot = solve_acceleration(mech, pos_result.q, q_dot, t=angle)

    return pos_result.q, q_dot, q_ddot


def analytical_fourbar_positions(
    crank_angle: float,
) -> tuple[float, float, float, float]:
    """Compute coupler and rocker angles analytically via two-circle intersection.

    Returns: (bx, by, cx, cy) where B is crank tip, C is coupler-rocker joint.
    """
    bx = np.cos(crank_angle)
    by = np.sin(crank_angle)

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

    return bx, by, cx, cy


# --- Benchmark Tests ---

class TestFourbarDOF:
    """Validate topology and constraint analysis."""

    def test_grubler_dof(self) -> None:
        """Grübler: M = 3*3 - (4*2 + 1*1) = 9 - 9 = 0 (fully determined with driver)."""
        mech = build_benchmark_fourbar()
        result = grubler_dof(mech, expected_dof=0)
        assert result.dof == 0
        assert not result.is_warning

    def test_jacobian_rank_full(self) -> None:
        """Jacobian should be full rank at a generic configuration."""
        mech = build_benchmark_fourbar()
        q, _, _ = solve_at_angle(mech, np.pi / 4)
        result = jacobian_rank_analysis(mech, q, t=np.pi / 4)
        assert result.constraint_rank == 9
        assert result.instantaneous_mobility == 0
        assert not result.has_redundant_constraints

    def test_body_count(self) -> None:
        mech = build_benchmark_fourbar()
        assert mech.state.n_moving_bodies == 3
        assert mech.state.n_coords == 9

    def test_constraint_count(self) -> None:
        mech = build_benchmark_fourbar()
        # 4 revolute * 2 + 1 driver = 9
        assert mech.n_constraints == 9


class TestFourbarPosition:
    """Validate position solve against analytical geometry."""

    @pytest.mark.parametrize("angle_deg", [30, 45, 60, 90, 120, 150])
    def test_crank_tip_position(self, angle_deg: int) -> None:
        """Crank tip B should be at (cos θ, sin θ)."""
        mech = build_benchmark_fourbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        # B is at local (1.0, 0.0) on crank body
        b_global = mech.state.body_point_global(
            "crank", np.array([1.0, 0.0]), q
        )
        expected_bx = np.cos(angle)
        expected_by = np.sin(angle)
        np.testing.assert_allclose(b_global, [expected_bx, expected_by], atol=1e-8)

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_rocker_pivot_at_ground(self, angle_deg: int) -> None:
        """Rocker attachment D should remain at O4=(4, 0)."""
        mech = build_benchmark_fourbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        d_global = mech.state.body_point_global(
            "rocker", np.array([0.0, 0.0]), q
        )
        np.testing.assert_allclose(d_global, [4.0, 0.0], atol=1e-8)

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_coupler_rocker_joint_analytical(self, angle_deg: int) -> None:
        """Point C should match the analytical two-circle intersection."""
        mech = build_benchmark_fourbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        _, _, cx_expected, cy_expected = analytical_fourbar_positions(angle)

        # C is at local (3.0, 0.0) on coupler
        c_global = mech.state.body_point_global(
            "coupler", np.array([3.0, 0.0]), q
        )
        np.testing.assert_allclose(
            c_global, [cx_expected, cy_expected], atol=1e-6
        )

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_link_lengths_preserved(self, angle_deg: int) -> None:
        """All link lengths should be preserved after solve."""
        mech = build_benchmark_fourbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        # Crank: |B - A| = 1
        a = mech.state.body_point_global("crank", np.array([0.0, 0.0]), q)
        b = mech.state.body_point_global("crank", np.array([1.0, 0.0]), q)
        assert abs(np.linalg.norm(b - a) - 1.0) < 1e-8

        # Coupler: |C - B| = 3
        b2 = mech.state.body_point_global("coupler", np.array([0.0, 0.0]), q)
        c = mech.state.body_point_global("coupler", np.array([3.0, 0.0]), q)
        assert abs(np.linalg.norm(c - b2) - 3.0) < 1e-8

        # Rocker: |C - D| = 2
        d = mech.state.body_point_global("rocker", np.array([0.0, 0.0]), q)
        c2 = mech.state.body_point_global("rocker", np.array([2.0, 0.0]), q)
        assert abs(np.linalg.norm(c2 - d) - 2.0) < 1e-8

    def test_constraints_zero_after_solve(self) -> None:
        """All constraint residuals should be near-zero."""
        mech = build_benchmark_fourbar()
        angle = np.pi / 4
        q, _, _ = solve_at_angle(mech, angle)
        phi = assemble_constraints(mech, q, angle)
        assert np.linalg.norm(phi) < 1e-10


class TestFourbarVelocity:
    """Validate velocity solve."""

    def test_crank_angular_velocity(self) -> None:
        """Crank θ̇ should be 1.0 (identity driver)."""
        mech = build_benchmark_fourbar()
        q, q_dot, _ = solve_at_angle(mech, np.pi / 4)
        crank_theta_dot = q_dot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_dot - 1.0) < 1e-8

    def test_crank_origin_velocity_zero(self) -> None:
        """Crank origin (pivot A at ground) should have zero translational velocity."""
        mech = build_benchmark_fourbar()
        q, q_dot, _ = solve_at_angle(mech, np.pi / 4)
        idx = mech.state.get_index("crank")
        assert abs(q_dot[idx.x_idx]) < 1e-8
        assert abs(q_dot[idx.y_idx]) < 1e-8

    def test_rocker_origin_velocity_zero(self) -> None:
        """Rocker origin (pivot D at ground) should have zero translational velocity."""
        mech = build_benchmark_fourbar()
        q, q_dot, _ = solve_at_angle(mech, np.pi / 4)
        idx = mech.state.get_index("rocker")
        assert abs(q_dot[idx.x_idx]) < 1e-8
        assert abs(q_dot[idx.y_idx]) < 1e-8

    def test_velocity_fd_consistency(self) -> None:
        """Velocity should match finite-difference of position."""
        mech = build_benchmark_fourbar()
        dt = 1e-7
        angle = np.pi / 3

        q, q_dot, _ = solve_at_angle(mech, angle)
        q_plus, _, _ = solve_at_angle(mech, angle + dt)

        q_dot_fd = (q_plus - q) / dt
        np.testing.assert_allclose(q_dot, q_dot_fd, atol=1e-4)


class TestFourbarAcceleration:
    """Validate acceleration solve."""

    def test_crank_angular_acceleration_zero(self) -> None:
        """Constant speed driver: crank θ̈ = 0."""
        mech = build_benchmark_fourbar()
        q, q_dot, q_ddot = solve_at_angle(mech, np.pi / 4)
        crank_theta_ddot = q_ddot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_ddot) < 1e-8

    def test_acceleration_fd_consistency(self) -> None:
        """Acceleration should match FD of velocity."""
        mech = build_benchmark_fourbar()
        dt = 1e-5
        angle = np.pi / 4

        q, q_dot, q_ddot = solve_at_angle(mech, angle)
        _, q_dot_plus, _ = solve_at_angle(mech, angle + dt)

        q_ddot_fd = (q_dot_plus - q_dot) / dt
        np.testing.assert_allclose(q_ddot, q_ddot_fd, atol=1e-3)


class TestFourbarCouplerPoint:
    """Validate coupler point evaluation."""

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_coupler_point_on_body(self, angle_deg: int) -> None:
        """Coupler point P should be at the correct position on the coupler body."""
        mech = build_benchmark_fourbar()
        angle = np.radians(angle_deg)
        q, q_dot, q_ddot = solve_at_angle(mech, angle)

        pt_local = mech.bodies["coupler"].coupler_points["P"]
        pos, vel, accel = eval_coupler_point(
            mech.state, "coupler", pt_local, q, q_dot, q_ddot
        )

        # Verify P is at the right distance from B (coupler origin)
        b_global = mech.state.body_point_global("coupler", np.array([0.0, 0.0]), q)
        dist_bp = np.linalg.norm(pos - b_global)
        expected_dist = np.linalg.norm(pt_local)  # sqrt(1.5^2 + 0.5^2)
        assert abs(dist_bp - expected_dist) < 1e-8


class TestFourbarSweep:
    """Validate full position sweep."""

    def test_full_rotation_convergence(self) -> None:
        """Sweep through 360° with small steps should converge everywhere."""
        mech = build_benchmark_fourbar()
        angles = np.linspace(0.01, 2 * np.pi - 0.01, 72)

        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=angles[0])
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        assert sweep.n_converged == 72
        assert sweep.n_failed == 0

    def test_sweep_link_lengths_throughout(self) -> None:
        """Link lengths should be preserved at every sweep step."""
        mech = build_benchmark_fourbar()
        angles = np.linspace(0.1, np.pi, 20)

        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=angles[0])

        sweep = position_sweep(mech, result.q, angles)

        for i, q in enumerate(sweep.solutions):
            assert q is not None
            b = mech.state.body_point_global("crank", np.array([1.0, 0.0]), q)
            a = mech.state.body_point_global("crank", np.array([0.0, 0.0]), q)
            assert abs(np.linalg.norm(b - a) - 1.0) < 1e-6, f"Step {i}"

    def test_coupler_curve_continuity(self) -> None:
        """Coupler trace should be continuous (no jumps between steps)."""
        mech = build_benchmark_fourbar()
        angles = np.linspace(0.1, np.pi, 50)

        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=angles[0])

        sweep = position_sweep(mech, result.q, angles)

        pt_local = mech.bodies["coupler"].coupler_points["P"]
        prev_pos = None
        for q in sweep.solutions:
            if q is None:
                continue
            pos = mech.state.body_point_global("coupler", pt_local, q)
            if prev_pos is not None:
                jump = np.linalg.norm(pos - prev_pos)
                assert jump < 0.5, f"Coupler trace jump of {jump}"
            prev_pos = pos
