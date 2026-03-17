"""6-bar ternary link benchmark test suite.

Validates that multi-attachment-point (ternary) bodies work correctly
within the full kinematic pipeline. Uses a Watt I 6-bar mechanism
where the coupler is a ternary link with 3 revolute connections.

Mechanism (Watt I topology):
    Ground: O2=(0,0), O4=(3,1), O6=(6,0)
    Crank: length=1.5, pivot at O2
    Ternary coupler: 3 attachment points P1, P2, P3
    Rocker4: length=2.5, pivot at O6
    Link5: length=2.0, connects ternary P3 to output6
    Output6: length=2.0, pivot at O4

DOF: 3*5 - 7*2 = 15 - 14 = 1  (with driver: 0)
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.validation import (
    check_connectivity,
    grubler_dof,
    jacobian_rank_analysis,
)
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep


def make_ternary_link(
    body_id: str,
    p1_name: str,
    p2_name: str,
    p3_name: str,
    p2_local: tuple[float, float],
    p3_local: tuple[float, float],
) -> Body:
    """Create a ternary body with 3 attachment points.

    P1 is at the local origin (0,0). P2 and P3 are specified in local coords.
    """
    body = Body(id=body_id)
    body.add_attachment_point(p1_name, 0.0, 0.0)
    body.add_attachment_point(p2_name, p2_local[0], p2_local[1])
    body.add_attachment_point(p3_name, p3_local[0], p3_local[1])
    cg_x = (0.0 + p2_local[0] + p3_local[0]) / 3.0
    cg_y = (0.0 + p2_local[1] + p3_local[1]) / 3.0
    body.cg_local = np.array([cg_x, cg_y])
    return body


def build_watt_sixbar() -> Mechanism:
    """Build a Watt I 6-bar with ternary coupler and identity driver.

    Topology:
        Loop 1: Ground(O2)—Crank—Ternary(P1,P2)—Rocker4—Ground(O6)
        Loop 2: Ternary(P3)—Link5—Output6—Ground(O4)

    Joints:
        J1: Ground O2 → Crank A (revolute)
        J2: Crank B → Ternary P1 (revolute)
        J3: Ternary P2 → Rocker4 R4B (revolute)
        J4: Ground O6 → Rocker4 R4A (revolute)
        J5: Ternary P3 → Link5 L5A (revolute)
        J6: Link5 L5B → Output6 R6B (revolute)
        J7: Ground O4 → Output6 R6A (revolute)
        D1: Revolute driver, Ground → Crank
    """
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 1.0), O6=(6.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5)
    ternary = make_ternary_link(
        "ternary", "P1", "P2", "P3",
        p2_local=(3.0, 0.0),
        p3_local=(1.5, 1.0),
    )
    ternary.add_coupler_point("CP", 1.5, 0.5)
    rocker4 = make_bar("rocker4", "R4A", "R4B", length=2.5)
    link5 = make_bar("link5", "L5A", "L5B", length=2.0)
    output6 = make_bar("output6", "R6A", "R6B", length=2.0)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(ternary)
    mech.add_body(rocker4)
    mech.add_body(link5)
    mech.add_body(output6)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "ternary", "P1")
    mech.add_revolute_joint("J3", "ternary", "P2", "rocker4", "R4B")
    mech.add_revolute_joint("J4", "ground", "O6", "rocker4", "R4A")
    mech.add_revolute_joint("J5", "ternary", "P3", "link5", "L5A")
    mech.add_revolute_joint("J6", "link5", "L5B", "output6", "R6B")
    mech.add_revolute_joint("J7", "ground", "O4", "output6", "R6A")

    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )

    mech.build()
    return mech


def find_initial_config(mech: Mechanism, crank_angle: float) -> np.ndarray:  # type: ignore[type-arg]
    """Find an initial configuration by placing bodies approximately, then solving.

    Uses geometry to set reasonable starting poses, then refines with Newton-Raphson.
    """
    q = mech.state.make_q()

    # Crank
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)

    # Crank tip B
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)

    # Ternary: P1 at B, initial θ ≈ 0
    theta_tern = 0.15  # small angle
    mech.state.set_pose("ternary", q, bx, by, theta_tern)

    # Ternary P2 global (approximate)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st

    # Rocker4: R4A at O6=(6,0), R4B should be near P2
    dx = p2_gx - 6.0
    dy = p2_gy
    theta_r4 = np.arctan2(dy, dx)
    mech.state.set_pose("rocker4", q, 6.0, 0.0, theta_r4)

    # Ternary P3 global (approximate)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct

    # Link5: L5A at P3, pointing toward O4 region
    dx5 = 3.0 - p3_gx
    dy5 = 1.0 - p3_gy
    theta_l5 = np.arctan2(dy5, dx5)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)

    # Output6: R6A at O4=(3,1), R6B should be near link5 L5B
    l5b_gx = p3_gx + 2.0 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.0 * np.sin(theta_l5)
    dx6 = l5b_gx - 3.0
    dy6 = l5b_gy - 1.0
    theta_o6 = np.arctan2(dy6, dx6)
    mech.state.set_pose("output6", q, 3.0, 1.0, theta_o6)

    return q


def solve_at_angle(
    mech: Mechanism, angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Solve position, velocity, acceleration at a given crank angle."""
    q0 = find_initial_config(mech, angle)
    pos_result = solve_position(mech, q0, t=angle)
    assert pos_result.converged, f"Position solve failed at angle={np.degrees(angle):.1f}°"

    q_dot = solve_velocity(mech, pos_result.q, t=angle)
    q_ddot = solve_acceleration(mech, pos_result.q, q_dot, t=angle)

    return pos_result.q, q_dot, q_ddot


# --- Benchmark Tests ---

class TestSixbarTopology:
    """Validate topology: DOF, connectivity, body/constraint count."""

    def test_body_count(self) -> None:
        mech = build_watt_sixbar()
        assert mech.state.n_moving_bodies == 5
        assert mech.state.n_coords == 15

    def test_constraint_count(self) -> None:
        mech = build_watt_sixbar()
        # 7 revolute * 2 + 1 driver * 1 = 15
        assert mech.n_constraints == 15

    def test_grubler_dof(self) -> None:
        """Grübler: M = 3*5 - (7*2 + 1*1) = 15 - 15 = 0."""
        mech = build_watt_sixbar()
        result = grubler_dof(mech, expected_dof=0)
        assert result.dof == 0
        assert not result.is_warning

    def test_connectivity(self) -> None:
        """All bodies should be reachable from ground."""
        mech = build_watt_sixbar()
        result = check_connectivity(mech)
        assert result.is_connected
        assert result.n_components == 1

    def test_ternary_has_three_attachment_points(self) -> None:
        mech = build_watt_sixbar()
        ternary = mech.bodies["ternary"]
        assert len(ternary.attachment_points) == 3
        assert "P1" in ternary.attachment_points
        assert "P2" in ternary.attachment_points
        assert "P3" in ternary.attachment_points

    def test_ternary_has_coupler_point(self) -> None:
        mech = build_watt_sixbar()
        ternary = mech.bodies["ternary"]
        assert "CP" in ternary.coupler_points


class TestSixbarPosition:
    """Validate position solve for the 6-bar mechanism."""

    def test_converges_at_initial_angle(self) -> None:
        mech = build_watt_sixbar()
        q, _, _ = solve_at_angle(mech, 0.3)
        phi = assemble_constraints(mech, q, 0.3)
        assert float(np.linalg.norm(phi)) < 1e-10

    @pytest.mark.parametrize("angle_deg", [15, 30, 45, 60])
    def test_constraints_zero_at_angle(self, angle_deg: int) -> None:
        """All constraint residuals should be near-zero after solve."""
        mech = build_watt_sixbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)
        phi = assemble_constraints(mech, q, angle)
        assert float(np.linalg.norm(phi)) < 1e-10

    @pytest.mark.parametrize("angle_deg", [15, 30, 60])
    def test_link_lengths_preserved(self, angle_deg: int) -> None:
        """All binary link lengths should be preserved."""
        mech = build_watt_sixbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        # Crank: |B - A| = 1.5
        a = mech.state.body_point_global("crank", np.array([0.0, 0.0]), q)
        b = mech.state.body_point_global("crank", np.array([1.5, 0.0]), q)
        np.testing.assert_allclose(float(np.linalg.norm(b - a)), 1.5, atol=1e-8)

        # Rocker4: |R4B - R4A| = 2.5
        r4a = mech.state.body_point_global("rocker4", np.array([0.0, 0.0]), q)
        r4b = mech.state.body_point_global("rocker4", np.array([2.5, 0.0]), q)
        np.testing.assert_allclose(float(np.linalg.norm(r4b - r4a)), 2.5, atol=1e-8)

        # Link5: |L5B - L5A| = 2.0
        l5a = mech.state.body_point_global("link5", np.array([0.0, 0.0]), q)
        l5b = mech.state.body_point_global("link5", np.array([2.0, 0.0]), q)
        np.testing.assert_allclose(float(np.linalg.norm(l5b - l5a)), 2.0, atol=1e-8)

        # Output6: |R6B - R6A| = 2.0
        r6a = mech.state.body_point_global("output6", np.array([0.0, 0.0]), q)
        r6b = mech.state.body_point_global("output6", np.array([2.0, 0.0]), q)
        np.testing.assert_allclose(float(np.linalg.norm(r6b - r6a)), 2.0, atol=1e-8)

    @pytest.mark.parametrize("angle_deg", [15, 30, 60])
    def test_ternary_internal_distances_preserved(self, angle_deg: int) -> None:
        """Ternary body internal distances should be preserved (rigid body)."""
        mech = build_watt_sixbar()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        p1 = mech.state.body_point_global("ternary", np.array([0.0, 0.0]), q)
        p2 = mech.state.body_point_global("ternary", np.array([3.0, 0.0]), q)
        p3 = mech.state.body_point_global("ternary", np.array([1.5, 1.0]), q)

        # |P2 - P1| = 3.0
        np.testing.assert_allclose(float(np.linalg.norm(p2 - p1)), 3.0, atol=1e-8)
        # |P3 - P1| = sqrt(1.5^2 + 1.0^2) = sqrt(3.25)
        np.testing.assert_allclose(
            float(np.linalg.norm(p3 - p1)), np.sqrt(3.25), atol=1e-8
        )
        # |P3 - P2| = sqrt(1.5^2 + 1.0^2) = sqrt(3.25)
        np.testing.assert_allclose(
            float(np.linalg.norm(p3 - p2)), np.sqrt(3.25), atol=1e-8
        )

    def test_crank_tip_follows_circle(self) -> None:
        """Crank tip B should trace a circle of radius 1.5 centered at O2."""
        mech = build_watt_sixbar()
        for angle_deg in [15, 30, 45, 60]:
            angle = np.radians(angle_deg)
            q, _, _ = solve_at_angle(mech, angle)
            b = mech.state.body_point_global("crank", np.array([1.5, 0.0]), q)
            np.testing.assert_allclose(
                b, [1.5 * np.cos(angle), 1.5 * np.sin(angle)], atol=1e-8
            )

    def test_ground_pivots_fixed(self) -> None:
        """Ground-connected joints should stay at their ground positions."""
        mech = build_watt_sixbar()
        q, _, _ = solve_at_angle(mech, np.radians(30))

        # Rocker4 pivot at O6
        r4a = mech.state.body_point_global("rocker4", np.array([0.0, 0.0]), q)
        np.testing.assert_allclose(r4a, [6.0, 0.0], atol=1e-8)

        # Output6 pivot at O4
        r6a = mech.state.body_point_global("output6", np.array([0.0, 0.0]), q)
        np.testing.assert_allclose(r6a, [3.0, 1.0], atol=1e-8)

    def test_jacobian_rank_full(self) -> None:
        """Jacobian should be full rank at a generic configuration."""
        mech = build_watt_sixbar()
        q, _, _ = solve_at_angle(mech, np.radians(30))
        result = jacobian_rank_analysis(mech, q, t=np.radians(30))
        assert result.constraint_rank == 15
        assert result.instantaneous_mobility == 0
        assert not result.has_redundant_constraints


class TestSixbarVelocity:
    """Validate velocity solve."""

    def test_crank_angular_velocity(self) -> None:
        """Crank θ̇ should be 1.0 (identity driver)."""
        mech = build_watt_sixbar()
        q, q_dot, _ = solve_at_angle(mech, np.radians(30))
        crank_theta_dot = q_dot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_dot - 1.0) < 1e-8

    def test_velocity_fd_consistency(self) -> None:
        """Velocity should match finite-difference of position."""
        mech = build_watt_sixbar()
        dt = 1e-7
        angle = np.radians(30)

        q, q_dot, _ = solve_at_angle(mech, angle)
        q_plus, _, _ = solve_at_angle(mech, angle + dt)

        q_dot_fd = (q_plus - q) / dt
        np.testing.assert_allclose(q_dot, q_dot_fd, atol=1e-4)


class TestSixbarAcceleration:
    """Validate acceleration solve."""

    def test_crank_angular_acceleration_zero(self) -> None:
        """Constant speed driver: crank θ̈ = 0."""
        mech = build_watt_sixbar()
        q, q_dot, q_ddot = solve_at_angle(mech, np.radians(30))
        crank_theta_ddot = q_ddot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_ddot) < 1e-8

    def test_acceleration_fd_consistency(self) -> None:
        """Acceleration should match FD of velocity."""
        mech = build_watt_sixbar()
        dt = 1e-5
        angle = np.radians(30)

        q, q_dot, q_ddot = solve_at_angle(mech, angle)
        _, q_dot_plus, _ = solve_at_angle(mech, angle + dt)

        q_ddot_fd = (q_dot_plus - q_dot) / dt
        np.testing.assert_allclose(q_ddot, q_ddot_fd, atol=1e-3)


class TestSixbarSweep:
    """Validate position sweep."""

    def test_sweep_convergence(self) -> None:
        """Sweep through a range of angles should converge."""
        mech = build_watt_sixbar()
        angles = np.linspace(0.1, 1.0, 20)

        q0 = find_initial_config(mech, angles[0])
        result = solve_position(mech, q0, t=angles[0])
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        assert sweep.n_converged == 20
        assert sweep.n_failed == 0

    def test_sweep_constraints_satisfied(self) -> None:
        """All constraint residuals should be near-zero at every step."""
        mech = build_watt_sixbar()
        angles = np.linspace(0.1, 0.8, 10)

        q0 = find_initial_config(mech, angles[0])
        result = solve_position(mech, q0, t=angles[0])
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        for i, q in enumerate(sweep.solutions):
            assert q is not None, f"Step {i} failed"
            phi = assemble_constraints(mech, q, angles[i])
            assert float(np.linalg.norm(phi)) < 1e-10, f"Residual at step {i}"

    def test_sweep_coupler_trace_continuity(self) -> None:
        """Coupler trace should be continuous (no jumps)."""
        mech = build_watt_sixbar()
        angles = np.linspace(0.1, 0.8, 15)

        q0 = find_initial_config(mech, angles[0])
        result = solve_position(mech, q0, t=angles[0])
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)

        pt_local = mech.bodies["ternary"].coupler_points["CP"]
        prev_pos = None
        for q in sweep.solutions:
            if q is None:
                continue
            pos = mech.state.body_point_global("ternary", pt_local, q)
            if prev_pos is not None:
                jump = float(np.linalg.norm(pos - prev_pos))
                assert jump < 0.5, f"Coupler trace jump of {jump}"
            prev_pos = pos
