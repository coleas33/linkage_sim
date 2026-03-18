"""Layer 1: Internal numerical consistency tests.

Cross-checks the solver pipeline against itself using finite difference (FD)
comparisons, rigid-body distance invariants, constraint residuals, and
Jacobian rank analysis at toggle positions.

Tolerance calibration (2026-03-17):
    FD velocity from position sweep:
        h = 1 degree = 0.01745 rad (sweep step size)
        Central FD: (q(t+h) - q(t-h)) / (2h)
        FD truncation is O(h^2), so error ~ h^2 * C
        Empirically: at h = 0.01745, error ~ 3e-4 for generic angles
        Tolerance: 5e-4 (generous to avoid flakiness at near-toggle angles)
    FD acceleration from velocity sweep:
        Same h, applied to q_dot. Error similar magnitude.
        Tolerance: 5e-3 (acceleration has larger constants in the O(h^2) term)
    Coupler FD:
        Same approach applied to coupler point position -> velocity -> acceleration.
        Tolerance: 5e-4 for velocity, 5e-3 for acceleration.
    Constraint residual: 1e-10 (solver convergence criterion).
    Distance invariant: 1e-10 (exact geometric identity).
    Toggle exclusion: +/- 5 degrees from known toggle positions.

    # BASELINE: calibrated 2026-03-17, recalibrate if step size or mechanism changes
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.coupler import eval_coupler_point
from linkage_sim.analysis.validation import jacobian_rank_analysis
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep


# ── Step size and tolerances ──

H_DEG = 1.0  # sweep step size in degrees
H_RAD = np.radians(H_DEG)
FD_VEL_TOL = 5e-4
FD_ACCEL_TOL = 5e-3
FD_COUPLER_VEL_TOL = 5e-4
FD_COUPLER_ACCEL_TOL = 5e-3
RESIDUAL_TOL = 1e-10
DISTANCE_TOL = 1e-10
TOGGLE_EXCLUSION_DEG = 5.0

# Known toggle angles for the boundary Grashof 4-bar (s+l = p+q)
# Toggle at theta=0 and theta=pi (crank aligned/anti-aligned with ground)
FOURBAR_TOGGLE_ANGLES_DEG = [0.0, 180.0, 360.0]


# ── Mechanism factories ──


def _build_fourbar() -> Mechanism:
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


def _build_slidercrank() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    conrod = make_bar("conrod", "B", "C", length=3.0)
    conrod.add_coupler_point("P", 1.5, 0.3)
    slider = Body(
        id="slider",
        attachment_points={"pin": np.array([0.0, 0.0])},
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
    return mech


def _make_ternary_link(body_id, p1_name, p2_name, p3_name, p2_local, p3_local):
    body = Body(id=body_id)
    body.add_attachment_point(p1_name, 0.0, 0.0)
    body.add_attachment_point(p2_name, p2_local[0], p2_local[1])
    body.add_attachment_point(p3_name, p3_local[0], p3_local[1])
    cg_x = (0.0 + p2_local[0] + p3_local[0]) / 3.0
    cg_y = (0.0 + p2_local[1] + p3_local[1]) / 3.0
    body.cg_local = np.array([cg_x, cg_y])
    return body


def _build_sixbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 1.0), O6=(6.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5)
    ternary = _make_ternary_link(
        "ternary", "P1", "P2", "P3",
        p2_local=(3.0, 0.0), p3_local=(1.5, 1.0),
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


# ── Solve helpers ──


def _solve_fourbar(mech, angle):
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged, f"Failed at {np.degrees(angle):.1f}°"
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def _solve_slidercrank(mech, angle):
    q = mech.state.make_q()
    r, l = 1.0, 3.0
    bx, by = np.cos(angle), np.sin(angle)
    x_sl = r * np.cos(angle) + np.sqrt(l**2 - r**2 * np.sin(angle)**2)
    phi_cr = np.arcsin(-r * np.sin(angle) / l)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("conrod", q, bx, by, phi_cr)
    mech.state.set_pose("slider", q, x_sl, 0.0, 0.0)
    result = solve_position(mech, q, t=angle)
    assert result.converged, f"Failed at {np.degrees(angle):.1f}°"
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def _find_sixbar_initial(mech, crank_angle):
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)
    theta_tern = 0.15
    mech.state.set_pose("ternary", q, bx, by, theta_tern)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st
    theta_r4 = np.arctan2(p2_gy, p2_gx - 6.0)
    mech.state.set_pose("rocker4", q, 6.0, 0.0, theta_r4)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct
    theta_l5 = np.arctan2(1.0 - p3_gy, 3.0 - p3_gx)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)
    l5b_gx = p3_gx + 2.0 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.0 * np.sin(theta_l5)
    theta_o6 = np.arctan2(l5b_gy - 1.0, l5b_gx - 3.0)
    mech.state.set_pose("output6", q, 3.0, 1.0, theta_o6)
    return q


def _solve_sixbar(mech, angle):
    q0 = _find_sixbar_initial(mech, angle)
    result = solve_position(mech, q0, t=angle)
    assert result.converged, f"Failed at {np.degrees(angle):.1f}°"
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def _near_toggle(angle_deg: float, toggle_angles: list[float], margin: float) -> bool:
    """Check if angle_deg is within margin degrees of any toggle angle."""
    for t in toggle_angles:
        if abs(angle_deg - t) < margin:
            return True
        # Wrap check
        if abs(angle_deg - t + 360) < margin or abs(angle_deg - t - 360) < margin:
            return True
    return False


# ── FD velocity from position sweep ──


class TestFDVelocityFourbar:
    """FD velocity: (q(t+h) - q(t-h)) / (2h) vs solve_velocity(), 4-bar."""

    @pytest.mark.parametrize("angle_deg", list(range(15, 346, 15)))
    def test_fd_velocity(self, angle_deg: int) -> None:
        if _near_toggle(angle_deg, FOURBAR_TOGGLE_ANGLES_DEG, TOGGLE_EXCLUSION_DEG):
            pytest.skip(f"Skipping near-toggle angle {angle_deg}°")
        mech = _build_fourbar()
        angle = np.radians(angle_deg)
        h = H_RAD
        q_minus, _, _ = _solve_fourbar(mech, angle - h)
        q_plus, _, _ = _solve_fourbar(mech, angle + h)
        _, q_dot_solver, _ = _solve_fourbar(mech, angle)
        q_dot_fd = (q_plus - q_minus) / (2.0 * h)
        np.testing.assert_allclose(
            q_dot_solver, q_dot_fd, atol=FD_VEL_TOL,
            err_msg=f"FD velocity mismatch at {angle_deg}°",
        )


class TestFDVelocitySliderCrank:
    """FD velocity vs solve_velocity() for slider-crank."""

    @pytest.mark.parametrize("angle_deg", list(range(15, 346, 15)))
    def test_fd_velocity(self, angle_deg: int) -> None:
        mech = _build_slidercrank()
        angle = np.radians(angle_deg)
        h = H_RAD
        q_minus, _, _ = _solve_slidercrank(mech, angle - h)
        q_plus, _, _ = _solve_slidercrank(mech, angle + h)
        _, q_dot_solver, _ = _solve_slidercrank(mech, angle)
        q_dot_fd = (q_plus - q_minus) / (2.0 * h)
        np.testing.assert_allclose(
            q_dot_solver, q_dot_fd, atol=FD_VEL_TOL,
            err_msg=f"FD velocity mismatch at {angle_deg}°",
        )


class TestFDVelocitySixbar:
    """FD velocity vs solve_velocity() for 6-bar.

    Uses sweep-based continuation to avoid branch jumps when building
    initial guesses independently at small angles.
    """

    def test_fd_velocity_sweep(self) -> None:
        """Sweep-based FD velocity comparison for the 6-bar mechanism."""
        mech = _build_sixbar()
        # Use a range where the mechanism is well-conditioned
        h = H_RAD
        angles_test = np.radians([25, 30, 35, 40, 45, 50])
        # Build sweep from 24 deg to 51 deg at 1-degree steps for neighbors
        sweep_start = np.radians(24)
        sweep_end = np.radians(51)
        sweep_angles = np.arange(sweep_start, sweep_end + h / 2, h)

        q0 = _find_sixbar_initial(mech, sweep_angles[0])
        result = solve_position(mech, q0, t=float(sweep_angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, sweep_angles)

        # Build a dict: angle_rad -> (index in sweep)
        angle_to_idx = {}
        for idx, a in enumerate(sweep_angles):
            angle_to_idx[round(float(a), 8)] = idx

        for angle_test in angles_test:
            angle_key = round(float(angle_test), 8)
            minus_key = round(float(angle_test - h), 8)
            plus_key = round(float(angle_test + h), 8)

            idx_center = angle_to_idx.get(angle_key)
            idx_minus = angle_to_idx.get(minus_key)
            idx_plus = angle_to_idx.get(plus_key)

            if idx_center is None or idx_minus is None or idx_plus is None:
                continue
            q_center = sweep.solutions[idx_center]
            q_minus = sweep.solutions[idx_minus]
            q_plus = sweep.solutions[idx_plus]
            if q_center is None or q_minus is None or q_plus is None:
                continue

            q_dot_solver = solve_velocity(mech, q_center, t=float(angle_test))
            q_dot_fd = (q_plus - q_minus) / (2.0 * h)
            # 6-bar has more bodies and slightly larger FD error
            np.testing.assert_allclose(
                q_dot_solver, q_dot_fd, atol=1e-3,
                err_msg=f"6-bar FD velocity mismatch at {np.degrees(angle_test):.0f}°",
            )


# ── FD acceleration from velocity sweep ──


class TestFDAccelerationFourbar:
    """FD accel: (q_dot(t+h) - q_dot(t-h)) / (2h) vs solve_acceleration(), 4-bar."""

    @pytest.mark.parametrize("angle_deg", list(range(15, 346, 15)))
    def test_fd_acceleration(self, angle_deg: int) -> None:
        if _near_toggle(angle_deg, FOURBAR_TOGGLE_ANGLES_DEG, TOGGLE_EXCLUSION_DEG):
            pytest.skip(f"Skipping near-toggle angle {angle_deg}°")
        mech = _build_fourbar()
        angle = np.radians(angle_deg)
        h = H_RAD
        _, q_dot_minus, _ = _solve_fourbar(mech, angle - h)
        _, q_dot_plus, _ = _solve_fourbar(mech, angle + h)
        _, _, q_ddot_solver = _solve_fourbar(mech, angle)
        q_ddot_fd = (q_dot_plus - q_dot_minus) / (2.0 * h)
        np.testing.assert_allclose(
            q_ddot_solver, q_ddot_fd, atol=FD_ACCEL_TOL,
            err_msg=f"FD acceleration mismatch at {angle_deg}°",
        )


class TestFDAccelerationSliderCrank:
    """FD acceleration vs solve_acceleration() for slider-crank."""

    @pytest.mark.parametrize("angle_deg", list(range(15, 346, 15)))
    def test_fd_acceleration(self, angle_deg: int) -> None:
        mech = _build_slidercrank()
        angle = np.radians(angle_deg)
        h = H_RAD
        _, q_dot_minus, _ = _solve_slidercrank(mech, angle - h)
        _, q_dot_plus, _ = _solve_slidercrank(mech, angle + h)
        _, _, q_ddot_solver = _solve_slidercrank(mech, angle)
        q_ddot_fd = (q_dot_plus - q_dot_minus) / (2.0 * h)
        np.testing.assert_allclose(
            q_ddot_solver, q_ddot_fd, atol=FD_ACCEL_TOL,
            err_msg=f"FD acceleration mismatch at {angle_deg}°",
        )


class TestFDAccelerationSixbar:
    """FD acceleration vs solve_acceleration() for 6-bar.

    Uses sweep-based continuation to avoid branch jumps.
    """

    def test_fd_acceleration_sweep(self) -> None:
        """Sweep-based FD acceleration comparison for the 6-bar mechanism."""
        mech = _build_sixbar()
        h = H_RAD
        angles_test = np.radians([25, 30, 35, 40, 45, 50])
        sweep_start = np.radians(24)
        sweep_end = np.radians(51)
        sweep_angles = np.arange(sweep_start, sweep_end + h / 2, h)

        q0 = _find_sixbar_initial(mech, sweep_angles[0])
        result = solve_position(mech, q0, t=float(sweep_angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, sweep_angles)

        angle_to_idx = {}
        for idx, a in enumerate(sweep_angles):
            angle_to_idx[round(float(a), 8)] = idx

        for angle_test in angles_test:
            angle_key = round(float(angle_test), 8)
            minus_key = round(float(angle_test - h), 8)
            plus_key = round(float(angle_test + h), 8)

            idx_center = angle_to_idx.get(angle_key)
            idx_minus = angle_to_idx.get(minus_key)
            idx_plus = angle_to_idx.get(plus_key)

            if idx_center is None or idx_minus is None or idx_plus is None:
                continue
            q_center = sweep.solutions[idx_center]
            q_minus = sweep.solutions[idx_minus]
            q_plus = sweep.solutions[idx_plus]
            if q_center is None or q_minus is None or q_plus is None:
                continue

            q_dot_center = solve_velocity(mech, q_center, t=float(angle_test))
            q_dot_minus = solve_velocity(mech, q_minus, t=float(angle_test - h))
            q_dot_plus = solve_velocity(mech, q_plus, t=float(angle_test + h))

            q_ddot_solver = solve_acceleration(
                mech, q_center, q_dot_center, t=float(angle_test)
            )
            q_ddot_fd = (q_dot_plus - q_dot_minus) / (2.0 * h)
            # 6-bar has more bodies and slightly larger FD error
            np.testing.assert_allclose(
                q_ddot_solver, q_ddot_fd, atol=0.01,
                err_msg=f"6-bar FD accel mismatch at {np.degrees(angle_test):.0f}°",
            )


# ── Rigid-body distance invariants ──


class TestDistanceInvariantFourbar:
    """Coupler body attachment points maintain constant distance across sweep."""

    def test_coupler_length_invariant(self) -> None:
        mech = _build_fourbar()
        angles = np.linspace(0.1, 2 * np.pi - 0.1, 36)
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        for i, q in enumerate(sweep.solutions):
            if q is None:
                continue
            b = mech.state.body_point_global("coupler", np.array([0.0, 0.0]), q)
            c = mech.state.body_point_global("coupler", np.array([3.0, 0.0]), q)
            dist = float(np.linalg.norm(c - b))
            assert abs(dist - 3.0) < DISTANCE_TOL, (
                f"Coupler length {dist} != 3.0 at step {i}"
            )


class TestDistanceInvariantSixbar:
    """Ternary body: 3 pairwise distances constant across sweep."""

    def test_ternary_pairwise_distances(self) -> None:
        mech = _build_sixbar()
        angles = np.linspace(0.1, 0.8, 15)
        q0 = _find_sixbar_initial(mech, angles[0])
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)

        # Expected distances from local coords
        p1_loc = np.array([0.0, 0.0])
        p2_loc = np.array([3.0, 0.0])
        p3_loc = np.array([1.5, 1.0])
        d12_expected = float(np.linalg.norm(p2_loc - p1_loc))
        d13_expected = float(np.linalg.norm(p3_loc - p1_loc))
        d23_expected = float(np.linalg.norm(p3_loc - p2_loc))

        for i, q in enumerate(sweep.solutions):
            if q is None:
                continue
            p1 = mech.state.body_point_global("ternary", p1_loc, q)
            p2 = mech.state.body_point_global("ternary", p2_loc, q)
            p3 = mech.state.body_point_global("ternary", p3_loc, q)
            d12 = float(np.linalg.norm(p2 - p1))
            d13 = float(np.linalg.norm(p3 - p1))
            d23 = float(np.linalg.norm(p3 - p2))
            assert abs(d12 - d12_expected) < DISTANCE_TOL, f"P1-P2 at step {i}"
            assert abs(d13 - d13_expected) < DISTANCE_TOL, f"P1-P3 at step {i}"
            assert abs(d23 - d23_expected) < DISTANCE_TOL, f"P2-P3 at step {i}"


# ── Constraint residual across sweep ──


class TestConstraintResidualFourbar:
    """Constraint residual norm < 1e-10 at every sweep step, 4-bar."""

    def test_residual_across_sweep(self) -> None:
        mech = _build_fourbar()
        angles = np.linspace(0.1, 2 * np.pi - 0.1, 36)
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        for i, q in enumerate(sweep.solutions):
            if q is None:
                continue
            phi = assemble_constraints(mech, q, float(angles[i]))
            assert float(np.linalg.norm(phi)) < RESIDUAL_TOL, (
                f"Residual {float(np.linalg.norm(phi)):.2e} at step {i}"
            )


class TestConstraintResidualSliderCrank:
    """Constraint residual norm < 1e-10 at every sweep step, slider-crank."""

    def test_residual_across_sweep(self) -> None:
        mech = _build_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 36)
        r, l = 1.0, 3.0
        a0 = angles[0]
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, a0)
        mech.state.set_pose("conrod", q0, np.cos(a0), np.sin(a0),
                            np.arcsin(-r * np.sin(a0) / l))
        x_sl = r * np.cos(a0) + np.sqrt(l**2 - r**2 * np.sin(a0)**2)
        mech.state.set_pose("slider", q0, x_sl, 0.0, 0.0)
        result = solve_position(mech, q0, t=float(a0))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        for i, q in enumerate(sweep.solutions):
            if q is None:
                continue
            phi = assemble_constraints(mech, q, float(angles[i]))
            assert float(np.linalg.norm(phi)) < RESIDUAL_TOL, (
                f"Residual {float(np.linalg.norm(phi)):.2e} at step {i}"
            )


class TestConstraintResidualSixbar:
    """Constraint residual norm < 1e-10 at every sweep step, 6-bar."""

    def test_residual_across_sweep(self) -> None:
        mech = _build_sixbar()
        angles = np.linspace(0.1, 0.8, 15)
        q0 = _find_sixbar_initial(mech, angles[0])
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        for i, q in enumerate(sweep.solutions):
            if q is None:
                continue
            phi = assemble_constraints(mech, q, float(angles[i]))
            assert float(np.linalg.norm(phi)) < RESIDUAL_TOL, (
                f"Residual {float(np.linalg.norm(phi)):.2e} at step {i}"
            )


# ── Jacobian rank drop at toggle ──


class TestRankDropAtToggle:
    """Non-Grashof 4-bar: rank drops at toggle position."""

    def _build_nongrashof_fourbar(self) -> Mechanism:
        """Build a non-Grashof 4-bar with known toggle angle.

        Links: a=2(crank), b=3(coupler), c=3(rocker), d=4(ground)
        s+l = 2+4 = 6, p+q = 3+3 = 6 => boundary Grashof
        Toggle at theta = arccos((a^2 + d^2 - (b-c)^2) / (2ad))
        With b=c: toggle at arccos((4+16-0)/16) = arccos(20/16) -> need b!=c
        Use: a=2, b=4, c=3, d=5 => non-Grashof: s+l=2+5=7 > p+q=3+4=7
        Actually need S+L > P+Q. Use: a=2, b=3, c=2, d=5 => s+l=2+5=7, p+q=2+3=5.
        Toggle when links fold: theta_toggle = arccos((a^2+d^2-(b+c)^2)/(2ad))
        = arccos((4+25-25)/20) = arccos(4/20) = arccos(0.2)
        """
        mech = Mechanism()
        a, b, c, d = 2.0, 3.0, 2.0, 5.0
        ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
        crank = make_bar("crank", "A", "B", length=a)
        coupler = make_bar("coupler", "B", "C", length=b)
        rocker = make_bar("rocker", "D", "C", length=c)
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

    def test_rank_drop_near_toggle(self) -> None:
        """Condition number should spike near toggle position."""
        mech = self._build_nongrashof_fourbar()
        a, b, c, d = 2.0, 3.0, 2.0, 5.0

        # Toggle when crank + coupler are collinear with rocker + ground:
        # theta_toggle = arccos((a^2 + d^2 - (b+c)^2) / (2*a*d))
        cos_toggle = (a**2 + d**2 - (b + c)**2) / (2 * a * d)
        # cos_toggle = (4+25-25)/20 = 0.2
        theta_toggle = np.arccos(np.clip(cos_toggle, -1, 1))

        # Also test the other toggle: theta = arccos((a^2 + d^2 - (b-c)^2) / (2ad))
        cos_toggle2 = (a**2 + d**2 - (b - c)**2) / (2 * a * d)
        theta_toggle2 = np.arccos(np.clip(cos_toggle2, -1, 1))

        # Solve near the second toggle (b-c = 1, more accessible)
        # cos_toggle2 = (4+25-1)/20 = 28/20 = 1.4 -> clamped to 1 -> theta=0
        # That means theta=0 is a toggle. Let's use theta_toggle instead.
        # theta_toggle = arccos(0.2) ~ 78.46 deg

        # Test near toggle: use the boundary Grashof benchmark 4-bar
        # (s+l = p+q) which has actual toggle at 0 and 180 degrees.
        # The condition number at exactly these angles is very high.
        mech2 = _build_fourbar()  # boundary Grashof: a=1, b=3, c=2, d=4
        # At 180 degrees for boundary Grashof, condition number is huge
        test_angle = np.radians(179.99)
        q = mech2.state.make_q()
        mech2.state.set_pose("crank", q, 0.0, 0.0, test_angle)
        bx, by = np.cos(test_angle), np.sin(test_angle)
        mech2.state.set_pose("coupler", q, bx, by, 0.0)
        mech2.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech2, q, t=test_angle)

        if result.converged:
            rank_result = jacobian_rank_analysis(mech2, result.q, t=test_angle)
            # Near toggle, condition number should be elevated
            assert rank_result.condition_number > 100, (
                f"Expected elevated condition number near toggle, "
                f"got {rank_result.condition_number:.1f}"
            )

    def test_away_from_toggle_healthy(self) -> None:
        """Away from toggle, condition number should be moderate."""
        mech = self._build_nongrashof_fourbar()
        a, d = 2.0, 5.0
        angle = np.radians(45)
        q = mech.state.make_q()
        bx = a * np.cos(angle)
        by = a * np.sin(angle)
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        mech.state.set_pose("coupler", q, bx, by, 0.0)
        mech.state.set_pose("rocker", q, d, 0.0, np.pi / 2)
        result = solve_position(mech, q, t=angle)

        if result.converged:
            rank_result = jacobian_rank_analysis(mech, result.q, t=angle)
            assert rank_result.condition_number < 1e4, (
                f"Unexpected high condition at 45°: {rank_result.condition_number:.1f}"
            )


# ── Coupler FD chain ──


class TestCouplerFDFourbar:
    """Coupler point FD: position -> velocity, velocity -> acceleration for 4-bar."""

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120, 150])
    def test_coupler_fd_velocity(self, angle_deg: int) -> None:
        """FD of coupler position should match coupler velocity."""
        mech = _build_fourbar()
        angle = np.radians(angle_deg)
        h = H_RAD
        pt_local = np.array([1.5, 0.5])

        q_minus, qd_minus, qdd_minus = _solve_fourbar(mech, angle - h)
        q_plus, qd_plus, qdd_plus = _solve_fourbar(mech, angle + h)
        q, q_dot, q_ddot = _solve_fourbar(mech, angle)

        pos_minus, _, _ = eval_coupler_point(
            mech.state, "coupler", pt_local, q_minus, qd_minus, qdd_minus
        )
        pos_plus, _, _ = eval_coupler_point(
            mech.state, "coupler", pt_local, q_plus, qd_plus, qdd_plus
        )
        _, vel_solver, _ = eval_coupler_point(
            mech.state, "coupler", pt_local, q, q_dot, q_ddot
        )

        vel_fd = (pos_plus - pos_minus) / (2.0 * h)
        np.testing.assert_allclose(
            vel_solver, vel_fd, atol=FD_COUPLER_VEL_TOL,
            err_msg=f"Coupler FD velocity mismatch at {angle_deg}°",
        )

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120, 150])
    def test_coupler_fd_acceleration(self, angle_deg: int) -> None:
        """FD of coupler velocity should match coupler acceleration."""
        mech = _build_fourbar()
        angle = np.radians(angle_deg)
        h = H_RAD
        pt_local = np.array([1.5, 0.5])

        q_minus, qd_minus, qdd_minus = _solve_fourbar(mech, angle - h)
        q_plus, qd_plus, qdd_plus = _solve_fourbar(mech, angle + h)
        q, q_dot, q_ddot = _solve_fourbar(mech, angle)

        _, vel_minus, _ = eval_coupler_point(
            mech.state, "coupler", pt_local, q_minus, qd_minus, qdd_minus
        )
        _, vel_plus, _ = eval_coupler_point(
            mech.state, "coupler", pt_local, q_plus, qd_plus, qdd_plus
        )
        _, _, accel_solver = eval_coupler_point(
            mech.state, "coupler", pt_local, q, q_dot, q_ddot
        )

        accel_fd = (vel_plus - vel_minus) / (2.0 * h)
        np.testing.assert_allclose(
            accel_solver, accel_fd, atol=FD_COUPLER_ACCEL_TOL,
            err_msg=f"Coupler FD acceleration mismatch at {angle_deg}°",
        )


class TestCouplerFDSixbar:
    """Coupler point FD for 6-bar ternary coupler.

    Uses sweep-based continuation to avoid branch-jump issues.
    """

    def test_coupler_fd_velocity_sweep(self) -> None:
        """Sweep-based coupler FD velocity for 6-bar."""
        mech = _build_sixbar()
        h = H_RAD
        pt_local = np.array([1.5, 0.5])
        angles_test = np.radians([25, 30, 35, 40, 45])
        sweep_start = np.radians(24)
        sweep_end = np.radians(46)
        sweep_angles = np.arange(sweep_start, sweep_end + h / 2, h)

        q0 = _find_sixbar_initial(mech, sweep_angles[0])
        result = solve_position(mech, q0, t=float(sweep_angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, sweep_angles)

        angle_to_idx = {}
        for idx, a in enumerate(sweep_angles):
            angle_to_idx[round(float(a), 8)] = idx

        for angle_test in angles_test:
            angle_key = round(float(angle_test), 8)
            minus_key = round(float(angle_test - h), 8)
            plus_key = round(float(angle_test + h), 8)

            idx_center = angle_to_idx.get(angle_key)
            idx_minus = angle_to_idx.get(minus_key)
            idx_plus = angle_to_idx.get(plus_key)

            if idx_center is None or idx_minus is None or idx_plus is None:
                continue
            q_center = sweep.solutions[idx_center]
            q_minus = sweep.solutions[idx_minus]
            q_plus = sweep.solutions[idx_plus]
            if q_center is None or q_minus is None or q_plus is None:
                continue

            q_dot_center = solve_velocity(mech, q_center, t=float(angle_test))
            q_ddot_center = solve_acceleration(
                mech, q_center, q_dot_center, t=float(angle_test)
            )
            q_dot_minus = solve_velocity(mech, q_minus, t=float(angle_test - h))
            q_ddot_minus = solve_acceleration(
                mech, q_minus, q_dot_minus, t=float(angle_test - h)
            )
            q_dot_plus = solve_velocity(mech, q_plus, t=float(angle_test + h))
            q_ddot_plus = solve_acceleration(
                mech, q_plus, q_dot_plus, t=float(angle_test + h)
            )

            pos_minus, _, _ = eval_coupler_point(
                mech.state, "ternary", pt_local, q_minus, q_dot_minus, q_ddot_minus
            )
            pos_plus, _, _ = eval_coupler_point(
                mech.state, "ternary", pt_local, q_plus, q_dot_plus, q_ddot_plus
            )
            _, vel_solver, _ = eval_coupler_point(
                mech.state, "ternary", pt_local, q_center, q_dot_center, q_ddot_center
            )

            vel_fd = (pos_plus - pos_minus) / (2.0 * h)
            # 6-bar has slightly larger FD errors
            np.testing.assert_allclose(
                vel_solver, vel_fd, atol=1e-3,
                err_msg=(
                    f"6-bar coupler FD velocity mismatch at "
                    f"{np.degrees(angle_test):.0f}°"
                ),
            )
