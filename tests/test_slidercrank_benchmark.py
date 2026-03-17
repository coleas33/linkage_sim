"""Slider-crank benchmark test suite.

Validates the prismatic joint integration with the complete kinematic pipeline
(position, velocity, acceleration, DOF, sweep) against analytically derived
reference values for a standard slider-crank mechanism.

Mechanism:
    Ground pivot O2 at (0,0), horizontal rail.
    Crank length r=1, connecting rod length l=3.
    Slider constrained to x-axis via prismatic joint.

Analytical reference:
    x_slider = r·cos θ + √(l² - r²·sin²θ)
    ẋ_slider = -r·sin θ · (1 + r·cos θ / √(l² - r²·sin²θ))  (for ω=1)
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.validation import grubler_dof, jacobian_rank_analysis
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep


# --- Analytical reference ---

def analytical_slider_position(theta: float, r: float = 1.0, l: float = 3.0) -> float:
    """Analytical slider x-position for an inline slider-crank."""
    return r * np.cos(theta) + np.sqrt(l**2 - r**2 * np.sin(theta) ** 2)


def analytical_slider_velocity(theta: float, r: float = 1.0, l: float = 3.0) -> float:
    """Analytical slider ẋ for ω=1 (dθ/dt=1)."""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    denom = np.sqrt(l**2 - r**2 * sin_t**2)
    return -r * sin_t * (1.0 + r * cos_t / denom)


def analytical_conrod_angle(theta: float, r: float = 1.0, l: float = 3.0) -> float:
    """Connecting rod angle φ from positive x-axis."""
    return np.arcsin(-r * np.sin(theta) / l)


# --- Mechanism factory ---

def build_benchmark_slidercrank() -> Mechanism:
    """Build the benchmark slider-crank with identity driver f(t) = t.

    Layout:
        - Ground: O2 at origin, rail reference at origin
        - Crank: length 1.0, A at O2, B at crank tip
        - Connecting rod: length 3.0, B at crank tip, C at wrist pin
        - Slider: block on horizontal rail

    Joints:
        - J1: Revolute, ground O2 → crank A
        - J2: Revolute, crank B → conrod B
        - J3: Revolute, conrod C → slider pin
        - P1: Prismatic, ground rail → slider pin, axis=(1,0)
        - D1: Revolute driver, ground → crank, f(t)=t
    """
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    conrod = make_bar("conrod", "B", "C", length=3.0)
    conrod.add_coupler_point("P", 1.5, 0.3)  # midpoint with offset
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


def solve_at_angle(
    mech: Mechanism, angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Solve position, velocity, acceleration at a given crank angle."""
    q = mech.state.make_q()

    # Initial guess using analytical geometry
    bx = np.cos(angle)
    by = np.sin(angle)
    x_slider = analytical_slider_position(angle)
    phi_conrod = analytical_conrod_angle(angle)

    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("conrod", q, bx, by, phi_conrod)
    mech.state.set_pose("slider", q, x_slider, 0.0, 0.0)

    pos_result = solve_position(mech, q, t=angle)
    assert pos_result.converged, f"Position solve failed at angle={np.degrees(angle):.1f}°"

    q_dot = solve_velocity(mech, pos_result.q, t=angle)
    q_ddot = solve_acceleration(mech, pos_result.q, q_dot, t=angle)

    return pos_result.q, q_dot, q_ddot


# --- Benchmark Tests ---

class TestSliderCrankDOF:
    """Validate topology and constraint analysis."""

    def test_grubler_dof(self) -> None:
        """Grübler: M = 3*3 - (3*2 + 1*2 + 1*1) = 9 - 9 = 0."""
        mech = build_benchmark_slidercrank()
        result = grubler_dof(mech, expected_dof=0)
        assert result.dof == 0
        assert not result.is_warning

    def test_jacobian_rank_full(self) -> None:
        """Jacobian should be full rank at a generic configuration."""
        mech = build_benchmark_slidercrank()
        q, _, _ = solve_at_angle(mech, np.pi / 4)
        result = jacobian_rank_analysis(mech, q, t=np.pi / 4)
        assert result.constraint_rank == 9
        assert result.instantaneous_mobility == 0
        assert not result.has_redundant_constraints

    def test_body_count(self) -> None:
        mech = build_benchmark_slidercrank()
        assert mech.state.n_moving_bodies == 3
        assert mech.state.n_coords == 9

    def test_constraint_count(self) -> None:
        mech = build_benchmark_slidercrank()
        # 3 revolute * 2 + 1 prismatic * 2 + 1 driver * 1 = 9
        assert mech.n_constraints == 9


class TestSliderCrankPosition:
    """Validate position solve against analytical geometry."""

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 60, 90, 120, 150, 180, 270])
    def test_slider_position_analytical(self, angle_deg: int) -> None:
        """Slider x-position should match analytical formula."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        x_slider = q[mech.state.get_index("slider").x_idx]
        x_expected = analytical_slider_position(angle)
        np.testing.assert_allclose(x_slider, x_expected, atol=1e-8)

    @pytest.mark.parametrize("angle_deg", [30, 90, 150, 270])
    def test_slider_on_rail(self, angle_deg: int) -> None:
        """Slider y-position should be zero (on the rail)."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        y_slider = q[mech.state.get_index("slider").y_idx]
        np.testing.assert_allclose(y_slider, 0.0, atol=1e-10)

    @pytest.mark.parametrize("angle_deg", [30, 90, 150, 270])
    def test_slider_angle_zero(self, angle_deg: int) -> None:
        """Slider orientation should be zero (locked by prismatic joint)."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        theta_slider = q[mech.state.get_index("slider").theta_idx]
        np.testing.assert_allclose(theta_slider, 0.0, atol=1e-10)

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_crank_tip_position(self, angle_deg: int) -> None:
        """Crank tip B should be at (cos θ, sin θ)."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        b_global = mech.state.body_point_global("crank", np.array([1.0, 0.0]), q)
        np.testing.assert_allclose(
            b_global, [np.cos(angle), np.sin(angle)], atol=1e-8
        )

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120])
    def test_link_lengths_preserved(self, angle_deg: int) -> None:
        """Crank and connecting rod lengths should be preserved."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, _, _ = solve_at_angle(mech, angle)

        # Crank: |B - A| = 1
        a = mech.state.body_point_global("crank", np.array([0.0, 0.0]), q)
        b = mech.state.body_point_global("crank", np.array([1.0, 0.0]), q)
        assert abs(float(np.linalg.norm(b - a)) - 1.0) < 1e-8

        # Conrod: |C - B| = 3
        b2 = mech.state.body_point_global("conrod", np.array([0.0, 0.0]), q)
        c = mech.state.body_point_global("conrod", np.array([3.0, 0.0]), q)
        assert abs(float(np.linalg.norm(c - b2)) - 3.0) < 1e-8

    def test_constraints_zero_after_solve(self) -> None:
        """All constraint residuals should be near-zero."""
        mech = build_benchmark_slidercrank()
        angle = np.pi / 4
        q, _, _ = solve_at_angle(mech, angle)
        phi = assemble_constraints(mech, q, angle)
        assert float(np.linalg.norm(phi)) < 1e-10

    def test_slider_stroke(self) -> None:
        """Slider stroke = 2r for inline slider-crank."""
        mech = build_benchmark_slidercrank()
        # At θ=0: x = r + l = 4.0 (TDC)
        # At θ=π: x = -r + l = 2.0 (BDC)
        q0, _, _ = solve_at_angle(mech, 0.0)
        q_pi, _, _ = solve_at_angle(mech, np.pi)

        x_tdc = q0[mech.state.get_index("slider").x_idx]
        x_bdc = q_pi[mech.state.get_index("slider").x_idx]

        np.testing.assert_allclose(x_tdc, 4.0, atol=1e-8)  # r + l
        np.testing.assert_allclose(x_bdc, 2.0, atol=1e-8)  # -r + l
        np.testing.assert_allclose(x_tdc - x_bdc, 2.0, atol=1e-8)  # stroke = 2r


class TestSliderCrankVelocity:
    """Validate velocity solve."""

    def test_crank_angular_velocity(self) -> None:
        """Crank θ̇ should be 1.0 (identity driver)."""
        mech = build_benchmark_slidercrank()
        q, q_dot, _ = solve_at_angle(mech, np.pi / 4)
        crank_theta_dot = q_dot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_dot - 1.0) < 1e-8

    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120, 150])
    def test_slider_velocity_analytical(self, angle_deg: int) -> None:
        """Slider ẋ should match analytical formula."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, q_dot, _ = solve_at_angle(mech, angle)

        vx_slider = q_dot[mech.state.get_index("slider").x_idx]
        vx_expected = analytical_slider_velocity(angle)
        np.testing.assert_allclose(vx_slider, vx_expected, atol=1e-6)

    @pytest.mark.parametrize("angle_deg", [30, 90, 150])
    def test_slider_y_velocity_zero(self, angle_deg: int) -> None:
        """Slider ẏ should be zero (constrained to rail)."""
        mech = build_benchmark_slidercrank()
        angle = np.radians(angle_deg)
        q, q_dot, _ = solve_at_angle(mech, angle)

        vy_slider = q_dot[mech.state.get_index("slider").y_idx]
        np.testing.assert_allclose(vy_slider, 0.0, atol=1e-10)

    def test_velocity_fd_consistency(self) -> None:
        """Velocity should match finite-difference of position."""
        mech = build_benchmark_slidercrank()
        dt = 1e-7
        angle = np.pi / 3

        q, q_dot, _ = solve_at_angle(mech, angle)
        q_plus, _, _ = solve_at_angle(mech, angle + dt)

        q_dot_fd = (q_plus - q) / dt
        np.testing.assert_allclose(q_dot, q_dot_fd, atol=1e-4)

    def test_slider_velocity_zero_at_tdc(self) -> None:
        """At θ=0 (TDC), slider velocity should be zero."""
        mech = build_benchmark_slidercrank()
        q, q_dot, _ = solve_at_angle(mech, 0.0)
        vx_slider = q_dot[mech.state.get_index("slider").x_idx]
        np.testing.assert_allclose(vx_slider, 0.0, atol=1e-8)


class TestSliderCrankAcceleration:
    """Validate acceleration solve."""

    def test_crank_angular_acceleration_zero(self) -> None:
        """Constant speed driver: crank θ̈ = 0."""
        mech = build_benchmark_slidercrank()
        q, q_dot, q_ddot = solve_at_angle(mech, np.pi / 4)
        crank_theta_ddot = q_ddot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_ddot) < 1e-8

    def test_acceleration_fd_consistency(self) -> None:
        """Acceleration should match FD of velocity."""
        mech = build_benchmark_slidercrank()
        dt = 1e-5
        angle = np.pi / 4

        q, q_dot, q_ddot = solve_at_angle(mech, angle)
        _, q_dot_plus, _ = solve_at_angle(mech, angle + dt)

        q_ddot_fd = (q_dot_plus - q_dot) / dt
        np.testing.assert_allclose(q_ddot, q_ddot_fd, atol=1e-3)

    def test_slider_y_acceleration_zero(self) -> None:
        """Slider ÿ should be zero (constrained to rail)."""
        mech = build_benchmark_slidercrank()
        q, q_dot, q_ddot = solve_at_angle(mech, np.pi / 4)
        ay_slider = q_ddot[mech.state.get_index("slider").y_idx]
        np.testing.assert_allclose(ay_slider, 0.0, atol=1e-8)


class TestSliderCrankSweep:
    """Validate full position sweep."""

    def test_full_rotation_convergence(self) -> None:
        """Sweep through 360° should converge everywhere."""
        mech = build_benchmark_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 73)  # include both endpoints

        q0 = mech.state.make_q()
        angle0 = angles[0]
        bx = np.cos(angle0)
        by = np.sin(angle0)
        x_s = analytical_slider_position(angle0)
        phi_cr = analytical_conrod_angle(angle0)

        mech.state.set_pose("crank", q0, 0.0, 0.0, angle0)
        mech.state.set_pose("conrod", q0, bx, by, phi_cr)
        mech.state.set_pose("slider", q0, x_s, 0.0, 0.0)
        result = solve_position(mech, q0, t=angle0)
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        assert sweep.n_converged == 73
        assert sweep.n_failed == 0

    def test_sweep_slider_on_rail_throughout(self) -> None:
        """Slider should stay on the rail at every sweep step."""
        mech = build_benchmark_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 36)

        q0 = mech.state.make_q()
        angle0 = angles[0]
        mech.state.set_pose("crank", q0, 0.0, 0.0, angle0)
        mech.state.set_pose("conrod", q0, np.cos(angle0), np.sin(angle0),
                            analytical_conrod_angle(angle0))
        mech.state.set_pose("slider", q0, analytical_slider_position(angle0), 0.0, 0.0)
        result = solve_position(mech, q0, t=angle0)
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        slider_y_idx = mech.state.get_index("slider").y_idx

        for i, q in enumerate(sweep.solutions):
            assert q is not None, f"Step {i} failed to converge"
            assert abs(q[slider_y_idx]) < 1e-10, f"Slider off rail at step {i}"

    def test_sweep_matches_analytical_positions(self) -> None:
        """Every sweep solution should match the analytical slider position."""
        mech = build_benchmark_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 36)

        q0 = mech.state.make_q()
        angle0 = angles[0]
        mech.state.set_pose("crank", q0, 0.0, 0.0, angle0)
        mech.state.set_pose("conrod", q0, np.cos(angle0), np.sin(angle0),
                            analytical_conrod_angle(angle0))
        mech.state.set_pose("slider", q0, analytical_slider_position(angle0), 0.0, 0.0)
        result = solve_position(mech, q0, t=angle0)
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        slider_x_idx = mech.state.get_index("slider").x_idx

        for i, q in enumerate(sweep.solutions):
            assert q is not None
            x_actual = q[slider_x_idx]
            x_expected = analytical_slider_position(angles[i])
            np.testing.assert_allclose(
                x_actual, x_expected, atol=1e-6,
                err_msg=f"Step {i}, angle={np.degrees(angles[i]):.1f}°",
            )

    def test_sweep_link_lengths_preserved(self) -> None:
        """Link lengths should be preserved at every sweep step."""
        mech = build_benchmark_slidercrank()
        angles = np.linspace(0.0, np.pi, 20)

        q0 = mech.state.make_q()
        angle0 = angles[0]
        mech.state.set_pose("crank", q0, 0.0, 0.0, angle0)
        mech.state.set_pose("conrod", q0, np.cos(angle0), np.sin(angle0),
                            analytical_conrod_angle(angle0))
        mech.state.set_pose("slider", q0, analytical_slider_position(angle0), 0.0, 0.0)
        result = solve_position(mech, q0, t=angle0)
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)

        for i, q in enumerate(sweep.solutions):
            assert q is not None
            # Crank: |B - A| = 1
            a = mech.state.body_point_global("crank", np.array([0.0, 0.0]), q)
            b = mech.state.body_point_global("crank", np.array([1.0, 0.0]), q)
            assert abs(float(np.linalg.norm(b - a)) - 1.0) < 1e-6, f"Crank at step {i}"

            # Conrod: |C - B| = 3
            b2 = mech.state.body_point_global("conrod", np.array([0.0, 0.0]), q)
            c = mech.state.body_point_global("conrod", np.array([3.0, 0.0]), q)
            assert abs(float(np.linalg.norm(c - b2)) - 3.0) < 1e-6, f"Conrod at step {i}"
