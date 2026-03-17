"""Tests for velocity and acceleration solvers."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)


def build_driven_crank(omega: float = 2.0) -> Mechanism:
    """Single crank pinned to ground with constant-speed driver.

    Ground pivot at O=(0,0). Crank length 1.0.
    Driver: f(t) = omega * t, fully determined system.
    """
    mech = Mechanism()
    ground = make_ground(O=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
    mech.add_constant_speed_driver("D1", "ground", "crank", omega=omega)
    mech.build()
    return mech


def build_driven_fourbar() -> Mechanism:
    """4-bar with identity driver: f(t) = t.

    Crank=1, Coupler=3, Rocker=2, Ground=4.
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

    # Identity driver: f(t) = t
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t,
        f_dot=lambda t: 1.0,
        f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def get_fourbar_q(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    """Get a valid 4-bar configuration at given crank angle via NR solve."""
    q = mech.state.make_q()
    # Rough initial guess
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx = np.cos(angle)
    by = np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)

    result = solve_position(mech, q, t=angle)
    assert result.converged, f"Failed to converge at angle={angle}"
    return result.q


class TestSolveVelocity:
    def test_crank_velocity(self) -> None:
        """Driven crank at ω=2: q̇ should be [0, 0, 2]."""
        omega = 2.0
        mech = build_driven_crank(omega)
        # At t=0, crank at θ=0, body origin at (0,0)
        q = mech.state.make_q()
        result = solve_position(mech, q, t=0.0)
        assert result.converged

        q_dot = solve_velocity(mech, result.q, t=0.0)
        # Body origin is pinned at (0,0) so ẋ=ẏ=0, θ̇=ω
        np.testing.assert_array_almost_equal(q_dot, [0.0, 0.0, omega])

    def test_crank_velocity_at_t1(self) -> None:
        """At t=1, crank has rotated to θ=2. Velocity still [0, 0, 2]."""
        omega = 2.0
        mech = build_driven_crank(omega)
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, omega * 1.0)
        result = solve_position(mech, q, t=1.0)
        assert result.converged

        q_dot = solve_velocity(mech, result.q, t=1.0)
        np.testing.assert_array_almost_equal(q_dot, [0.0, 0.0, omega])

    def test_fourbar_velocity_shape(self) -> None:
        mech = build_driven_fourbar()
        q = get_fourbar_q(mech, np.pi / 4)
        q_dot = solve_velocity(mech, q, t=np.pi / 4)
        assert q_dot.shape == (9,)

    def test_fourbar_crank_angular_velocity(self) -> None:
        """Crank θ̇ should be 1.0 (identity driver f'(t)=1)."""
        mech = build_driven_fourbar()
        angle = np.pi / 6
        q = get_fourbar_q(mech, angle)
        q_dot = solve_velocity(mech, q, t=angle)

        # Crank is the second body in sorted order (coupler, crank, rocker)
        crank_theta_dot = q_dot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_dot - 1.0) < 1e-8

    def test_velocity_satisfies_constraint_rate(self) -> None:
        """Φ_q * q̇ + Φ_t = 0 should hold."""
        mech = build_driven_fourbar()
        angle = np.pi / 3
        q = get_fourbar_q(mech, angle)
        q_dot = solve_velocity(mech, q, t=angle)

        from linkage_sim.solvers.assembly import assemble_jacobian, assemble_phi_t

        phi_q = assemble_jacobian(mech, q, angle)
        phi_t = assemble_phi_t(mech, q, angle)
        residual = phi_q @ q_dot + phi_t
        np.testing.assert_array_almost_equal(residual, np.zeros(9), decimal=8)

    def test_velocity_finite_difference(self) -> None:
        """Velocity from FD of position should match analytical velocity."""
        mech = build_driven_fourbar()
        dt = 1e-7
        angle = np.pi / 4

        q = get_fourbar_q(mech, angle)
        q_plus = get_fourbar_q(mech, angle + dt)

        q_dot_analytical = solve_velocity(mech, q, t=angle)
        q_dot_fd = (q_plus - q) / dt

        np.testing.assert_array_almost_equal(
            q_dot_analytical, q_dot_fd, decimal=4
        )

    def test_requires_built_mechanism(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(RuntimeError, match="must be built"):
            solve_velocity(mech, np.zeros(0))


class TestSolveAcceleration:
    def test_crank_acceleration_constant_speed(self) -> None:
        """Constant speed driver: q̈ should be [0, 0, 0] for pinned crank."""
        omega = 2.0
        mech = build_driven_crank(omega)
        q = mech.state.make_q()
        result = solve_position(mech, q, t=0.0)
        assert result.converged

        q_dot = solve_velocity(mech, result.q, t=0.0)
        q_ddot = solve_acceleration(mech, result.q, q_dot, t=0.0)

        # Pinned at origin with constant ω: no translational accel,
        # angular accel = 0
        np.testing.assert_array_almost_equal(q_ddot, [0.0, 0.0, 0.0])

    def test_fourbar_acceleration_shape(self) -> None:
        mech = build_driven_fourbar()
        q = get_fourbar_q(mech, np.pi / 4)
        q_dot = solve_velocity(mech, q, t=np.pi / 4)
        q_ddot = solve_acceleration(mech, q, q_dot, t=np.pi / 4)
        assert q_ddot.shape == (9,)

    def test_fourbar_crank_angular_acceleration(self) -> None:
        """Identity driver f''=0: crank angular acceleration should be 0."""
        mech = build_driven_fourbar()
        angle = np.pi / 6
        q = get_fourbar_q(mech, angle)
        q_dot = solve_velocity(mech, q, t=angle)
        q_ddot = solve_acceleration(mech, q, q_dot, t=angle)

        crank_theta_ddot = q_ddot[mech.state.get_index("crank").theta_idx]
        assert abs(crank_theta_ddot) < 1e-8

    def test_acceleration_satisfies_constraint_accel(self) -> None:
        """Φ_q * q̈ = γ should hold."""
        mech = build_driven_fourbar()
        angle = np.pi / 3
        q = get_fourbar_q(mech, angle)
        q_dot = solve_velocity(mech, q, t=angle)
        q_ddot = solve_acceleration(mech, q, q_dot, t=angle)

        from linkage_sim.solvers.assembly import assemble_gamma, assemble_jacobian

        phi_q = assemble_jacobian(mech, q, angle)
        gamma = assemble_gamma(mech, q, q_dot, angle)
        residual = phi_q @ q_ddot - gamma
        np.testing.assert_array_almost_equal(residual, np.zeros(9), decimal=8)

    def test_acceleration_finite_difference(self) -> None:
        """Acceleration from FD of velocity should match analytical."""
        mech = build_driven_fourbar()
        dt = 1e-5
        angle = np.pi / 4

        q = get_fourbar_q(mech, angle)
        q_plus = get_fourbar_q(mech, angle + dt)

        q_dot = solve_velocity(mech, q, t=angle)
        q_dot_plus = solve_velocity(mech, q_plus, t=angle + dt)

        q_ddot_analytical = solve_acceleration(mech, q, q_dot, t=angle)
        q_ddot_fd = (q_dot_plus - q_dot) / dt

        np.testing.assert_array_almost_equal(
            q_ddot_analytical, q_ddot_fd, decimal=3
        )

    def test_requires_built_mechanism(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(RuntimeError, match="must be built"):
            solve_acceleration(mech, np.zeros(0), np.zeros(0))
