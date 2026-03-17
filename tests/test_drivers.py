"""Tests for driver constraints (revolute driver)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.drivers import (
    RevoluteDriver,
    constant_speed_driver,
    make_revolute_driver,
)
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import State
from linkage_sim.solvers.assembly import assemble_constraints, assemble_jacobian
from linkage_sim.solvers.kinematics import solve_position


def build_driven_crank() -> Mechanism:
    """Single crank pinned to ground with a revolute driver.

    Ground pivot at O=(0,0). Crank length 1.0.
    Driver prescribes: θ = 2*t (constant speed ω=2 rad/s).
    DOF = 3*1 - 2(revolute) - 1(driver) = 0 (fully determined).
    """
    mech = Mechanism()
    ground = make_ground(O=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
    mech.add_constant_speed_driver("D1", "ground", "crank", omega=2.0)
    mech.build()
    return mech


class TestRevoluteDriverConstraint:
    def test_constraint_at_t0(self) -> None:
        """At t=0, driver requires θ_crank = 0."""
        mech = build_driven_crank()
        q = mech.state.make_q()
        # Crank at θ=0, body origin at (0,0) so A=(0,0) matches ground O
        mech.state.set_pose("crank", q, 0.0, 0.0, 0.0)

        phi = assemble_constraints(mech, q, 0.0)
        # J1 revolute: 2 eqs, D1 driver: 1 eq => 3 total
        assert phi.shape == (3,)
        # At valid config, all constraints near zero
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0, 0.0])

    def test_constraint_at_t_nonzero(self) -> None:
        """At t=π/4, driver requires θ_crank = π/2."""
        mech = build_driven_crank()
        q = mech.state.make_q()
        # Place crank at θ=π/2, body origin at (0,0) so A=(0,0) matches ground O
        mech.state.set_pose("crank", q, 0.0, 0.0, np.pi / 2)

        phi = assemble_constraints(mech, q, np.pi / 4)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0, 0.0])

    def test_constraint_violated(self) -> None:
        """Crank at wrong angle produces nonzero driver residual."""
        mech = build_driven_crank()
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.5, 0.0, 0.0)  # θ=0

        phi = assemble_constraints(mech, q, 1.0)  # expects θ=2.0
        # Driver constraint: θ_crank - f(1.0) = 0 - 2.0 = -2.0
        assert abs(phi[2] - (-2.0)) < 1e-10

    def test_n_constraints(self) -> None:
        mech = build_driven_crank()
        # 2 (revolute) + 1 (driver) = 3
        assert mech.n_constraints == 3

    def test_jacobian_shape(self) -> None:
        mech = build_driven_crank()
        q = mech.state.make_q()
        jac = assemble_jacobian(mech, q, 0.0)
        assert jac.shape == (3, 3)

    def test_jacobian_driver_row(self) -> None:
        """Driver Jacobian row: [0, 0, +1] for ground-to-crank."""
        mech = build_driven_crank()
        q = mech.state.make_q()
        jac = assemble_jacobian(mech, q, 0.0)
        # Driver row is the 3rd row (index 2)
        np.testing.assert_array_almost_equal(jac[2, :], [0.0, 0.0, 1.0])


class TestRevoluteDriverFD:
    """Finite-difference verification of the driver Jacobian."""

    @given(
        theta=st.floats(min_value=-np.pi, max_value=np.pi),
        x=st.floats(min_value=-2.0, max_value=2.0),
        y=st.floats(min_value=-2.0, max_value=2.0),
        t=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_jacobian_fd(
        self, theta: float, x: float, y: float, t: float
    ) -> None:
        """Analytical Jacobian matches finite-difference Jacobian."""
        mech = build_driven_crank()
        q = np.array([x, y, theta])

        analytical = assemble_jacobian(mech, q, t)

        eps = 1e-7
        n = len(q)
        m = mech.n_constraints
        numerical = np.zeros((m, n))
        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            phi_plus = assemble_constraints(mech, q_plus, t)
            phi_minus = assemble_constraints(mech, q_minus, t)
            numerical[:, i] = (phi_plus - phi_minus) / (2 * eps)

        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)


class TestRevoluteDriverGamma:
    def test_gamma_constant_speed(self) -> None:
        """For constant speed driver, f''=0 so γ_driver = 0."""
        mech = build_driven_crank()
        q = mech.state.make_q()
        q_dot = mech.state.make_q()
        from linkage_sim.solvers.assembly import assemble_gamma

        gamma = assemble_gamma(mech, q, q_dot, 0.0)
        # 3 constraint equations total
        assert gamma.shape == (3,)
        # Driver gamma (last element) should be 0 for constant speed
        assert abs(gamma[2]) < 1e-15

    def test_gamma_accelerating_driver(self) -> None:
        """For f(t) = t², f''(t) = 2, so γ_driver = 2."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
        mech.add_revolute_driver(
            "D1",
            "ground",
            "crank",
            f=lambda t: t**2,
            f_dot=lambda t: 2 * t,
            f_ddot=lambda t: 2.0,
        )
        mech.build()

        q = mech.state.make_q()
        q_dot = mech.state.make_q()
        from linkage_sim.solvers.assembly import assemble_gamma

        gamma = assemble_gamma(mech, q, q_dot, 5.0)
        assert abs(gamma[2] - 2.0) < 1e-15


class TestDrivenFourbar:
    """Integration tests: driven 4-bar linkage."""

    def build_driven_fourbar(self, crank_angle: float) -> Mechanism:
        """4-bar with revolute driver fixing crank angle."""
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

        mech.add_revolute_driver(
            "D1",
            "ground",
            "crank",
            f=lambda t: crank_angle,
            f_dot=lambda t: 0.0,
            f_ddot=lambda t: 0.0,
        )

        mech.build()
        return mech

    def test_driven_fourbar_n_constraints(self) -> None:
        mech = self.build_driven_fourbar(np.pi / 4)
        # 4 revolute * 2 + 1 driver = 9
        assert mech.n_constraints == 9

    def test_driven_fourbar_fully_determined(self) -> None:
        """Driven 4-bar: 9 constraints, 9 coords -> DOF=0 with driver."""
        mech = self.build_driven_fourbar(np.pi / 6)
        from linkage_sim.analysis.validation import grubler_dof

        result = grubler_dof(mech, expected_dof=0)
        assert result.dof == 0

    def test_driven_fourbar_solve(self) -> None:
        """Newton-Raphson converges for a driven 4-bar."""
        angle = np.pi / 4
        mech = self.build_driven_fourbar(angle)

        # Start from a rough initial guess
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.3, 0.3, angle)
        mech.state.set_pose("coupler", q, 1.0, 0.5, 0.0)
        mech.state.set_pose("rocker", q, 3.5, 0.5, 1.0)

        result = solve_position(mech, q)
        assert result.converged
        assert result.residual_norm < 1e-10

        # Verify crank angle matches driver
        crank_theta = mech.state.get_angle("crank", result.q)
        assert abs(crank_theta - angle) < 1e-8


class TestDriverProperties:
    def test_dof_removed(self) -> None:
        driver = constant_speed_driver("D", "ground", "crank", omega=1.0)
        assert driver.dof_removed == 1

    def test_n_equations(self) -> None:
        driver = constant_speed_driver("D", "ground", "crank", omega=1.0)
        assert driver.n_equations == 1

    def test_id(self) -> None:
        driver = constant_speed_driver("mydriver", "ground", "crank", omega=1.0)
        assert driver.id == "mydriver"

    def test_body_ids(self) -> None:
        driver = constant_speed_driver("D", "gnd", "crk", omega=1.0)
        assert driver.body_i_id == "gnd"
        assert driver.body_j_id == "crk"

    def test_constant_speed_f_values(self) -> None:
        driver = constant_speed_driver(
            "D", "ground", "crank", omega=3.0, theta_0=1.0
        )
        state = State()
        state.register_body("crank")
        q = state.make_q()

        # At t=0: f(0) = 1.0, so constraint = θ_crank - 1.0
        phi = driver.constraint(state, q, 0.0)
        assert abs(phi[0] - (0.0 - 1.0)) < 1e-15

        # At t=2: f(2) = 1.0 + 3.0*2 = 7.0
        phi = driver.constraint(state, q, 2.0)
        assert abs(phi[0] - (0.0 - 7.0)) < 1e-15

    def test_cannot_add_driver_after_build(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.build()

        with pytest.raises(RuntimeError, match="after build"):
            mech.add_revolute_driver(
                "D1",
                "ground",
                "crank",
                f=lambda t: t,
                f_dot=lambda t: 1.0,
                f_ddot=lambda t: 0.0,
            )

    def test_driver_unknown_body_raises(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)

        with pytest.raises(KeyError, match="not found"):
            mech.add_revolute_driver(
                "D1",
                "ground",
                "missing",
                f=lambda t: t,
                f_dot=lambda t: 1.0,
                f_ddot=lambda t: 0.0,
            )
