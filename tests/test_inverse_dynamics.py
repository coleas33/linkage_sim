"""Tests for mass matrix assembly and inverse dynamics solver."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.solvers.inverse_dynamics import solve_inverse_dynamics
from linkage_sim.solvers.kinematics import solve_acceleration, solve_position, solve_velocity
from linkage_sim.solvers.mass_matrix import assemble_mass_matrix
from linkage_sim.solvers.statics import solve_statics


def build_fourbar(mass: float = 0.0, Izz: float = 0.0) -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=mass, Izz_cg=Izz)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=mass, Izz_cg=Izz)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=mass, Izz_cg=Izz)
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


def solve_full(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_s = result.q
    q_dot = solve_velocity(mech, q_s, t=angle)
    q_ddot = solve_acceleration(mech, q_s, q_dot, t=angle)
    return q_s, q_dot, q_ddot


class TestMassMatrix:
    def test_shape(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.1)
        M = assemble_mass_matrix(mech)
        n = mech.state.n_coords
        assert M.shape == (n, n)

    def test_block_diagonal(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.1)
        M = assemble_mass_matrix(mech)
        # Check crank block: crank has length=1, CG at (0.5, 0)
        # M_θθ = Izz_cg + m * |s_cg|^2 = 0.1 + 2.0 * 0.25 = 0.6
        idx = mech.state.get_index("crank")
        assert M[idx.x_idx, idx.x_idx] == pytest.approx(2.0)
        assert M[idx.y_idx, idx.y_idx] == pytest.approx(2.0)
        assert M[idx.theta_idx, idx.theta_idx] == pytest.approx(0.6)

    def test_symmetric(self) -> None:
        mech = build_fourbar(mass=3.0, Izz=0.5)
        M = assemble_mass_matrix(mech)
        np.testing.assert_array_equal(M, M.T)

    def test_zero_mass_zero_matrix(self) -> None:
        mech = build_fourbar(mass=0.0, Izz=0.0)
        M = assemble_mass_matrix(mech)
        np.testing.assert_array_equal(M, np.zeros_like(M))


class TestInverseDynamics:
    def test_zero_mass_matches_statics(self) -> None:
        """With zero mass, inverse dynamics = statics (no inertial terms)."""
        mech = build_fourbar(mass=0.0)
        angle = np.pi / 4
        q, q_dot, q_ddot = solve_full(mech, angle)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        static_result = solve_statics(mech, q, [gravity], t=angle)
        id_result = solve_inverse_dynamics(
            mech, q, q_dot, q_ddot, [gravity], t=angle
        )

        np.testing.assert_allclose(
            id_result.lambdas, static_result.lambdas, atol=1e-10
        )

    def test_residual_small(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.05)
        angle = np.pi / 3
        q, q_dot, q_ddot = solve_full(mech, angle)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        result = solve_inverse_dynamics(
            mech, q, q_dot, q_ddot, [gravity], t=angle
        )
        assert result.residual_norm < 1e-10

    def test_inertia_changes_driver_torque(self) -> None:
        """With mass/inertia, driver torque differs from statics."""
        mech = build_fourbar(mass=5.0, Izz=0.5)
        angle = np.pi / 4
        q, q_dot, q_ddot = solve_full(mech, angle)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        static_result = solve_statics(mech, q, [gravity], t=angle)
        id_result = solve_inverse_dynamics(
            mech, q, q_dot, q_ddot, [gravity], t=angle
        )

        tau_static = static_result.lambdas[-1]
        tau_dynamic = id_result.lambdas[-1]

        # With significant mass and acceleration, they should differ
        assert abs(tau_static - tau_dynamic) > 0.01

    def test_sweep_residuals(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.1)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        angles = np.linspace(np.radians(30), np.radians(150), 13)
        for angle in angles:
            q, q_dot, q_ddot = solve_full(mech, angle)
            result = solve_inverse_dynamics(
                mech, q, q_dot, q_ddot, [gravity], t=angle
            )
            assert result.residual_norm < 1e-10, (
                f"Residual {result.residual_norm:.2e} at {np.degrees(angle):.0f}°"
            )

    def test_no_forces_pure_inertia(self) -> None:
        """No applied forces: driver torque comes purely from inertia."""
        mech = build_fourbar(mass=3.0, Izz=0.2)
        angle = np.pi / 4
        q, q_dot, q_ddot = solve_full(mech, angle)

        result = solve_inverse_dynamics(
            mech, q, q_dot, q_ddot, [], t=angle
        )
        assert result.residual_norm < 1e-10
        # With nonzero acceleration, driver torque should be nonzero
        assert abs(result.lambdas[-1]) > 0.001

    def test_m_q_ddot_nonzero_with_mass(self) -> None:
        mech = build_fourbar(mass=2.0, Izz=0.1)
        angle = np.pi / 3
        q, q_dot, q_ddot = solve_full(mech, angle)

        result = solve_inverse_dynamics(mech, q, q_dot, q_ddot, [], t=angle)
        assert float(np.linalg.norm(result.M_q_ddot)) > 0.01
