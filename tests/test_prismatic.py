"""Tests for prismatic joint constraint equations, Jacobians, and gamma.

The critical test: analytical Jacobian must match finite-difference numerical
Jacobian for all configurations. Gamma is verified via FD of the velocity RHS.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from linkage_sim.core.constraints import (
    PrismaticJoint,
    make_prismatic_joint,
)
from linkage_sim.core.state import GROUND_ID, State


def finite_difference_jacobian(
    constraint_func: object,
    state: State,
    q: NDArray[np.float64],
    t: float,
    n_eqs: int,
    eps: float = 1e-7,
) -> NDArray[np.float64]:
    """Compute Jacobian by central finite differences."""
    n = len(q)
    jac = np.zeros((n_eqs, n))
    for i in range(n):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += eps
        q_minus[i] -= eps
        phi_plus = constraint_func(state, q_plus, t)  # type: ignore[operator]
        phi_minus = constraint_func(state, q_minus, t)  # type: ignore[operator]
        jac[:, i] = (phi_plus - phi_minus) / (2 * eps)
    return jac


class TestPrismaticJointTwoBodies:
    """Prismatic joint between two moving bodies."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body_i")
        self.state.register_body("body_j")

        self.joint = make_prismatic_joint(
            joint_id="P1",
            body_i_id="body_i",
            point_i_name="A",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="body_j",
            point_j_name="B",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=np.array([1.0, 0.0]),
            delta_theta_0=0.0,
        )

    def test_properties(self) -> None:
        assert self.joint.id == "P1"
        assert self.joint.n_equations == 2
        assert self.joint.dof_removed == 2
        assert self.joint.body_i_id == "body_i"
        assert self.joint.body_j_id == "body_j"

    def test_constraint_satisfied_on_axis(self) -> None:
        """Body j displaced along axis => Φ[0]=0, same angle => Φ[1]=0."""
        # body_i at origin θ=0, body_j at (2, 0) θ=0
        # slide axis is (1,0), perpendicular is (0,1)
        # d = (2,0), n̂_g = (0,1), n̂_g · d = 0 ✓
        q = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_constraint_violated_perpendicular(self) -> None:
        """Body j displaced perpendicular to axis => Φ[0] ≠ 0."""
        # body_j at (0, 1): d = (0,1), n̂_g · d = 1 ≠ 0
        q = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        assert abs(phi[0]) > 0.5
        assert abs(phi[1]) < 1e-10  # same angle

    def test_constraint_violated_rotation(self) -> None:
        """Different angles => Φ[1] ≠ 0."""
        q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.5])
        phi = self.joint.constraint(self.state, q, 0.0)
        assert abs(phi[1]) > 0.1

    def test_constraint_with_rotated_body_i(self) -> None:
        """When body_i rotates, the slide axis rotates with it."""
        theta_i = np.pi / 2
        # axis_local = (1,0), after rotation by π/2: axis_global = (0,1)
        # n̂_global = (-1, 0)
        # body_j at (0, 3) with θ = π/2: d = (0,3)
        # n̂_g · d = (-1)*0 + 0*3 = 0 ✓ (on-axis)
        q = np.array([0.0, 0.0, theta_i, 0.0, 3.0, theta_i])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_constraint_with_delta_theta(self) -> None:
        """Non-zero delta_theta_0 shifts the rotation constraint."""
        joint = make_prismatic_joint(
            joint_id="P2",
            body_i_id="body_i",
            point_i_name="A",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="body_j",
            point_j_name="B",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=np.array([1.0, 0.0]),
            delta_theta_0=np.pi / 4,
        )
        # θⱼ - θᵢ = π/4 should satisfy Φ[1]=0
        q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, np.pi / 4])
        phi = joint.constraint(self.state, q, 0.0)
        assert abs(phi[1]) < 1e-10

    def test_jacobian_shape(self) -> None:
        q = np.zeros(6)
        jac = self.joint.jacobian(self.state, q, 0.0)
        assert jac.shape == (2, 6)

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.1, 0.2, 0.5, 0.3, 0.4, 1.2])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    @given(
        xi=st.floats(-1.0, 1.0),
        yi=st.floats(-1.0, 1.0),
        ti=st.floats(-np.pi, np.pi),
        xj=st.floats(-1.0, 1.0),
        yj=st.floats(-1.0, 1.0),
        tj=st.floats(-np.pi, np.pi),
    )
    @settings(max_examples=200)
    def test_jacobian_matches_finite_difference_hypothesis(
        self,
        xi: float,
        yi: float,
        ti: float,
        xj: float,
        yj: float,
        tj: float,
    ) -> None:
        """Property-based: analytical Jacobian matches FD at random configs."""
        q = np.array([xi, yi, ti, xj, yj, tj])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    def test_gamma_shape(self) -> None:
        q = np.zeros(6)
        q_dot = np.zeros(6)
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        assert gamma.shape == (2,)

    def test_gamma_zero_at_rest(self) -> None:
        """With zero velocities, gamma should be zero."""
        q = np.array([0.1, 0.2, 0.5, 0.3, 0.4, 1.0])
        q_dot = np.zeros(6)
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        np.testing.assert_array_almost_equal(gamma, [0.0, 0.0])

    def test_gamma_nonzero_with_angular_velocity(self) -> None:
        """With angular velocity, gamma should have terms."""
        q = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.5])
        q_dot = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 2.0])
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        # Rotation constraint gamma[1] should be 0 (linear in θ)
        assert abs(gamma[1]) < 1e-10

    @given(
        ti=st.floats(-np.pi, np.pi),
        tj=st.floats(-np.pi, np.pi),
        tdi=st.floats(-5.0, 5.0),
        tdj=st.floats(-5.0, 5.0),
    )
    @settings(max_examples=100)
    def test_gamma_matches_finite_difference_of_velocity_rhs(
        self,
        ti: float,
        tj: float,
        tdi: float,
        tdj: float,
    ) -> None:
        """Verify gamma via FD: when q̈=0, d(Φ_q*q̇)/dt = -γ."""
        q = np.array([0.1, 0.2, ti, 0.3, 0.4, tj])
        q_dot = np.array([0.5, -0.3, tdi, -0.2, 0.1, tdj])

        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)

        dt = 1e-7
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        jac_plus = self.joint.jacobian(self.state, q_plus, 0.0)
        jac_minus = self.joint.jacobian(self.state, q_minus, 0.0)

        d_phi_q_qdot_dt = (jac_plus @ q_dot - jac_minus @ q_dot) / (2 * dt)
        np.testing.assert_array_almost_equal(d_phi_q_qdot_dt, -gamma, decimal=4)

    def test_phi_t_is_zero(self) -> None:
        """Geometric constraint has no explicit time dependence."""
        q = np.array([0.1, 0.2, 0.5, 0.3, 0.4, 1.0])
        phi_t = self.joint.phi_t(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi_t, [0.0, 0.0])


class TestPrismaticJointGroundToBody:
    """Prismatic joint between ground and a moving body (common slider case)."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("slider")

        # Horizontal rail on ground, slider slides along X
        self.joint = make_prismatic_joint(
            joint_id="P_rail",
            body_i_id=GROUND_ID,
            point_i_name="rail",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="slider",
            point_j_name="pin",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=np.array([1.0, 0.0]),
            delta_theta_0=0.0,
        )

    def test_constraint_satisfied_on_x_axis(self) -> None:
        """Slider at (3, 0, 0) is on the horizontal rail."""
        q = np.array([3.0, 0.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_constraint_violated_off_axis(self) -> None:
        """Slider at (3, 1, 0) is off the horizontal rail."""
        q = np.array([3.0, 1.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        assert abs(phi[0]) > 0.5  # perpendicular violation

    def test_constraint_violated_rotated(self) -> None:
        """Slider rotated relative to ground."""
        q = np.array([3.0, 0.0, 0.5])
        phi = self.joint.constraint(self.state, q, 0.0)
        assert abs(phi[1]) > 0.1  # rotation violation

    def test_jacobian_shape(self) -> None:
        q = np.zeros(3)
        jac = self.joint.jacobian(self.state, q, 0.0)
        assert jac.shape == (2, 3)

    def test_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.5, 0.1, 0.3])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    @given(
        x=st.floats(-2.0, 2.0),
        y=st.floats(-2.0, 2.0),
        theta=st.floats(-np.pi, np.pi),
    )
    @settings(max_examples=200)
    def test_jacobian_matches_finite_difference_hypothesis(
        self, x: float, y: float, theta: float
    ) -> None:
        q = np.array([x, y, theta])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    def test_gamma_zero_at_rest(self) -> None:
        q = np.array([1.0, 0.0, 0.0])
        q_dot = np.zeros(3)
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        np.testing.assert_array_almost_equal(gamma, [0.0, 0.0])

    @given(
        theta=st.floats(-np.pi, np.pi),
        td=st.floats(-5.0, 5.0),
        vx=st.floats(-3.0, 3.0),
    )
    @settings(max_examples=100)
    def test_gamma_matches_finite_difference(
        self, theta: float, td: float, vx: float
    ) -> None:
        """Verify gamma via FD for ground-to-body slider."""
        q = np.array([0.5, 0.1, theta])
        q_dot = np.array([vx, 0.0, td])

        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)

        dt = 1e-7
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        jac_plus = self.joint.jacobian(self.state, q_plus, 0.0)
        jac_minus = self.joint.jacobian(self.state, q_minus, 0.0)

        d_phi_q_qdot_dt = (jac_plus @ q_dot - jac_minus @ q_dot) / (2 * dt)
        np.testing.assert_array_almost_equal(d_phi_q_qdot_dt, -gamma, decimal=4)


class TestPrismaticJointDiagonalAxis:
    """Prismatic joint with a non-axis-aligned slide direction."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body_i")
        self.state.register_body("body_j")

        # Slide axis at 45° in body_i's local frame
        axis = np.array([1.0, 1.0])  # will be normalized
        self.joint = make_prismatic_joint(
            joint_id="P_diag",
            body_i_id="body_i",
            point_i_name="A",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="body_j",
            point_j_name="B",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=axis,
            delta_theta_0=0.0,
        )

    def test_axis_normalized(self) -> None:
        """Factory should normalize the axis vector."""
        np.testing.assert_almost_equal(
            np.linalg.norm(self.joint._axis_local_i), 1.0
        )

    def test_n_hat_perpendicular(self) -> None:
        """n̂ should be perpendicular to ê."""
        dot = np.dot(self.joint._axis_local_i, self.joint._n_hat_local_i)
        assert abs(dot) < 1e-10

    def test_constraint_satisfied_along_diagonal(self) -> None:
        """Displacement along 45° axis should satisfy perpendicular constraint."""
        # body_i at origin θ=0: axis = (1/√2, 1/√2), n̂ = (-1/√2, 1/√2)
        # body_j at (1, 1): d = (1, 1), n̂ · d = -1/√2 + 1/√2 = 0 ✓
        q = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_almost_equal(phi[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(phi[1], 0.0, decimal=10)

    @given(
        xi=st.floats(-1.0, 1.0),
        yi=st.floats(-1.0, 1.0),
        ti=st.floats(-np.pi, np.pi),
        xj=st.floats(-1.0, 1.0),
        yj=st.floats(-1.0, 1.0),
        tj=st.floats(-np.pi, np.pi),
    )
    @settings(max_examples=200)
    def test_jacobian_matches_finite_difference_hypothesis(
        self,
        xi: float,
        yi: float,
        ti: float,
        xj: float,
        yj: float,
        tj: float,
    ) -> None:
        q = np.array([xi, yi, ti, xj, yj, tj])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    @given(
        ti=st.floats(-np.pi, np.pi),
        tj=st.floats(-np.pi, np.pi),
        tdi=st.floats(-5.0, 5.0),
        tdj=st.floats(-5.0, 5.0),
    )
    @settings(max_examples=100)
    def test_gamma_matches_finite_difference(
        self, ti: float, tj: float, tdi: float, tdj: float
    ) -> None:
        q = np.array([0.1, 0.2, ti, 0.3, 0.4, tj])
        q_dot = np.array([0.5, -0.3, tdi, -0.2, 0.1, tdj])

        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)

        dt = 1e-7
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        jac_plus = self.joint.jacobian(self.state, q_plus, 0.0)
        jac_minus = self.joint.jacobian(self.state, q_minus, 0.0)

        d_phi_q_qdot_dt = (jac_plus @ q_dot - jac_minus @ q_dot) / (2 * dt)
        np.testing.assert_array_almost_equal(d_phi_q_qdot_dt, -gamma, decimal=4)


class TestPrismaticJointWithOffsetPoints:
    """Prismatic joint with non-origin attachment points."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body_i")
        self.state.register_body("body_j")

        self.joint = make_prismatic_joint(
            joint_id="P_off",
            body_i_id="body_i",
            point_i_name="rail_start",
            point_i_local=np.array([0.1, 0.05]),
            body_j_id="body_j",
            point_j_name="slide_pt",
            point_j_local=np.array([0.0, 0.02]),
            axis_local_i=np.array([1.0, 0.0]),
            delta_theta_0=0.0,
        )

    @given(
        xi=st.floats(-1.0, 1.0),
        yi=st.floats(-1.0, 1.0),
        ti=st.floats(-np.pi, np.pi),
        xj=st.floats(-1.0, 1.0),
        yj=st.floats(-1.0, 1.0),
        tj=st.floats(-np.pi, np.pi),
    )
    @settings(max_examples=200)
    def test_jacobian_matches_finite_difference_hypothesis(
        self,
        xi: float,
        yi: float,
        ti: float,
        xj: float,
        yj: float,
        tj: float,
    ) -> None:
        q = np.array([xi, yi, ti, xj, yj, tj])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)

    @given(
        ti=st.floats(-np.pi, np.pi),
        tj=st.floats(-np.pi, np.pi),
        tdi=st.floats(-5.0, 5.0),
        tdj=st.floats(-5.0, 5.0),
    )
    @settings(max_examples=100)
    def test_gamma_matches_finite_difference(
        self, ti: float, tj: float, tdi: float, tdj: float
    ) -> None:
        q = np.array([0.1, 0.2, ti, 0.3, 0.4, tj])
        q_dot = np.array([0.5, -0.3, tdi, -0.2, 0.1, tdj])

        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)

        dt = 1e-7
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        jac_plus = self.joint.jacobian(self.state, q_plus, 0.0)
        jac_minus = self.joint.jacobian(self.state, q_minus, 0.0)

        d_phi_q_qdot_dt = (jac_plus @ q_dot - jac_minus @ q_dot) / (2 * dt)
        np.testing.assert_array_almost_equal(d_phi_q_qdot_dt, -gamma, decimal=4)


class TestPrismaticJointFactory:
    """Tests for make_prismatic_joint factory."""

    def test_zero_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            make_prismatic_joint(
                joint_id="bad",
                body_i_id="a",
                point_i_name="p",
                point_i_local=np.array([0.0, 0.0]),
                body_j_id="b",
                point_j_name="q",
                point_j_local=np.array([0.0, 0.0]),
                axis_local_i=np.array([0.0, 0.0]),
            )

    def test_axis_normalization(self) -> None:
        joint = make_prismatic_joint(
            joint_id="norm_test",
            body_i_id="a",
            point_i_name="p",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b",
            point_j_name="q",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=np.array([3.0, 4.0]),
        )
        np.testing.assert_almost_equal(np.linalg.norm(joint._axis_local_i), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(joint._n_hat_local_i), 1.0)

    def test_perpendicular_direction(self) -> None:
        """n̂ should be 90° CCW from ê."""
        joint = make_prismatic_joint(
            joint_id="perp_test",
            body_i_id="a",
            point_i_name="p",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="b",
            point_j_name="q",
            point_j_local=np.array([0.0, 0.0]),
            axis_local_i=np.array([1.0, 0.0]),
        )
        # For axis (1,0), perpendicular should be (0,1) or (-0, 1)
        np.testing.assert_array_almost_equal(joint._n_hat_local_i, [0.0, 1.0])

    def test_copies_input_arrays(self) -> None:
        """Input arrays should be copied, not aliased."""
        pt_i = np.array([1.0, 2.0])
        pt_j = np.array([3.0, 4.0])
        axis = np.array([1.0, 0.0])
        joint = make_prismatic_joint(
            joint_id="copy_test",
            body_i_id="a",
            point_i_name="p",
            point_i_local=pt_i,
            body_j_id="b",
            point_j_name="q",
            point_j_local=pt_j,
            axis_local_i=axis,
        )
        pt_i[0] = 999.0
        pt_j[0] = 999.0
        axis[0] = 999.0
        assert joint._point_i_local[0] != 999.0
        assert joint._point_j_local[0] != 999.0
        assert joint._axis_local_i[0] != 999.0


class TestPrismaticJointInMechanism:
    """Integration: prismatic joint works correctly inside a Mechanism."""

    def test_add_prismatic_joint(self) -> None:
        from linkage_sim.core.bodies import Body, make_ground
        from linkage_sim.core.mechanism import Mechanism

        mech = Mechanism()
        ground = make_ground(rail=(0.0, 0.0))
        slider = Body(
            id="slider",
            attachment_points={"pin": np.array([0.0, 0.0])},
        )
        mech.add_body(ground)
        mech.add_body(slider)
        mech.add_prismatic_joint(
            "P1", "ground", "rail", "slider", "pin",
            axis_local_i=np.array([1.0, 0.0]),
        )
        mech.build()

        assert mech.n_constraints == 2
        assert len(mech.joints) == 1

    def test_after_build_raises(self) -> None:
        from linkage_sim.core.bodies import Body, make_ground
        from linkage_sim.core.mechanism import Mechanism

        mech = Mechanism()
        ground = make_ground(rail=(0.0, 0.0))
        slider = Body(
            id="slider",
            attachment_points={"pin": np.array([0.0, 0.0])},
        )
        mech.add_body(ground)
        mech.add_body(slider)
        mech.build()

        with pytest.raises(RuntimeError, match="after build"):
            mech.add_prismatic_joint(
                "P1", "ground", "rail", "slider", "pin",
                axis_local_i=np.array([1.0, 0.0]),
            )
