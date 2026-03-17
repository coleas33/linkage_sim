"""Tests for joint constraint equations and Jacobians.

The critical test: analytical Jacobian must match finite-difference numerical
Jacobian for all constraint types, at arbitrary configurations. This is
tested with hypothesis for property-based coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linkage_sim.core.constraints import RevoluteJoint, make_revolute_joint
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
    from numpy.typing import NDArray

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


class TestRevoluteJointTwoBodies:
    """Revolute joint between two moving bodies."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("body_i")
        self.state.register_body("body_j")

        self.joint = make_revolute_joint(
            joint_id="J1",
            body_i_id="body_i",
            point_i_name="B",
            point_i_local=np.array([0.1, 0.0]),
            body_j_id="body_j",
            point_j_name="A",
            point_j_local=np.array([0.0, 0.0]),
        )

    def test_constraint_satisfied_when_coincident(self) -> None:
        """Points coincident => Φ = 0."""
        q = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        # body_i at origin, θ=0: point B at (0.1, 0)
        # body_j at (0.1, 0), θ=0: point A at (0.1, 0)
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_constraint_nonzero_when_separated(self) -> None:
        """Points not coincident => Φ ≠ 0."""
        q = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        assert not np.allclose(phi, [0.0, 0.0])

    def test_constraint_with_rotation(self) -> None:
        """Verify constraint handles rotated bodies correctly."""
        # body_i at origin, θ=π/2: point B (local [0.1, 0]) maps to global (0, 0.1)
        # body_j at (0, 0.1), θ=0: point A (local [0, 0]) maps to global (0, 0.1)
        q = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.1, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_jacobian_shape(self) -> None:
        q = np.zeros(6)
        jac = self.joint.jacobian(self.state, q, 0.0)
        assert jac.shape == (2, 6)

    def test_jacobian_matches_finite_difference(self) -> None:
        """Analytical Jacobian vs finite-difference at a specific config."""
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

    def test_n_equations(self) -> None:
        assert self.joint.n_equations == 2

    def test_gamma_shape(self) -> None:
        q = np.zeros(6)
        q_dot = np.zeros(6)
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        assert gamma.shape == (2,)

    def test_gamma_zero_at_rest(self) -> None:
        """With zero velocities, gamma should be zero (no centripetal terms)."""
        q = np.array([0.0, 0.0, 0.5, 0.1, 0.2, 1.0])
        q_dot = np.zeros(6)
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        np.testing.assert_array_almost_equal(gamma, [0.0, 0.0])

    def test_gamma_nonzero_with_angular_velocity(self) -> None:
        """With angular velocity, gamma should have centripetal terms."""
        q = np.array([0.0, 0.0, 0.5, 0.1, 0.0, 0.0])
        q_dot = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        # Centripetal: A_i * s_i * θ̇ᵢ² should be nonzero
        assert not np.allclose(gamma, [0.0, 0.0])

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
        """Verify gamma by checking: Φ_q * q̈ = γ is consistent.

        If we perturb q along q̇ * dt and recompute Φ_q * q̇, the change
        should be approximately -γ * dt.

        More precisely: d(Φ_q * q̇)/dt = Φ_q * q̈ + (d(Φ_q)/dt) * q̇
        And γ = -(d(Φ_q)/dt) * q̇, so d(Φ_q * q̇)/dt = Φ_q * q̈ + (-γ)
        When q̈ = 0: d(Φ_q * q̇)/dt = -γ
        """
        q = np.array([0.1, 0.2, ti, 0.3, 0.4, tj])
        q_dot = np.array([0.5, -0.3, tdi, -0.2, 0.1, tdj])

        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)

        # Finite difference: (Φ_q(q+dq) * q̇ - Φ_q(q-dq) * q̇) / (2*dt)
        dt = 1e-7
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        jac_plus = self.joint.jacobian(self.state, q_plus, 0.0)
        jac_minus = self.joint.jacobian(self.state, q_minus, 0.0)

        # d(Φ_q * q̇)/dt with q̈=0
        d_phi_q_qdot_dt = (jac_plus @ q_dot - jac_minus @ q_dot) / (2 * dt)

        # Should equal -gamma
        np.testing.assert_array_almost_equal(d_phi_q_qdot_dt, -gamma, decimal=4)


class TestRevoluteJointGroundToBody:
    """Revolute joint between ground and a moving body."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("crank")

        self.joint = make_revolute_joint(
            joint_id="J_ground",
            body_i_id=GROUND_ID,
            point_i_name="O2",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="crank",
            point_j_name="A",
            point_j_local=np.array([0.0, 0.0]),
        )

    def test_constraint_satisfied_at_origin(self) -> None:
        q = np.array([0.0, 0.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [0.0, 0.0])

    def test_constraint_violated_when_displaced(self) -> None:
        q = np.array([0.5, 0.0, 0.0])
        phi = self.joint.constraint(self.state, q, 0.0)
        np.testing.assert_array_almost_equal(phi, [-0.5, 0.0])

    def test_jacobian_shape(self) -> None:
        q = np.zeros(3)
        jac = self.joint.jacobian(self.state, q, 0.0)
        assert jac.shape == (2, 3)

    def test_jacobian_ground_columns_not_present(self) -> None:
        """Ground has no entries in q, so only body_j columns should appear."""
        q = np.zeros(3)
        jac = self.joint.jacobian(self.state, q, 0.0)
        # For body j at origin with point A at local (0,0):
        # ∂Φ/∂xⱼ = -1, ∂Φ/∂yⱼ = -1, ∂Φ/∂θⱼ = -B*s = 0
        expected = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        np.testing.assert_array_almost_equal(jac, expected)

    @given(
        x=st.floats(-1.0, 1.0),
        y=st.floats(-1.0, 1.0),
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

    def test_gamma_ground_only_body_j_contributes(self) -> None:
        """Ground has no velocity, so only body_j centripetal term appears."""
        q = np.array([0.0, 0.0, 0.5])
        q_dot = np.array([0.0, 0.0, 3.0])
        gamma = self.joint.gamma(self.state, q, q_dot, 0.0)
        # body_j point A is at local (0,0), so A_j * s_j = (0,0)
        # gamma contribution = -A_j * s_j * θ̇² = (0,0)
        np.testing.assert_array_almost_equal(gamma, [0.0, 0.0])


class TestRevoluteJointGroundWithOffset:
    """Ground joint with offset attachment point on the moving body."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("crank")

        # Ground pivot at (0,0), crank local frame origin connects at crank's "A" point
        # which is at local (0,0). The crank tip "B" is at local (0.05, 0).
        self.joint = make_revolute_joint(
            joint_id="J1",
            body_i_id=GROUND_ID,
            point_i_name="O2",
            point_i_local=np.array([0.0, 0.0]),
            body_j_id="crank",
            point_j_name="A",
            point_j_local=np.array([0.0, 0.0]),
        )

    def test_crank_rotates_around_pivot(self) -> None:
        """Crank pinned at origin should satisfy constraint regardless of angle."""
        for angle in [0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]:
            q = np.array([0.0, 0.0, angle])
            phi = self.joint.constraint(self.state, q, 0.0)
            np.testing.assert_array_almost_equal(phi, [0.0, 0.0])


class TestRevoluteJointWithOffsetPoints:
    """Revolute joint where both bodies have non-origin attachment points."""

    def setup_method(self) -> None:
        self.state = State()
        self.state.register_body("link1")
        self.state.register_body("link2")

        self.joint = make_revolute_joint(
            joint_id="J_offset",
            body_i_id="link1",
            point_i_name="B",
            point_i_local=np.array([0.05, 0.0]),
            body_j_id="link2",
            point_j_name="A",
            point_j_local=np.array([0.0, 0.02]),
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
    def test_jacobian_matches_fd_with_offsets(
        self,
        xi: float,
        yi: float,
        ti: float,
        xj: float,
        yj: float,
        tj: float,
    ) -> None:
        """The most important test: offset points on both bodies, random configs."""
        q = np.array([xi, yi, ti, xj, yj, tj])
        analytical = self.joint.jacobian(self.state, q, 0.0)
        numerical = finite_difference_jacobian(
            self.joint.constraint, self.state, q, 0.0, 2
        )
        np.testing.assert_array_almost_equal(analytical, numerical, decimal=5)
