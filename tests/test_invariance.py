"""Layer 3a: Deterministic invariance tests.

Tests symmetry, frame independence, rotation matrix identities,
equivalent mechanisms, and force helper validation via virtual work.

Tolerance calibration (2026-03-17):
    Mirror symmetry: 1e-8 (round-trip through negate + solve).
    Frame rotation: 1e-8 (round-trip through rotation transform).
    A/B matrix identities: 1e-10 (exact mathematical identities).
    Force invariance: 1e-8 (round-trip through coordinate rotation).
    Virtual work: 1e-8 (linearization error with small perturbation).
    # BASELINE: calibrated 2026-03-17
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID, State
from linkage_sim.forces.assembly import assemble_Q
from linkage_sim.forces.helpers import (
    body_torque_to_Q,
    gravity_to_Q,
    point_force_to_Q,
)
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)


# ── Test 1: Mirror symmetry ──


class TestMirrorSymmetry:
    """Negate all Y-coordinates; verify all output Y-values are negated.

    This catches sign errors in A(theta), B(theta).
    """

    def test_mirror_fourbar(self) -> None:
        """Build 4-bar and its Y-mirror. Solve both. Assert Y-values negated."""
        angle = np.radians(60)

        # Original
        mech_orig = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        coupler = make_bar("coupler", "B", "C", length=3.0)
        rocker = make_bar("rocker", "D", "C", length=2.0)
        mech_orig.add_body(ground)
        mech_orig.add_body(crank)
        mech_orig.add_body(coupler)
        mech_orig.add_body(rocker)
        mech_orig.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech_orig.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech_orig.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech_orig.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech_orig.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech_orig.build()

        q_orig = mech_orig.state.make_q()
        mech_orig.state.set_pose("crank", q_orig, 0.0, 0.0, angle)
        bx, by = np.cos(angle), np.sin(angle)
        mech_orig.state.set_pose("coupler", q_orig, bx, by, 0.0)
        mech_orig.state.set_pose("rocker", q_orig, 4.0, 0.0, np.pi / 2)
        result_orig = solve_position(mech_orig, q_orig, t=angle)
        assert result_orig.converged
        q_dot_orig = solve_velocity(mech_orig, result_orig.q, t=angle)

        # Mirror: negate all Y, use negative crank angle
        mech_mirror = Mechanism()
        ground_m = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank_m = make_bar("crank", "A", "B", length=1.0)
        coupler_m = make_bar("coupler", "B", "C", length=3.0)
        rocker_m = make_bar("rocker", "D", "C", length=2.0)
        mech_mirror.add_body(ground_m)
        mech_mirror.add_body(crank_m)
        mech_mirror.add_body(coupler_m)
        mech_mirror.add_body(rocker_m)
        mech_mirror.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech_mirror.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech_mirror.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech_mirror.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech_mirror.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech_mirror.build()

        # Mirror angle = -angle
        mirror_angle = -angle
        q_mirror = mech_mirror.state.make_q()
        mech_mirror.state.set_pose("crank", q_mirror, 0.0, 0.0, mirror_angle)
        mech_mirror.state.set_pose("coupler", q_mirror, bx, -by, 0.0)
        mech_mirror.state.set_pose("rocker", q_mirror, 4.0, 0.0, -np.pi / 2)
        result_mirror = solve_position(mech_mirror, q_mirror, t=mirror_angle)
        assert result_mirror.converged
        q_dot_mirror = solve_velocity(mech_mirror, result_mirror.q, t=mirror_angle)

        # Compare: for each body, x should be same, y should be negated,
        # theta should be negated
        for body_id in ["crank", "coupler", "rocker"]:
            idx = mech_orig.state.get_index(body_id)
            # x same
            np.testing.assert_allclose(
                result_mirror.q[idx.x_idx], result_orig.q[idx.x_idx],
                atol=1e-8, err_msg=f"{body_id} x not preserved",
            )
            # y negated
            np.testing.assert_allclose(
                result_mirror.q[idx.y_idx], -result_orig.q[idx.y_idx],
                atol=1e-8, err_msg=f"{body_id} y not negated",
            )
            # theta negated
            np.testing.assert_allclose(
                result_mirror.q[idx.theta_idx], -result_orig.q[idx.theta_idx],
                atol=1e-8, err_msg=f"{body_id} theta not negated",
            )


# ── Test 3: A(theta) orthogonality ──


class TestRotationMatrixOrthogonality:
    """A(theta)^T * A(theta) = I and det(A(theta)) = 1."""

    @pytest.mark.parametrize("angle_deg", list(range(0, 360, 18)))
    def test_orthogonality_and_det(self, angle_deg: int) -> None:
        state = State()
        theta = np.radians(angle_deg)
        A = state.rotation_matrix(theta)

        # A^T * A = I
        product = A.T @ A
        np.testing.assert_allclose(product, np.eye(2), atol=1e-15)

        # det(A) = 1
        det = np.linalg.det(A)
        np.testing.assert_allclose(det, 1.0, atol=1e-15)


# ── Test 4: B(theta) = dA/dtheta ──


class TestRotationMatrixDerivative:
    """B(theta) should match (A(theta+h) - A(theta-h)) / (2h)."""

    @pytest.mark.parametrize("angle_deg", list(range(0, 360, 18)))
    def test_b_matches_fd(self, angle_deg: int) -> None:
        state = State()
        theta = np.radians(angle_deg)
        h = 1e-8

        B = state.rotation_matrix_derivative(theta)
        A_plus = state.rotation_matrix(theta + h)
        A_minus = state.rotation_matrix(theta - h)
        B_fd = (A_plus - A_minus) / (2.0 * h)

        np.testing.assert_allclose(B, B_fd, atol=1e-6)

    @pytest.mark.parametrize("angle_deg", list(range(0, 360, 18)))
    def test_b_identity_dBdtheta_equals_neg_A(self, angle_deg: int) -> None:
        """dB/dtheta = -A. Verify via FD."""
        state = State()
        theta = np.radians(angle_deg)
        h = 1e-8

        A = state.rotation_matrix(theta)
        B_plus = state.rotation_matrix_derivative(theta + h)
        B_minus = state.rotation_matrix_derivative(theta - h)
        dB_dtheta = (B_plus - B_minus) / (2.0 * h)

        np.testing.assert_allclose(dB_dtheta, -A, atol=1e-6)


# ── Test 5: Equivalent mechanism (fixed joint vs revolute + driver) ──


class TestEquivalentMechanism:
    """A fixed joint should produce identical kinematics to a revolute + locked driver."""

    def test_fixed_vs_revolute_plus_driver(self) -> None:
        """Build same mechanism two ways: fixed joint vs revolute + driver.

        Single bar grounded at one end.
        Way 1: Fixed joint ground-to-bar (3 DOF removed)
        Way 2: Revolute joint (2 DOF) + revolute driver locking angle (1 DOF)
        """
        target_angle = np.radians(45)

        # Way 1: Fixed joint
        mech1 = Mechanism()
        g1 = make_ground(O=(0.0, 0.0))
        bar1 = make_bar("bar", "A", "B", length=2.0)
        mech1.add_body(g1)
        mech1.add_body(bar1)
        mech1.add_fixed_joint("F1", "ground", "O", "bar", "A",
                              delta_theta_0=target_angle)
        mech1.build()

        q1 = mech1.state.make_q()
        mech1.state.set_pose("bar", q1, 0.0, 0.0, target_angle)
        result1 = solve_position(mech1, q1, t=0.0)
        assert result1.converged

        # Way 2: Revolute + driver
        mech2 = Mechanism()
        g2 = make_ground(O=(0.0, 0.0))
        bar2 = make_bar("bar", "A", "B", length=2.0)
        mech2.add_body(g2)
        mech2.add_body(bar2)
        mech2.add_revolute_joint("J1", "ground", "O", "bar", "A")
        mech2.add_revolute_driver(
            "D1", "ground", "bar",
            f=lambda t: target_angle,
            f_dot=lambda t: 0.0,
            f_ddot=lambda t: 0.0,
        )
        mech2.build()

        q2 = mech2.state.make_q()
        mech2.state.set_pose("bar", q2, 0.0, 0.0, target_angle)
        result2 = solve_position(mech2, q2, t=0.0)
        assert result2.converged

        # Same position
        np.testing.assert_allclose(result1.q, result2.q, atol=1e-10)

        # Same velocity (both should be zero)
        q_dot1 = solve_velocity(mech1, result1.q, t=0.0)
        q_dot2 = solve_velocity(mech2, result2.q, t=0.0)
        np.testing.assert_allclose(q_dot1, q_dot2, atol=1e-10)


# ── Test 6: Force frame rotation invariance ──


class TestForceFrameRotationInvariance:
    """Rotate both mechanism geometry AND force definitions. Q should transform.

    For each body's (Qx, Qy) pair, apply the 2x2 rotation. Q_theta unchanged.
    """

    def test_point_force_frame_rotation(self) -> None:
        """Point force invariance under global frame rotation."""
        state = State()
        state.register_body("body1")

        alpha = np.radians(37)  # arbitrary non-special angle
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])

        # Original configuration
        q_orig = np.array([1.0, 2.0, 0.5])
        s_local = np.array([0.3, 0.1])
        F_orig = np.array([5.0, -3.0])

        Q_orig = point_force_to_Q(state, "body1", s_local, F_orig, q_orig)

        # Rotated configuration: rotate body position and angle
        pos_rotated = R @ np.array([q_orig[0], q_orig[1]])
        theta_rotated = q_orig[2] + alpha
        q_rotated = np.array([pos_rotated[0], pos_rotated[1], theta_rotated])
        F_rotated = R @ F_orig

        Q_rotated = point_force_to_Q(state, "body1", s_local, F_rotated, q_rotated)

        # Expected: (Qx, Qy) of original rotated by R, Q_theta unchanged
        Qxy_orig = np.array([Q_orig[0], Q_orig[1]])
        Qxy_expected = R @ Qxy_orig
        np.testing.assert_allclose(Q_rotated[0], Qxy_expected[0], atol=1e-8)
        np.testing.assert_allclose(Q_rotated[1], Qxy_expected[1], atol=1e-8)
        np.testing.assert_allclose(Q_rotated[2], Q_orig[2], atol=1e-8)

    def test_gravity_frame_rotation(self) -> None:
        """Gravity force rotation invariance."""
        state = State()
        state.register_body("bar")
        bodies = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=2.0, mass=3.0),
        }

        alpha = np.radians(37)
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])

        q_orig = np.array([1.0, 0.5, 0.3])
        g_orig = np.array([0.0, -9.81])

        Q_orig = gravity_to_Q(state, bodies, q_orig, g_orig)

        # Rotated
        pos_rot = R @ np.array([q_orig[0], q_orig[1]])
        q_rot = np.array([pos_rot[0], pos_rot[1], q_orig[2] + alpha])
        g_rot = R @ g_orig

        Q_rot = gravity_to_Q(state, bodies, q_rot, g_rot)

        Qxy_expected = R @ np.array([Q_orig[0], Q_orig[1]])
        np.testing.assert_allclose(Q_rot[0], Qxy_expected[0], atol=1e-8)
        np.testing.assert_allclose(Q_rot[1], Qxy_expected[1], atol=1e-8)
        np.testing.assert_allclose(Q_rot[2], Q_orig[2], atol=1e-8)


# ── Test 7: Virtual work consistency (point force) ──


class TestVirtualWorkPointForce:
    """F dot delta_r_P = Q dot delta_q for a point force."""

    @pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, 2.0, 3.0])
    def test_virtual_work_point_force(self, theta: float) -> None:
        state = State()
        state.register_body("body1")

        q = np.array([0.5, 0.3, theta])
        s_local = np.array([0.2, 0.1])
        F = np.array([3.0, -2.0])

        Q = point_force_to_Q(state, "body1", s_local, F, q)

        # Small perturbation
        dq = np.array([1e-7, -2e-7, 5e-8])

        # Generalized virtual work
        dW_Q = float(np.dot(Q, dq))

        # Actual virtual work: F dot delta_r_P
        r_P = state.body_point_global("body1", s_local, q)
        r_P_plus = state.body_point_global("body1", s_local, q + dq)
        dr_P = r_P_plus - r_P
        dW_F = float(np.dot(F, dr_P))

        np.testing.assert_allclose(dW_Q, dW_F, rtol=1e-5)


# ── Test 8: Virtual work consistency (gravity) ──


class TestVirtualWorkGravity:
    """Q_gravity dot delta_q = -m*g*delta_h_cg for each body."""

    def test_virtual_work_gravity(self) -> None:
        state = State()
        state.register_body("bar")
        bodies = {
            GROUND_ID: make_ground(),
            "bar": make_bar("bar", "A", "B", length=2.0, mass=3.0),
        }
        g = np.array([0.0, -9.81])
        q = np.array([0.5, 0.3, 0.8])

        Q = gravity_to_Q(state, bodies, q, g)

        # Perturbation
        dq = np.array([1e-7, -2e-7, 5e-8])

        # Q dot dq
        dW_Q = float(np.dot(Q, dq))

        # -m*g*delta_h = m * (-g_y) * delta_y_cg
        cg_local = bodies["bar"].cg_local
        cg_before = state.body_point_global("bar", cg_local, q)
        cg_after = state.body_point_global("bar", cg_local, q + dq)
        delta_h = cg_after[1] - cg_before[1]  # change in height
        dW_gravity = bodies["bar"].mass * g[1] * delta_h  # F_y * delta_y

        # dW_gravity = m * g_y * delta_y (not negated because g_y is already negative)
        # Actually: F_gravity = m * g, so work = F dot dr_cg
        dW_expected = float(np.dot(bodies["bar"].mass * g, cg_after - cg_before))

        np.testing.assert_allclose(dW_Q, dW_expected, rtol=1e-5)


# ── Test 9: Virtual work consistency (torque) ──


class TestVirtualWorkTorque:
    """tau * delta_theta = Q dot delta_q for a pure torque."""

    def test_virtual_work_torque(self) -> None:
        state = State()
        state.register_body("body1")

        tau = 5.0
        Q = body_torque_to_Q(state, "body1", tau)

        # Perturbation: only theta changes
        dq = np.array([0.0, 0.0, 1e-7])

        # Q dot dq should equal tau * delta_theta
        dW_Q = float(np.dot(Q, dq))
        dW_expected = tau * dq[2]

        np.testing.assert_allclose(dW_Q, dW_expected, atol=1e-15)

    def test_virtual_work_torque_with_translation(self) -> None:
        """Pure torque: translational perturbation should not contribute to work."""
        state = State()
        state.register_body("body1")

        tau = 5.0
        Q = body_torque_to_Q(state, "body1", tau)

        # Perturbation: only x,y change (no theta)
        dq = np.array([1e-7, -2e-7, 0.0])
        dW_Q = float(np.dot(Q, dq))
        assert abs(dW_Q) < 1e-15, "Torque should not do work on translations"


# ── Test 10: Force superposition ──


class TestForceSuperposition:
    """F1 alone -> Q1, F2 alone -> Q2, both -> Q12. Assert Q12 = Q1 + Q2."""

    def test_superposition(self) -> None:
        state = State()
        state.register_body("body1")

        q = np.array([1.0, 2.0, 0.5])
        s1 = np.array([0.3, 0.1])
        s2 = np.array([0.5, -0.2])
        F1 = np.array([3.0, -1.0])
        F2 = np.array([-2.0, 4.0])

        Q1 = point_force_to_Q(state, "body1", s1, F1, q)
        Q2 = point_force_to_Q(state, "body1", s2, F2, q)
        Q_both = Q1 + Q2

        # Also verify via assemble with two custom force elements
        class _ConstFE:
            def __init__(self, id_: str, Q_val):
                self._id = id_
                self._Q = Q_val
            @property
            def id(self):
                return self._id
            def evaluate(self, state, q, q_dot, t):
                return self._Q.copy()

        fe1 = _ConstFE("f1", Q1)
        fe2 = _ConstFE("f2", Q2)
        Q_assembled = assemble_Q(state, [fe1, fe2], q, np.zeros(3), 0.0)

        np.testing.assert_allclose(Q_assembled, Q_both, atol=1e-15)


# ── Test 11: Zero-force baseline ──


class TestZeroForceBaseline:
    """No force elements: assemble_Q returns zero vector."""

    def test_zero_force(self) -> None:
        state = State()
        state.register_body("body1")
        state.register_body("body2")

        q = np.zeros(6)
        q_dot = np.zeros(6)
        Q = assemble_Q(state, [], q, q_dot, 0.0)

        np.testing.assert_array_equal(Q, np.zeros(6))

    def test_zero_force_nonzero_state(self) -> None:
        """Zero forces even with nonzero configuration."""
        state = State()
        state.register_body("body1")

        q = np.array([1.0, 2.0, 0.5])
        q_dot = np.array([0.1, -0.2, 0.3])
        Q = assemble_Q(state, [], q, q_dot, 1.0)

        np.testing.assert_array_equal(Q, np.zeros(3))
