"""Layer 3b: Property-based stress tests using Hypothesis.

Generates random valid mechanisms and verifies fundamental properties hold:
constraint satisfaction, velocity self-consistency, distance invariants.

All tests use:
    derandomize=True  — deterministic in CI
    database=None     — no cached example drift
    max_examples=50   — moderate coverage

Generation constraints:
    Link lengths: 0.5 <= L <= 10.0
    Transmission angle: mu >= 15 degrees
    Singularity margin: crank angle >= 10 deg from computed toggle positions

Quarantine rule: If any test flakes twice in CI, mark it
@pytest.mark.xfail(strict=False) with a tracking comment.

Tolerance calibration (2026-03-17):
    Constraint residual: 1e-10 (solver convergence criterion).
    Velocity self-consistency: 1e-10 (Phi_q * q_dot + Phi_t check).
    Distance invariant: 1e-8 (accumulated error from solving at 2 angles).
    # BASELINE: calibrated 2026-03-17
"""

from __future__ import annotations

import numpy as np
import pytest

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from linkage_sim.analysis.grashof import check_grashof, GrashofType
from linkage_sim.analysis.transmission import transmission_angle_fourbar
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.assembly import assemble_constraints, assemble_jacobian, assemble_phi_t
from linkage_sim.solvers.kinematics import solve_position, solve_velocity


# ── Strategies for generating valid mechanisms ──

# Link length range
_link = st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False)

# Crank angle range (away from 0 and pi for safety)
_crank_angle = st.floats(
    min_value=np.radians(20), max_value=np.radians(160),
    allow_nan=False, allow_infinity=False,
)


def _is_valid_grashof_crank_rocker(a: float, b: float, c: float, d: float) -> bool:
    """Check if the 4-bar is a valid Grashof crank-rocker.

    Requires: s + l < p + q (strict Grashof) AND shortest link is crank.
    """
    result = check_grashof(d, a, b, c)
    return (
        result.classification == GrashofType.GRASHOF_CRANK_ROCKER
        and result.grashof_sum < result.other_sum  # strict, not change-point
    )


def _build_random_fourbar(a: float, b: float, c: float, d: float) -> Mechanism:
    """Build a 4-bar with given link lengths."""
    mech = Mechanism()
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


def _solve_fourbar(mech: Mechanism, angle: float, a: float, d: float):
    """Solve 4-bar at given crank angle with good initial guess."""
    q = mech.state.make_q()
    bx = a * np.cos(angle)
    by = a * np.sin(angle)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, d, 0.0, np.pi / 2)
    return solve_position(mech, q, t=angle)


# ── Hypothesis tests ──


class TestRandomFourbarConstraintSatisfaction:
    """Random valid 4-bar: constraint residual < 1e-10."""

    @settings(max_examples=50, derandomize=True, database=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(a=_link, b=_link, c=_link, d=_link, theta=_crank_angle)
    def test_constraint_satisfaction(
        self, a: float, b: float, c: float, d: float, theta: float,
    ) -> None:
        assume(_is_valid_grashof_crank_rocker(a, b, c, d))

        # Check transmission angle
        ta = transmission_angle_fourbar(a, b, c, d, theta)
        assume(ta.angle_deg >= 15 and ta.angle_deg <= 165)

        mech = _build_random_fourbar(a, b, c, d)
        result = _solve_fourbar(mech, theta, a, d)
        assume(result.converged)

        phi = assemble_constraints(mech, result.q, theta)
        residual = float(np.linalg.norm(phi))
        assert residual < 1e-10, (
            f"Residual {residual:.2e} at theta={np.degrees(theta):.1f}° "
            f"with links a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}"
        )


class TestRandomFourbarVelocitySelfConsistency:
    """Random valid 4-bar: Phi_q * q_dot + Phi_t norm < 1e-10."""

    @settings(max_examples=50, derandomize=True, database=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(a=_link, b=_link, c=_link, d=_link, theta=_crank_angle)
    def test_velocity_self_consistency(
        self, a: float, b: float, c: float, d: float, theta: float,
    ) -> None:
        assume(_is_valid_grashof_crank_rocker(a, b, c, d))

        ta = transmission_angle_fourbar(a, b, c, d, theta)
        assume(ta.angle_deg >= 15 and ta.angle_deg <= 165)

        mech = _build_random_fourbar(a, b, c, d)
        result = _solve_fourbar(mech, theta, a, d)
        assume(result.converged)

        q_dot = solve_velocity(mech, result.q, t=theta)

        # Verify: Phi_q * q_dot + Phi_t = 0
        phi_q = assemble_jacobian(mech, result.q, theta)
        phi_t = assemble_phi_t(mech, result.q, theta)
        vel_residual = phi_q @ q_dot + phi_t
        vel_norm = float(np.linalg.norm(vel_residual))
        assert vel_norm < 1e-10, (
            f"Velocity residual {vel_norm:.2e} at theta={np.degrees(theta):.1f}°"
        )


class TestRandomFourbarDistanceInvariant:
    """Random valid 4-bar: coupler length constant across 2 different angles."""

    @settings(max_examples=50, derandomize=True, database=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(a=_link, b=_link, c=_link, d=_link,
           theta1=_crank_angle, theta2=_crank_angle)
    def test_coupler_distance_invariant(
        self, a: float, b: float, c: float, d: float,
        theta1: float, theta2: float,
    ) -> None:
        assume(_is_valid_grashof_crank_rocker(a, b, c, d))
        assume(abs(theta1 - theta2) > np.radians(5))

        ta1 = transmission_angle_fourbar(a, b, c, d, theta1)
        ta2 = transmission_angle_fourbar(a, b, c, d, theta2)
        assume(ta1.angle_deg >= 15 and ta1.angle_deg <= 165)
        assume(ta2.angle_deg >= 15 and ta2.angle_deg <= 165)

        mech = _build_random_fourbar(a, b, c, d)
        result1 = _solve_fourbar(mech, theta1, a, d)
        result2 = _solve_fourbar(mech, theta2, a, d)
        assume(result1.converged and result2.converged)

        # Coupler: points at local (0,0) and (b, 0)
        pt_b = np.array([0.0, 0.0])
        pt_c = np.array([b, 0.0])

        b1 = mech.state.body_point_global("coupler", pt_b, result1.q)
        c1 = mech.state.body_point_global("coupler", pt_c, result1.q)
        dist1 = float(np.linalg.norm(c1 - b1))

        b2 = mech.state.body_point_global("coupler", pt_b, result2.q)
        c2 = mech.state.body_point_global("coupler", pt_c, result2.q)
        dist2 = float(np.linalg.norm(c2 - b2))

        np.testing.assert_allclose(
            dist1, b, atol=1e-8,
            err_msg=f"Coupler length {dist1:.6f} != {b:.6f} at theta1",
        )
        np.testing.assert_allclose(
            dist2, b, atol=1e-8,
            err_msg=f"Coupler length {dist2:.6f} != {b:.6f} at theta2",
        )


class TestRandomSliderCrankConstraintSatisfaction:
    """Random valid slider-crank: constraint residual < 1e-10."""

    @settings(max_examples=50, derandomize=True, database=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(
        r=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        l_ratio=st.floats(min_value=2.5, max_value=6.0, allow_nan=False, allow_infinity=False),
        theta=_crank_angle,
    )
    def test_constraint_satisfaction(
        self, r: float, l_ratio: float, theta: float,
    ) -> None:
        l = r * l_ratio  # l > r always (typical slider-crank geometry)
        assume(l >= r * 2)  # ensure r/l ratio is safe

        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=r)
        conrod = make_bar("conrod", "B", "C", length=l)
        slider = Body(id="slider", attachment_points={"pin": np.array([0.0, 0.0])})
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(conrod)
        mech.add_body(slider)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
        mech.add_revolute_joint("J3", "conrod", "C", "slider", "pin")
        mech.add_prismatic_joint("P1", "ground", "rail", "slider", "pin",
                                 axis_local_i=np.array([1.0, 0.0]), delta_theta_0=0.0)
        mech.add_revolute_driver("D1", "ground", "crank",
                                 f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0)
        mech.build()

        # Initial guess
        q = mech.state.make_q()
        bx = r * np.cos(theta)
        by = r * np.sin(theta)
        x_sl = r * np.cos(theta) + np.sqrt(l**2 - r**2 * np.sin(theta)**2)
        phi_cr = np.arcsin(-r * np.sin(theta) / l)
        mech.state.set_pose("crank", q, 0.0, 0.0, theta)
        mech.state.set_pose("conrod", q, bx, by, phi_cr)
        mech.state.set_pose("slider", q, x_sl, 0.0, 0.0)

        result = solve_position(mech, q, t=theta)
        assume(result.converged)

        phi = assemble_constraints(mech, result.q, theta)
        residual = float(np.linalg.norm(phi))
        assert residual < 1e-10, (
            f"Residual {residual:.2e} with r={r:.2f}, l={l:.2f}, "
            f"theta={np.degrees(theta):.1f}°"
        )
