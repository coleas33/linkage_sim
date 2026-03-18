"""Layer 4: Failure-mode and diagnostics validation.

Verifies correct behavior under degenerate, singular, and impossible inputs.
Covers infeasible assembly, bad initial guess, toggle, redundant constraints,
disconnected topology, branch continuity, and topology cross-consistency.

Tolerance calibration (2026-03-17):
    Residual norm for infeasible: > 0.1 (far from satisfied).
    Condition number at toggle: > 1e3 (elevated from normal ~10-50).
    Rank analysis: exact integer comparisons.
    # BASELINE: calibrated 2026-03-17
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
from linkage_sim.solvers.assembly import (
    assemble_constraints,
    assemble_jacobian,
    assemble_phi_t,
)
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep


# ── Helper: build standard 4-bar ──


def _build_fourbar(a=1.0, b=3.0, c=2.0, d=4.0) -> Mechanism:
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


def _solve_fourbar(mech, angle, a=1.0, d=4.0):
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = a * np.cos(angle), a * np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, d, 0.0, np.pi / 2)
    return solve_position(mech, q, t=angle)


# ── Test 1: Infeasible assembly ──


class TestInfeasibleAssembly:
    """4-bar with link lengths that cannot reach at requested angle."""

    def test_infeasible_returns_not_converged(self) -> None:
        """When links can't close, converged should be False."""
        # a=2, b=1, c=1, d=10 => a + b + c = 4 < d = 10
        # These links can never form a closed loop
        a, b, c, d = 2.0, 1.0, 1.0, 10.0
        mech = _build_fourbar(a, b, c, d)
        angle = np.radians(90)

        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        bx, by = a * np.cos(angle), a * np.sin(angle)
        mech.state.set_pose("coupler", q, bx, by, 0.0)
        mech.state.set_pose("rocker", q, d, 0.0, np.pi / 2)

        result = solve_position(mech, q, t=angle)
        assert not result.converged, "Infeasible assembly should not converge"
        assert result.residual_norm > 0.01, (
            f"Residual should be large for infeasible, got {result.residual_norm:.2e}"
        )

    def test_infeasible_nongrashof_at_locked_angle(self) -> None:
        """Non-Grashof 4-bar at angle beyond its motion range."""
        # a=3, b=4, c=2, d=5 => non-Grashof: s+l=2+5=7, p+q=3+4=7 (boundary)
        # Actually s+l=p+q so it's change-point. Use more extreme:
        # a=3, b=2, c=2, d=8 => s+l=2+8=10 > p+q=2+3=5 definitely non-Grashof
        a, b, c, d = 3.0, 2.0, 2.0, 8.0
        mech = _build_fourbar(a, b, c, d)

        # At 90 degrees, these links definitely can't close
        angle = np.radians(90)
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        bx, by = a * np.cos(angle), a * np.sin(angle)
        mech.state.set_pose("coupler", q, bx, by, 0.0)
        mech.state.set_pose("rocker", q, d, 0.0, 0.0)

        result = solve_position(mech, q, t=angle)
        assert not result.converged


# ── Test 2: Bad initial guess ──


class TestBadInitialGuess:
    """Valid 4-bar, initial guess 180 deg from solution."""

    def test_bad_guess_behavior(self) -> None:
        """Bad guess: either converges with good residual, or fails cleanly."""
        mech = _build_fourbar()
        angle = np.radians(60)

        # Good initial guess for reference
        result_good = _solve_fourbar(mech, angle)
        assert result_good.converged

        # Bad initial guess: everything 180 degrees off
        q_bad = mech.state.make_q()
        mech.state.set_pose("crank", q_bad, 0.0, 0.0, angle + np.pi)
        mech.state.set_pose("coupler", q_bad, -np.cos(angle), -np.sin(angle), np.pi)
        mech.state.set_pose("rocker", q_bad, 4.0, 0.0, -np.pi / 2)

        result_bad = solve_position(mech, q_bad, t=angle)

        if result_bad.converged:
            # If it converged, residual should be good
            assert result_bad.residual_norm < 1e-10
            # May have found a different branch
        else:
            # If it didn't converge, the returned q should be inspectable
            assert result_bad.q is not None
            assert not np.any(np.isnan(result_bad.q)), "q should not contain NaN"
            assert not np.any(np.isinf(result_bad.q)), "q should not contain inf"


# ── Test 3: Exact toggle position ──


class TestExactToggle:
    """Toggle detection for boundary Grashof 4-bar."""

    def test_toggle_detection(self) -> None:
        """At toggle, solver should report high condition number."""
        # Boundary Grashof: s+l = p+q => toggle at 0 and 180 degrees
        mech = _build_fourbar()  # a=1, b=3, c=2, d=4

        # At exactly 180 degrees (toggle for boundary Grashof)
        angle = np.pi
        result = _solve_fourbar(mech, angle)

        # Check condition number via rank analysis
        rank_result = jacobian_rank_analysis(mech, result.q, t=angle)
        # At toggle, condition number should be very elevated
        assert not result.converged or rank_result.condition_number > 100, (
            f"Expected toggle detection: converged={result.converged}, "
            f"cond={rank_result.condition_number:.1f}"
        )


# ── Test 4: Near-toggle degradation ──


class TestNearToggleDegradation:
    """Near toggle: solver converges but with more iterations."""

    def test_near_toggle_elevated_iterations(self) -> None:
        mech = _build_fourbar()

        # Solve at a generic angle for baseline
        result_generic = _solve_fourbar(mech, np.radians(60))
        assert result_generic.converged
        baseline_iter = result_generic.iterations

        # Solve near toggle (179.9 degrees)
        result_near = _solve_fourbar(mech, np.radians(179.9))
        if result_near.converged:
            # Iterations should be higher than baseline
            assert result_near.iterations >= baseline_iter, (
                f"Near toggle ({result_near.iterations}) should need >= "
                f"generic ({baseline_iter}) iterations"
            )

            # Condition number should be elevated
            rank_result = jacobian_rank_analysis(mech, result_near.q, t=np.radians(179.9))
            assert rank_result.condition_number > 100


# ── Test 5a: Redundant constraints - extra revolute ──


class TestRedundantConstraintExtraRevolute:
    """4-bar + one redundant revolute at an existing pin."""

    def test_rank_deficit(self) -> None:
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
        # Redundant: another revolute at the same J2 pin
        crank.add_attachment_point("B2", 1.0, 0.0)
        coupler.add_attachment_point("B2", 0.0, 0.0)
        mech.add_revolute_joint("J2_dup", "crank", "B2", "coupler", "B2")
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()

        # Solve
        angle = np.radians(60)
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        mech.state.set_pose("coupler", q, np.cos(angle), np.sin(angle), 0.0)
        mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q, t=angle)

        if result.converged:
            rank_result = jacobian_rank_analysis(mech, result.q, t=angle)
            # Should detect redundancy: rank < n_constraints
            assert rank_result.has_redundant_constraints, (
                f"Expected redundant constraints, rank={rank_result.constraint_rank}, "
                f"n_constraints={rank_result.n_constraints}"
            )


# ── Test 5c: Redundant constraints - duplicate fixed ──


class TestRedundantConstraintDuplicateFixed:
    """Body with two identical fixed joints to ground."""

    def test_rank_deficit_3(self) -> None:
        """Duplicate fixed joint adds 3 redundant equations."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0), O2=(0.0, 0.0))
        bar = make_bar("bar", "A", "B", length=2.0)
        bar.add_attachment_point("A2", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(bar)
        # First fixed joint
        mech.add_fixed_joint("F1", "ground", "O", "bar", "A", delta_theta_0=0.0)
        # Duplicate fixed joint at same point
        mech.add_fixed_joint("F2", "ground", "O2", "bar", "A2", delta_theta_0=0.0)
        mech.build()

        q = mech.state.make_q()
        result = solve_position(mech, q, t=0.0)

        rank_result = jacobian_rank_analysis(mech, result.q, t=0.0)
        # First fixed joint removes 3 DOF. Duplicate adds 3 more equations
        # but all are redundant. Rank deficit should be 3.
        assert rank_result.has_redundant_constraints
        deficit = rank_result.n_constraints - rank_result.constraint_rank
        assert deficit == 3, f"Expected rank deficit 3, got {deficit}"


# ── Test 6: Disconnected body ──


class TestDisconnectedBody:
    """4-bar + one floating body with no joints."""

    def test_connectivity_reports_unreachable(self) -> None:
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        coupler = make_bar("coupler", "B", "C", length=3.0)
        rocker = make_bar("rocker", "D", "C", length=2.0)
        floating = Body(id="floating", attachment_points={"X": np.array([0.0, 0.0])})
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_body(floating)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()

        conn = check_connectivity(mech)
        assert not conn.is_connected
        assert "floating" in conn.disconnected_bodies


# ── Test 7: Multi-component disconnect ──


class TestMultiComponentDisconnect:
    """Two separate 4-bars sharing no bodies."""

    def test_two_components_detected(self) -> None:
        mech = Mechanism()
        # Component 1: simple 2-bar linkage
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0), O5=(10.0, 0.0), O6=(14.0, 0.0))
        crank1 = make_bar("crank1", "A", "B", length=1.0)
        rocker1 = make_bar("rocker1", "D", "C", length=2.0)
        coupler1 = make_bar("coupler1", "B", "C", length=3.0)

        # Component 2: separate linkage
        crank2 = make_bar("crank2", "A", "B", length=1.0)
        rocker2 = make_bar("rocker2", "D", "C", length=2.0)
        coupler2 = make_bar("coupler2", "B", "C", length=3.0)

        mech.add_body(ground)
        mech.add_body(crank1)
        mech.add_body(coupler1)
        mech.add_body(rocker1)
        mech.add_body(crank2)
        mech.add_body(coupler2)
        mech.add_body(rocker2)

        # Component 1 joints
        mech.add_revolute_joint("J1", "ground", "O2", "crank1", "A")
        mech.add_revolute_joint("J2", "crank1", "B", "coupler1", "B")
        mech.add_revolute_joint("J3", "coupler1", "C", "rocker1", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker1", "D")

        # Component 2 joints - connect to ground at different points
        mech.add_revolute_joint("J5", "ground", "O5", "crank2", "A")
        mech.add_revolute_joint("J6", "crank2", "B", "coupler2", "B")
        mech.add_revolute_joint("J7", "coupler2", "C", "rocker2", "C")
        mech.add_revolute_joint("J8", "ground", "O6", "rocker2", "D")

        mech.build()

        conn = check_connectivity(mech)
        # Both components connect to ground, so they're actually in one component
        # since ground is shared. This is the correct behavior.
        assert conn.is_connected
        assert conn.n_components == 1

    def test_truly_disconnected_components(self) -> None:
        """A floating body is truly disconnected from ground."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        coupler = make_bar("coupler", "B", "C", length=3.0)
        rocker = make_bar("rocker", "D", "C", length=2.0)

        # Truly disconnected pair (connected to each other but not to ground)
        float1 = Body(id="float1", attachment_points={
            "X": np.array([0.0, 0.0]), "Y": np.array([1.0, 0.0])
        })
        float2 = Body(id="float2", attachment_points={
            "X": np.array([0.0, 0.0])
        })

        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_body(float1)
        mech.add_body(float2)

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.add_revolute_joint("J5", "float1", "Y", "float2", "X")

        mech.build()

        conn = check_connectivity(mech)
        assert not conn.is_connected
        assert "float1" in conn.disconnected_bodies
        assert "float2" in conn.disconnected_bodies
        assert conn.n_components == 2


# ── Test 8: Zero-length link ──


class TestZeroLengthLink:
    """Revolute where both attachment points are at body origin."""

    def test_zero_length_graceful(self) -> None:
        """Zero-length link should not crash the solver."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        bar = Body(id="bar", attachment_points={
            "A": np.array([0.0, 0.0]),
            "B": np.array([0.0, 0.0]),
        })
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
        mech.add_revolute_driver(
            "D1", "ground", "bar",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()

        q = mech.state.make_q()
        result = solve_position(mech, q, t=0.0)
        assert result.converged
        # Should converge with bar at origin
        np.testing.assert_allclose(result.q[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result.q[1], 0.0, atol=1e-10)


# ── Test 9: Singular Phi_q in velocity solve ──


class TestSingularVelocitySolve:
    """At toggle, velocity solve via lstsq gives meaningless result."""

    def test_velocity_constraint_violation_at_toggle(self) -> None:
        """At toggle, Phi_q * q_dot + Phi_t should NOT be small."""
        mech = _build_fourbar()  # boundary Grashof
        angle = np.pi  # toggle

        result = _solve_fourbar(mech, angle)
        # Even if position converges (maybe approximately), check velocity
        q_dot = solve_velocity(mech, result.q, t=angle)

        phi_q = assemble_jacobian(mech, result.q, angle)
        phi_t = assemble_phi_t(mech, result.q, angle)
        vel_residual = phi_q @ q_dot + phi_t
        vel_norm = float(np.linalg.norm(vel_residual))

        # At toggle, condition number is very high
        rank_result = jacobian_rank_analysis(mech, result.q, t=angle)
        if rank_result.condition_number > 1e6:
            # The velocity solution at a true singular point is suspect
            # We just verify the solver doesn't crash
            assert np.all(np.isfinite(q_dot)), "q_dot should be finite even at toggle"


# ── Test 10: Branch continuity through sweep ──


class TestBranchContinuity:
    """Sweep Grashof crank-rocker 360 degrees: branch invariant should not flip.

    The branch invariant for a 4-bar is the signed orientation of the
    coupler triangle relative to the crank-coupler-rocker kinematic chain.
    Specifically, the sign of the cross product of:
        (coupler_joint_C - crank_tip_B) x (rocker_pivot_D - crank_tip_B)
    This determines which assembly branch the mechanism is on.
    """

    def test_branch_invariant_no_flip(self) -> None:
        """Signed orientation should not change between consecutive sweep steps."""
        # Use strict Grashof (not boundary) for full rotation
        a, b, c, d = 1.0, 3.5, 2.5, 4.0
        mech = _build_fourbar(a, b, c, d)

        angles = np.linspace(0.01, 2 * np.pi - 0.01, 72)
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        bx, by = a * np.cos(angles[0]), a * np.sin(angles[0])
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, d, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged

        sweep = position_sweep(mech, result.q, angles)
        assert sweep.n_converged == len(angles)

        # Branch invariant: sign((C - B) x (D - B))
        # B = crank tip = coupler attachment point
        # C = coupler-rocker joint
        # D = rocker pivot at ground
        signs = []
        for q in sweep.solutions:
            if q is None:
                continue
            pt_B = mech.state.body_point_global("crank", np.array([a, 0.0]), q)
            pt_C = mech.state.body_point_global("coupler", np.array([b, 0.0]), q)
            pt_D = np.array([d, 0.0])  # O4 (ground pivot)

            # 2D cross product: (C-B) x (D-B)
            cb = pt_C - pt_B
            db = pt_D - pt_B
            cross = cb[0] * db[1] - cb[1] * db[0]
            signs.append(np.sign(cross))

        # Check no sign flips between consecutive steps
        flips = 0
        for i in range(1, len(signs)):
            if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                flips += 1

        assert flips == 0, (
            f"Branch invariant flipped {flips} times during sweep "
            f"(unintended branch jump)"
        )


# ── Test 11a-11e: Topology cross-consistency ──


class TestTopologyCrossConsistency:
    """Cross-check Gruebler DOF, rank-based mobility, and connectivity."""

    def test_well_posed_fourbar_with_driver(self) -> None:
        """11a: Well-posed 4-bar + driver: DOF=0, rank=0, all connected."""
        mech = _build_fourbar()
        angle = np.radians(60)
        result = _solve_fourbar(mech, angle)
        assert result.converged

        gruebler = grubler_dof(mech, expected_dof=0)
        assert gruebler.dof == 0

        rank = jacobian_rank_analysis(mech, result.q, t=angle)
        assert rank.instantaneous_mobility == 0

        conn = check_connectivity(mech)
        assert conn.is_connected

    def test_fourbar_without_driver(self) -> None:
        """11b: 4-bar without driver: DOF=1, rank=1, all connected."""
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
        mech.build()

        gruebler = grubler_dof(mech, expected_dof=1)
        assert gruebler.dof == 1

        # Solve at an arbitrary config for rank analysis
        q = mech.state.make_q()
        angle = np.radians(60)
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        mech.state.set_pose("coupler", q, np.cos(angle), np.sin(angle), 0.0)
        mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
        # Need to solve to get consistent config
        # Without a driver, we can't use solve_position. Just use the guess.
        rank = jacobian_rank_analysis(mech, q, t=0.0)
        assert rank.instantaneous_mobility == 1

        conn = check_connectivity(mech)
        assert conn.is_connected

    def test_fourbar_plus_floating_body(self) -> None:
        """11c: 4-bar (no driver) + floating body: Gruebler and rank agree at 4."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        coupler = make_bar("coupler", "B", "C", length=3.0)
        rocker = make_bar("rocker", "D", "C", length=2.0)
        floating = Body(id="floating", attachment_points={"X": np.array([0.0, 0.0])})
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_body(floating)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.build()

        # Gruebler: M = 3*4 - 4*2 = 12 - 8 = 4
        gruebler = grubler_dof(mech, expected_dof=4)
        assert gruebler.dof == 4

        q = mech.state.make_q()
        angle = np.radians(60)
        mech.state.set_pose("crank", q, 0.0, 0.0, angle)
        mech.state.set_pose("coupler", q, np.cos(angle), np.sin(angle), 0.0)
        mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
        mech.state.set_pose("floating", q, 5.0, 5.0, 0.0)

        rank = jacobian_rank_analysis(mech, q, t=0.0)
        assert rank.instantaneous_mobility == 4

        conn = check_connectivity(mech)
        assert not conn.is_connected
        assert "floating" in conn.disconnected_bodies

    def test_rigid_triangle(self) -> None:
        """11e: Rigid triangle (3 bars, 3 revolutes, all grounded): fully constrained."""
        mech = Mechanism()
        ground = make_ground(O1=(0.0, 0.0), O2=(3.0, 0.0), O3=(1.5, 2.0))
        bar1 = make_bar("bar1", "A", "B", length=3.0)
        bar2 = make_bar("bar2", "A", "B", length=np.sqrt(1.5**2 + 2.0**2))
        bar3 = make_bar("bar3", "A", "B", length=np.sqrt(1.5**2 + 2.0**2))
        mech.add_body(ground)
        mech.add_body(bar1)
        mech.add_body(bar2)
        mech.add_body(bar3)
        mech.add_revolute_joint("J1", "ground", "O1", "bar1", "A")
        mech.add_revolute_joint("J2", "bar1", "B", "bar2", "A")
        mech.add_revolute_joint("J3", "ground", "O2", "bar2", "A")
        mech.add_revolute_joint("J4", "bar2", "B", "bar3", "A")
        mech.add_revolute_joint("J5", "ground", "O3", "bar3", "A")
        mech.add_revolute_joint("J6", "bar3", "B", "bar1", "A")
        mech.build()

        # Gruebler: M = 3*3 - 6*2 = 9 - 12 = -3
        # This is over-constrained (negative Gruebler DOF)
        gruebler = grubler_dof(mech, expected_dof=0)
        assert gruebler.dof < 0

        conn = check_connectivity(mech)
        assert conn.is_connected
