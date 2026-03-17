"""Tests for Grübler DOF count, Jacobian rank, and mechanism validation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.validation import (
    GrublerResult,
    JacobianRankResult,
    grubler_dof,
    jacobian_rank_analysis,
)
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism


def build_fourbar() -> Mechanism:
    """Standard 4-bar: 3 moving bodies, 4 revolute joints -> DOF = 1."""
    mech = Mechanism()

    ground = make_ground(O2=(0.0, 0.0), O4=(0.038, 0.0))
    crank = make_bar("crank", "A", "B", length=0.010)
    coupler = make_bar("coupler", "B", "C", length=0.040)
    rocker = make_bar("rocker", "D", "C", length=0.030)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")

    mech.build()
    return mech


class TestGrublerDOF:
    def test_fourbar_dof_is_one(self) -> None:
        """4-bar linkage: M = 3*3 - 4*2 = 9 - 8 = 1."""
        mech = build_fourbar()
        result = grubler_dof(mech)
        assert result.dof == 1
        assert not result.is_warning

    def test_fourbar_result_fields(self) -> None:
        mech = build_fourbar()
        result = grubler_dof(mech)
        assert result.n_moving_bodies == 3
        assert result.total_dof_removed == 8
        assert result.dof == 1
        assert result.expected_dof == 1

    def test_single_body_single_revolute(self) -> None:
        """One body pinned to ground: M = 3*1 - 1*2 = 1."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        bar = make_bar("bar", "A", "B", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
        mech.build()

        result = grubler_dof(mech)
        assert result.dof == 1
        assert result.n_moving_bodies == 1
        assert result.total_dof_removed == 2

    def test_single_body_fixed_to_ground(self) -> None:
        """One body welded to ground: M = 3*1 - 1*3 = 0."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        body = Body(id="welded")
        body.add_attachment_point("A", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(body)
        mech.add_fixed_joint("F1", "ground", "O", "welded", "A")
        mech.build()

        result = grubler_dof(mech, expected_dof=0)
        assert result.dof == 0
        assert not result.is_warning

    def test_overconstrained_warning(self) -> None:
        """Body with 2 revolutes to ground: M = 3*1 - 2*2 = -1 (overconstrained)."""
        mech = Mechanism()
        ground = make_ground(O1=(0.0, 0.0), O2=(0.1, 0.0))
        bar = make_bar("bar", "A", "B", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O1", "bar", "A")
        mech.add_revolute_joint("J2", "ground", "O2", "bar", "B")
        mech.build()

        result = grubler_dof(mech)
        assert result.dof == -1
        assert result.is_warning

    def test_underconstrained_warning(self) -> None:
        """Two free bodies, one revolute between them: M = 3*2 - 1*2 = 4."""
        mech = Mechanism()
        ground = make_ground()
        bar1 = make_bar("bar1", "A", "B", length=0.1)
        bar2 = make_bar("bar2", "C", "D", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar1)
        mech.add_body(bar2)
        mech.add_revolute_joint("J1", "bar1", "B", "bar2", "C")
        mech.build()

        result = grubler_dof(mech)
        assert result.dof == 4
        assert result.is_warning

    def test_custom_expected_dof(self) -> None:
        """Verify that expected_dof parameter controls the warning flag."""
        mech = build_fourbar()
        result_ok = grubler_dof(mech, expected_dof=1)
        assert not result_ok.is_warning

        result_bad = grubler_dof(mech, expected_dof=2)
        assert result_bad.is_warning

    def test_requires_built_mechanism(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(RuntimeError, match="must be built"):
            grubler_dof(mech)

    def test_no_joints_three_dof_per_body(self) -> None:
        """Two free bodies, no joints: M = 3*2 - 0 = 6."""
        mech = Mechanism()
        ground = make_ground()
        bar1 = make_bar("bar1", "A", "B", length=0.1)
        bar2 = make_bar("bar2", "C", "D", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar1)
        mech.add_body(bar2)
        mech.build()

        result = grubler_dof(mech, expected_dof=6)
        assert result.dof == 6
        assert not result.is_warning

    def test_result_is_frozen(self) -> None:
        """GrublerResult is immutable."""
        result = GrublerResult(
            n_moving_bodies=3,
            total_dof_removed=8,
            dof=1,
            expected_dof=1,
            is_warning=False,
        )
        with pytest.raises(AttributeError):
            result.dof = 2  # type: ignore[misc]

    def test_mixed_joint_types(self) -> None:
        """One revolute + one fixed: M = 3*2 - (2+3) = 1."""
        mech = Mechanism()
        ground = make_ground(O1=(0.0, 0.0), O2=(0.1, 0.0))
        body1 = Body(id="body1")
        body1.add_attachment_point("A", 0.0, 0.0)
        body1.add_attachment_point("B", 0.05, 0.0)
        body2 = Body(id="body2")
        body2.add_attachment_point("C", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(body1)
        mech.add_body(body2)
        mech.add_revolute_joint("J1", "ground", "O1", "body1", "A")
        mech.add_fixed_joint("F1", "ground", "O2", "body2", "C")
        mech.build()

        result = grubler_dof(mech)
        assert result.dof == 1
        assert result.n_moving_bodies == 2
        assert result.total_dof_removed == 5


class TestJacobianRankAnalysis:
    def test_fourbar_at_valid_config(self) -> None:
        """4-bar at a valid config: rank=8, mobility=1."""
        mech = build_fourbar()
        q = mech.state.make_q()
        # Place crank at θ=60°, pointing A at origin, B at (0.005, 0.00866)
        mech.state.set_pose("crank", q, 0.0, 0.0, np.pi / 3)
        # Coupler from B: set a plausible pose
        mech.state.set_pose("coupler", q, 0.005, 0.00866, -0.2)
        # Rocker pivoted at O4=(0.038,0): set near a valid pose
        mech.state.set_pose("rocker", q, 0.038, 0.0, np.pi / 2)

        result = jacobian_rank_analysis(mech, q)
        assert result.constraint_rank == 8
        assert result.instantaneous_mobility == 1
        assert result.n_constraints == 8
        assert result.n_coords == 9
        assert not result.has_redundant_constraints

    def test_fourbar_singular_values_shape(self) -> None:
        mech = build_fourbar()
        q = mech.state.make_q()
        result = jacobian_rank_analysis(mech, q)
        # min(m, n) = min(8, 9) = 8 singular values
        assert result.singular_values.shape == (8,)

    def test_fourbar_condition_number_finite(self) -> None:
        """At a generic configuration, condition number should be finite."""
        mech = build_fourbar()
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, 0.5)
        mech.state.set_pose("coupler", q, 0.005, 0.004, -0.1)
        mech.state.set_pose("rocker", q, 0.038, 0.0, 1.2)

        result = jacobian_rank_analysis(mech, q)
        assert np.isfinite(result.condition_number)
        assert result.condition_number >= 1.0

    def test_fixed_body_zero_mobility(self) -> None:
        """Body fixed to ground: rank=3, mobility=0."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        body = Body(id="welded")
        body.add_attachment_point("A", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(body)
        mech.add_fixed_joint("F1", "ground", "O", "welded", "A")
        mech.build()

        q = mech.state.make_q()
        result = jacobian_rank_analysis(mech, q)
        assert result.constraint_rank == 3
        assert result.instantaneous_mobility == 0

    def test_overconstrained_has_redundant(self) -> None:
        """Body with 2 revolutes to ground at correct spacing: redundant constraints."""
        mech = Mechanism()
        ground = make_ground(O1=(0.0, 0.0), O2=(0.1, 0.0))
        bar = make_bar("bar", "A", "B", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar)
        mech.add_revolute_joint("J1", "ground", "O1", "bar", "A")
        mech.add_revolute_joint("J2", "ground", "O2", "bar", "B")
        mech.build()

        # Place bar in the valid config: A at (0,0), B at (0.1,0), θ=0
        q = mech.state.make_q()
        mech.state.set_pose("bar", q, 0.0, 0.0, 0.0)

        result = jacobian_rank_analysis(mech, q)
        # 4 constraint eqs, but only 3 coords -> rank <= 3
        assert result.n_constraints == 4
        assert result.n_coords == 3
        assert result.has_redundant_constraints
        assert result.constraint_rank <= 3

    def test_underconstrained_high_mobility(self) -> None:
        """Two free bodies, one revolute: mobility = 4."""
        mech = Mechanism()
        ground = make_ground()
        bar1 = make_bar("bar1", "A", "B", length=0.1)
        bar2 = make_bar("bar2", "C", "D", length=0.1)
        mech.add_body(ground)
        mech.add_body(bar1)
        mech.add_body(bar2)
        mech.add_revolute_joint("J1", "bar1", "B", "bar2", "C")
        mech.build()

        q = mech.state.make_q()
        result = jacobian_rank_analysis(mech, q)
        assert result.n_coords == 6
        assert result.constraint_rank == 2
        assert result.instantaneous_mobility == 4
        assert result.grubler_agrees

    def test_grubler_agrees_fourbar(self) -> None:
        mech = build_fourbar()
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, 0.5)
        mech.state.set_pose("coupler", q, 0.005, 0.004, -0.1)
        mech.state.set_pose("rocker", q, 0.038, 0.0, 1.2)

        result = jacobian_rank_analysis(mech, q)
        assert result.grubler_agrees

    def test_requires_built_mechanism(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(RuntimeError, match="must be built"):
            jacobian_rank_analysis(mech, np.zeros(0))

    def test_singular_values_descending(self) -> None:
        mech = build_fourbar()
        q = np.random.default_rng(42).uniform(-1, 1, size=9)
        result = jacobian_rank_analysis(mech, q)
        # SVD returns singular values in descending order
        for i in range(len(result.singular_values) - 1):
            assert result.singular_values[i] >= result.singular_values[i + 1]

    def test_custom_rank_tolerance(self) -> None:
        """Very large rank_tol should reduce apparent rank."""
        mech = build_fourbar()
        q = mech.state.make_q()
        mech.state.set_pose("crank", q, 0.0, 0.0, 0.5)
        mech.state.set_pose("coupler", q, 0.005, 0.004, -0.1)
        mech.state.set_pose("rocker", q, 0.038, 0.0, 1.2)

        result_default = jacobian_rank_analysis(mech, q)
        result_strict = jacobian_rank_analysis(mech, q, rank_tol=1e30)

        assert result_default.constraint_rank > result_strict.constraint_rank
        assert result_strict.constraint_rank == 0

    def test_result_is_frozen(self) -> None:
        mech = build_fourbar()
        q = mech.state.make_q()
        result = jacobian_rank_analysis(mech, q)
        with pytest.raises(AttributeError):
            result.constraint_rank = 99  # type: ignore[misc]
