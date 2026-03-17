"""Tests for Grübler DOF count and mechanism validation."""

from __future__ import annotations

import pytest

from linkage_sim.analysis.validation import GrublerResult, grubler_dof
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
