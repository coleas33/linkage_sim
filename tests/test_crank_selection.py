"""Tests for crank selection analysis."""

from __future__ import annotations

import pytest

from linkage_sim.analysis.crank_selection import (
    CrankRecommendation,
    FourbarTopology,
    detect_fourbar_topology,
    estimate_driven_range,
    recommend_crank_fourbar,
)
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


def _build_fourbar(d: float, a: float, b: float, c: float) -> Mechanism:
    """Build 4-bar: ground=d, crank=a, coupler=b, rocker=c. No driver."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.5)
    coupler = make_bar("coupler", "B", "C", length=b, mass=0.5)
    rocker = make_bar("rocker", "D", "C", length=c, mass=0.5)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    return mech


class TestDetectFourbarTopology:
    """Verify detection of 4-bar topology from mechanism structure."""

    def test_standard_fourbar_detected(self) -> None:
        """A standard 4-bar is detected with correct ground length."""
        mech = _build_fourbar(d=4.0, a=2.0, b=4.0, c=3.0)
        result = detect_fourbar_topology(mech)
        assert result is not None
        assert isinstance(result, FourbarTopology)
        assert result.ground_length == pytest.approx(4.0)

    def test_fourbar_with_driver_detected(self) -> None:
        """Adding a revolute driver does not break 4-bar detection."""
        mech = _build_fourbar(d=4.0, a=2.0, b=4.0, c=3.0)
        mech.add_constant_speed_driver("D1", "ground", "crank", omega=1.0)
        result = detect_fourbar_topology(mech)
        assert result is not None
        assert result.ground_length == pytest.approx(4.0)
        assert result.driven_body_id == "crank"

    def test_sixbar_not_detected(self) -> None:
        """A mechanism with 5+ moving bodies returns None."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0), O6=(8.0, 0.0))
        mech.add_body(ground)
        for i in range(5):
            bar = make_bar(f"bar{i}", "A", "B", length=2.0, mass=0.5)
            mech.add_body(bar)
        result = detect_fourbar_topology(mech)
        assert result is None

    def test_driver_on_rocker_detected(self) -> None:
        """A driver on the rocker link is correctly identified."""
        mech = _build_fourbar(d=4.0, a=2.0, b=4.0, c=3.0)
        mech.add_constant_speed_driver("D1", "ground", "rocker", omega=1.0)
        result = detect_fourbar_topology(mech)
        assert result is not None
        assert result.driven_body_id == "rocker"

    def test_underjointed_not_detected(self) -> None:
        """3 moving bodies with only 3 revolute joints is not a 4-bar."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        coupler = make_bar("coupler", "B", "C", length=4.0, mass=0.5)
        rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.5)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        # Only 3 revolute joints instead of 4
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        # Missing: mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        result = detect_fourbar_topology(mech)
        assert result is None


class TestRecommendCrankFourbar:
    """Verify crank recommendation logic for various 4-bar types."""

    def test_crank_rocker_recommends_shortest(self) -> None:
        """Crank-rocker: shortest link (crank=2) gets 360 full rotation."""
        mech = _build_fourbar(d=4.0, a=2.0, b=4.0, c=3.0)
        recs = recommend_crank_fourbar(mech)
        assert len(recs) > 0
        best = recs[0]
        assert best.full_rotation is True
        assert best.estimated_range_deg == pytest.approx(360.0)
        assert best.body_id == "crank"
        assert best.reason != ""

    def test_double_crank_either_works(self) -> None:
        """Double crank: both ground-adjacent links get 360."""
        mech = _build_fourbar(d=2.0, a=4.0, b=3.5, c=3.0)
        recs = recommend_crank_fourbar(mech)
        full_recs = [r for r in recs if r.full_rotation]
        assert len(full_recs) == 2
        for r in full_recs:
            assert r.estimated_range_deg == pytest.approx(360.0)

    def test_grashof_double_rocker_recommends_best(self) -> None:
        """Grashof double-rocker: neither link gets full rotation."""
        mech = _build_fourbar(d=4.0, a=5.0, b=2.0, c=5.0)
        recs = recommend_crank_fourbar(mech)
        assert len(recs) > 0
        for r in recs:
            assert r.full_rotation is False

    def test_non_grashof_no_full_rotation(self) -> None:
        """Non-Grashof: no full rotation for any link."""
        mech = _build_fourbar(d=5.0, a=3.0, b=4.0, c=7.0)
        recs = recommend_crank_fourbar(mech)
        assert len(recs) > 0
        for r in recs:
            assert r.full_rotation is False
            assert r.estimated_range_deg < 360.0

    def test_chebyshev_recommends_short_link(self) -> None:
        """Chebyshev linkage: crank=2 is shortest, gets 360."""
        mech = _build_fourbar(d=4.0, a=2.0, b=5.0, c=5.0)
        recs = recommend_crank_fourbar(mech)
        assert len(recs) > 0
        best = recs[0]
        assert best.body_id == "crank"
        assert best.full_rotation is True
        assert best.estimated_range_deg == pytest.approx(360.0)
        assert best.reason != ""

    def test_change_point_full_rotation(self) -> None:
        """Change-point (S+L == P+Q): both grounded links get full rotation."""
        mech = _build_fourbar(3.0, 3.0, 3.0, 3.0)
        recs = recommend_crank_fourbar(mech)
        full_recs = [r for r in recs if r.full_rotation]
        assert len(full_recs) == 2

    def test_not_a_fourbar_returns_empty(self) -> None:
        """A mechanism with 1 body returns empty recommendations."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        mech.add_body(ground)
        recs = recommend_crank_fourbar(mech)
        assert recs == []


class TestEstimateDrivenRange:
    """Numerical probing of driven link range."""

    def test_crank_rocker_full_range(self) -> None:
        """Grashof crank-rocker: driven link achieves ~360 deg."""
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        q0 = mech.state.make_q()
        est = estimate_driven_range(mech, q0, n_probes=72)
        assert est >= 355.0  # ~360 with probing tolerance

    def test_non_grashof_limited_range(self) -> None:
        """Non-Grashof: driven link has limited range."""
        mech = _build_fourbar(5.0, 3.0, 4.0, 7.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        q0 = mech.state.make_q()
        est = estimate_driven_range(mech, q0, n_probes=72)
        assert est < 360.0
        assert est > 0.0
