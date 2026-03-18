"""Tests for Grashof condition analysis."""

from __future__ import annotations

import pytest

from linkage_sim.analysis.grashof import GrashofType, check_grashof


class TestGrashofClassification:
    """Verify correct 4-bar classification."""

    def test_crank_rocker(self) -> None:
        """Shortest link is crank → crank-rocker."""
        # ground=4, crank=1, coupler=3, rocker=2
        # S=1, L=4, P=2, Q=3. S+L=5 = P+Q=5 → change point
        # Use slightly different to get strict Grashof
        result = check_grashof(4.0, 1.0, 3.0, 2.5)
        assert result.is_grashof
        assert not result.is_change_point
        assert result.classification == GrashofType.GRASHOF_CRANK_ROCKER

    def test_double_crank(self) -> None:
        """Shortest link is ground → double-crank (drag link)."""
        # ground=1, crank=2, coupler=3, rocker=2.5
        # S=1, L=3, P=2, Q=2.5. S+L=4 < P+Q=4.5
        result = check_grashof(1.0, 2.0, 3.0, 2.5)
        assert result.is_grashof
        assert result.classification == GrashofType.GRASHOF_DOUBLE_CRANK
        assert result.shortest_is == "ground"

    def test_double_rocker_grashof(self) -> None:
        """Shortest link is coupler → Grashof double-rocker."""
        # ground=3, crank=2.5, coupler=1, rocker=2
        # S=1, L=3, P=2, Q=2.5. S+L=4 < P+Q=4.5
        result = check_grashof(3.0, 2.5, 1.0, 2.0)
        assert result.is_grashof
        assert result.classification == GrashofType.GRASHOF_DOUBLE_ROCKER
        assert result.shortest_is == "coupler"

    def test_non_grashof(self) -> None:
        """S + L > P + Q → non-Grashof (triple rocker)."""
        # ground=2, crank=4, coupler=3, rocker=1.5
        # S=1.5, L=4, P=2, Q=3. S+L=5.5 > P+Q=5
        result = check_grashof(2.0, 4.0, 3.0, 1.5)
        assert not result.is_grashof
        assert result.classification == GrashofType.NON_GRASHOF

    def test_change_point(self) -> None:
        """S + L = P + Q → change point."""
        # ground=4, crank=1, coupler=3, rocker=2
        # S=1, L=4, P=2, Q=3. S+L=5 = P+Q=5
        result = check_grashof(4.0, 1.0, 3.0, 2.0)
        assert result.is_change_point
        assert result.is_grashof  # change point is a Grashof boundary
        assert result.classification == GrashofType.CHANGE_POINT

    def test_all_equal_links(self) -> None:
        """All links equal → change point (parallelogram or deltoid)."""
        result = check_grashof(2.0, 2.0, 2.0, 2.0)
        assert result.is_change_point
        assert result.classification == GrashofType.CHANGE_POINT

    def test_crank_rocker_rocker_is_shortest(self) -> None:
        """Shortest is rocker → also crank-rocker."""
        # ground=3, crank=2, coupler=2.5, rocker=1
        # S=1, L=3, P=2, Q=2.5. S+L=4 < P+Q=4.5
        result = check_grashof(3.0, 2.0, 2.5, 1.0)
        assert result.is_grashof
        assert result.classification == GrashofType.GRASHOF_CRANK_ROCKER
        assert result.shortest_is == "rocker"


class TestGrashofValues:
    """Verify computed values."""

    def test_sum_values(self) -> None:
        result = check_grashof(4.0, 1.0, 3.0, 2.5)
        assert result.shortest == 1.0
        assert result.longest == 4.0
        assert result.grashof_sum == pytest.approx(5.0)
        assert result.other_sum == pytest.approx(5.5)

    def test_link_lengths_stored(self) -> None:
        result = check_grashof(4.0, 1.0, 3.0, 2.0)
        assert result.link_lengths == (4.0, 1.0, 3.0, 2.0)
