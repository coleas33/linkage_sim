"""Tests for toggle/dead-point detection."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.analysis.toggle import detect_toggle
from linkage_sim.solvers.kinematics import solve_position


def build_fourbar() -> Mechanism:
    """Standard 4-bar: crank=1, coupler=3, rocker=2, ground=4."""
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
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def build_non_grashof() -> Mechanism:
    """Non-Grashof 4-bar that has toggle positions.
    Links: ground=3, crank=2, coupler=4, rocker=3.5
    S=2, L=4, P=3, Q=3.5. S+L=6 > P+Q=6.5? No, 6 < 6.5.
    Hmm, need S+L > P+Q for non-Grashof.
    Links: ground=2, crank=4, coupler=3, rocker=1.5
    S=1.5, L=4, P=2, Q=3. S+L=5.5 > P+Q=5 ✓ Non-Grashof.
    But crank can't rotate fully. Use rocker-rocker range.
    """
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(2.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5)
    coupler = make_bar("coupler", "B", "C", length=3.0)
    rocker = make_bar("rocker", "D", "C", length=4.0)
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


def solve_at(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    if "coupler" in mech.bodies:
        coupler_len = 3.0
        mech.state.set_pose("coupler", q, bx, by, 0.0)
    ground_x = 4.0 if mech.bodies["ground"].attachment_points.get("O4") is not None \
        and float(mech.bodies["ground"].attachment_points["O4"][0]) > 3.0 else 2.0
    mech.state.set_pose("rocker", q, ground_x, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    if not result.converged:
        return result.q  # return anyway for toggle detection
    return result.q


class TestToggleDetection:
    """Toggle/dead-point detection via σ_min monitoring."""

    def test_regular_config_not_toggle(self) -> None:
        """Well-posed config has high σ_min, not near toggle."""
        mech = build_fourbar()
        q = solve_at(mech, np.pi / 4)
        result = detect_toggle(mech, q, t=np.pi / 4)

        assert result.sigma_min > 0.1
        assert not result.is_near_toggle
        assert np.isfinite(result.condition_number)

    def test_sigma_min_positive(self) -> None:
        """σ_min is always non-negative."""
        mech = build_fourbar()
        for angle_deg in range(30, 160, 10):
            angle = np.radians(angle_deg)
            q = solve_at(mech, angle)
            result = detect_toggle(mech, q, t=angle)
            assert result.sigma_min >= 0

    def test_condition_number_finite_away_from_toggle(self) -> None:
        """Condition number is finite at regular configs."""
        mech = build_fourbar()
        q = solve_at(mech, np.pi / 3)
        result = detect_toggle(mech, q, t=np.pi / 3)
        assert np.isfinite(result.condition_number)
        assert result.condition_number > 1.0  # always >= 1

    def test_sigma_min_varies_with_angle(self) -> None:
        """σ_min changes across the sweep."""
        mech = build_fourbar()
        sigma_values = []
        for angle_deg in range(30, 160, 10):
            angle = np.radians(angle_deg)
            q = solve_at(mech, angle)
            result = detect_toggle(mech, q, t=angle)
            sigma_values.append(result.sigma_min)

        # Should vary
        assert np.ptp(sigma_values) > 0.01

    def test_custom_threshold(self) -> None:
        """Custom threshold changes toggle detection."""
        mech = build_fourbar()
        q = solve_at(mech, np.pi / 4)

        r_strict = detect_toggle(mech, q, t=np.pi / 4, threshold=100.0)
        r_loose = detect_toggle(mech, q, t=np.pi / 4, threshold=1e-10)

        assert r_strict.is_near_toggle  # very high threshold → always near toggle
        assert not r_loose.is_near_toggle  # very low threshold → never near toggle
