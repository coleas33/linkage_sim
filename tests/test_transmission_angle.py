"""Tests for transmission angle computation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.transmission import transmission_angle_fourbar


class TestTransmissionAngleFourbar:
    """Analytical 4-bar transmission angle."""

    def test_known_geometry_at_zero(self) -> None:
        """At θ=0 for crank=1, coupler=3, rocker=2, ground=4:
        cos μ = (9 + 4 - 1 - 16 + 8) / (12) = 4/12 = 1/3
        μ = arccos(1/3) ≈ 70.53°
        """
        result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=0.0)
        expected = np.degrees(np.arccos(1.0 / 3.0))
        assert result.angle_deg == pytest.approx(expected, abs=1e-6)

    def test_ideal_at_90_deg(self) -> None:
        """Find a configuration where μ ≈ 90° (cos μ ≈ 0).
        cos μ = (b²+c²-a²-d²+2ad·cosθ) / (2bc)
        For μ=90°: 2ad·cosθ = a²+d²-b²-c²
        With a=1,b=3,c=2,d=4: cosθ = (1+16-9-4)/8 = 4/8 = 0.5 → θ=60°
        """
        result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=np.pi / 3)
        assert result.angle_deg == pytest.approx(90.0, abs=1e-6)
        assert result.deviation_from_ideal == pytest.approx(0.0, abs=1e-6)
        assert not result.is_poor

    def test_poor_transmission(self) -> None:
        """Near toggle: transmission angle should be flagged as poor."""
        # At θ near 180°, the 4-bar approaches toggle
        result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=np.pi)
        # cos μ = (9+4-1-16-8)/12 = -12/12 = -1 → μ=180°
        # But this is likely infeasible. Test with θ close to toggle.
        # At θ = 170° (close to π):
        result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=np.radians(170))
        # Should have large deviation
        assert result.deviation_from_ideal > 40.0

    def test_transmission_angle_range(self) -> None:
        """Transmission angle stays in [0, 180] degrees."""
        for theta_deg in range(0, 360, 15):
            theta = np.radians(theta_deg)
            result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=theta)
            assert 0.0 <= result.angle_deg <= 180.0

    def test_symmetry(self) -> None:
        """cos μ(θ) = cos μ(-θ) since formula uses cos(θ)."""
        r_pos = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=np.pi / 4)
        r_neg = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=-np.pi / 4)
        assert r_pos.angle_deg == pytest.approx(r_neg.angle_deg, abs=1e-10)

    def test_continuous_across_sweep(self) -> None:
        """Transmission angle varies smoothly (no jumps)."""
        angles = np.linspace(0, np.pi, 50)
        mu_values = []
        for theta in angles:
            result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=theta)
            mu_values.append(result.angle_deg)

        # Check no jumps > 20 degrees between adjacent steps
        diffs = np.abs(np.diff(mu_values))
        assert np.max(diffs) < 20.0

    def test_is_poor_flag(self) -> None:
        """is_poor is True when deviation > 50°."""
        # At θ=0: μ ≈ 70.5° → deviation ≈ 19.5° → not poor
        result = transmission_angle_fourbar(a=1.0, b=3.0, c=2.0, d=4.0, theta=0.0)
        assert not result.is_poor

    def test_equal_links(self) -> None:
        """Parallelogram: all links equal, μ should be defined."""
        result = transmission_angle_fourbar(a=2.0, b=2.0, c=2.0, d=2.0, theta=np.pi / 3)
        assert 0.0 <= result.angle_deg <= 180.0


class TestTransmissionAngleFormula:
    """Verify formula correctness against hand calculations."""

    def test_formula_matches_law_of_cosines(self) -> None:
        """At θ, the diagonal p² = a² + d² - 2ad·cos(π-θ) = a² + d² + 2ad·cosθ.
        Then by law of cosines on triangle (b, c, p):
            cos μ = (b² + c² - p²) / (2bc)
            = (b² + c² - a² - d² - 2ad·cosθ) / (2bc)

        Wait — the standard formula uses:
            cos μ = (b² + c² - a² - d² + 2ad·cosθ) / (2bc)

        This is for the diagonal from crank tip to rocker ground pivot.
        Let's verify with specific numbers.
        """
        # a=1, b=3, c=2, d=4, θ=π/4
        theta = np.pi / 4
        a, b, c, d = 1.0, 3.0, 2.0, 4.0

        # Compute via formula
        result = transmission_angle_fourbar(a, b, c, d, theta)

        # Compute via diagonal + law of cosines
        # Diagonal from O2 to O4: this is d itself
        # Diagonal from crank tip B to rocker ground pivot O4:
        # B at (a·cosθ, a·sinθ), O4 at (d, 0)
        bx = a * np.cos(theta)
        by = a * np.sin(theta)
        p_sq = (bx - d) ** 2 + by**2

        # Law of cosines for triangle B-C-O4 (sides b, c, p)
        cos_mu = (b**2 + c**2 - p_sq) / (2 * b * c)
        mu_expected = np.degrees(np.arccos(np.clip(cos_mu, -1, 1)))

        assert result.angle_deg == pytest.approx(mu_expected, abs=1e-8)
