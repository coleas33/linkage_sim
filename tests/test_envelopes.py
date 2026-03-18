"""Tests for result envelope computation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.envelopes import compute_envelope


class TestComputeEnvelope:
    def test_basic_sinusoid(self) -> None:
        angles = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(angles)
        env = compute_envelope(values, angles)

        assert env.peak == pytest.approx(1.0, abs=0.02)
        assert env.maximum == pytest.approx(1.0, abs=0.02)
        assert env.minimum == pytest.approx(-1.0, abs=0.02)
        assert env.range == pytest.approx(2.0, abs=0.04)
        assert env.rms == pytest.approx(1 / np.sqrt(2), abs=0.02)
        assert env.mean == pytest.approx(0.0, abs=0.02)

    def test_constant_signal(self) -> None:
        angles = np.linspace(0, np.pi, 10)
        values = np.full(10, 5.0)
        env = compute_envelope(values, angles)

        assert env.peak == 5.0
        assert env.rms == 5.0
        assert env.mean == 5.0
        assert env.range == 0.0

    def test_peak_angle_correct(self) -> None:
        angles = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 5.0, 3.0, -2.0])
        env = compute_envelope(values, angles)

        assert env.peak == 5.0
        assert env.peak_angle == 1.0
        assert env.max_angle == 1.0
        assert env.min_angle == 3.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_envelope(np.array([]), np.array([]))

    def test_mismatched_shapes_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            compute_envelope(np.array([1.0, 2.0]), np.array([1.0]))

    def test_single_value(self) -> None:
        env = compute_envelope(np.array([3.0]), np.array([0.0]))
        assert env.peak == 3.0
        assert env.rms == 3.0
        assert env.range == 0.0
