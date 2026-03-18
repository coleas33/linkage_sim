"""Result envelopes: peak, RMS, min/max extraction over a sweep.

Given arrays of values computed at each step of a position sweep,
extract summary statistics useful for engineering sizing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnvelopeResult:
    """Summary statistics for a signal over a sweep.

    Attributes:
        values: Raw values at each sweep step.
        angles: Corresponding input angles (radians).
        peak: Maximum absolute value.
        peak_angle: Angle at which peak occurs (radians).
        minimum: Minimum value.
        min_angle: Angle at which minimum occurs.
        maximum: Maximum value.
        max_angle: Angle at which maximum occurs.
        rms: Root mean square over the sweep.
        mean: Arithmetic mean.
        range: max - min.
    """

    values: NDArray[np.float64]
    angles: NDArray[np.float64]
    peak: float
    peak_angle: float
    minimum: float
    min_angle: float
    maximum: float
    max_angle: float
    rms: float
    mean: float
    range: float


def compute_envelope(
    values: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> EnvelopeResult:
    """Compute envelope statistics for a signal over a sweep.

    Args:
        values: Signal values at each sweep step (N,).
        angles: Corresponding input angles in radians (N,).

    Returns:
        EnvelopeResult with all summary statistics.

    Raises:
        ValueError: If arrays are empty or different lengths.
    """
    if values.size == 0:
        raise ValueError("Cannot compute envelope of empty array.")
    if values.shape != angles.shape:
        raise ValueError(
            f"Values and angles must have same shape: "
            f"{values.shape} vs {angles.shape}"
        )

    abs_values = np.abs(values)
    peak_idx = int(np.argmax(abs_values))
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))

    return EnvelopeResult(
        values=values,
        angles=angles,
        peak=float(abs_values[peak_idx]),
        peak_angle=float(angles[peak_idx]),
        minimum=float(values[min_idx]),
        min_angle=float(angles[min_idx]),
        maximum=float(values[max_idx]),
        max_angle=float(angles[max_idx]),
        rms=float(np.sqrt(np.mean(values**2))),
        mean=float(np.mean(values)),
        range=float(np.ptp(values)),
    )
