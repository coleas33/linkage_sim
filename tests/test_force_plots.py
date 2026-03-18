"""Tests for force-related plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from linkage_sim.viz.force_plots import (
    plot_input_torque,
    plot_joint_reactions,
    plot_transmission_angle,
)


class TestForcePlots:
    """Smoke tests for plotting functions."""

    def test_input_torque_plot(self) -> None:
        angles = np.linspace(0, 360, 25)
        torques = np.sin(np.radians(angles)) * 10
        fig = plot_input_torque(angles, torques)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_joint_reactions_plot(self) -> None:
        angles = np.linspace(0, 360, 25)
        reactions = {
            "J1": np.abs(np.sin(np.radians(angles))) * 50,
            "J4": np.abs(np.cos(np.radians(angles))) * 30,
        }
        fig = plot_joint_reactions(angles, reactions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_transmission_angle_plot(self) -> None:
        angles = np.linspace(0, 180, 25)
        mu = 90 - 30 * np.cos(np.radians(angles * 2))
        fig = plot_transmission_angle(angles, mu)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
