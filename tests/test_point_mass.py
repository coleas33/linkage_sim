"""Tests for PointMass element and composite mass recomputation."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar
from linkage_sim.core.point_mass import PointMass, add_point_mass


class TestAddPointMass:
    """Composite mass property recomputation."""

    def test_add_to_zero_mass_body(self) -> None:
        """Adding point mass to zero-mass body: CG = point mass location."""
        body = make_bar("b1", "A", "B", 2.0, mass=0.0, Izz_cg=0.0)
        pm = PointMass(label="tip", mass=3.0, local_position=np.array([1.5, 0.0]))
        add_point_mass(body, pm)

        assert body.mass == pytest.approx(3.0)
        np.testing.assert_allclose(body.cg_local, [1.5, 0.0])
        assert body.Izz_cg == pytest.approx(0.0)  # point mass at its own CG

    def test_add_to_existing_mass(self) -> None:
        """CG shifts toward point mass."""
        body = make_bar("b1", "A", "B", 2.0, mass=2.0, Izz_cg=0.0)
        # Bar CG at (1, 0) by default
        pm = PointMass(label="tip", mass=2.0, local_position=np.array([2.0, 0.0]))
        add_point_mass(body, pm)

        assert body.mass == pytest.approx(4.0)
        # New CG = (2*1 + 2*2) / 4 = 1.5
        np.testing.assert_allclose(body.cg_local, [1.5, 0.0])

    def test_izz_parallel_axis(self) -> None:
        """Moment of inertia uses parallel axis theorem correctly."""
        body = Body(
            id="b1",
            mass=1.0,
            cg_local=np.array([0.0, 0.0]),
            Izz_cg=0.5,
        )
        pm = PointMass(label="tip", mass=2.0, local_position=np.array([1.0, 0.0]))
        add_point_mass(body, pm)

        # New CG = (1*0 + 2*1) / 3 = (2/3, 0)
        expected_cg = np.array([2.0 / 3.0, 0.0])
        np.testing.assert_allclose(body.cg_local, expected_cg, atol=1e-10)

        # Old body: Izz at new CG = 0.5 + 1.0 * (2/3)^2 = 0.5 + 4/9
        d_old = 2.0 / 3.0
        izz_old_shifted = 0.5 + 1.0 * d_old**2

        # Point mass: Izz at new CG = 2.0 * (1 - 2/3)^2 = 2 * (1/3)^2 = 2/9
        d_pm = 1.0 / 3.0
        izz_pm = 2.0 * d_pm**2

        expected_izz = izz_old_shifted + izz_pm
        assert body.Izz_cg == pytest.approx(expected_izz)

    def test_multiple_point_masses(self) -> None:
        """Adding multiple point masses sequentially."""
        body = make_bar("b1", "A", "B", 2.0, mass=1.0, Izz_cg=0.0)
        # CG starts at (1, 0)

        pm1 = PointMass(label="m1", mass=1.0, local_position=np.array([0.0, 0.0]))
        pm2 = PointMass(label="m2", mass=1.0, local_position=np.array([2.0, 0.0]))
        add_point_mass(body, pm1)
        add_point_mass(body, pm2)

        assert body.mass == pytest.approx(3.0)
        # Final CG = (1*1 + 1*0 + 1*2) / 3 = 1.0
        np.testing.assert_allclose(body.cg_local, [1.0, 0.0], atol=1e-10)

    def test_off_axis_point_mass(self) -> None:
        """Point mass off the x-axis shifts CG in both directions."""
        body = Body(
            id="b1",
            mass=1.0,
            cg_local=np.array([0.0, 0.0]),
            Izz_cg=0.0,
        )
        pm = PointMass(label="tip", mass=1.0, local_position=np.array([0.0, 1.0]))
        add_point_mass(body, pm)

        assert body.mass == pytest.approx(2.0)
        np.testing.assert_allclose(body.cg_local, [0.0, 0.5])

    def test_total_mass_correct(self) -> None:
        """Total mass is sum of body + point mass."""
        body = make_bar("b1", "A", "B", 1.0, mass=5.0)
        pm = PointMass(label="tip", mass=3.0, local_position=np.array([0.5, 0.0]))
        add_point_mass(body, pm)
        assert body.mass == pytest.approx(8.0)

    def test_zero_mass_point_mass_no_change(self) -> None:
        """Zero-mass point mass doesn't change body properties."""
        body = make_bar("b1", "A", "B", 2.0, mass=3.0, Izz_cg=1.0)
        original_cg = body.cg_local.copy()
        original_izz = body.Izz_cg

        pm = PointMass(label="zero", mass=0.0, local_position=np.array([1.0, 1.0]))
        add_point_mass(body, pm)

        assert body.mass == pytest.approx(3.0)
        np.testing.assert_array_equal(body.cg_local, original_cg)
        assert body.Izz_cg == pytest.approx(original_izz)

    def test_symmetric_point_masses_cg_at_center(self) -> None:
        """Two symmetric point masses on zero-mass body: CG at midpoint."""
        body = Body(
            id="b1",
            mass=0.0,
            cg_local=np.array([0.0, 0.0]),
            Izz_cg=0.0,
        )
        pm1 = PointMass(label="left", mass=5.0, local_position=np.array([-1.0, 0.0]))
        pm2 = PointMass(label="right", mass=5.0, local_position=np.array([1.0, 0.0]))
        add_point_mass(body, pm1)
        add_point_mass(body, pm2)

        assert body.mass == pytest.approx(10.0)
        np.testing.assert_allclose(body.cg_local, [0.0, 0.0], atol=1e-10)
        # Izz = 2 * 5 * 1^2 = 10 (two point masses at distance 1 from CG)
        assert body.Izz_cg == pytest.approx(10.0)
