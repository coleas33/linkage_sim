"""Tests for body data structures."""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.state import GROUND_ID


class TestBody:
    def test_get_attachment_point(self) -> None:
        body = Body(
            id="link1",
            attachment_points={"A": np.array([0.0, 0.0]), "B": np.array([0.1, 0.0])},
        )
        pt = body.get_attachment_point("A")
        np.testing.assert_array_equal(pt, [0.0, 0.0])

    def test_get_attachment_point_missing_raises(self) -> None:
        body = Body(id="link1")
        with pytest.raises(KeyError, match="not found"):
            body.get_attachment_point("Z")

    def test_add_attachment_point(self) -> None:
        body = Body(id="link1")
        body.add_attachment_point("A", 0.0, 0.0)
        body.add_attachment_point("B", 0.05, 0.0)
        np.testing.assert_array_equal(body.get_attachment_point("A"), [0.0, 0.0])
        np.testing.assert_array_equal(body.get_attachment_point("B"), [0.05, 0.0])

    def test_add_duplicate_attachment_raises(self) -> None:
        body = Body(id="link1")
        body.add_attachment_point("A", 0.0, 0.0)
        with pytest.raises(ValueError, match="already exists"):
            body.add_attachment_point("A", 1.0, 1.0)

    def test_add_coupler_point(self) -> None:
        body = Body(id="link1")
        body.add_coupler_point("P", 0.03, 0.02)
        np.testing.assert_array_equal(body.coupler_points["P"], [0.03, 0.02])

    def test_add_duplicate_coupler_raises(self) -> None:
        body = Body(id="link1")
        body.add_coupler_point("P", 0.03, 0.02)
        with pytest.raises(ValueError, match="already exists"):
            body.add_coupler_point("P", 0.0, 0.0)


class TestMakeGround:
    def test_ground_id(self) -> None:
        ground = make_ground(O2=(0.0, 0.0))
        assert ground.id == GROUND_ID

    def test_ground_zero_mass(self) -> None:
        ground = make_ground(O2=(0.0, 0.0))
        assert ground.mass == 0.0
        assert ground.Izz_cg == 0.0

    def test_ground_attachment_points(self) -> None:
        ground = make_ground(O2=(0.0, 0.0), O4=(0.038, 0.0))
        np.testing.assert_array_equal(
            ground.get_attachment_point("O2"), [0.0, 0.0]
        )
        np.testing.assert_array_equal(
            ground.get_attachment_point("O4"), [0.038, 0.0]
        )


class TestMakeBar:
    def test_bar_two_points(self) -> None:
        bar = make_bar("crank", "A", "B", length=0.010)
        np.testing.assert_array_equal(bar.get_attachment_point("A"), [0.0, 0.0])
        np.testing.assert_array_almost_equal(
            bar.get_attachment_point("B"), [0.010, 0.0]
        )

    def test_bar_cg_at_midpoint(self) -> None:
        bar = make_bar("crank", "A", "B", length=0.1)
        np.testing.assert_array_almost_equal(bar.cg_local, [0.05, 0.0])

    def test_bar_mass_properties(self) -> None:
        bar = make_bar("crank", "A", "B", length=0.1, mass=0.5, Izz_cg=1e-4)
        assert bar.mass == 0.5
        assert bar.Izz_cg == 1e-4

    def test_bar_id(self) -> None:
        bar = make_bar("my_link", "P1", "P2", length=0.05)
        assert bar.id == "my_link"

    def test_bar_default_zero_mass(self) -> None:
        bar = make_bar("link", "A", "B", length=0.1)
        assert bar.mass == 0.0
        assert bar.Izz_cg == 0.0


class TestTernaryBody:
    """Verify that bodies with 3+ attachment points work correctly."""

    def test_ternary_plate(self) -> None:
        plate = Body(
            id="ternary",
            attachment_points={
                "A": np.array([0.0, 0.0]),
                "B": np.array([0.05, 0.0]),
                "C": np.array([0.025, 0.03]),
            },
            mass=0.1,
            cg_local=np.array([0.025, 0.01]),
            Izz_cg=2e-5,
        )
        assert len(plate.attachment_points) == 3
        np.testing.assert_array_equal(plate.get_attachment_point("C"), [0.025, 0.03])
