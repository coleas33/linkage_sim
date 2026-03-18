"""Tests for multi-coupler trace support."""
from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import CouplerTrace, precompute_sweep


def _build_fourbar_two_coupler_points() -> tuple[Mechanism, list]:
    """Build a 4-bar with coupler points on TWO different bodies."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        crank.add_coupler_point("CP_crank", 1.0, 0.3)
        coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
        coupler.add_coupler_point("CP_mid", 2.0, 0.0)
        rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
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
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    return mech, [gravity]


class TestMultiCouplerTrace:
    def test_finds_two_traces(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sweep = precompute_sweep(mech, forces, n_steps=36)
        assert len(sweep.coupler_traces) == 2

    def test_trace_has_correct_body_ids(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sweep = precompute_sweep(mech, forces, n_steps=36)
        body_ids = {t.body_id for t in sweep.coupler_traces}
        assert body_ids == {"crank", "coupler"}

    def test_trace_arrays_correct_length(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sweep = precompute_sweep(mech, forces, n_steps=36)
        for trace in sweep.coupler_traces:
            assert len(trace.x) == 36
            assert len(trace.y) == 36

    def test_single_coupler_point_still_works(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mech = Mechanism()
            ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
            crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
            coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
            coupler.add_coupler_point("P", 2.0, 0.0)
            rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
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
            sweep = precompute_sweep(mech, [], n_steps=36)
        assert len(sweep.coupler_traces) == 1
        assert sweep.coupler_traces[0].body_id == "coupler"

    def test_no_coupler_points_empty_list(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mech = Mechanism()
            ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
            crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
            coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
            rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
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
            sweep = precompute_sweep(mech, [], n_steps=36)
        assert len(sweep.coupler_traces) == 0
