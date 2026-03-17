"""Tests for JSON serialization and deserialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.io.serialization import (
    SCHEMA_VERSION,
    dict_to_mechanism,
    load_mechanism,
    mechanism_to_dict,
    save_mechanism,
)


def build_fourbar() -> Mechanism:
    """Standard 4-bar for serialization testing."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(0.038, 0.0))
    crank = make_bar("crank", "A", "B", length=0.010)
    coupler = make_bar("coupler", "B", "C", length=0.040)
    coupler.add_coupler_point("P", 0.020, 0.005)
    rocker = make_bar("rocker", "D", "C", length=0.030)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    mech.build()
    return mech


class TestMechanismToDict:
    def test_schema_version(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        assert data["schema_version"] == SCHEMA_VERSION

    def test_bodies_present(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        assert "ground" in data["bodies"]
        assert "crank" in data["bodies"]
        assert "coupler" in data["bodies"]
        assert "rocker" in data["bodies"]

    def test_attachment_points_serialized(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        crank = data["bodies"]["crank"]
        assert "A" in crank["attachment_points"]
        assert "B" in crank["attachment_points"]
        assert crank["attachment_points"]["A"] == [0.0, 0.0]
        assert crank["attachment_points"]["B"] == [0.010, 0.0]

    def test_coupler_points_serialized(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        coupler = data["bodies"]["coupler"]
        assert "coupler_points" in coupler
        assert coupler["coupler_points"]["P"] == [0.020, 0.005]

    def test_joints_present(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        assert len(data["joints"]) == 4
        assert "J1" in data["joints"]
        assert data["joints"]["J1"]["type"] == "revolute"

    def test_joint_body_references(self) -> None:
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        j1 = data["joints"]["J1"]
        assert j1["body_i"] == "ground"
        assert j1["body_j"] == "crank"
        assert j1["point_i"] == "O2"
        assert j1["point_j"] == "A"

    def test_mass_and_inertia(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        bar = make_bar("bar", "A", "B", length=1.0, mass=2.5, Izz_cg=0.1)
        mech.add_body(ground)
        mech.add_body(bar)
        mech.build()

        data = mechanism_to_dict(mech)
        assert data["bodies"]["bar"]["mass"] == 2.5
        assert data["bodies"]["bar"]["Izz_cg"] == 0.1

    def test_json_serializable(self) -> None:
        """Output dict must be JSON-serializable."""
        mech = build_fourbar()
        data = mechanism_to_dict(mech)
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_fixed_joint_serialized(self) -> None:
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        body = Body(id="welded")
        body.add_attachment_point("A", 0.0, 0.0)
        mech.add_body(ground)
        mech.add_body(body)
        mech.add_fixed_joint("F1", "ground", "O", "welded", "A", delta_theta_0=0.5)
        mech.build()

        data = mechanism_to_dict(mech)
        f1 = data["joints"]["F1"]
        assert f1["type"] == "fixed"
        assert f1["delta_theta_0"] == 0.5


class TestDictToMechanism:
    def test_roundtrip_fourbar(self) -> None:
        """Serialize then deserialize should produce equivalent mechanism."""
        mech_orig = build_fourbar()
        data = mechanism_to_dict(mech_orig)
        mech_loaded = dict_to_mechanism(data)

        assert mech_loaded.state.n_moving_bodies == 3
        assert mech_loaded.n_constraints == 8
        assert set(mech_loaded.bodies.keys()) == {"ground", "crank", "coupler", "rocker"}

    def test_roundtrip_attachment_points(self) -> None:
        mech_orig = build_fourbar()
        data = mechanism_to_dict(mech_orig)
        mech_loaded = dict_to_mechanism(data)

        for body_id in ["crank", "coupler", "rocker"]:
            for pt_name in mech_orig.bodies[body_id].attachment_points:
                orig = mech_orig.bodies[body_id].get_attachment_point(pt_name)
                loaded = mech_loaded.bodies[body_id].get_attachment_point(pt_name)
                np.testing.assert_array_equal(orig, loaded)

    def test_roundtrip_coupler_points(self) -> None:
        mech_orig = build_fourbar()
        data = mechanism_to_dict(mech_orig)
        mech_loaded = dict_to_mechanism(data)

        assert "P" in mech_loaded.bodies["coupler"].coupler_points
        np.testing.assert_array_almost_equal(
            mech_loaded.bodies["coupler"].coupler_points["P"],
            [0.020, 0.005],
        )

    def test_roundtrip_constraint_residuals(self) -> None:
        """Loaded mechanism should produce same constraint residuals."""
        mech_orig = build_fourbar()
        data = mechanism_to_dict(mech_orig)
        mech_loaded = dict_to_mechanism(data)

        q = mech_orig.state.make_q()
        from linkage_sim.solvers.assembly import assemble_constraints

        phi_orig = assemble_constraints(mech_orig, q, 0.0)
        phi_loaded = assemble_constraints(mech_loaded, q, 0.0)
        np.testing.assert_array_almost_equal(phi_orig, phi_loaded)

    def test_unsupported_version_raises(self) -> None:
        data = {"schema_version": "99.0.0", "bodies": {}, "joints": {}}
        with pytest.raises(ValueError, match="Unsupported schema version"):
            dict_to_mechanism(data)

    def test_unknown_joint_type_raises(self) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {"O": [0.0, 0.0]}, "mass": 0.0, "cg_local": [0.0, 0.0], "Izz_cg": 0.0},
                "bar": {"attachment_points": {"A": [0.0, 0.0]}, "mass": 0.0, "cg_local": [0.0, 0.0], "Izz_cg": 0.0},
            },
            "joints": {
                "J1": {"type": "magical", "body_i": "ground", "body_j": "bar"},
            },
        }
        with pytest.raises(ValueError, match="Unknown joint type"):
            dict_to_mechanism(data)

    def test_driver_skipped_on_load(self) -> None:
        """Revolute driver entries should be skipped (not crash)."""
        mech = Mechanism()
        ground = make_ground(O=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=1.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_revolute_joint("J1", "ground", "O", "crank", "A")
        mech.add_constant_speed_driver("D1", "ground", "crank", omega=2.0)
        mech.build()

        data = mechanism_to_dict(mech)
        mech_loaded = dict_to_mechanism(data)

        # Driver was skipped, only revolute joint remains
        assert mech_loaded.n_constraints == 2


class TestFileIO:
    def test_save_and_load(self) -> None:
        mech_orig = build_fourbar()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_mechanism(mech_orig, path)
            assert path.exists()

            mech_loaded = load_mechanism(path)
            assert mech_loaded.state.n_moving_bodies == 3
            assert mech_loaded.n_constraints == 8
        finally:
            path.unlink(missing_ok=True)

    def test_saved_file_is_valid_json(self) -> None:
        mech = build_fourbar()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_mechanism(mech, path)
            with open(path) as f:
                data = json.load(f)
            assert "schema_version" in data
            assert "bodies" in data
            assert "joints" in data
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_file(self) -> None:
        """Full file round-trip preserves mechanism structure."""
        mech_orig = build_fourbar()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_mechanism(mech_orig, path)
            mech_loaded = load_mechanism(path)

            q = mech_orig.state.make_q()
            from linkage_sim.solvers.assembly import assemble_constraints

            phi_orig = assemble_constraints(mech_orig, q, 0.0)
            phi_loaded = assemble_constraints(mech_loaded, q, 0.0)
            np.testing.assert_array_almost_equal(phi_orig, phi_loaded)
        finally:
            path.unlink(missing_ok=True)
