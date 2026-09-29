"""API behaviour a GUI relies on."""
import json

from magcoupling import DesignInputs, compute_all, headline, input_schema, result_schema, set_input, to_dict


def test_input_schema_has_units_labels_and_cells():
    rows = input_schema()
    assert len(rows) > 150
    npole = next(r for r in rows if r["path"] == "coupling.npole")
    assert npole["unit"] == "-" and npole["cell"] == "Calculator!C5" and npole["value"] == 10
    gap = next(r for r in rows if r["path"] == "metal.face_gap_mm")
    assert gap["unit"] == "mm" and gap["label"]


def test_set_input_is_non_destructive():
    base = DesignInputs()
    new = set_input(base, "coupling.npole", 12)
    assert base.coupling.npole == 10 and new.coupling.npole == 12


def test_changing_an_input_changes_results():
    base = compute_all()
    hotter = compute_all(set_input(DesignInputs(), "coupling.op_temp_C", 80))
    assert hotter.model.pullout_Nm < base.model.pullout_Nm
    wider = compute_all(set_input(DesignInputs(), "metal.face_gap_mm", 2.0))
    assert wider.model.pullout_Nm < base.model.pullout_Nm


def test_results_serialize_to_json():
    res = compute_all()
    text = json.dumps(to_dict(res), default=str)
    assert "pullout_Nm" in text
    assert any(r["cell"] == "Calculator!C93" for r in result_schema(res))


def test_headline_keys():
    h = headline(compute_all())
    assert {"pullout_at_op_temp_Nm", "temperature_verdict", "clamp_screw"} <= set(h)


def test_unknown_part_falls_back_to_manual_magnet():
    inp = set_input(DesignInputs(), "coupling.magnets.part_inner", "")
    res = compute_all(inp)
    assert res.model.inner_tmax_C == "n/a" and res.model.inner_temp_check == "unknown"


def test_no_screw_fits_is_reported_not_raised():
    inp = set_input(set_input(DesignInputs(), "clamps.boss_od_mm", 22), "clamps.clamp_length_mm", 10)
    res = compute_all(inp)
    assert res.clamps.index == 0 and res.clamps.recommended.startswith("None")
