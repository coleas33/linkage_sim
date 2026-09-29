"""Optional modules: skipped when magpylib / matplotlib are not installed."""
import pytest

from magcoupling import DesignInputs, compute_all


def test_fields3d_reproduces_workbook_inputs():
    pytest.importorskip("magpylib")
    from magcoupling import fields3d
    inp = DesignInputs()
    fr = fields3d.run(inp, compute_all(inp))
    d = inp.temperature.demag
    for got, want in ((fr.h_rev_aligned_kA_m, d.h_rev_aligned_kA_m), (fr.h_rev_pullout_kA_m, d.h_rev_pullout_kA_m),
                      (fr.h_rev_likepole_kA_m, d.h_rev_likepole_kA_m), (fr.h_rev_single_ring_kA_m, d.h_rev_single_ring_kA_m)):
        assert abs(got - want) / want < 0.01


def test_clamp_drawing_renders():
    pytest.importorskip("matplotlib")
    from magcoupling.drawing import clamp_layout
    inp = DesignInputs()
    fig = clamp_layout(inp, compute_all(inp))
    assert len(fig.axes) == 2
