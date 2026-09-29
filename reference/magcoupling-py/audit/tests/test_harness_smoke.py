import pytest

from audit.common import defaults, mismatch, rel_err, run, vary


@pytest.mark.family("constants")
def test_harness_runs_engine_defaults():
    """Harness sanity: defaults run and reproduce the README headline pull-out (2.647 N·m at 50 °C)."""
    res = run()
    assert abs(res.model.pullout_Nm - 2.647) < 5e-4, mismatch("pull-out at op temp", "Calculator", res.model.pullout_Nm, 2.647, 5e-4)


@pytest.mark.family("constants")
def test_vary_changes_only_named_input():
    """Harness sanity: vary() returns a changed copy and leaves the original untouched."""
    base = defaults()
    changed = vary(base, {"coupling.npole": 12})
    assert changed.coupling.npole == 12 and base.coupling.npole != 12
    assert rel_err(2.0, 1.0) == 1.0
