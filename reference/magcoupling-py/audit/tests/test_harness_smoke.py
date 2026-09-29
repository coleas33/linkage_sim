import subprocess
import sys
from pathlib import Path

import pytest

from audit.common import defaults, mismatch, rel_err, run, vary

CONFTEST = Path(__file__).with_name("conftest.py")
MARKED_SOURCE = 'import pytest\n\n\n@pytest.mark.family("torque")\ndef test_marked():\n    pass\n'
UNMARKED_SOURCE = "def test_unmarked():\n    pass\n"
UNKNOWN_FAMILY_SOURCE = 'import pytest\n\n\n@pytest.mark.family("nonsense")\ndef test_unmarked():\n    pass\n'


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


def _pytest_in_scratch_tree(tmp_path, audit_test_source, outside_test_source):
    """Run pytest over ``audit/tests`` (with a copy of the audit conftest) and a sibling ``tests`` dir in one session.

    A fresh interpreter, so the copy of the conftest is the only one in play and the rule is exercised by a
    real collection. The copy computes its audit root from its own location, so ``audit/tests`` inside
    ``tmp_path`` counts as the audit tree and ``tmp_path/tests`` does not.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")  # pins the rootdir here
    audit_tests = tmp_path / "audit" / "tests"
    audit_tests.mkdir(parents=True)
    (audit_tests / "conftest.py").write_text(CONFTEST.read_text(encoding="utf-8"), encoding="utf-8")
    (audit_tests / "test_audit_case.py").write_text(audit_test_source, encoding="utf-8")
    outside = tmp_path / "tests"
    outside.mkdir()
    (outside / "test_outside_case.py").write_text(outside_test_source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "audit/tests"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )


@pytest.mark.family("constants")
@pytest.mark.parametrize("audit_source", [UNMARKED_SOURCE, UNKNOWN_FAMILY_SOURCE], ids=["no-marker", "unknown-family"])
def test_conftest_rejects_audit_item_without_valid_family(tmp_path, audit_source):
    """Harness sanity: an audit check without a valid family marker aborts the session, even when unmarked vendored-style tests are collected with it."""
    proc = _pytest_in_scratch_tree(tmp_path, audit_source, UNMARKED_SOURCE)
    output = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "every audit check needs" in output, output
    assert "audit/tests/test_audit_case.py::test_unmarked" in output.replace("\\", "/"), output


@pytest.mark.family("constants")
def test_conftest_skips_family_rule_for_items_outside_audit_tree(tmp_path):
    """Harness sanity: unmarked items outside the audit tree (the vendored tests/) are not rejected and run in the same session."""
    proc = _pytest_in_scratch_tree(tmp_path, MARKED_SOURCE, UNMARKED_SOURCE)
    output = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.OK, output
    assert "2 passed" in output, output
