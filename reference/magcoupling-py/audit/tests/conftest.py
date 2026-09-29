"""Audit test plumbing: the ``family`` marker and a machine-readable results file.

Every check is marked ``@pytest.mark.family("<name>")`` with one of FAMILIES.
After a run, ``audit/out/results.json`` lists every check with its family,
outcome and failure message; the findings report is built from that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FAMILIES = ("torque", "geometry", "metal", "materials", "temperature", "thermal", "clamps", "sweeps", "constants")
_RESULTS: list[dict] = []


def pytest_configure(config):
    config.addinivalue_line("markers", "family(name): formula family this audit check belongs to")


def pytest_collection_modifyitems(items):
    for item in items:
        m = item.get_closest_marker("family")
        if m is None or not m.args or m.args[0] not in FAMILIES:
            raise pytest.UsageError(f"{item.nodeid}: every audit check needs @pytest.mark.family(<one of {FAMILIES}>)")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        _RESULTS.append({
            "id": item.nodeid,
            "family": item.get_closest_marker("family").args[0],
            "outcome": rep.outcome,
            "message": (str(rep.longrepr.reprcrash.message) if rep.failed and hasattr(rep.longrepr, "reprcrash") else ""),
            "doc": (item.function.__doc__ or "").strip(),
        })


def pytest_sessionfinish(session, exitstatus):
    out = Path(__file__).resolve().parents[1] / "out"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(_RESULTS, indent=1), encoding="utf-8")
