"""Shared helpers for every audit check. Keep this file small and dependency-free."""
from __future__ import annotations

import math
from typing import Any

from magcoupling import DesignInputs, compute_all, set_input
from magcoupling.api import DesignResults

#: Exact vacuum permeability (the workbook uses the rounded 1.256637e-6).
MU0_EXACT = 4e-7 * math.pi

#: Tolerance for checks that re-derive the same closed-form algebra.
TOL_ALGEBRA = 1e-9

# Engine analytical model vs an exact or independent reference. 2 %: the default design misses the hot-torque
# requirement by ~10 %, so a systematic model bias of a few percent is material and must surface as a candidate
# finding.
TOL_MODEL = 0.02

#: Peak-angle check: torque at the engine's pull-out angle (a quarter pole-pair pitch) vs the true maximum of the
#: torque-angle curve. The 3D curve (test_torque_laws.py) is converged to ~1e-7 and its maximum is located to 1e-4 of
#: a pole pitch; the exact 2D curve (test_torque_field2d.py) is exact to < 1e-3 (BEM) and its maximum is located to
#: 1e-7 rad. So 1e-3 is a resolution bound.
TOL_PEAK = 1e-3


def defaults() -> DesignInputs:
    """Workbook-default inputs (a fresh object every call)."""
    return DesignInputs()


def run(inp: DesignInputs | None = None) -> DesignResults:
    """Run the engine; defaults when ``inp`` is None."""
    return compute_all(inp or defaults())


def vary(inp: DesignInputs, changes: dict[str, Any]) -> DesignInputs:
    """Return a copy of ``inp`` with dotted-path inputs changed, e.g.
    ``vary(defaults(), {"coupling.npole": 12, "metal.face_gap_mm": 1.2})``."""
    out = inp
    for path, value in changes.items():
        out = set_input(out, path, value)
    return out


def rel_err(value: float, reference: float) -> float:
    """|value - reference| / |reference| (absolute error when reference is 0)."""
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def mismatch(what: str, cells: str, engine: float, reference: float, tol: float) -> str:
    """Assertion message every failing check uses, so findings are uniform:
    what was compared, which workbook cells, both values, the relative error, and the tolerance."""
    return (f"{what} [{cells}]: engine={engine!r} reference={reference!r} "
            f"rel_err={rel_err(engine, reference):.3e} tol={tol:.1e}")


# --------------------------------------------------------------------------- multi-cell checks
#: assert_all prints at most this many failing comparisons in full, then only counts the rest.
MAX_REPORTED = 12


def assert_all(items) -> None:
    """Fail once, listing every failing comparison.

    ``items`` is an iterable of (what, cells, engine, reference, tol) tuples. Each passes when
    rel_err(engine, reference) <= tol; NaN never passes. The message starts with the failure count and
    formats each failure with mismatch(), so one check reports all of its wrong cells."""
    items = list(items)
    bad = [mismatch(*it) for it in items if not rel_err(it[2], it[3]) <= it[4]]
    shown = " || ".join(bad[:MAX_REPORTED])
    more = f" || ... and {len(bad) - MAX_REPORTED} more" if len(bad) > MAX_REPORTED else ""
    assert not bad, f"{len(bad)} of {len(items)} comparisons failed: {shown}{more}"


def flag_item(what: str, cells: str, holds: bool) -> tuple:
    """A yes/no condition as an assert_all item (engine 1.0 when it holds, reference 1.0).
    Put the numbers behind the condition in ``what``."""
    return (f"{what} (1 = holds)", cells, 1.0 if holds else 0.0, 1.0, 0.0)


def text_item(what: str, cells: str, engine_text: str, expected_text: str) -> tuple:
    """A text comparison as an assert_all item, with both texts in the description."""
    return flag_item(f"{what}: engine {engine_text!r}, expected {expected_text!r}", cells, engine_text == expected_text)


class ScenarioCache:
    """``cache[name]`` -> (inputs, results) for vary(defaults(), scenarios[name]); each scenario runs once."""

    def __init__(self, scenarios: dict[str, dict[str, Any]]):
        self.scenarios = scenarios
        self._done: dict[str, tuple[DesignInputs, DesignResults]] = {}

    def __getitem__(self, name: str) -> tuple[DesignInputs, DesignResults]:
        if name not in self._done:
            inp = vary(defaults(), self.scenarios[name])
            self._done[name] = (inp, run(inp))
        return self._done[name]
