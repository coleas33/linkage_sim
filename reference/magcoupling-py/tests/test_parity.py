"""Parity tests: every Python result that names a workbook cell must match that cell's cached value.

The reference snapshot comes from tools/extract_reference.py. Numbers must agree to 1e-9 relative;
text must match exactly.
"""
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from magcoupling import DesignInputs, compute_all
from magcoupling.clamps import TABLE_COLUMNS, TABLE_ROWS
from magcoupling.sweeps import SWEEP_COLUMNS

REF = json.loads((Path(__file__).parent / "reference_values.json").read_text())
RES = compute_all(DesignInputs())


def _close(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def _cells(obj, kind):
    """Yield (cell, value, path) for every dataclass field with a cell reference."""
    def walk(o, path):
        for f in fields(o):
            v = getattr(o, f.name)
            if is_dataclass(v):
                yield from walk(v, f"{path}.{f.name}")
                continue
            cell = f.metadata.get("cell")
            if cell and f.metadata.get("kind") == kind:
                yield cell, v, f"{path}.{f.name}"
    yield from walk(obj, "")


RESULT_CELLS = list(_cells(RES, "result"))
INPUT_CELLS = [c for c in _cells(DesignInputs(), "input") if c[1] is not None]


@pytest.mark.parametrize("cell,value,path", RESULT_CELLS, ids=[c[0] for c in RESULT_CELLS])
def test_result_matches_workbook(cell, value, path):
    assert cell in REF, f"{cell} not in the reference snapshot"
    assert _close(value, REF[cell]), f"{path}: python={value!r} workbook={REF[cell]!r}"


@pytest.mark.parametrize("cell,value,path", INPUT_CELLS, ids=[c[0] for c in INPUT_CELLS])
def test_default_input_matches_workbook(cell, value, path):
    assert cell in REF, f"{cell} not in the reference snapshot"
    assert _close(value, REF[cell]), f"{path}: default={value!r} workbook={REF[cell]!r}"


def _sweep_cases():
    for sheet, rows in (("Gap sweep", RES.gap_sweep), ("Pole sweep", RES.pole_sweep)):
        for i, row in enumerate(rows):
            for name, col in SWEEP_COLUMNS.items():
                yield f"{sheet}!{col}{6 + i}", getattr(row, name)


@pytest.mark.parametrize("cell,value", list(_sweep_cases()))
def test_sweeps(cell, value):
    assert _close(value, REF[cell]), f"python={value!r} workbook={REF[cell]!r}"


def _table_cases():
    for j, row in enumerate(RES.clamps.table):
        col = TABLE_COLUMNS[j]
        for name, r in TABLE_ROWS.items():
            yield f"Clamp screw sizes!{col}{r}", getattr(row, name)


@pytest.mark.parametrize("cell,value", list(_table_cases()))
def test_screw_table(cell, value):
    assert _close(value, REF[cell]), f"python={value!r} workbook={REF[cell]!r}"


def test_coverage():
    """Guard against silently dropping outputs: a healthy port checks several hundred cells."""
    assert len(RESULT_CELLS) > 300
