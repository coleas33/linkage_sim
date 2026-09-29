"""Small helpers that attach GUI-friendly metadata to dataclass fields.

Every input and result field carries:
    unit   - display unit (the model works in mm, N·m, °C, T, W, J, s, rpm)
    label  - short human label (matches the workbook row label)
    help   - longer explanation / provenance note
    cell   - the workbook cell this value came from, e.g. "Calculator!C93"
    choices- optional {value: text} mapping for selector inputs

A GUI can call `schema(obj)` to build forms and result tables without
hard-coding anything.
"""
from __future__ import annotations

import math
from dataclasses import field, fields, is_dataclass
from typing import Any


def param(default: Any, unit: str = "", label: str = "", help: str = "",
          cell: str | None = None, choices: dict | None = None):
    """Declare an editable input with metadata."""
    return field(default=default, metadata={
        "unit": unit, "label": label, "help": help, "cell": cell, "choices": choices, "kind": "input"})


def out(unit: str = "", label: str = "", help: str = "", cell: str | None = None):
    """Declare a computed result with metadata (value is filled in by the model)."""
    return field(default=None, metadata={
        "unit": unit, "label": label, "help": help, "cell": cell, "kind": "result"})


def schema(obj_or_cls) -> list[dict]:
    """Flatten a (possibly nested) dataclass into a list of field descriptions.

    Each entry: {"path", "label", "unit", "help", "cell", "choices", "kind", "value"}.
    Nested dataclasses produce dotted paths ("layout.npole").
    """
    rows: list[dict] = []

    def walk(o, prefix):
        for f in fields(o):
            v = getattr(o, f.name) if not isinstance(o, type) else f.default
            path = f"{prefix}{f.name}"
            if is_dataclass(v) and not isinstance(v, type):
                walk(v, path + ".")
                continue
            if isinstance(v, list) and v and is_dataclass(v[0]):
                for i, item in enumerate(v):
                    walk(item, f"{path}[{i}].")
                continue
            md = dict(f.metadata)
            rows.append({"path": path, "label": md.get("label", f.name), "unit": md.get("unit", ""),
                         "help": md.get("help", ""), "cell": md.get("cell"), "choices": md.get("choices"),
                         "kind": md.get("kind", ""), "value": v})

    walk(obj_or_cls, "")
    return rows


def ceiling(x: float, significance: float) -> float:
    """Excel CEILING for positive numbers: round up to a multiple of `significance`."""
    q = x / significance
    return math.ceil(q - 1e-12) * significance


def floor_(x: float, significance: float) -> float:
    """Excel FLOOR for positive numbers."""
    q = x / significance
    return math.floor(q + 1e-12) * significance
