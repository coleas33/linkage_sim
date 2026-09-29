"""Differential test data for the Rust port (magcoupling-rs).

The workbook snapshot covers only the default inputs. This script runs the
Python engine (the oracle: imported from this checkout, never modified) on
seeded input sets and writes inputs and results to JSON; the Rust tests
(magcoupling-rs/tests/differential.rs) compare the port against them.

Usage, from reference/magcoupling-py:

    python tools/gen_differential.py           write the files
    python tools/gen_differential.py --check   exit 1 if a committed file is stale

Reads (slider ranges are defined once, in Rust):
    magcoupling-rs/tests/data/input_schema.json
        types, workbook defaults, choices and slider ranges of every input;
        rewrite it with `MAGCOUPLING_BLESS=1 cargo test --test schema`.
Writes:
    magcoupling-rs/tests/data/python_schema.json
        Python metadata of every input and result (label, unit, help, cell,
        choices, input defaults) and the layout of each table (field order, row
        count, cell of every field and row), for the Rust metadata-parity test.
    magcoupling-rs/tests/data/differential/<group>.json
        seeded cases for each ported result group (MODULES): columnar, the
        input and result paths once in the header, one case per line.
    magcoupling-rs/tests/data/differential/helpers.json
        a corpus for the rounding and formatting helpers.
    magcoupling-rs/tests/data/static_data.json
        the static engine tables (magnet library, the harmonic set, the validation checklist, the aluminium alloys, the adhesive candidates, the screw sizes, the machining steps, the screw table columns and the sweep variables so far), for tests/static_data.rs.

Cases per module, all inside the slider ranges: the workbook defaults; each
input at each end of its range, each selector at each choice and each text
input at each of its TEXT_CHOICES (others at their defaults); the module's
boundary probes; then seeded random sets (every input of the module's input
groups varied at once), enough to reach CASES_PER_MODULE and never fewer than
RANDOM_MIN. Python must not raise on any case, and every result must be
finite: a failure means a slider range lets the engine leave its domain and
must be fixed (or the behaviour becomes a registered deviation).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve()
ORACLE = HERE.parents[1]  # reference/magcoupling-py
REPO = HERE.parents[3]
DATA = REPO / "magcoupling-rs" / "tests" / "data"

sys.path.insert(0, str(ORACLE))  # this checkout's engine, never an installed copy
import magcoupling  # noqa: E402
from magcoupling import DesignInputs, compute_all, input_schema, result_schema, set_input  # noqa: E402
from magcoupling._fields import ceiling, floor_  # noqa: E402
from magcoupling.clamps import SCREW_SIZES, TABLE_COLUMNS, TABLE_ROWS, ScrewRow, _fmt_num  # noqa: E402
from magcoupling import clamps as py_clamps  # noqa: E402
from magcoupling import metal_design as py_metal_design  # noqa: E402
from magcoupling import model as py_model  # noqa: E402
from magcoupling import sweeps as py_sweeps  # noqa: E402
from magcoupling.library import _ROWS, MAGNET_LIBRARY  # noqa: E402
from magcoupling.materials import Aluminium  # noqa: E402
from magcoupling import temperature as py_temperature  # noqa: E402
from magcoupling.sweeps import GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES, SWEEP_COLUMNS, SweepRow  # noqa: E402
from magcoupling.temperature import _text0  # noqa: E402

if Path(magcoupling.__file__).resolve().parent != ORACLE / "magcoupling":
    sys.exit(f"imported the engine from {magcoupling.__file__}, expected {ORACLE / 'magcoupling'}")

SEED = 20260929
CASES_PER_MODULE = 300
RANDOM_MIN = 100            # at least this many random cases, however many fixed cases a module has
MAX_FILE_BYTES = 4_000_000  # a data file above this means: vary fewer groups or cut cases
# Ported result groups -> the input groups each case varies (every group the
# results read, directly or through upstream sheets). Add a line when a
# module's Rust port lands, and list its groups in magcoupling-rs/tests/common/mod.rs.
MODULES = {
    "calibration": ["calibration"],
    "model": ["coupling", "metal", "materials", "calibration"],
    "retainers": ["coupling", "metal"],
    "mass": ["coupling", "metal"],
    "metal": ["coupling", "metal", "calibration"],
    "materials": ["coupling", "metal", "materials", "calibration"],
    "temperature": ["coupling", "metal", "calibration", "materials", "temperature"],
    "clamps": ["coupling", "metal", "calibration", "materials", "clamps"],
    "gap_sweep": ["coupling", "metal", "calibration"],
    "pole_sweep": ["coupling", "metal", "calibration"],
}
# every library part, blank (manual magnet) and a near miss of the default part
PART_CHOICES = list(MAGNET_LIBRARY) + ["", "b842sh"]
# Text inputs have no slider: the values each case may take (sampled like a selector).
TEXT_CHOICES: dict[str, list[str]] = {
    "coupling.magnets.part_inner": PART_CHOICES,
    "coupling.magnets.part_outer": PART_CHOICES,
}
# Hand-placed cases on branch boundaries, keyed by result group: (tag, input overrides).
PROBES = {
    # The 3D interpolation span 1.0 <= corner gap <= 1.5 is inclusive. With the
    # corner definition the corner gap equals the spacing, so these land on and
    # just beside both ends.
    "calibration": [
        ("interpolation span, low end", {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.0}),
        ("interpolation span, high end", {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.5}),
        ("just below the interpolation span", {"calibration.gap_definition": 0, "calibration.spacing_mm": 0.999}),
        ("just above the interpolation span", {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.501}),
    ],
    # The measured calibration factor applies only to the prototype's circuit
    # (model.select_calibration_factor); random sampling almost never hits all four conditions.
    "model": [
        ("prototype circuit: measured calibration factor",
         {"coupling.backiron": 0, "coupling.npole": 10, "calibration.total_magnets": 20,
          "coupling.magnets.part_inner": "B842SH", "coupling.magnets.part_outer": "B842SH"}),
        ("prototype circuit but 12 poles", {"coupling.backiron": 0, "coupling.npole": 12}),
        ("prototype circuit, 12 poles against a 24-magnet prototype",
         {"coupling.backiron": 0, "coupling.npole": 12, "calibration.total_magnets": 24}),
        ("prototype circuit but an N52 inner part",
         {"coupling.backiron": 0, "coupling.magnets.part_inner": "B842-N52"}),
        ("operating temperature at the 150 C rating", {"coupling.op_temp_C": 150.0}),
        ("operating temperature above the 80 C rating of the outer part",
         {"coupling.op_temp_C": 80.5, "coupling.magnets.part_outer": "B842"}),
        ("manual magnets in both rings", {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""}),
    ],
    # A bench drag replaces 'not measured' (random cases leave it None 20 % of the time).
    "metal": [
        ("measured drag at the slider minimum", {"metal.measured_drag_Nm": 0.001}),
        ("measured drag 0.05 N m", {"metal.measured_drag_Nm": 0.05}),
    ],
    # The Temperature design branches random sampling reaches only by chance: a hot-day start above
    # the limit, the bench drag at and above its slider minimum, a manual inner magnet (the onset stays
    # uncalibrated) and a low conductance (the steady state rises above the limit).
    "temperature": [
        ("hot-day start above the limit", {"temperature.duty.driving_rise_C": 40.0}),
        ("bench drag entered", {"metal.measured_drag_Nm": 0.05}),
        ("bench drag at the slider minimum", {"metal.measured_drag_Nm": 0.001}),
        ("manual inner magnet: uncalibrated onset", {"coupling.magnets.part_inner": ""}),
        ("low conductance: steady state above the limit", {"temperature.thermal.conductance_W_K": 0.02}),
    ],
    # Shaft clamps: the size that works changes with the boss and the clamp length (the boss stays 22 mm
    # so only M3 fits, and the clamp length decides how many screws fit); stripping governs when the
    # engagement is short in the softer alloy; the M6 probe reaches the 4 mm vent-port limit.
    "clamps": [
        ("22 mm boss, 10 mm clamp: nothing fits", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 10.0}),
        ("22 mm boss, 14.5 mm clamp: two M3", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 14.5}),
        ("22 mm boss, 18 mm clamp: three M2.5", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 18.0}),
        ("stripping governs: 6061 with 1 x d engagement", {"clamps.alloy": 2, "clamps.engagement_x_d": 1.0}),
        # No fixed case reaches "No: key too large" (M6 must be the first size that works; about 0.5 %
        # of random clamp samples do). Checked in the oracle: recommended "ISO 4762 M6 x 26, class 12.9".
        ("M6 first size that works: key too large for the vent port",
         {"clamps.safety_factor": 3.0, "clamps.friction": 0.1, "clamps.boss_od_mm": 40.0, "clamps.clamp_length_mm": 13.0}),
    ],
    # Each status of the workbook's priority order (sweeps.py docstring).
    "gap_sweep": [
        ("small envelope: outside OD envelope", {"metal.max_diameter_mm": 20.0}),
        ("low requirement: nominal everywhere", {"metal.required_min_Nm": 0.1, "coupling.drive_torque_Nm": 0.0}),
        ("wide manual inner block: inner flat too narrow",
         {"coupling.magnets.part_inner": "", "coupling.magnets.manual_inner_width_mm": 25.4}),
        ("wide manual outer block: outer flat too narrow",
         {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 25.4}),
    ],
    "pole_sweep": [
        ("small envelope: outside OD envelope", {"metal.max_diameter_mm": 20.0}),
        ("low requirement: nominal everywhere", {"metal.required_min_Nm": 0.1, "coupling.drive_torque_Nm": 0.0}),
        ("wide manual outer block: outer flat too narrow",
         {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 25.4}),
    ],
}


def group_of(path: str) -> str:
    """Top-level group of a dotted path; table rows ('gap_sweep[3].x') count as their table."""
    return re.split(r"[.\[]", path, maxsplit=1)[0]


def dumps(obj, compact=False) -> str:
    """Deterministic JSON: sorted keys, UTF-8 text, no NaN or infinity."""
    if compact:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1)


def plain(value, where: str):
    """A JSON-safe scalar; fails loudly on anything else."""
    if value is None or isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite value {value!r}")
        return value
    raise TypeError(f"{where}: unexpected {type(value).__name__} {value!r}")


# --------------------------------------------------------------------------- python schema
def table_layouts() -> dict:
    """Each table's field order, row count and the cell of every (field, row): test_parity.py's mapping."""
    def sweep(sheet: str, n: int) -> dict:
        return {"fields": [f.name for f in dc_fields(SweepRow)], "rows": n,
                "cells": {name: [f"{sheet}!{col}{6 + i}" for i in range(n)] for name, col in SWEEP_COLUMNS.items()}}
    screw = {"fields": [f.name for f in dc_fields(ScrewRow)], "rows": len(SCREW_SIZES),
             "cells": {name: [f"Clamp screw sizes!{col}{row}" for col in TABLE_COLUMNS] for name, row in TABLE_ROWS.items()}}
    return {"clamps.table": screw, "gap_sweep": sweep("Gap sweep", len(GAP_SWEEP_CORNER_GAPS_MM)),
            "pole_sweep": sweep("Pole sweep", len(POLE_SWEEP_POLES))}


def python_schema() -> str:
    inp = DesignInputs()
    res = compute_all(inp)
    rows = []
    for r in input_schema(inp) + [r for r in result_schema(res) if r["kind"] == "result"]:
        row = {"path": r["path"], "kind": r["kind"], "label": r["label"], "unit": r["unit"], "help": r["help"],
               "cell": r["cell"], "choices": [[code, text] for code, text in (r["choices"] or {}).items()]}
        if r["kind"] == "input":
            row["default"] = plain(r["value"], r["path"])
        rows.append(row)
    doc = {"about": "Python metadata of every magcoupling input and result (kinds 'input' and 'result'; "
                    "table rows without metadata are not listed; table layouts under 'tables'). Written by "
                    "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
           "engine": f"magcoupling {magcoupling.__version__}", "rows": rows, "tables": table_layouts()}
    return dumps(doc) + "\n"


# --------------------------------------------------------------------------- module cases
def typed(field: dict, x):
    """A range end in the field's Python type."""
    return int(round(x)) if field["type"] == "i64" else float(x)


def sample(field: dict, rng: random.Random):
    """A random value inside the field's slider range (or among its choices)."""
    if field["choices"]:
        return rng.choice([code for code, _ in field["choices"]])
    if field["path"] in TEXT_CHOICES:
        return rng.choice(TEXT_CHOICES[field["path"]])
    rng_ = field["range"]
    if rng_ is None:
        return field["default"]  # text inputs keep their default
    lo, hi, step = rng_["min"], rng_["max"], rng_["step"]
    if field["type"] == "i64":
        return int(lo) + int(step) * rng.randint(0, int(round((hi - lo) / step)))
    if field["type"] == "opt_f64" and rng.random() < 0.2:
        return None
    if rng_["log"]:
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


def module_cases(module: str, fields: list, rng: random.Random) -> list:
    defaults = {f["path"]: f["default"] for f in fields}
    cases = [("workbook defaults", dict(defaults))]
    for f in fields:
        if f["choices"]:
            for code, _ in f["choices"]:
                cases.append((f"{f['path']} = choice {code}", {**defaults, f["path"]: code}))
        elif f["range"]:
            for end in ("min", "max"):
                cases.append((f"{f['path']} = range {end}", {**defaults, f["path"]: typed(f, f["range"][end])}))
        elif f["path"] in TEXT_CHOICES:
            for text in TEXT_CHOICES[f["path"]]:
                cases.append((f"{f['path']} = {text!r}", {**defaults, f["path"]: text}))
    for tag, overrides in PROBES.get(module, []):
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise KeyError(f"probe {tag!r} names inputs outside {module}'s groups: {sorted(unknown)}")
        cases.append((tag, {**defaults, **overrides}))
    for _ in range(max(CASES_PER_MODULE - len(cases), RANDOM_MIN)):
        cases.append(("random", {f["path"]: sample(f, rng) for f in fields}))
    return cases


def run_case(module: str, tag: str, inputs: dict) -> dict:
    inp = DesignInputs()
    for path, value in inputs.items():
        inp = set_input(inp, path, value)
    try:
        res = compute_all(inp)
    except Exception as exc:  # noqa: BLE001 - reported with the case, then re-raised
        raise RuntimeError(f"{module} case {tag!r} raised {type(exc).__name__}: {exc}\ninputs: {inputs}") from exc
    return {r["path"]: plain(r["value"], f"{tag}: {r['path']}") for r in result_schema(res)
            if r["kind"] in ("result", "") and group_of(r["path"]) == module}


def module_file(module: str, groups: list, schema: list) -> str:
    fields = [f for f in schema if group_of(f["path"]) in groups]
    missing = sorted(set(groups) - {group_of(f["path"]) for f in fields})
    if missing:
        raise KeyError(f"input_schema.json has no inputs under {missing}")
    input_paths = [f["path"] for f in fields]
    rng = random.Random(f"{SEED}:{module}")
    result_paths, cases = None, []
    for i, (tag, inputs) in enumerate(module_cases(module, fields, rng)):
        results = run_case(module, tag, inputs)
        if result_paths is None:
            result_paths = list(results)
        elif list(results) != result_paths:
            raise ValueError(f"{module} case {tag!r}: result paths differ from case 0")
        cases.append({"id": i, "tag": tag,
                      "inputs": [plain(inputs[p], f"{tag}: {p}") for p in input_paths],
                      "results": list(results.values())})
    header = {"about": f"Python engine results for seeded inputs, result group {module}; compared by "
                       "magcoupling-rs/tests/differential.rs. Paths are listed once; each case holds "
                       "values in that order. Written by reference/magcoupling-py/tools/"
                       "gen_differential.py. Do not edit by hand.",
              "engine": f"magcoupling {magcoupling.__version__}", "module": module, "seed": SEED,
              "input_groups": groups, "input_paths": input_paths, "result_paths": result_paths}
    head = dumps(header, compact=True)
    body = ",\n".join(dumps(c, compact=True) for c in cases)
    text = head[:-1] + ',"cases":[\n' + body + "\n]}\n"
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"differential/{module}.json is {len(text.encode('utf-8'))} bytes, over {MAX_FILE_BYTES}")
    return text


# --------------------------------------------------------------------------- helpers corpus
def helpers_file() -> str:
    rng = random.Random(f"{SEED}:helpers")
    xs = [0.0, -0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.125, 0.375, 0.625, -0.125, 2.675, 1.005, 0.045,
          0.05, 0.95, 1.29, -0.0012, 1.256637e-06, 4.5e6, 99.5, 100.5, 7.5, 8.5, -7.5, 12.0, 14.0, 13.66,
          7.335, 13.9999999999, 12.000000000001, 0.1 + 0.2, 1 / 3, 2 / 3, 1e-5, 1e-4, 9.999e-5, 0.0001,
          1e15, 1e16, 1e16 + 2, 1.5e16, 1e22, 1e23, 123456789.0, 5e-324, 2.2250738585072014e-308,
          1.7976931348623157e308, -1.7976931348623157e308, -0.03, -0.05, -0.3]
    for _ in range(600):  # wide magnitudes, both signs
        xs.append(rng.choice((1, -1)) * 10 ** rng.uniform(-8, 17))
    for _ in range(300):  # exact binary ties at one and two decimals
        xs.append(rng.randint(-4000, 4000) / rng.choice((8, 16, 32)))
    for _ in range(300):  # a hair off a multiple of 0.1, 1 or 2: the 1e-12 guard of ceiling/floor_
        x = rng.randint(1, 400) * rng.choice((0.1, 1.0, 2.0))
        xs.append(x * (1 + rng.choice((-1, 1)) * 10 ** rng.uniform(-16, -10)))
    entries = []
    for x in xs:
        small = abs(x) < 1e12  # ceiling/floor_ raise on an infinite quotient; the engine rounds mm values
        entries.append({"x": x, "repr": repr(x), "fmt_num": _fmt_num(x), "fixed1": format(x, ".1f"),
                        "fixed2": format(x, ".2f"), "text0": _text0(x),
                        "ceiling_0_1": ceiling(x, 0.1) if small else None,
                        "ceiling_2": ceiling(x, 2) if small else None,
                        "floor_1": floor_(x, 1) if small else None})
    header = {"about": "Python outputs of repr, clamps._fmt_num, f-string .1f/.2f, temperature._text0, "
                       "_fields.ceiling and _fields.floor_ on a seeded corpus; compared by "
                       "magcoupling-rs/tests/differential.rs. Written by "
                       "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
              "engine": f"magcoupling {magcoupling.__version__}", "seed": SEED}
    head = dumps(header, compact=True)
    body = ",\n".join(dumps(e, compact=True) for e in entries)
    return head[:-1] + ',"entries":[\n' + body + "\n]}\n"


# --------------------------------------------------------------------------- static data
def static_data() -> str:
    """Static engine tables, for magcoupling-rs/tests/static_data.rs."""
    doc = {
        "about": "Static tables of the magcoupling engine (plain data, no metadata). Written by "
                 "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
        "engine": f"magcoupling {magcoupling.__version__}",
        "magnet_library": [dataclasses.asdict(m) for m in _ROWS],
        "harmonics": list(py_model.HARMONICS),
        "validation_items": [[label, status, action]
                             for label, (status, action) in py_metal_design.VALIDATION_ITEMS.items()],
        "aluminium": [dataclasses.asdict(a) for a in (Aluminium().al7075, Aluminium().al6061)],
        "adhesives": [dataclasses.asdict(a) for a in py_temperature.default_adhesives()],
        "screw_sizes": [dataclasses.asdict(s) for s in py_clamps.SCREW_SIZES],
        "machining_steps": list(py_clamps.MACHINING_STEPS),
        "table_columns": list(py_clamps.TABLE_COLUMNS),
        "sweeps": {"gap_sweep_corner_gaps_mm": list(py_sweeps.GAP_SWEEP_CORNER_GAPS_MM),
                   "pole_sweep_poles": list(py_sweeps.POLE_SWEEP_POLES)},
    }
    return dumps(doc) + "\n"


# --------------------------------------------------------------------------- main
def outputs() -> dict:
    schema_path = DATA / "input_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["inputs"]
    files = {DATA / "python_schema.json": python_schema(),
             DATA / "differential" / "helpers.json": helpers_file(),
             DATA / "static_data.json": static_data()}
    for module, groups in MODULES.items():
        files[DATA / "differential" / f"{module}.json"] = module_file(module, groups, schema)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed file differs from a fresh run")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        rel = path.relative_to(REPO).as_posix()
        if args.check:
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
            if current != text:
                stale.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {rel} ({text.count(chr(10))} lines)")
    if args.check:
        if stale:
            print("stale differential data (run tools/gen_differential.py):\n  " + "\n  ".join(stale))
            return 1
        print(f"differential data is current ({', '.join(MODULES)} + helpers + python schema + static data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
