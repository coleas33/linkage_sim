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
        choices, input defaults), for the Rust metadata-parity test.
    magcoupling-rs/tests/data/differential/<module>.json
        seeded cases for each ported module.
    magcoupling-rs/tests/data/differential/helpers.json
        a corpus for the rounding and formatting helpers.
    magcoupling-rs/tests/data/static_data.json
        the static engine tables (magnet library so far), for tests/static_data.rs.

Cases per module, all inside the slider ranges: the workbook defaults; each
input at each end of its range and each selector at each choice (others at
their defaults); the module's boundary probes; then seeded random sets (every
input of the module varied at once) up to CASES_PER_MODULE. Python must not
raise on any case, and every result must be finite: a failure means a slider
range lets the engine leave its domain and must be fixed (or the behaviour
becomes a registered deviation).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ORACLE = HERE.parents[1]  # reference/magcoupling-py
REPO = HERE.parents[3]
DATA = REPO / "magcoupling-rs" / "tests" / "data"

sys.path.insert(0, str(ORACLE))  # this checkout's engine, never an installed copy
import magcoupling  # noqa: E402
from magcoupling import DesignInputs, compute_all, input_schema, result_schema, set_input  # noqa: E402
from magcoupling._fields import ceiling, floor_  # noqa: E402
from magcoupling.clamps import _fmt_num  # noqa: E402
from magcoupling.library import _ROWS  # noqa: E402
from magcoupling.temperature import _text0  # noqa: E402

if Path(magcoupling.__file__).resolve().parent != ORACLE / "magcoupling":
    sys.exit(f"imported the engine from {magcoupling.__file__}, expected {ORACLE / 'magcoupling'}")

SEED = 20260929
CASES_PER_MODULE = 300
# Ported modules, as top-level groups of DesignInputs/DesignResults. Add one
# when its Rust port lands (and list it in magcoupling-rs/tests/common/mod.rs).
MODULES = ("calibration",)
# Hand-placed cases on branch boundaries, per module: (tag, input overrides).
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
}


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
                    "table rows without metadata are not listed). Written by "
                    "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
           "engine": f"magcoupling {magcoupling.__version__}", "rows": rows}
    return dumps(doc) + "\n"


# --------------------------------------------------------------------------- module cases
def typed(field: dict, x):
    """A range end in the field's Python type."""
    return int(round(x)) if field["type"] == "i64" else float(x)


def sample(field: dict, rng: random.Random):
    """A random value inside the field's slider range (or among its choices)."""
    if field["choices"]:
        return rng.choice([code for code, _ in field["choices"]])
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
    for tag, overrides in PROBES.get(module, []):
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise KeyError(f"probe {tag!r} names inputs outside {module}: {sorted(unknown)}")
        cases.append((tag, {**defaults, **overrides}))
    if len(cases) > CASES_PER_MODULE:
        raise ValueError(f"{module}: {len(cases)} fixed cases exceed CASES_PER_MODULE")
    while len(cases) < CASES_PER_MODULE:
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
    prefix = module + "."
    return {r["path"]: plain(r["value"], f"{tag}: {r['path']}")
            for r in result_schema(res) if r["kind"] == "result" and r["path"].startswith(prefix)}


def module_file(module: str, schema: list) -> str:
    fields = [f for f in schema if f["path"].startswith(module + ".")]
    if not fields:
        raise KeyError(f"input_schema.json has no inputs under {module!r}")
    rng = random.Random(f"{SEED}:{module}")
    cases = []
    for i, (tag, inputs) in enumerate(module_cases(module, fields, rng)):
        inputs = {path: plain(value, f"{tag}: {path}") for path, value in inputs.items()}
        cases.append({"id": i, "tag": tag, "inputs": inputs, "results": run_case(module, tag, inputs)})
    header = {"about": f"Python engine results for seeded {module} inputs; compared by "
                       "magcoupling-rs/tests/differential.rs. Written by "
                       "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
              "engine": f"magcoupling {magcoupling.__version__}", "module": module, "seed": SEED}
    head = dumps(header, compact=True)
    body = ",\n".join(dumps(c, compact=True) for c in cases)
    return head[:-1] + ',"cases":[\n' + body + "\n]}\n"


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
    }
    return dumps(doc) + "\n"


# --------------------------------------------------------------------------- main
def outputs() -> dict:
    schema_path = DATA / "input_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["inputs"]
    files = {DATA / "python_schema.json": python_schema(),
             DATA / "differential" / "helpers.json": helpers_file(),
             DATA / "static_data.json": static_data()}
    for module in MODULES:
        files[DATA / "differential" / f"{module}.json"] = module_file(module, schema)
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
