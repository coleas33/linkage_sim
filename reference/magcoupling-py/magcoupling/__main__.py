"""Command line: python -m magcoupling [--json out.json] [--set path=value ...]

Examples
    python -m magcoupling
    python -m magcoupling --set coupling.npole=12 --set metal.face_gap_mm=1.2
    python -m magcoupling --json results.json
"""
from __future__ import annotations

import argparse
import json

from . import DesignInputs, compute_all, headline, set_input, to_dict


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("none", "null"):
        return None
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="magcoupling", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="override an input by dotted path")
    ap.add_argument("--json", metavar="FILE", help="write all results to a JSON file")
    args = ap.parse_args(argv)
    inp = DesignInputs()
    for item in args.set:
        path, _, val = item.partition("=")
        inp = set_input(inp, path.strip(), _parse_value(val.strip()))
    res = compute_all(inp)
    for k, v in headline(res).items():
        print(f"{k:32s} {v:.4g}" if isinstance(v, float) else f"{k:32s} {v}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"inputs": to_dict(inp), "results": to_dict(res)}, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
