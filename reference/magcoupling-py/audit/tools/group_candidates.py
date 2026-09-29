"""Group failing audit checks by root cause so each root cause is verified once (Task 8).

A check's group is its declared tag (a docstring line 'Root-cause group: <tag>') or, failing that,
its test function name without pytest parameters, so all parametrized cases of one check form one
group. Writes audit/out/candidate_groups.json and prints one line per group.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.group_candidates
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
TAG = re.compile(r"Root-cause group:\s*([A-Za-z0-9_-]+)")


def group_key(result: dict) -> str:
    tag = TAG.search(result.get("doc") or "")
    if tag:
        return tag.group(1)
    name = result["id"].split("::")[-1]
    return name.split("[", 1)[0]


def main() -> None:
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in results:
        if r["outcome"] == "passed":
            continue
        groups.setdefault(group_key(r), []).append(
            {"id": r["id"], "family": r["family"], "message": r["message"], "doc": r["doc"]})
    payload = [{"group": k, "members": v} for k, v in groups.items()]
    (OUT / "candidate_groups.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    total = sum(len(v) for v in groups.values())
    print(f"{total} candidates in {len(groups)} root-cause groups")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} ({v[0]['family']})")


if __name__ == "__main__":
    main()
