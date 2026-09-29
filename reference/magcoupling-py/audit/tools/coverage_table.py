"""Print the 'confirmed correct' coverage table for the findings report (Task 8).

Only passing checks of the ENGINE count as confirmations. Excluded and counted separately:
reference self-tests (test functions named test_sanity*), the harness smoke tests, and checks whose
docstring says 'not an independent check'.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.coverage_table
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "out" / "results.json"


def classify(result: dict) -> str:
    module, _, name = result["id"].rpartition("::")
    if name.split("[", 1)[0].startswith("test_sanity"):
        return "reference-self-test"
    if module.endswith("test_harness_smoke.py"):
        return "harness"
    if "not an independent check" in (result.get("doc") or "").lower():
        return "consistency-only"
    return "engine"


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    for r in results:
        if r["outcome"] != "passed":
            continue
        kind = classify(r)
        if kind != "engine":
            excluded[kind] += 1
            continue
        first_line = ((r.get("doc") or "").splitlines() or [""])[0]
        by_family[r["family"]].append((r["id"].split("::")[-1], first_line))
    print("| Family | Check | What it confirms |")
    print("|---|---|---|")
    for fam in sorted(by_family):
        for name, doc in sorted(by_family[fam]):
            print(f"| {fam} | `{name}` | {doc.replace('|', '/')} |")
    total = sum(len(v) for v in by_family.values())
    print(f"\n{total} engine checks passed across {len(by_family)} families.")
    print("Excluded from engine coverage: " + ", ".join(f"{n} {k}" for k, n in sorted(excluded.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
