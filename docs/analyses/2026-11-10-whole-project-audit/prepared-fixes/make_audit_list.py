"""Build the index of audit findings from the skeptics' `-verified.md` files.

Usage: python -I make_audit_list.py <findings_dir> <status.json> <detail_dir_name> [<out.md>]

status.json: {"fixed": {id: {"commit": sha, "what": text}}, "held": {id: reason}}.
Prints markdown tables, one per area: id, finding, verdict, what happened. A finding the skeptic
called safe that is neither fixed nor held is marked UNHANDLED and makes the exit code 1, and so
does a finder heading that no skeptic section covers.
"""
import json
import re
import sys
from pathlib import Path

findings = Path(sys.argv[1])
status = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
detail = sys.argv[3]
fixed, held = status.get("fixed", {}), status.get("held", {})
HEAD = re.compile(r"^## ([A-Z]+-\d+)\s*:\s*(.*)$")


def sections(path):
    """Yield (id, title, body lines) for each `## ID: title` section of a file."""
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            if cur:
                yield cur
            cur = (m.group(1), m.group(2).strip(), [])
        elif line.startswith("## "):
            if cur:
                yield cur
            cur = None
        elif cur:
            cur[2].append(line)
    if cur:
        yield cur


def field(body, name):
    for line in body:
        m = re.match(rf"^- {name}[^:]*:\s*(.*)$", line)
        if m:
            return m.group(1).strip()
    return ""


def cell(text, limit=150):
    text = text.replace("|", "/").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


out, bad, counts = [], [], {"fixed": 0, "listed": 0, "dropped": 0}
for ver in sorted(findings.glob("*-verified.md")):
    key = ver.name[: -len("-verified.md")]
    seen = {}
    rows = []
    for fid, title, body in sections(ver):
        verdict, safe = field(body, "Verdict"), field(body, "Safe to apply")
        seen[fid] = True
        if fid in fixed:
            what = f"**Fixed** in `{fixed[fid]['commit']}`: {fixed[fid]['what']}"
            counts["fixed"] += 1
        elif re.match(r"(REFUTED|NOT CONFIRMED|NOT A)", verdict, re.I):
            what = "Dropped: the check refuted it."
            counts["dropped"] += 1
        elif fid in held:
            what = f"For you: {held[fid]}"
            counts["listed"] += 1
        elif safe.lower().startswith("yes"):
            what = "UNHANDLED: called safe, neither fixed nor held"
            bad.append(fid)
        else:
            what = f"For you: {safe or 'the check gave no safe change'}"
            counts["listed"] += 1
        rows.append(f"| {fid} | {cell(title, 200)} | {cell(verdict, 90)} | {cell(what, 260)} |")
    finder = findings / f"{key}.md"
    if finder.exists():
        for fid, title, _ in sections(finder):
            if fid not in seen:
                rows.append(f"| {fid} | {cell(title, 200)} | NOT CHECKED | For you: no skeptic section covers it |")
                counts["listed"] += 1
                bad.append(f"{fid} (unchecked)")
    out += [f"### {key}", "", f"Detail: `{detail}/{key}-verified.md` (the check) and `{detail}/{key}.md` (the finder).", "",
            "| id | finding | verdict | what happened |", "|---|---|---|---|", *rows, ""]

text = "\n".join(out)
summary = f"{counts['fixed']} fixed, {counts['listed']} listed for review, {counts['dropped']} dropped"
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(text, encoding="utf-8", newline="\n")
print(summary)
if bad:
    sys.exit("NEEDS A DECISION: " + ", ".join(bad))
