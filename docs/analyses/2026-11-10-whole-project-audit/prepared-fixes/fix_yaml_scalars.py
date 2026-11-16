"""Make a YAML file parse by quoting the plain multi-line scalars that break it (": " or " #" inside).

Usage: python -I fix_yaml_scalars.py <file> [--write]

Only quoting changes: each broken scalar is wrapped in double quotes, or single quotes, or made a
folded block (>-), whichever needs no escaping. The words are not changed. Self-check: the file
parses, and every fixed scalar's parsed value equals its original words (whitespace-folded).
Without --write nothing is saved.
"""
import re
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
write = "--write" in sys.argv
raw = path.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
lines = raw.split(nl)


def indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


def fold(parts) -> str:
    return " ".join(" ".join(parts).split())


fixed = []  # (first line number at the time of the fix, style, expected folded words)
for _ in range(40):
    try:
        doc = yaml.safe_load(nl.join(lines))
        break
    except yaml.YAMLError as e:
        # A scanner error's context is the broken scalar itself; a parser error's context is only the
        # enclosing list, and its problem mark is the line where the cut-off scalar goes on.
        if isinstance(e, yaml.parser.ParserError):
            mark = getattr(e, "problem_mark", None)
        else:
            mark = getattr(e, "context_mark", None) or getattr(e, "problem_mark", None)
        if mark is None:
            sys.exit(f"FAIL: an error with no position: {e}")
        cand = mark.line
        # A "simple key" error points at the line after the scalar when the scalar's last line is the one read as a key.
        while cand > 0 and not lines[cand].strip():
            cand -= 1
        # The scalar starts at the nearest line above with a smaller indent (its "- " or "key:" line), or here.
        start = cand
        if not lines[cand].lstrip().startswith("- "):
            while start > 0 and (not lines[start].strip() or indent(lines[start]) >= indent(lines[cand])):
                start -= 1
        base = indent(lines[start])
        end = start
        for j in range(start + 1, len(lines)):
            if lines[j].strip() and indent(lines[j]) <= base:
                break
            if lines[j].strip():
                end = j
        head = lines[start]
        m = re.match(r"^( *)(- |[A-Za-z_][\w.-]*: )(.*)$", head)
        if not m or m.group(3)[:1] in ('"', "'", ">", "|", "[", "{") or not m.group(3):
            sys.exit(f"FAIL: line {start + 1} is not a plain scalar this script can quote: {head[:100]!r} ({e.problem} at line {mark.line + 1})")
        lead, opener, first = m.groups()
        words = [first] + [l.strip() for l in lines[start + 1:end + 1]]
        if any(not w for w in words):
            sys.exit(f"FAIL: a blank line inside the scalar at line {start + 1}")
        body = nl.join(lines[start:end + 1])
        if '"' not in body and "\\" not in body:
            style = "double"
            lines[start] = f'{lead}{opener}"{first}'
            lines[end] = lines[end].rstrip() + '"'
        elif "'" not in body:
            style = "single"
            lines[start] = f"{lead}{opener}'{first}"
            lines[end] = lines[end].rstrip() + "'"
        else:
            style = "folded"
            pad = " " * (base + 2)
            lines[start:end + 1] = [f"{lead}{opener}>-"] + [pad + w for w in words]
        fixed.append((start + 1, style, fold(words)))
else:
    sys.exit("FAIL: still does not parse after 40 fixes")

leaves = set()


def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            walk(k)
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
    elif isinstance(x, str):
        leaves.add(fold([x]))


walk(doc)
bad = [(n, s) for n, s, want in fixed if want not in leaves]
for n, style, want in fixed:
    print(f"line {n:4d}  {style:6s}  {want[:90]}")
if bad:
    sys.exit(f"FAIL: the parsed value differs from the words at {bad}")
out = nl.join(lines)
print(f"parses; {len(fixed)} scalars quoted; words kept: {fold([out.replace('>-', '')]).replace(chr(34), '').replace(chr(39), '') == fold([raw]).replace(chr(34), '').replace(chr(39), '')}")
if write:
    path.write_bytes(out.encode("utf-8"))
    print("written", path)
