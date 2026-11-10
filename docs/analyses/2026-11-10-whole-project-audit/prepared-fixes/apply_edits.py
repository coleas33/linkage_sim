"""Apply exact text replacements to files under a root.

Usage: python -I apply_edits.py <root> <edits.json> [--write]

edits.json is a list of {"file", "old", "new", "count" (default 1)}. "\n" inside old/new stands for
the file's own line ending. Every old text must be found exactly `count` times, or nothing is
written at all. Without --write it only reports.
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
edits = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
write = "--write" in sys.argv

texts = {}
problems = []
for e in edits:
    path = root / e["file"]
    if path not in texts:
        texts[path] = path.read_bytes().decode("utf-8")
    text = texts[path]
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = e["old"].replace("\n", nl), e["new"].replace("\n", nl)
    want = e.get("count", 1)
    found = text.count(old)
    print(f"{'ok  ' if found == want else 'BAD '} {e['file']}: found {found}, want {want}: {e['old'][:70]!r}")
    if found != want:
        problems.append(e)
        continue
    texts[path] = text.replace(old, new)

if problems:
    sys.exit(f"FAIL: {len(problems)} edit(s) did not match; nothing written")
if write:
    for path, text in texts.items():
        path.write_bytes(text.encode("utf-8"))
    print(f"written {len(texts)} file(s)")
