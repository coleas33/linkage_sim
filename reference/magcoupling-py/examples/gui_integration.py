"""Framework-agnostic pattern for wiring the calculator into a GUI.

1. Build the input form from `input_schema()`: every field has a dotted path,
   label, unit, help text, source cell and (for selectors) choices.
2. On edit, apply the value with `set_input()` and recompute with `compute_all()`
   (well under 10 ms, safe to run on every change).
3. Show `headline()` first, then any section of the results; `result_schema()`
   gives label/unit/cell for every computed value.
4. Run the optional 3D check (`fields3d.run`, about 2 s) in a worker thread.
"""
from collections import defaultdict

from magcoupling import DesignInputs, compute_all, headline, input_schema, result_schema, set_input

inp = DesignInputs()

# 1) group inputs into form sections by their top-level path
sections = defaultdict(list)
for f in input_schema(inp):
    sections[f["path"].split(".")[0]].append(f)
for name, rows in sections.items():
    print(f"[{name}] {len(rows)} inputs, e.g. {rows[0]['label']} = {rows[0]['value']} {rows[0]['unit']}")

# 2) user edits the face gap and the operating temperature
inp = set_input(inp, "metal.face_gap_mm", 1.2)
inp = set_input(inp, "coupling.op_temp_C", 60)
res = compute_all(inp)

# 3) dashboard first, then a detail table for one section
print("\nHeadline")
for k, v in headline(res).items():
    print(f"  {k:30s} {v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"  {k:30s} {v}")

print("\nTemperature summary")
for r in result_schema(res):
    if r["path"].startswith("temperature.summary."):
        v = r["value"]
        num = isinstance(v, (int, float)) and not isinstance(v, bool)
        print(f"  {r['label']:55s} {v:.4g} {r['unit']}" if num else f"  {r['label']:55s} {v}")

# sweeps are lists of rows, ready for a table or a plot
print("\nGap sweep:", [(row.variable, round(row.pullout_op_Nm, 2)) for row in res.gap_sweep])
