"""Minimal Tkinter front end (standard library only) showing the integration pattern.

Run: python examples/tk_demo.py
"""
import tkinter as tk
from tkinter import ttk

from magcoupling import DesignInputs, compute_all, headline, input_schema, set_input

KEY_INPUTS = ["coupling.npole", "coupling.op_temp_C", "coupling.inner_back_apothem_mm", "metal.face_gap_mm",
              "metal.cup_wall_corner_mm", "metal.slip_rpm", "temperature.duty.hot_ambient_C",
              "temperature.thermal.conductance_W_K", "clamps.boss_od_mm", "clamps.clamp_length_mm"]


def main():
    root = tk.Tk()
    root.title("Magnetic coupling calculator")
    state = {"inp": DesignInputs()}
    meta = {r["path"]: r for r in input_schema(state["inp"])}
    entries = {}
    form = ttk.Frame(root, padding=8)
    form.grid(row=0, column=0, sticky="n")
    for i, path in enumerate(KEY_INPUTS):
        m = meta[path]
        ttk.Label(form, text=m["label"]).grid(row=i, column=0, sticky="w")
        var = tk.StringVar(value=str(m["value"]))
        ttk.Entry(form, textvariable=var, width=10).grid(row=i, column=1)
        ttk.Label(form, text=m["unit"]).grid(row=i, column=2, sticky="w")
        entries[path] = var
    out = tk.Text(root, width=70, height=20, font=("Courier", 10))
    out.grid(row=0, column=1, padx=8, pady=8)

    def recompute(*_):
        inp = DesignInputs()
        for path, var in entries.items():
            try:
                val = float(var.get())
                inp = set_input(inp, path, int(val) if val.is_integer() and isinstance(meta[path]["value"], int) else val)
            except ValueError:
                pass
        res = compute_all(inp)
        out.delete("1.0", tk.END)
        for k, v in headline(res).items():
            out.insert(tk.END, (f"{k:30s} {v:.4g}\n" if isinstance(v, float) else f"{k:30s} {v}\n"))

    ttk.Button(form, text="Recompute", command=recompute).grid(row=len(KEY_INPUTS), column=0, columnspan=3, pady=6)
    recompute()
    root.mainloop()


if __name__ == "__main__":
    main()
