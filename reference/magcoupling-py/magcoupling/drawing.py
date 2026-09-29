"""Optional cut-layout drawing of the recommended shaft clamp (needs matplotlib).

Returns a matplotlib Figure, so a GUI can embed it (e.g. FigureCanvasQTAgg or
FigureCanvasTkAgg) or save it:

    from magcoupling import DesignInputs, compute_all
    from magcoupling.drawing import clamp_layout
    inp = DesignInputs(); res = compute_all(inp)
    fig = clamp_layout(inp, res); fig.savefig("clamp.png", dpi=150)
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg") if matplotlib.get_backend().lower() in ("", "agg") else None
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

INK, CUT, HID, DIM = "#1f2d3d", "#c0392b", "#7f8c8d", "#2a6f97"


def _dim(ax, p, q, txt, off=(0, 0), fs=9):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<->", color=DIM, lw=1))
    ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], txt, color=DIM, fontsize=fs, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.5))


def clamp_layout(inp, res):
    """End view and top view of the one-piece slotted clamp for the recommended screw."""
    c, cl, md = inp.clamps, res.clamps, inp.metal
    if not cl.index:
        raise ValueError("No screw size fits; enlarge the boss or the clamp length first.")
    row = cl.table[cl.index - 1]
    D, d, L = c.boss_od_mm, res.clamps.shaft_mm, c.clamp_length_mm
    R, r = D / 2, d / 2
    e, hole, cb, tap, thr = row.offset_mm, row.hole_mm, row.cbore_dia_mm, row.tap_drill_mm, row.d_mm
    x_out = math.sqrt(R ** 2 - e ** 2)
    x_seat = row.grip_mm + c.slit_mm / 2
    key_w, key_d = c.key_width_mm, inp.coupling.keyway_depth_mm

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for a in (a1, a2):
        a.set_aspect("equal")
        a.axis("off")

    # end view
    boss = Circle((0, 0), R, fill=False, lw=2, ec=INK)
    a1.add_patch(boss)
    a1.add_patch(Circle((0, 0), r, fill=False, lw=2, ec=INK))
    a1.add_patch(Rectangle((r - 0.2, -key_w / 2), key_d + 0.2, key_w, fc="white", ec=INK, lw=1.5))
    for p in (Rectangle((-c.slit_mm / 2, r), c.slit_mm, R - r + 1.0, fc=CUT, ec=CUT),
              Rectangle((-R - 1, e - cb / 2), R + 1 - x_seat, cb, fc="#f4d9d5", ec=CUT, lw=1.2, ls="--"),
              Rectangle((-x_seat, e - hole / 2), x_seat - c.slit_mm / 2, hole, fc="#f4d9d5", ec=CUT, lw=1.2, ls="--"),
              Rectangle((c.slit_mm / 2, e - thr / 2), R + 1 - c.slit_mm / 2, thr, fc="#f4d9d5", ec=CUT, lw=1.2, ls="--")):
        a1.add_patch(p)
        p.set_clip_path(boss)
    for y in (e, 0):
        a1.plot([-R - 2, R + 2], [y, y], color=HID, lw=0.8, ls="-.")
    a1.plot([0, 0], [-R - 2, R + 2], color=HID, lw=0.8, ls="-.")
    _dim(a1, (R + 3.5, 0), (R + 3.5, e), f"{e:.2f}", off=(1.6, 0))
    _dim(a1, (-R, -R - 3), (R, -R - 3), f"Ø{D:g} boss", off=(0, -1.2))
    _dim(a1, (-r, -2.2), (r, -2.2), f"Ø{d:g} H7", off=(0, -1.1), fs=8)
    a1.annotate(f"Slit {c.slit_mm:g} wide,\nbore to OD", xy=(0, R - 1), xytext=(4.5, R + 4.5), color=CUT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=CUT))
    a1.annotate(f"Counterbore Ø{cb:g} from the OD;\nhead seat {x_seat:.2f} from the slit", xy=(-x_out + 2.0, e + 1.0),
                xytext=(-R - 11, R + 3), color=CUT, fontsize=9, arrowprops=dict(arrowstyle="->", color=CUT))
    a1.annotate(f"Clearance Ø{hole:g}", xy=(-2.5, e - hole / 2), xytext=(-R - 9, 1.5), color=CUT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=CUT))
    a1.annotate(f"{row.size} tapped\n(drill Ø{tap:g})", xy=(x_out - 2, e + thr / 2), xytext=(R + 1.0, R + 3.0), color=CUT,
                fontsize=9, arrowprops=dict(arrowstyle="->", color=CUT))
    a1.annotate(f"Keyway {key_w:g} wide,\n90° from slit", xy=(r + key_d, 0), xytext=(R + 1.5, -7.5), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    a1.set_xlim(-R - 12, R + 12)
    a1.set_ylim(-R - 6, R + 8)
    a1.set_title("End view from the free end (screw hidden, dashed)", fontsize=11, color=INK)

    # top view onto the slit
    fl_D, fl_t, pl_D, pl_t = md.adapter_flange_dia_mm, md.adapter_flange_mm, md.adapter_pilot_dia_mm, md.adapter_pilot_mm
    a2.add_patch(Rectangle((0, -R), L, D, fc="white", ec=INK, lw=2))
    a2.add_patch(Rectangle((L, -R), c.relief_mm, D, fc="#f4d9d5", ec=CUT, lw=1.2))
    a2.add_patch(Rectangle((L + c.relief_mm, -fl_D / 2), fl_t, fl_D, fc="white", ec=INK, lw=2))
    a2.add_patch(Rectangle((L + c.relief_mm + fl_t, -pl_D / 2), pl_t, pl_D, fc="white", ec=INK, lw=2))
    a2.add_patch(Rectangle((0, -c.slit_mm / 2), L, c.slit_mm, fc=CUT, ec=CUT))
    a2.plot([-2, L + c.relief_mm + fl_t + pl_t + 2], [0, 0], color=HID, lw=0.8, ls="-.")
    for i in range(int(row.screws_needed)):
        zc = cl.layout_first_mm + i * cl.layout_pitch_mm
        a2.plot([zc, zc], [-R - 1.5, R + 1.5], color=HID, lw=0.8, ls="-.")
        a2.add_patch(Rectangle((zc - cb / 2, -x_out), cb, x_out - x_seat, fc="none", ec=CUT, lw=1.2, ls="--"))
        a2.add_patch(Rectangle((zc - hole / 2, -x_seat), hole, x_seat - c.slit_mm / 2, fc="none", ec=CUT, lw=1.2, ls="--"))
        a2.add_patch(Rectangle((zc - thr / 2, c.slit_mm / 2), thr, x_out - c.slit_mm / 2, fc="none", ec=CUT, lw=1.2, ls="--"))
    _dim(a2, (0, R + 3), (L, R + 3), f"{L:g} clamp", off=(0, 1.2))
    _dim(a2, (0, -R - 3), (cl.layout_first_mm, -R - 3), f"{cl.layout_first_mm:g}", off=(0, -1.2))
    a2.annotate(f"Relief cut {c.relief_mm:g} wide,\n{D - c.hinge_mm:g} deep from the slit side\n(leaves {c.hinge_mm:g} hinge)",
                xy=(L + c.relief_mm / 2, R - 2), xytext=(L + 7, R + 7), color=CUT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=CUT))
    a2.annotate("Screw axis\n(square to the slit)", xy=(cl.layout_first_mm, -R + 3), xytext=(-9, -R - 5), color=HID, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=HID))
    a2.annotate("Flange to the steel cup web\n(pilot + 3 x M3 + dowel)", xy=(L + c.relief_mm + fl_t / 2, -fl_D / 2 + 2),
                xytext=(L + 9, -R - 7), color=INK, fontsize=9, arrowprops=dict(arrowstyle="->", color=INK))
    a2.set_xlim(-12, L + c.relief_mm + fl_t + pl_t + 16)
    a2.set_ylim(-R - 10, R + 12)
    a2.set_title("Top view onto the slit", fontsize=11, color=INK)

    fig.suptitle(f"One-piece slotted clamp, Ø{d:g} keyed shaft: {cl.recommended}, {cl.tightening_Nm:.1f} N·m, "
                 f"{cl.hex_mm:g} mm key", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
