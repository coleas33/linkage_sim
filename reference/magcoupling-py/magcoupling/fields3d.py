"""Optional 3D magnetic field check (needs `pip install magpylib`).

The Temperature design sheet uses a handful of numbers that only a 3D field
calculation can give: the worst reverse field inside a block (demagnetization)
and the field amplitudes that drive eddy losses while slipping. They were
computed for the current geometry. Run this module after changing poles, gap,
block size, grade or back iron, then feed the results back with
`apply_to_inputs`.

Model: each block is a uniformly magnetized cuboid (magpylib). Steel back iron
is approximated with first-order images (mirror copies of each ring across the
hub and cup surfaces), which is a lower-bound treatment of real iron.

    from magcoupling import DesignInputs, compute_all
    from magcoupling import fields3d
    inp = DesignInputs(); res = compute_all(inp)
    fr = fields3d.run(inp, res)
    inp2 = fields3d.apply_to_inputs(inp, fr); res2 = compute_all(inp2)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

try:
    import magpylib as magpy
except ImportError as exc:  # pragma: no cover
    raise ImportError("fields3d needs magpylib: pip install magpylib") from exc

MM = 1e-3
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 renamed trapz


@dataclass
class Fields3DResults:
    h_rev_aligned_kA_m: float       # worst point, max of inner/outer ring
    h_rev_pullout_kA_m: float
    h_rev_likepole_kA_m: float
    h_rev_single_ring_kA_m: float
    b_hub_T: float                  # opposite-ring fundamental at the hub steel surface, doubled
    b_cup_T: float
    b_sleeve_T: float
    b_liner_T: float
    cap_integral_T2m4: float
    web_integral_T2m2: float
    b_magnet_T: float               # alternating radial field inside the inner blocks (rms of amplitude)
    force_inner_radial_N: dict      # {"aligned"|"pullout"|"likepole": N}, negative = pressed onto the hub
    force_outer_radial_N: dict      # positive = pressed into the cup
    force_inner_tangential_pullout_N: float
    torque_pullout_3d_Nm: float     # first-order images, 20 °C


class _Geom:
    def __init__(self, inp, res):
        m, md, ret = res.model, inp.metal, res.retainers
        self.N = inp.coupling.npole
        self.L, self.W, self.T = m.inner_length_mm, m.inner_width_mm, m.inner_thickness_mm
        self.To = m.outer_thickness_mm
        self.br = m.inner_br_T
        self.backiron = inp.coupling.backiron
        self.a_i = inp.coupling.inner_back_apothem_mm
        self.hub_surf = self.a_i - md.bond_inner_mm
        self.A_o = m.outer_face_apothem_mm
        self.cup_surf = m.outer_back_apothem_mm + md.bond_outer_mm
        self.r_i = self.a_i + self.T / 2
        self.r_o = self.A_o + self.To / 2
        self.img_i = self.hub_surf - (self.r_i - self.hub_surf)
        self.img_o = self.cup_surf + (self.cup_surf - self.r_o)
        self.r_sleeve = (ret.sleeve_id_mm + ret.sleeve_od_mm) / 4
        self.r_liner = (ret.liner_od_mm + ret.liner_id_mm) / 4
        self.cap_r0, self.cap_r1 = ret.liner_id_mm / 2, md.cap_od_mm / 2

    def ring(self, r_mid, phase_deg, thickness):
        mags = []
        for i in range(self.N):
            ang = phase_deg + 360.0 * i / self.N
            s = 1 if i % 2 == 0 else -1
            c = magpy.magnet.Cuboid(dimension=(thickness * MM, self.W * MM, self.L * MM), polarization=(s * self.br, 0, 0))
            c.rotate_from_angax(ang, "z")
            c.position = (r_mid * MM * math.cos(math.radians(ang)), r_mid * MM * math.sin(math.radians(ang)), 0)
            mags.append(c)
        return mags

    def inner(self, phase=0.0):
        s = self.ring(self.r_i, phase, self.T)
        return s + (self.ring(self.img_i, phase, self.T) if self.backiron else [])

    def outer(self, phase=0.0):
        s = self.ring(self.r_o, phase, self.To)
        return s + (self.ring(self.img_o, phase, self.To) if self.backiron else [])


def _coll(lst):
    return magpy.Collection(*lst, override_parent=True)


def _block_points(g, r_c, ang, thickness, n=(7, 9, 9)):
    fr, ft, fa = (np.linspace(-0.47, 0.47, k) for k in n)
    loc = np.array([(a * thickness, b * g.W, c * g.L) for a in fr for b in ft for c in fa]) * MM
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pts = np.c_[r_c * MM + loc[:, 0], loc[:, 1], loc[:, 2]] @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return pts, np.array([c, s, 0.0])


def _reverse_field(g, sources, r_c, ang, thickness):
    pts, mhat = _block_points(g, r_c, ang, thickness)
    H = magpy.getH(_coll(sources), pts)
    return float((-(H @ mhat) / 1e3).max())          # kA/m, positive = opposing the magnetization


def _fundamental_br(g, sources, r_mm):
    th = np.radians(np.linspace(0, 720 / g.N, 145)[:-1])
    amps = []
    for z in np.linspace(-0.45, 0.45, 11) * g.L * MM:
        p = np.c_[r_mm * MM * np.cos(th), r_mm * MM * np.sin(th), np.full(th.size, z)]
        B = magpy.getB(_coll(sources), p)
        br = B[:, 0] * np.cos(th) + B[:, 1] * np.sin(th)
        pp = g.N / 2
        amps.append(math.hypot(2 * np.mean(br * np.cos(pp * th)), 2 * np.mean(br * np.sin(pp * th))))
    return float(np.sqrt(np.mean(np.square(amps))))


def run(inp, res, cap_gap_mm: float = 1.0, web_gap_mm: float = 1.5) -> Fields3DResults:
    """3D check for the geometry in `res` (from compute_all(inp)). Takes a few seconds."""
    g = _Geom(inp, res)
    half, pitch = 180.0 / g.N, 360.0 / g.N           # pull-out and like-pole relative angles

    def worst(phase):
        src = g.inner(0) + g.outer(phase)
        return max(_reverse_field(g, src, g.r_i, 0.0, g.T), _reverse_field(g, src, g.r_o, phase, g.To))

    h_al, h_po, h_lp = worst(0.0), worst(half), worst(pitch)
    h_single = max(_reverse_field(g, g.inner(0), g.r_i, 0.0, g.T), _reverse_field(g, g.outer(0), g.r_o, 0.0, g.To))

    b_hub = 2 * _fundamental_br(g, g.outer(0), g.hub_surf)
    b_cup = 2 * _fundamental_br(g, g.inner(0), g.cup_surf)
    b_slv = _fundamental_br(g, g.outer(0), g.r_sleeve)
    b_lin = _fundamental_br(g, g.inner(0), g.r_liner)

    th = np.radians(np.linspace(0, 720 / g.N, 145)[:-1])

    def disk(r0, r1, z_mm, fundamental):
        rs = np.linspace(r0, r1, 41)
        R, TH = np.meshgrid(rs, th, indexing="ij")
        p = np.c_[(R * MM * np.cos(TH)).ravel(), (R * MM * np.sin(TH)).ravel(), np.full(R.size, z_mm * MM)]
        Bz = magpy.getB(_coll(g.inner(0)), p)[:, 2].reshape(R.shape)
        if fundamental:
            pp = g.N / 2
            f2 = np.array([math.hypot(2 * np.mean(Bz[i] * np.cos(pp * th)), 2 * np.mean(Bz[i] * np.sin(pp * th))) ** 2
                           for i in range(len(rs))])
            return float(_trapz(f2 * rs * MM, rs * MM) * 2 * math.pi)
        return float(_trapz(np.mean(Bz ** 2, axis=1) * (rs * MM) ** 3, rs * MM) * 2 * math.pi)

    cap_int = disk(g.cap_r0, g.cap_r1, g.L / 2 + cap_gap_mm, False)
    web_int = disk(9.0, 18.0, g.L / 2 + web_gap_mm, True)

    pts, mhat = _block_points(g, g.r_i, 0.0, g.T)
    Ba = magpy.getB(_coll(g.inner(0) + g.outer(0)), pts) @ mhat
    Bl = magpy.getB(_coll(g.inner(0) + g.outer(pitch)), pts) @ mhat
    b_mag = float(np.sqrt(np.mean(((Ba - Bl) / 2) ** 2)))

    f_in, f_out, ft_po = {}, {}, None
    for name, ph in (("aligned", 0.0), ("pullout", half), ("likepole", pitch)):
        inner, outer = g.ring(g.r_i, 0, g.T), g.ring(g.r_o, ph, g.To)
        imgs = (g.ring(g.img_i, 0, g.T) + g.ring(g.img_o, ph, g.To)) if g.backiron else []
        for which, tgt, ang in (("in", inner[0], 0.0), ("out", outer[0], ph)):
            tgt.meshing = (4, 6, 10)
            others = [x for x in inner + outer + imgs if x is not tgt]
            F, _ = magpy.getFT(_coll(others), tgt, pivot=(0, 0, 0))
            F = np.asarray(F).ravel()
            er = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0])
            et = np.array([-er[1], er[0], 0])
            (f_in if which == "in" else f_out)[name] = float(F @ er)
            if which == "in" and name == "pullout":
                ft_po = float(F @ et)

    outer = g.ring(g.r_o, half, g.To)
    for c in outer:
        c.meshing = (4, 6, 10)
    srcs = g.ring(g.r_i, 0, g.T) + ((g.ring(g.img_i, 0, g.T) + g.ring(g.img_o, half, g.To)) if g.backiron else [])
    _, T = magpy.getFT(_coll(srcs), outer, pivot=(0, 0, 0))
    torque = float(abs(np.asarray(T).reshape(-1, 3)[:, 2].sum()))

    return Fields3DResults(h_al, h_po, h_lp, h_single, b_hub, b_cup, b_slv, b_lin, cap_int, web_int, b_mag,
                           f_in, f_out, ft_po, torque)


def apply_to_inputs(inp, fr: Fields3DResults):
    """Copy of `inp` with the Temperature design 3D inputs replaced by `fr` (rounded like the workbook)."""
    new = copy.deepcopy(inp)
    d, s = new.temperature.demag, new.temperature.slip_loss
    d.h_rev_aligned_kA_m = round(fr.h_rev_aligned_kA_m)
    d.h_rev_pullout_kA_m = round(fr.h_rev_pullout_kA_m)
    d.h_rev_likepole_kA_m = round(fr.h_rev_likepole_kA_m)
    d.h_rev_single_ring_kA_m = round(fr.h_rev_single_ring_kA_m)
    s.b_hub_T, s.b_cup_T = round(fr.b_hub_T, 3), round(fr.b_cup_T, 3)
    s.b_sleeve_T, s.b_liner_T = round(fr.b_sleeve_T, 3), round(fr.b_liner_T, 3)
    s.cap_integral_T2m4 = float(f"{fr.cap_integral_T2m4:.3g}")
    s.web_integral_T2m2 = float(f"{fr.web_integral_T2m2:.4g}")
    s.b_magnet_T = round(fr.b_magnet_T, 2)
    return new
