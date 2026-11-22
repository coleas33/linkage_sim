"""The 3D fields the Rust calculator's starting design stores (backlog BL-080).

Runs the reference routine, `magcoupling.fields3d.run`, unchanged, for the geometry of
`magcoupling-rs/src/app.rs::starting_design`, and prints the fourteen values the Rust
engine reads: as the routine returns them (`raw`) and as they are entered (`entered`).
The Rust test `the_starting_design_stores_the_3d_fields_of_its_own_geometry` pins the
raw values; after a change of the starting design's geometry, run this again and copy
both lists.

    python tools/starting_design_fields.py            (needs magpylib; about 20 s)
    python tools/starting_design_fields.py --example  (the self-check, below)

The self-check runs the same code for the example design (the workbook's values) at the
workbook's Br. Its `entered` list must come out as the Rust engine's defaults, apart
from two known residues: the liner field (0.42 against the stored 0.419) and the web
integral (4.139e-5 against 4.14e-5, which is four times the workbook's rounded 1.035e-5).

Three corrections of the Rust engine are carried, because the routine is the workbook's:

- E3: the magnets' Br is 1.30 T in the Rust engine (1.29 T here). Every field is
  proportional to Br, so the routine is given 1.30 T.
- E5: the Rust engine reads the rear-web integral doubled at the steel surface, four
  times what the routine returns.
- E17: the Rust engine has three free-space fields for a hub, cup and web that are not
  steel. They are the routine's hub, cup and web results with the steel's mirror
  images off and no doubling at the surface.

The keyway depth is not an argument: the routine reads the magnets, the bondlines, the
steel surfaces behind the magnets, the sleeve, the liner and the cap, and nothing of
the bore or its keyway.

It writes nothing.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from magcoupling import DesignInputs, compute_all, fields3d  # noqa: E402

RUST_BR_T = 1.30          # correction E3
WEB_STEEL_FACTOR = 4.0    # correction E5


def starting_inputs() -> DesignInputs:
    """The inputs of the starting design that the routine or its geometry reads."""
    inp = DesignInputs()
    inp.coupling.npole = 8
    inp.coupling.inner_back_apothem_mm = 9.0
    inp.metal.face_gap_mm = 1.8
    inp.metal.cup_wall_corner_mm = 2.2
    inp.metal.cap_thread_dia_mm = 41.7
    inp.metal.required_min_Nm = 1.35
    return inp


def fields(inp: DesignInputs, br_T: float = RUST_BR_T) -> dict:
    """`raw`: the fourteen values at full precision; `entered`: rounded as stored."""
    res = compute_all(inp)
    res.model.inner_br_T = br_T                       # the routine reads Br from here
    steel = fields3d.run(inp, res)
    no_images = copy.deepcopy(inp)
    no_images.coupling.backiron = 0                   # same geometry (res), images off
    free = fields3d.run(no_images, res)
    raw = {
        "temperature.demag.h_rev_aligned_kA_m": steel.h_rev_aligned_kA_m,
        "temperature.demag.h_rev_pullout_kA_m": steel.h_rev_pullout_kA_m,
        "temperature.demag.h_rev_likepole_kA_m": steel.h_rev_likepole_kA_m,
        "temperature.demag.h_rev_single_ring_kA_m": steel.h_rev_single_ring_kA_m,
        "temperature.slip_loss.b_hub_T": steel.b_hub_T,
        "temperature.slip_loss.b_cup_T": steel.b_cup_T,
        "temperature.slip_loss.b_sleeve_T": steel.b_sleeve_T,
        "temperature.slip_loss.b_liner_T": steel.b_liner_T,
        "temperature.slip_loss.cap_integral_T2m4": steel.cap_integral_T2m4,
        "temperature.slip_loss.web_integral_T2m2": WEB_STEEL_FACTOR * steel.web_integral_T2m2,
        "temperature.slip_loss.b_magnet_T": steel.b_magnet_T,
        # The routine doubles the hub and cup fields at the steel surface; free space is not doubled.
        "temperature.slip_loss.b_hub_free_T": free.b_hub_T / 2,
        "temperature.slip_loss.b_cup_free_T": free.b_cup_T / 2,
        "temperature.slip_loss.web_integral_free_T2m2": free.web_integral_T2m2,
    }
    # The first eleven as fields3d.apply_to_inputs rounds them (its web integral is the
    # steel one here); the free-space three to four significant figures, as the Rust
    # engine's defaults are (correction E17).
    entered = {}
    for path, value in raw.items():
        name = path.rsplit(".", 1)[1]
        if name.startswith("h_rev"):
            entered[path] = float(round(value))
        elif name == "b_magnet_T":
            entered[path] = round(value, 2)
        elif name == "cap_integral_T2m4":
            entered[path] = float(f"{value:.3g}")
        elif "free" in name or name == "web_integral_T2m2":
            entered[path] = float(f"{value:.4g}")
        else:
            entered[path] = round(value, 3)
    # Everything the routine reads (fields3d._Geom), by the Rust engine's paths; it uses the
    # inner ring's length, width and Br for both rings.
    model, retainers = res.model, res.retainers
    reads = {
        "coupling.npole": inp.coupling.npole,
        "coupling.backiron": inp.coupling.backiron,
        "coupling.inner_back_apothem_mm": inp.coupling.inner_back_apothem_mm,
        "metal.bond_inner_mm": inp.metal.bond_inner_mm,
        "metal.bond_outer_mm": inp.metal.bond_outer_mm,
        "metal.cap_od_mm": inp.metal.cap_od_mm,
        "model.inner_length_mm": model.inner_length_mm,
        "model.inner_width_mm": model.inner_width_mm,
        "model.inner_thickness_mm": model.inner_thickness_mm,
        "model.outer_thickness_mm": model.outer_thickness_mm,
        "model.inner_br_T": model.inner_br_T,
        "model.outer_face_apothem_mm": model.outer_face_apothem_mm,
        "model.outer_back_apothem_mm": model.outer_back_apothem_mm,
        "retainers.sleeve_id_mm": retainers.sleeve_id_mm,
        "retainers.sleeve_od_mm": retainers.sleeve_od_mm,
        "retainers.liner_id_mm": retainers.liner_id_mm,
        "retainers.liner_od_mm": retainers.liner_od_mm,
    }
    return {
        "reads": reads,
        "raw": raw,
        "entered": entered,
        "torque_pullout_3d_Nm": steel.torque_pullout_3d_Nm,
    }


if __name__ == "__main__":
    if "--example" in sys.argv[1:]:
        example = DesignInputs()
        print(json.dumps(fields(example, compute_all(example).model.inner_br_T), indent=1))
    else:
        print(json.dumps(fields(starting_inputs()), indent=1))
