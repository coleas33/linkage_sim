"""Public entry point: one inputs object in, one results object out.

    from magcoupling import DesignInputs, compute_all
    res = compute_all(DesignInputs())          # workbook defaults
    res.model.pullout_Nm                       # 2.647 N·m at 50 °C

Calculation order mirrors the workbook's dependencies:
    Calibration → Calculator (model) → Metal design retainers → Calculator mass
    → Metal design → Materials → Temperature design → Shaft clamps → sweeps
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, is_dataclass

from . import calibration, clamps, materials, metal_design, model, sweeps, temperature
from ._fields import schema


@dataclass
class DesignInputs:
    coupling: model.CouplingInputs = None
    metal: metal_design.MetalDesignInputs = None
    calibration: calibration.CalibrationInputs = None
    materials: materials.MaterialsInputs = None
    temperature: temperature.TemperatureInputs = None
    clamps: clamps.ClampInputs = None

    def __post_init__(self):
        self.coupling = self.coupling or model.CouplingInputs()
        self.metal = self.metal or metal_design.MetalDesignInputs()
        self.calibration = self.calibration or calibration.CalibrationInputs()
        self.materials = self.materials or materials.MaterialsInputs()
        self.temperature = self.temperature or temperature.TemperatureInputs()
        self.clamps = self.clamps or clamps.ClampInputs()


@dataclass
class DesignResults:
    calibration: calibration.CalibrationResults
    model: model.ModelResults
    mass: model.MassResults
    retainers: metal_design.RetainerResults
    metal: metal_design.MetalDesignResults
    materials: materials.MaterialsResults
    temperature: temperature.TemperatureResults
    clamps: clamps.ClampResults
    gap_sweep: list
    pole_sweep: list


def compute_all(inp: DesignInputs | None = None) -> DesignResults:
    inp = inp or DesignInputs()
    ci, md, cal_in, mat_in = inp.coupling, inp.metal, inp.calibration, inp.materials

    cal = calibration.compute(cal_in)
    f_cal = model.select_calibration_factor(ci.backiron, ci.npole, ci.magnets.part_inner, ci.magnets.part_outer,
                                            cal.poles_per_ring, cal.f_cal_updated, cal_in.f_cal_original)
    m = model.compute(ci, md.face_gap_mm, md.bond_inner_mm, md.bond_outer_mm, md.cup_wall_corner_mm,
                      cal_in.alpha_br_per_C, mat_in.steel.bsat_T, f_cal, cal_in.f_cal_original, md.slip_rpm, md.required_min_Nm)
    ret = metal_design.retainers(md, ci.inner_back_apothem_mm, m.inner_thickness_mm, m.inner_width_mm,
                                 m.outer_face_apothem_mm, ci.bore_mm)
    mass = model.mass_estimate(ci, m, md.bond_inner_mm, md.bond_outer_mm, md.cup_depth_mm, md.web_mm, md.hub_length_mm,
                               md.boss_length_mm, md.boss_od_mm, md.steel_density_g_mm3, md.al_density_g_mm3,
                               ret.retainers_g, md.hardware_g, ret.cap_g, ret.endplates_g)
    mdr = metal_design.compute(md, m.pullout_Nm, m.pullout_20C_Nm, ci.op_temp_C, cal_in.alpha_br_per_C, m.corner_gap_mm,
                               m.face_gap_mm, m.cup_od_mm, ci.npole, ci.bore_mm, ci.gear_ratio, ci.gear_efficiency,
                               mass.total_g, mass.boss_g, ret, cal_in.measured_torque_Nm, cal_in.test_temp_C)
    matr = materials.compute(mat_in, m.backiron_needed_mm, md.cup_wall_corner_mm)

    links = temperature.TemperatureLinks(
        op_temp_C=ci.op_temp_C, npole=ci.npole, br20_T=m.inner_br_T, alpha_br=cal_in.alpha_br_per_C, tmax_lib_C=m.inner_tmax_C,
        mu0=ci.mu0, pullout_op_Nm=m.pullout_Nm, pullout_20C_Nm=m.pullout_20C_Nm, inner_back_apothem_mm=ci.inner_back_apothem_mm,
        inner_length_mm=m.inner_length_mm, inner_width_mm=m.inner_width_mm, inner_thickness_mm=m.inner_thickness_mm,
        hub_wall_mm=m.hub_wall_mm, active_length_mm=m.active_length_mm, outer_back_apothem_mm=m.outer_back_apothem_mm,
        mass_magnets_g=mass.magnets_g, mass_cup_g=mass.cup_g, mass_hub_g=mass.hub_g, mass_boss_g=mass.boss_g,
        slip_rpm=md.slip_rpm, slip_event_s=md.slip_event_s, life_events=md.life_events, measured_drag_Nm=md.measured_drag_Nm,
        cold_high_Nm=mdr.torque_cold_high_Nm, required_min_Nm=md.required_min_Nm, variation=md.variation,
        min_temp_C=md.min_temp_C, magnetic_cycles=mdr.magnetic_cycles, bond_inner_mm=md.bond_inner_mm,
        bond_outer_mm=md.bond_outer_mm, sleeve_mm=md.sleeve_mm, liner_mm=md.liner_mm, sleeve_id_mm=ret.sleeve_id_mm,
        sleeve_od_mm=ret.sleeve_od_mm, liner_od_mm=ret.liner_od_mm, liner_id_mm=ret.liner_id_mm, cap_face_mm=md.cap_axial_mm,
        hardware_g=md.hardware_g, retainers_g=ret.retainers_g, cap_g=ret.cap_g, endplates_g=ret.endplates_g,
        steel_sigma_S_m=mat_in.steel.conductivity_S_m, steel_mu_r=mat_in.steel.mu_r_incremental,
        steel_c=mat_in.steel.specific_heat_J_kgK, steel_cte=mat_in.steel.cte_per_C, steel_E_GPa=mat_in.steel.modulus_GPa,
        al6061_sigma_S_m=mat_in.aluminium.al6061.conductivity_S_m)
    temp = temperature.compute(inp.temperature, links)

    alloy = mat_in.aluminium.al7075 if inp.clamps.alloy == 1 else mat_in.aluminium.al6061
    clr = clamps.compute(inp.clamps, ci.bore_mm, mdr.torque_cold_high_Nm, alloy, mat_in.screws.proof(inp.clamps.screw_class))

    ctx = sweeps.SweepContext(
        faceted=ci.faceted, backiron=ci.backiron, t_i=m.inner_thickness_mm, w_i=m.inner_width_mm, t_o=m.outer_thickness_mm,
        w_o=m.outer_width_mm, L=m.active_length_mm, br_i20=m.inner_br_T, br_o20=m.outer_br_T, br_i_op=m.br_inner_T_op,
        br_o_op=m.br_outer_T_op, bond_outer=md.bond_outer_mm, cup_wall_corner=md.cup_wall_corner_mm, c_end=ci.c_end, mu0=ci.mu0,
        gear_ratio=ci.gear_ratio, gear_eff=ci.gear_efficiency, required_floor_Nm=m.required_floor_Nm, max_diameter_mm=md.max_diameter_mm)
    gap = sweeps.gap_sweep(ctx, ci.npole, ci.inner_back_apothem_mm, f_cal)
    pole = sweeps.pole_sweep(ctx, m.corner_gap_mm, ci.bore_mm, ci.keyway_depth_mm, cal_in.f_cal_original)

    return DesignResults(cal, m, mass, ret, mdr, matr, temp, clr, gap, pole)


# --------------------------------------------------------------------------- helpers for GUIs
def to_dict(obj) -> dict:
    """Plain nested dict (JSON-serializable) of inputs or results."""
    return asdict(obj)


def input_schema(inp: DesignInputs | None = None) -> list[dict]:
    """Every editable input with path, label, unit, help, source cell and current value."""
    return [r for r in schema(inp or DesignInputs()) if r["kind"] == "input"]


def result_schema(res: DesignResults) -> list[dict]:
    """Every computed value with path, label, unit, source cell and value."""
    return schema(res)


def set_input(inp: DesignInputs, path: str, value) -> DesignInputs:
    """Return a copy of `inp` with a dotted-path field changed, e.g. set_input(inp, "coupling.npole", 12)."""
    new = copy.deepcopy(inp)
    obj = new
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"No input named {path!r}")
    setattr(obj, parts[-1], value)
    return new


def headline(res: DesignResults) -> dict:
    """The handful of numbers a dashboard should show first."""
    t = res.temperature.summary
    return {
        "pullout_at_op_temp_Nm": res.model.pullout_Nm,
        "pullout_at_20C_Nm": res.model.pullout_20C_Nm,
        "hot_low_with_variation_Nm": res.metal.torque_hot_low_Nm,
        "hot_min_check": res.metal.hot_min_check,
        "cold_high_with_variation_Nm": res.metal.torque_cold_high_Nm,
        "gearbox_input_ripple_Nm": res.model.gearbox_input_ripple_Nm,
        "cup_od_mm": res.model.cup_od_mm,
        "rotating_mass_g": res.mass.total_g,
        "running_clearance_mm": res.metal.min_running_clearance_mm,
        "clearance_check": res.metal.clearance_check,
        "cup_wall_check": res.materials.cup_wall_check,
        "governing_temp_limit_C": t.governing_limit_C,
        "hot_day_margin_C": t.margin_hot_day_C,
        "temperature_verdict": t.verdict,
        "clamp_screw": res.clamps.recommended,
    }
