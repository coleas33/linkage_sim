"""Metal prototype design ('Metal design' sheet).

Steel-backed candidate: keyed 4140 inner hub with ten external flats, one-piece
4140 cup (return ring + rear web + shaft boss) with ten internal flats, the same
20 × B842SH blocks and a 1.4 mm flat-face gap. Magnets are bonded and captured
by a rotating 0.10 mm 316L sleeve (inner) and a 0.20 mm 316L liner (outer),
with endplates and a threaded aluminium front cap. The housing is stationary
and sealed.

This module covers: torque at hot/cold limits with a production-variation
allowance, the radial clearance stack, retainer geometry and masses, the axial
stack against the envelope, the optional aluminium shaft adapter, and duty
(slip cycles, slip loss when a bench drag torque is entered).

Dimensions are shape-level, not released machining drawings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ._fields import out, param


@dataclass
class MetalDesignInputs:
    required_min_Nm: float = param(2.5, "N·m", "Required minimum service pull-out torque", "At the 50 °C magnet temperature.", "Metal design!C7")
    min_temp_C: float = param(-40, "°C", "Minimum magnet temperature", "Cold torque is reported, not capped.", "Metal design!C16")
    variation: float = param(0.15, "fraction", "Symmetric torque variation allowance", "Engineering allowance, not measured.", "Metal design!C18")
    # radial stack
    sleeve_mm: float = param(0.1, "mm", "Inner rotating retaining sleeve thickness", "316L band over the inner magnets.", "Metal design!C25")
    liner_mm: float = param(0.2, "mm", "Outer rotating keeper liner thickness", "316L liner inside the outer magnets.", "Metal design!C26")
    shaft_displacement_mm: float = param(0.4, "mm", "Relative shaft radial displacement allowance", "UNCONFIRMED.", "Metal design!C28")
    runout_mm: float = param(0.05, "mm", "Combined assembled runout allowance", "", "Metal design!C29")
    deflection_mm: float = param(0.05, "mm", "Additional load deflection / tilt allowance", "", "Metal design!C30")
    thermal_mm: float = param(0.03, "mm", "Differential thermal movement allowance", "", "Metal design!C31")
    sleeve_form_mm: float = param(0.05, "mm", "Sleeve fit / thickness / form allowance", "", "Metal design!C32")
    magnet_position_mm: float = param(0.2, "mm", "Magnet position / size allowance", "UNCONFIRMED.", "Metal design!C33")
    residual_target_mm: float = param(0.2, "mm", "Minimum desired residual running clearance", "", "Metal design!C36")
    # densities
    al_density_g_mm3: float = param(0.0027, "g/mm³", "Aluminium cap / adapter density", "", "Metal design!C42")
    sleeve_density_g_mm3: float = param(0.008, "g/mm³", "Sleeve density", "", "Metal design!C44")
    steel_density_g_mm3: float = param(0.00785, "g/mm³", "Steel density (4140)", "", "Metal design!C132")
    # duty
    slip_rpm: float = param(2000, "rpm", "Relative slip speed", "User: 2,000 rpm at the wheel.", "Metal design!C85")
    slip_event_s: float = param(0.1, "s", "Slip duration per event", "Illustrative; replace with the recorded value.", "Metal design!C87")
    life_events: float = param(2e7, "events", "Life events", "Supplied life target.", "Metal design!C88")
    measured_drag_Nm: float | None = param(None, "N·m", "Measured mean slip drag torque",
                                           "Enter the bench result; None = not measured.", "Metal design!C90")
    # geometry
    face_gap_mm: float = param(1.4, "mm", "Candidate flat-face magnetic gap", "Same as the measured prototype.", "Metal design!C119")
    bond_inner_mm: float = param(0.05, "mm", "Inner magnet back bondline", "", "Metal design!C120")
    bond_outer_mm: float = param(0.05, "mm", "Outer magnet back bondline", "", "Metal design!C121")
    cup_wall_corner_mm: float = param(1.8, "mm", "Minimum outer return ring wall",
                                      "At the pocket corners. 4140 at 1.5 T needs about 1.9 mm (Materials check).", "Metal design!C122")
    hub_length_mm: float = param(13, "mm", "Steel inner hub axial length", "", "Metal design!C123")
    cup_depth_mm: float = param(15.5, "mm", "Cup cavity axial depth", "", "Metal design!C124")
    web_mm: float = param(2.5, "mm", "Integral steel rear web thickness", "", "Metal design!C125")
    boss_length_mm: float = param(13, "mm", "Integral steel boss axial length", "", "Metal design!C126")
    boss_od_mm: float = param(22, "mm", "Shaft boss outside diameter", "", "Metal design!C127")
    hardware_g: float = param(6, "g", "Keys / screws / lock tab mass allowance", "", "Metal design!C128")
    max_large_dia_axial_mm: float = param(20, "mm", "Maximum large-diameter axial region", "User supplied.", "Metal design!C129")
    max_overall_axial_mm: float = param(35, "mm", "Maximum overall axial length", "User supplied.", "Metal design!C130")
    max_diameter_mm: float = param(43, "mm", "Maximum rotating coupling diameter", "User supplied.", "Metal design!C131")
    cap_axial_mm: float = param(0.8, "mm", "Front cap axial addition", "", "Metal design!C133")
    cap_od_mm: float = param(42.8, "mm", "Threaded cap OD", "", "Metal design!C166")
    cap_thread_engagement_mm: float = param(2, "mm", "Cap thread engagement length", "", "Metal design!C168")
    cap_thread_dia_mm: float = param(41, "mm", "Cap thread nominal diameter", "M41 × 0.5 concept.", "Metal design!C169")
    front_endplate_mm: float = param(0.5, "mm", "Inner front endplate thickness", "", "Metal design!C170")
    rear_endplate_mm: float = param(1, "mm", "Inner rear endplate thickness", "", "Metal design!C171")
    retainer_span_mm: float = param(14.5, "mm", "Nominal retainer axial span", "", "Metal design!C172")
    sleeve_bedding_mm: float = param(0.025, "mm", "Inner sleeve minimum bedding clearance", "", "Metal design!C173")
    liner_bedding_mm: float = param(0.025, "mm", "Outer liner minimum bedding clearance", "", "Metal design!C174")
    rear_endplate_hole_mm: float = param(4.5, "mm", "Rear endplate screw clearance diameter", "", "Metal design!C182")
    adapter_flange_dia_mm: float = param(30, "mm", "Optional adapter flange diameter", "", "Metal design!C183")
    adapter_flange_mm: float = param(4.5, "mm", "Optional adapter flange thickness", "", "Metal design!C184")
    adapter_pilot_dia_mm: float = param(18, "mm", "Optional adapter pilot diameter", "", "Metal design!C185")
    adapter_pilot_mm: float = param(2, "mm", "Optional adapter pilot length", "", "Metal design!C186")
    adapter_boss_mm: float = param(10, "mm", "Optional adapter boss extension", "", "Metal design!C187")
    adapter_hardware_g: float = param(2, "g", "Optional joint extra hardware allowance", "", "Metal design!C190")


# Open validation items from the sheet (label -> (status, what to do)); useful as a GUI checklist.
VALIDATION_ITEMS = {
    "Prototype metrology": ("Open", "Record corner gap, flat gap, radii, overlap, magnet orientation and magnet temperature."),
    "Torque repeatability": ("Open", "Slow torque-angle tests in both directions and multiple positions."),
    "Metal A/B test": ("Open", "Test the metal candidate at the same measured face gap and temperature."),
    "Hot / cold torque": ("Open", "Verify minimum 2.5 N·m and gearbox-safe torque at service temperature limits."),
    "Slip loss test": ("Open", "Measure drag at the design slip speed; add laminations only if needed."),
    "Retention qualification": ("Open", "Magnetic forces, hoop stress, liner buckling, attachment fatigue, bond-loss capture."),
    "Lifetime cycling": ("Open", "Convert slip events to magnetic load cycles; test reversals, shock, vibration, dwell."),
    "Environmental test": ("Open", "Thermal cycling, humidity/condensation and ingress, then recheck torque."),
    "Robot integration": ("Open", "Record gearbox speed and torque, bus voltage, swing tracking, re-engagement."),
    "Fault handling": ("Open", "Detect sustained clutch slip using wheel/vehicle feedback."),
    "Control tuning": ("Open", "Measure resonance and slip ripple; tune the traction loop."),
    "Production acceptance": ("Open", "Measure each assembly or establish process capability."),
}


@dataclass
class RetainerResults:
    retainer_span_mm: float = out("mm", "Proposed retainer axial span", cell="Metal design!C45")
    retainers_g: float = out("g", "Approximate sleeve / liner mass", cell="Metal design!C46")
    sleeve_id_mm: float = out("mm", "Inner sleeve nominal ID", cell="Metal design!C175")
    sleeve_od_mm: float = out("mm", "Inner sleeve nominal OD", cell="Metal design!C176")
    liner_od_mm: float = out("mm", "Outer liner nominal OD", cell="Metal design!C177")
    liner_id_mm: float = out("mm", "Outer liner nominal ID", cell="Metal design!C178")
    endplate_od_mm: float = out("mm", "Inner endplate nominal OD", cell="Metal design!C179")
    cap_g: float = out("g", "Aluminium cap estimated gross mass", cell="Metal design!C180")
    endplates_g: float = out("g", "Two inner endplates estimated mass", cell="Metal design!C181")


@dataclass
class MetalDesignResults:
    torque_op_Nm: float = out("N·m", "Predicted candidate torque at operating temperature", cell="Metal design!C5")
    torque_20C_Nm: float = out("N·m", "Predicted candidate torque at 20 °C", cell="Metal design!C6")
    torque_cold_Nm: float = out("N·m", "Predicted candidate torque at cold temperature", cell="Metal design!C8")
    torque_hot_low_Nm: float = out("N·m", "Hot-side low torque including assumed variation", cell="Metal design!C9")
    torque_cold_high_Nm: float = out("N·m", "Cold torque with assumed positive variation", cell="Metal design!C10")
    hot_min_check: str = out("", "Hot minimum with variation allowance", cell="Metal design!C11")
    running_clearance_mm: float = out("mm", "Remaining radial clearance, screening estimate", cell="Metal design!C12")
    op_temp_C: float = out("°C", "Operating / maximum magnet temperature", cell="Metal design!C15")
    alpha_br_per_C: float = out("1/°C", "Br temperature coefficient", cell="Metal design!C17")
    required_20C_Nm: float = out("N·m", "Required nominal 20 °C torque for hot minimum", cell="Metal design!C19")
    hot_margin: float = out("fraction", "Nominal hot margin above the requirement", cell="Metal design!C20")
    corner_gap_mm: float = out("mm", "Magnet corner-to-opposing-face gap", cell="Metal design!C24")
    sleeve_liner_clearance_mm: float = out("mm", "Nominal sleeve-to-liner radial clearance", cell="Metal design!C27")
    adverse_movement_mm: float = out("mm", "Total adverse radial movement", cell="Metal design!C34")
    min_running_clearance_mm: float = out("mm", "Minimum running clearance after allowances", cell="Metal design!C35")
    clearance_check: str = out("", "Clearance screening", cell="Metal design!C37")
    rotating_mass_g: float = out("g", "Modeled rotating coupling mass", cell="Metal design!C47")
    slip_freq_Hz: float = out("Hz", "Pole-pair slip frequency", cell="Metal design!C86")
    magnetic_cycles: float = out("cycles", "Approximate magnetic cycles over life", cell="Metal design!C89")
    slip_loss_W: object = out("W", "Slip-loss power", "'not measured' until a bench drag torque is entered.", "Metal design!C91")
    slip_energy_J: object = out("J", "Slip-loss energy per event", cell="Metal design!C92")
    required_20C_zero_scatter_Nm: float = out("N·m", "20 °C torque needed at hot limit, zero scatter", cell="Metal design!C111")
    torque_cold_zero_var_Nm: float = out("N·m", "Predicted cold torque, zero variation", cell="Metal design!C112")
    axial_stack_mm: float = out("mm", "Proposed total axial stack", cell="Metal design!C134")
    rotating_od_mm: float = out("mm", "Proposed rotating outside diameter", cell="Metal design!C135")
    diameter_reserve_mm: float = out("mm", "Diameter reserve", cell="Metal design!C136")
    large_dia_stack_mm: float = out("mm", "Proposed large-diameter axial stack", cell="Metal design!C137")
    large_dia_reserve_mm: float = out("mm", "Large-diameter axial reserve", cell="Metal design!C138")
    axial_reserve_mm: float = out("mm", "Overall axial reserve", cell="Metal design!C139")
    installed_magnets: float = out("count", "Installed magnets", cell="Metal design!C140")
    assembled_face_gap_mm: float = out("mm", "Assembled face gap", cell="Metal design!C141")
    corner_clearance_mm: float = out("mm", "Corner clearance before sleeves", cell="Metal design!C142")
    nominal_sleeve_liner_mm: float = out("mm", "Nominal sleeve-to-liner clearance", cell="Metal design!C143")
    allowed_radial_disp_mm: float = out("mm", "Allowed relative radial displacement", cell="Metal design!C144")
    steel_cup_mass_g: float = out("g", "Selected one-piece steel-cup mass", cell="Metal design!C147")
    adapter_variant_mass_g: float = out("g", "Optional aluminium-adapter variant mass", cell="Metal design!C148")
    adapter_mass_saved_g: float = out("g", "Mass saved by optional aluminium adapter", cell="Metal design!C149")
    retainers_mass_g: float = out("g", "Estimated mass of both thin retainers", cell="Metal design!C150")
    noiron_baseline_hot_Nm: float = out("N·m", "No-back-iron baseline hot torque", "Measured prototype, temperature-scaled.", "Metal design!C151")
    cold_for_hot_min_Nm: float = out("N·m", "Cold torque corresponding to the hot minimum", cell="Metal design!C155")
    cold_for_hot_min_input_Nm: float = out("N·m", "Input-equivalent torque for value above", cell="Metal design!C156")
    cold_high_Nm: float = out("N·m", "Predicted candidate cold high, with allowance", cell="Metal design!C157")
    cold_high_input_Nm: float = out("N·m", "Candidate cold high, input equivalent", cell="Metal design!C158")
    cup_body_od_mm: float = out("mm", "Steel cup body OD", cell="Metal design!C165")
    cap_face_mm: float = out("mm", "Front cap face thickness", cell="Metal design!C167")
    adapter_g: float = out("g", "Optional aluminium adapter gross mass", cell="Metal design!C188")
    adapter_steel_removed_g: float = out("g", "Steel removed for optional larger pilot bore", cell="Metal design!C189")
    hybrid_mass_g: float = out("g", "Optional hybrid gross mass", cell="Metal design!C191")
    hybrid_length_mm: float = out("mm", "Optional hybrid overall length", cell="Metal design!C192")


def retainers(md: MetalDesignInputs, inner_back_apothem_mm: float, inner_thickness_mm: float, inner_width_mm: float,
              outer_face_apothem_mm: float, bore_mm: float) -> RetainerResults:
    """Sleeve, liner, endplate and cap geometry and mass (rows 45–46, 175–181)."""
    sleeve_id = 2 * (math.sqrt((inner_back_apothem_mm + inner_thickness_mm) ** 2 + (inner_width_mm / 2) ** 2) + md.sleeve_bedding_mm)
    sleeve_od = sleeve_id + 2 * md.sleeve_mm
    liner_od = 2 * (outer_face_apothem_mm - md.liner_bedding_mm)
    liner_id = liner_od - 2 * md.liner_mm
    span = md.retainer_span_mm
    m_ret = math.pi / 4 * (sleeve_od ** 2 - sleeve_id ** 2 + liner_od ** 2 - liner_id ** 2) * span * md.sleeve_density_g_mm3
    cap = (math.pi / 4 * (md.cap_od_mm ** 2 - liner_id ** 2) * md.cap_axial_mm
           + math.pi / 4 * (md.cap_od_mm ** 2 - md.cap_thread_dia_mm ** 2) * md.cap_thread_engagement_mm) * md.al_density_g_mm3
    endplate_od = sleeve_id
    endplates = math.pi / 4 * ((endplate_od ** 2 - bore_mm ** 2) * md.front_endplate_mm
                               + (endplate_od ** 2 - md.rear_endplate_hole_mm ** 2) * md.rear_endplate_mm) * md.sleeve_density_g_mm3
    return RetainerResults(retainer_span_mm=span, retainers_g=m_ret, sleeve_id_mm=sleeve_id, sleeve_od_mm=sleeve_od,
                           liner_od_mm=liner_od, liner_id_mm=liner_id, endplate_od_mm=endplate_od, cap_g=cap, endplates_g=endplates)


def compute(md: MetalDesignInputs, torque_op_Nm: float, torque_20C_Nm: float, op_temp_C: float, alpha_br: float,
            corner_gap_mm: float, face_gap_mm: float, cup_od_mm: float, npole: int, bore_mm: float,
            gear_ratio: float, gear_eff: float, mass_total_g: float, boss_mass_g: float, ret: RetainerResults,
            proto_measured_Nm: float, proto_test_temp_C: float) -> MetalDesignResults:
    th = lambda T: 1 + alpha_br * (T - 20)
    cold = torque_20C_Nm * th(md.min_temp_C) ** 2
    hot_low = torque_op_Nm * (1 - md.variation)
    cold_high = cold * (1 + md.variation)
    clearance = (ret.liner_id_mm - ret.sleeve_od_mm) / 2
    adverse = (md.shaft_displacement_mm + md.runout_mm + md.deflection_mm + md.thermal_mm
               + md.sleeve_form_mm + md.magnet_position_mm)
    min_run = clearance - adverse
    slip_f = npole / 2 * md.slip_rpm / 60
    if isinstance(md.measured_drag_Nm, (int, float)):
        loss = md.measured_drag_Nm * 2 * math.pi * md.slip_rpm / 60
        energy = loss * md.slip_event_s
    else:
        loss = energy = "not measured"
    stack = md.cap_axial_mm + md.cup_depth_mm + md.web_mm + md.boss_length_mm
    rot_od = max(cup_od_mm, md.cap_od_mm)
    large = md.cap_axial_mm + md.cup_depth_mm + md.web_mm
    adapter = math.pi / 4 * ((md.adapter_flange_dia_mm ** 2 - bore_mm ** 2) * md.adapter_flange_mm
                             + (md.boss_od_mm ** 2 - bore_mm ** 2) * md.adapter_boss_mm
                             + (md.adapter_pilot_dia_mm ** 2 - bore_mm ** 2) * md.adapter_pilot_mm) * md.al_density_g_mm3
    removed = math.pi / 4 * (md.adapter_pilot_dia_mm ** 2 - bore_mm ** 2) * md.web_mm * md.steel_density_g_mm3
    hybrid = mass_total_g - boss_mass_g - removed + adapter + md.adapter_hardware_g
    cold_for_min = md.required_min_Nm * (th(md.min_temp_C) / th(op_temp_C)) ** 2
    return MetalDesignResults(
        torque_op_Nm=torque_op_Nm, torque_20C_Nm=torque_20C_Nm, torque_cold_Nm=cold, torque_hot_low_Nm=hot_low,
        torque_cold_high_Nm=cold_high,
        hot_min_check="Below hot minimum" if hot_low < md.required_min_Nm else "Estimate covers hot min",
        running_clearance_mm=min_run, op_temp_C=op_temp_C, alpha_br_per_C=alpha_br,
        required_20C_Nm=md.required_min_Nm / (th(op_temp_C) ** 2 * (1 - md.variation)),
        hot_margin=torque_op_Nm / md.required_min_Nm - 1, corner_gap_mm=corner_gap_mm,
        sleeve_liner_clearance_mm=clearance, adverse_movement_mm=adverse, min_running_clearance_mm=min_run,
        clearance_check="Below target" if min_run < md.residual_target_mm else "Meets assumed target",
        rotating_mass_g=mass_total_g, slip_freq_Hz=slip_f, magnetic_cycles=slip_f * md.slip_event_s * md.life_events,
        slip_loss_W=loss, slip_energy_J=energy, required_20C_zero_scatter_Nm=md.required_min_Nm / th(op_temp_C) ** 2,
        torque_cold_zero_var_Nm=cold, axial_stack_mm=stack, rotating_od_mm=rot_od,
        diameter_reserve_mm=md.max_diameter_mm - rot_od, large_dia_stack_mm=large,
        large_dia_reserve_mm=md.max_large_dia_axial_mm - large, axial_reserve_mm=md.max_overall_axial_mm - stack,
        installed_magnets=2 * npole, assembled_face_gap_mm=face_gap_mm, corner_clearance_mm=corner_gap_mm,
        nominal_sleeve_liner_mm=clearance,
        allowed_radial_disp_mm=clearance - (md.runout_mm + md.deflection_mm + md.thermal_mm + md.sleeve_form_mm
                                            + md.magnet_position_mm) - md.residual_target_mm,
        steel_cup_mass_g=mass_total_g, adapter_variant_mass_g=hybrid, adapter_mass_saved_g=mass_total_g - hybrid,
        retainers_mass_g=ret.retainers_g,
        noiron_baseline_hot_Nm=proto_measured_Nm * (th(op_temp_C) / th(proto_test_temp_C)) ** 2,
        cold_for_hot_min_Nm=cold_for_min, cold_for_hot_min_input_Nm=cold_for_min / (gear_ratio * gear_eff),
        cold_high_Nm=cold_high, cold_high_input_Nm=cold_high / (gear_ratio * gear_eff), cup_body_od_mm=cup_od_mm,
        cap_face_mm=md.cap_axial_mm, adapter_g=adapter, adapter_steel_removed_g=removed, hybrid_mass_g=hybrid,
        hybrid_length_mm=md.cap_axial_mm + md.cup_depth_mm + md.web_mm + md.adapter_flange_mm + md.adapter_boss_mm,
    )
