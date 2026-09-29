"""Metal design checks (torque band, requirements, radial clearance stack, retainers, axial envelope,
slip duty, adapter variant) and the Calculator gearbox rows C99/C100, against independent
re-derivations. A failing check is a candidate finding for Task 8.

Owned elsewhere (not repeated here): the Br² temperature law on Calculator C93 (Task 3) and the
verdict thresholds Metal design C11 and C37 (Task 7). The values those verdicts compare,
C9, C12 and C35, are checked here."""
from __future__ import annotations

import math

import pytest

from audit.common import TOL_ALGEBRA, defaults, mismatch, rel_err, run, vary
from audit.references.block_geometry import (annulus_part, geometry_for_design, retainer_geometry_for_design,
                                             retainer_parts_for_design)
from audit.references.metal_stack import (gearbox_input_torque, gearbox_output_torque, linear_stack,
                                          pole_pair_frequency_Hz, shaft_power_W, sleeve_liner_clearance, torque_high,
                                          torque_low)
from audit.references.remanence import br_ratio, torque_at

#: 7075-T6 density, 2.81 g/cm³ (Aluminum Association, *Aluminum Standards and Data*; ASM Handbook Vol. 2).
AL7075_DENSITY_G_MM3 = 2.81e-3


# --------------------------------------------------------------------------- torque band
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("torque_cold_Nm", "Metal design!C8"), ("torque_hot_low_Nm", "Metal design!C9"),
                                        ("torque_cold_high_Nm", "Metal design!C10"),
                                        ("torque_cold_zero_var_Nm", "Metal design!C112"),
                                        ("cold_high_Nm", "Metal design!C157")], ids=["C8", "C9", "C10", "C112", "C157"])
def test_torque_band(field, cell):
    """Hot low = pull-out at the hot (operating) temperature × (1 − v); cold = pull-out moved to the minimum
    temperature with torque ∝ Br(T)²; cold high = cold × (1 + v). The reference starts from the 50 °C
    pull-out (C93) while the engine starts from the 20 °C value (C94), so the C94 scaling is cross-checked."""
    inp = defaults()
    res = run(inp)
    md, a = inp.metal, inp.calibration.alpha_br_per_C
    cold = torque_at(res.model.pullout_Nm, a, inp.coupling.op_temp_C, md.min_temp_C)
    ref = {"torque_cold_Nm": cold, "torque_hot_low_Nm": torque_low(res.model.pullout_Nm, md.variation),
           "torque_cold_high_Nm": torque_high(cold, md.variation), "torque_cold_zero_var_Nm": cold,
           "cold_high_Nm": torque_high(cold, md.variation)}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_cold_torque_equals_model_run_cold():
    """Metal design!C8 (cold torque by Br² scaling) equals the Calculator pull-out computed directly at the
    minimum temperature (an independent path through the engine's own torque model)."""
    inp = defaults()
    got = run(inp).metal.torque_cold_Nm
    ref = run(vary(inp, {"coupling.op_temp_C": inp.metal.min_temp_C})).model.pullout_Nm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("cold torque vs model run at min temp", "Metal design!C8, Calculator!C93",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("required_20C_Nm", "Metal design!C19"), ("required_20C_zero_scatter_Nm", "Metal design!C111"),
                                        ("hot_margin", "Metal design!C20"), ("cold_for_hot_min_Nm", "Metal design!C155"),
                                        ("cold_for_hot_min_input_Nm", "Metal design!C156"),
                                        ("cold_high_input_Nm", "Metal design!C158"),
                                        ("noiron_baseline_hot_Nm", "Metal design!C151")],
                         ids=["C19", "C111", "C20", "C155", "C156", "C158", "C151"])
def test_requirement_derivations(field, cell):
    """Requirements by inverting the torque band: the 20 °C nominal that leaves hot low = required
    (T20·(Br(hot)/Br20)²·(1 − v) = T_req), the same with v = 0, the nominal hot margin, the cold torque of a
    design whose hot nominal equals the requirement, gearbox-input equivalents T/(i·η) (driving), and the
    no-iron prototype moved from its test temperature to the operating temperature."""
    inp = defaults()
    res = run(inp)
    md, c, cal = inp.metal, inp.coupling, inp.calibration
    a, hot, req = cal.alpha_br_per_C, c.op_temp_C, md.required_min_Nm
    cold_for_min = torque_at(req, a, hot, md.min_temp_C)
    cold_high = torque_high(torque_at(res.model.pullout_Nm, a, hot, md.min_temp_C), md.variation)
    ref = {
        "required_20C_Nm": req / (br_ratio(a, hot) ** 2 * (1 - md.variation)),
        "required_20C_zero_scatter_Nm": torque_at(req, a, hot, 20.0),
        "hot_margin": res.model.pullout_Nm / req - 1,
        "cold_for_hot_min_Nm": cold_for_min,
        "cold_for_hot_min_input_Nm": gearbox_input_torque(cold_for_min, c.gear_ratio, c.gear_efficiency),
        "cold_high_input_Nm": gearbox_input_torque(cold_high, c.gear_ratio, c.gear_efficiency),
        "noiron_baseline_hot_Nm": torque_at(cal.measured_torque_Nm, a, cal.test_temp_C, hot),
    }[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


#: Gearbox scenarios: defaults, and another ratio and efficiency in the free-space circuit (a different C93).
GEAR_SCENARIOS = {"defaults": {},
                  "ratio 7, efficiency 0.8, free space": {"coupling.gear_ratio": 7, "coupling.gear_efficiency": 0.8,
                                                          "coupling.backiron": 0}}


@pytest.mark.family("torque")
@pytest.mark.parametrize("cell", ["Calculator!C99", "Calculator!C100"])
def test_gearbox_equivalents(cell):
    """Calculator!C99: the gearbox input torque while the coupling carries its pull-out torque C93, T/(i·η)
    (motor driving, the conservative form also used for Metal design C156/C158). Calculator!C100: the
    output-side torque that loads the gearbox input to its rating C47, i·η·T_rating (power balance, same
    convention), so C99 ≤ C47 exactly when C93 ≤ C100. Sole owner of C99 and C100; one check per cell covers
    every GEAR_SCENARIOS entry and lists each failing scenario."""
    failures = []
    for name, changes in GEAR_SCENARIOS.items():
        inp = vary(defaults(), changes)
        res = run(inp)
        c = inp.coupling
        if cell == "Calculator!C99":
            got, ref = res.model.gearbox_input_ripple_Nm, gearbox_input_torque(res.model.pullout_Nm, c.gear_ratio, c.gear_efficiency)
        else:
            got, ref = res.model.gearbox_reference_Nm, gearbox_output_torque(c.gearbox_input_rating_Nm, c.gear_ratio, c.gear_efficiency)
        if not rel_err(got, ref) < TOL_ALGEBRA:
            failures.append(mismatch(f"gearbox equivalent ({name})", f"{cell}, Calculator!C45, C46, C47", got, ref, TOL_ALGEBRA))
    assert not failures, " | ".join(failures)


# --------------------------------------------------------------------------- radial clearance stack
@pytest.mark.family("metal")
@pytest.mark.parametrize("changes", [{}, {"metal.shaft_displacement_mm": 0.1}, {"metal.face_gap_mm": 2.0}], ids=["defaults", "shaft 0.1", "gap 2.0"])
@pytest.mark.parametrize("field,cell", [("sleeve_liner_clearance_mm", "Metal design!C27"), ("nominal_sleeve_liner_mm", "Metal design!C143"),
                                        ("adverse_movement_mm", "Metal design!C34"), ("min_running_clearance_mm", "Metal design!C35"),
                                        ("running_clearance_mm", "Metal design!C12"), ("allowed_radial_disp_mm", "Metal design!C144")],
                         ids=["C27", "C143", "C34", "C35", "C12", "C144"])
def test_radial_clearance_stack(changes, field, cell):
    """Nominal sleeve-to-liner clearance = reference corner gap − sleeve (0.10) − liner (0.20) − both beddings;
    adverse movement = linear sum of shaft float, runout, deflection, thermal, sleeve form and magnet
    position; running clearance = nominal − adverse; allowed shaft displacement = nominal − the other five
    allowances − the residual target."""
    inp = vary(defaults(), changes)
    res = run(inp)
    md = inp.metal
    nominal = sleeve_liner_clearance(geometry_for_design(inp, res).corner_gap, md.sleeve_mm, md.liner_mm,
                                     md.sleeve_bedding_mm, md.liner_bedding_mm)
    others = [md.runout_mm, md.deflection_mm, md.thermal_mm, md.sleeve_form_mm, md.magnet_position_mm]
    adverse = linear_stack(md.shaft_displacement_mm, *others)
    ref = {"sleeve_liner_clearance_mm": nominal, "nominal_sleeve_liner_mm": nominal, "adverse_movement_mm": adverse,
           "min_running_clearance_mm": nominal - adverse, "running_clearance_mm": nominal - adverse,
           "allowed_radial_disp_mm": nominal - linear_stack(*others) - md.residual_target_mm}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{field} ({changes or 'defaults'})", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- retainers
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell,attr", [("sleeve_id_mm", "Metal design!C175", "sleeve_id"), ("sleeve_od_mm", "Metal design!C176", "sleeve_od"),
                                             ("liner_od_mm", "Metal design!C177", "liner_od"), ("liner_id_mm", "Metal design!C178", "liner_id"),
                                             ("endplate_od_mm", "Metal design!C179", "endplate_od")],
                         ids=["C175", "C176", "C177", "C178", "C179"])
def test_retainer_geometry(field, cell, attr):
    """Sleeve bore clears the inner ring's largest radius (block corners, from explicit rectangles) by the
    bedding clearance; liner OD clears the outer ring's smallest radius (point-to-segment distance from the
    axis to the outer block faces); walls add inward/outward; endplates cap the magnets to the sleeve bore.
    (Arc mode is test_geometry_mass.py::test_arc_mode_uses_arc_geometry.)"""
    inp = defaults()
    res = run(inp)
    ref = getattr(retainer_geometry_for_design(inp, res), attr)
    got = getattr(res.retainers, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell,parts", [("retainers_g", "Metal design!C46", ("sleeve", "liner")),
                                              ("cap_g", "Metal design!C180", ("cap",)),
                                              ("endplates_g", "Metal design!C181", ("endplates",))],
                         ids=["C46", "C180", "C181"])
def test_retainer_masses(field, cell, parts):
    """316L sleeve and liner tubes over the retainer span; aluminium cap face plate (cap OD to the liner-ID
    opening) plus threaded skirt (cap OD to thread diameter over the engagement); front endplate with the
    bore opening and rear endplate with the screw-clearance opening. Metal design!C45 span and C150 link
    are checked with the other links."""
    inp = defaults()
    res = run(inp)
    rp = retainer_parts_for_design(inp, res)
    ref = math.fsum(rp[p].mass_g for p in parts)
    got = getattr(res.retainers, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- axial envelope
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("axial_stack_mm", "Metal design!C134"), ("rotating_od_mm", "Metal design!C135"),
                                        ("diameter_reserve_mm", "Metal design!C136"), ("large_dia_stack_mm", "Metal design!C137"),
                                        ("large_dia_reserve_mm", "Metal design!C138"), ("axial_reserve_mm", "Metal design!C139"),
                                        ("hybrid_length_mm", "Metal design!C192")],
                         ids=["C134", "C135", "C136", "C137", "C138", "C139", "C192"])
def test_axial_envelope(field, cell):
    """Axial stacks: the large-diameter region is cap face + cup cavity + rear web (the cap skirt overlaps the
    cup body); overall adds the boss; the hybrid replaces the boss with adapter flange + boss extension.
    Reserves are envelope minus stack: 20 mm bay, 35 mm overall, 43 mm diameter; the rotating OD is the
    larger of the cup body and the cap."""
    inp = defaults()
    res = run(inp)
    md = inp.metal
    large = linear_stack(md.cap_axial_mm, md.cup_depth_mm, md.web_mm)
    overall = linear_stack(large, md.boss_length_mm)
    rot_od = max(geometry_for_design(inp, res).cup_od, md.cap_od_mm)
    ref = {"axial_stack_mm": overall, "rotating_od_mm": rot_od, "diameter_reserve_mm": md.max_diameter_mm - rot_od,
           "large_dia_stack_mm": large, "large_dia_reserve_mm": md.max_large_dia_axial_mm - large,
           "axial_reserve_mm": md.max_overall_axial_mm - overall,
           "hybrid_length_mm": linear_stack(large, md.adapter_flange_mm, md.adapter_boss_mm)}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- slip duty
@pytest.mark.family("metal")
def test_slip_duty():
    """Pole-pair frequency (N/2)·rpm/60 (Metal design!C86, also the ripple frequency Calculator!C97) and
    magnetic cycles over life = frequency × event duration × events (C89)."""
    inp = defaults()
    res = run(inp)
    md = inp.metal
    f = pole_pair_frequency_Hz(inp.coupling.npole, md.slip_rpm)
    for got, ref, cell in ((res.metal.slip_freq_Hz, f, "Metal design!C86"), (res.model.ripple_freq_Hz, f, "Calculator!C97"),
                           (res.metal.magnetic_cycles, f * md.slip_event_s * md.life_events, "Metal design!C89")):
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("slip duty", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("drag", [None, 0.04, 0.25])
def test_slip_loss_from_measured_drag(drag):
    """Slip loss P = T_drag·2π·rpm/60 (Metal design!C91) and energy per event P·t (C92); 'not measured' until a
    bench drag torque is entered."""
    inp = vary(defaults(), {"metal.measured_drag_Nm": drag})
    m = run(inp).metal
    if drag is None:
        for got, cell in ((m.slip_loss_W, "Metal design!C91"), (m.slip_energy_J, "Metal design!C92")):
            assert got == "not measured", mismatch(f"slip loss placeholder {got!r}", cell, float(got == "not measured"), 1.0, 0.0)
        return
    p = shaft_power_W(drag, inp.metal.slip_rpm)
    for got, ref, cell in ((m.slip_loss_W, p, "Metal design!C91"), (m.slip_energy_J, p * inp.metal.slip_event_s, "Metal design!C92")):
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"slip loss at {drag} N·m", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- adapter variant
def _adapter_mass(md, bore: float, density: float) -> float:
    """Optional aluminium adapter as three tubes on the bore: flange, boss extension (boss OD) and pilot."""
    return (annulus_part(md.adapter_flange_dia_mm, bore, md.adapter_flange_mm, density).mass_g
            + annulus_part(md.boss_od_mm, bore, md.adapter_boss_mm, density).mass_g
            + annulus_part(md.adapter_pilot_dia_mm, bore, md.adapter_pilot_mm, density).mass_g)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("adapter_g", "Metal design!C188"), ("adapter_steel_removed_g", "Metal design!C189"),
                                        ("hybrid_mass_g", "Metal design!C191"), ("adapter_variant_mass_g", "Metal design!C148"),
                                        ("adapter_mass_saved_g", "Metal design!C149")],
                         ids=["C188", "C189", "C191", "C148", "C149"])
def test_adapter_variant(field, cell):
    """Hybrid = steel-cup total − integral boss − web ring opened to the adapter pilot bore + adapter + its
    hardware, with the adapter at the workbook's aluminium density (Metal design!C42)."""
    inp = defaults()
    res = run(inp)
    md, bore = inp.metal, inp.coupling.bore_mm
    adapter = _adapter_mass(md, bore, md.al_density_g_mm3)
    removed = annulus_part(md.adapter_pilot_dia_mm, bore, md.web_mm, md.steel_density_g_mm3).mass_g
    hybrid = res.mass.total_g - res.mass.boss_g - removed + adapter + md.adapter_hardware_g
    ref = {"adapter_g": adapter, "adapter_steel_removed_g": removed, "hybrid_mass_g": hybrid,
           "adapter_variant_mass_g": hybrid, "adapter_mass_saved_g": res.mass.total_g - hybrid}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_adapter_mass_with_7075_density():
    """The Materials plan (materials.py docstring) puts the clamp collars and adapters in 7075-T6 (2.81 g/cm³)
    and the cap in 6061-T6 (2.70 g/cm³), but Metal design!C42 gives the adapter the 6061 density.
    Reference: adapter at 7075 density."""
    inp = defaults()
    ref = _adapter_mass(inp.metal, inp.coupling.bore_mm, AL7075_DENSITY_G_MM3)
    got = run(inp).metal.adapter_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("adapter mass at 7075 density", "Metal design!C188, Metal design!C42",
                                                      got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- links
@pytest.mark.family("metal")
@pytest.mark.parametrize("cell", ["Calculator!C35", "Metal design!C5", "Metal design!C6", "Metal design!C15", "Metal design!C17", "Metal design!C24",
                                  "Metal design!C45", "Metal design!C47", "Metal design!C140", "Metal design!C141",
                                  "Metal design!C142", "Metal design!C147", "Metal design!C150", "Metal design!C165",
                                  "Metal design!C167"])
def test_linked_cells(cell):
    """Cells that link another sheet or an input: each must carry exactly its source value."""
    inp = defaults()
    res = run(inp)
    m, md, g = res.metal, inp.metal, geometry_for_design(inp, res)
    got, ref = {
        "Calculator!C35": (res.model.alpha_br_per_C, inp.calibration.alpha_br_per_C),
        "Metal design!C5": (m.torque_op_Nm, res.model.pullout_Nm),
        "Metal design!C6": (m.torque_20C_Nm, res.model.pullout_20C_Nm),
        "Metal design!C15": (m.op_temp_C, inp.coupling.op_temp_C),
        "Metal design!C17": (m.alpha_br_per_C, inp.calibration.alpha_br_per_C),
        "Metal design!C24": (m.corner_gap_mm, g.corner_gap),
        "Metal design!C45": (res.retainers.retainer_span_mm, md.retainer_span_mm),
        "Metal design!C47": (m.rotating_mass_g, res.mass.total_g),
        "Metal design!C140": (m.installed_magnets, 2 * inp.coupling.npole),
        "Metal design!C141": (m.assembled_face_gap_mm, md.face_gap_mm),
        "Metal design!C142": (m.corner_clearance_mm, g.corner_gap),
        "Metal design!C147": (m.steel_cup_mass_g, res.mass.total_g),
        "Metal design!C150": (m.retainers_mass_g, res.retainers.retainers_g),
        "Metal design!C165": (m.cup_body_od_mm, g.cup_od),
        "Metal design!C167": (m.cap_face_mm, md.cap_axial_mm),
    }[cell]
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("linked value", cell, got, ref, TOL_ALGEBRA)
