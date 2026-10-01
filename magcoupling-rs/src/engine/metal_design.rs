//! Metal design sheet ('Metal design').
//!
//! Port of `reference/magcoupling-py/magcoupling/metal_design.py`: the inputs
//! ([`MetalDesignInputs`], 48 fields), the open validation checklist
//! ([`VALIDATION_ITEMS`]), the retainers ([`RetainerResults`], [`retainers`],
//! 9 result cells) and the sheet's results ([`MetalDesignResults`],
//! [`compute`], 49 result cells).
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied E8 (Metal design!C175)
//! and E16 (C189: the disc bored out of the web for the adapter pilot is priced
//! at the cup's density, aluminium with no back iron under E9).

use std::f64::consts::PI;

use super::compat::py_max;
use super::deviations::{DeviationId, Deviations};
use super::meta::{NumOrText, inputs, out, param, results};
use super::model::{br_factor, corner_radius, ring_pair_factor};

inputs! {
    /// Metal design inputs (Metal design!C7:C190).
    pub struct MetalDesignInputs {
        fields {
            required_min_Nm: f64 = 2.5 => param("N·m", "Required minimum service pull-out torque",
                "At the 50 °C magnet temperature.", "Metal design!C7")
                .range(0.1, 10.0, 0.01),
            min_temp_C: f64 = -40.0 => param("°C", "Minimum magnet temperature",
                "Cold torque is reported, not capped.", "Metal design!C16")
                .range(-60.0, 20.0, 0.5),
            variation: f64 = 0.15 => param("fraction", "Symmetric torque variation allowance",
                "Engineering allowance, not measured.", "Metal design!C18")
                .range(0.0, 0.5, 0.005)
                .assumption(),
            sleeve_mm: f64 = 0.1 => param("mm", "Inner rotating retaining sleeve thickness",
                "316L band over the inner magnets.", "Metal design!C25")
                .range(0.0, 1.0, 0.01),
            liner_mm: f64 = 0.2 => param("mm", "Outer rotating keeper liner thickness",
                "316L liner inside the outer magnets.", "Metal design!C26")
                .range(0.0, 1.0, 0.01),
            shaft_displacement_mm: f64 = 0.4 => param("mm", "Relative shaft radial displacement allowance",
                "UNCONFIRMED.", "Metal design!C28")
                .range(0.0, 2.0, 0.01),
            runout_mm: f64 = 0.05 => param("mm", "Combined assembled runout allowance",
                "", "Metal design!C29")
                .range(0.0, 0.5, 0.005),
            deflection_mm: f64 = 0.05 => param("mm", "Additional load deflection / tilt allowance",
                "", "Metal design!C30")
                .range(0.0, 0.5, 0.005),
            thermal_mm: f64 = 0.03 => param("mm", "Differential thermal movement allowance",
                "", "Metal design!C31")
                .range(0.0, 0.3, 0.005),
            sleeve_form_mm: f64 = 0.05 => param("mm", "Sleeve fit / thickness / form allowance",
                "", "Metal design!C32")
                .range(0.0, 0.5, 0.005),
            magnet_position_mm: f64 = 0.2 => param("mm", "Magnet position / size allowance",
                "UNCONFIRMED.", "Metal design!C33")
                .range(0.0, 1.0, 0.01),
            residual_target_mm: f64 = 0.2 => param("mm", "Minimum desired residual running clearance",
                "", "Metal design!C36")
                .range(0.0, 1.0, 0.01),
            al_density_g_mm3: f64 = 0.0027 => param("g/mm³", "Aluminium cap / adapter density",
                "", "Metal design!C42")
                .range(0.0025, 0.003, 0.00001),
            sleeve_density_g_mm3: f64 = 0.008 => param("g/mm³", "Sleeve density",
                "", "Metal design!C44")
                .range(0.004, 0.009, 0.0001),
            steel_density_g_mm3: f64 = 0.00785 => param("g/mm³", "Steel density (4140)",
                "", "Metal design!C132")
                .range(0.0075, 0.0081, 0.00001),
            slip_rpm: f64 = 2000.0 => param("rpm", "Relative slip speed",
                "User: 2,000 rpm at the wheel.", "Metal design!C85")
                .range(100.0, 6000.0, 10.0),
            slip_event_s: f64 = 0.1 => param("s", "Slip duration per event",
                "Illustrative; replace with the recorded value.", "Metal design!C87")
                .range(0.01, 10.0, 0.01)
                .log()
                .assumption(),
            life_events: f64 = 2e7 => param("events", "Life events",
                "Supplied life target.", "Metal design!C88")
                .range(1e4, 1e9, 1000.0)
                .log(),
            measured_drag_Nm: Option<f64> = None => param("N·m", "Measured mean slip drag torque",
                "Enter the bench result; None = not measured.", "Metal design!C90")
                .range(0.001, 1.0, 0.0001)
                .log(),
            face_gap_mm: f64 = 1.4 => param("mm", "Candidate flat-face magnetic gap",
                "Same as the measured prototype.", "Metal design!C119")
                .range(0.3, 5.0, 0.01),
            bond_inner_mm: f64 = 0.05 => param("mm", "Inner magnet back bondline",
                "", "Metal design!C120")
                .range(0.01, 0.2, 0.005),
            bond_outer_mm: f64 = 0.05 => param("mm", "Outer magnet back bondline",
                "", "Metal design!C121")
                .range(0.0, 0.2, 0.005),
            cup_wall_corner_mm: f64 = 1.8 => param("mm", "Minimum outer return ring wall",
                "At the pocket corners. 4140 at 1.5 T needs about 1.9 mm (Materials check).", "Metal design!C122")
                .range(0.5, 6.0, 0.05),
            hub_length_mm: f64 = 13.0 => param("mm", "Steel inner hub axial length",
                "", "Metal design!C123")
                .range(3.0, 40.0, 0.1),
            cup_depth_mm: f64 = 15.5 => param("mm", "Cup cavity axial depth",
                "", "Metal design!C124")
                .range(3.0, 40.0, 0.1),
            web_mm: f64 = 2.5 => param("mm", "Integral steel rear web thickness",
                "", "Metal design!C125")
                .range(0.5, 8.0, 0.1),
            boss_length_mm: f64 = 13.0 => param("mm", "Integral steel boss axial length",
                "", "Metal design!C126")
                .range(0.0, 40.0, 0.1),
            boss_od_mm: f64 = 22.0 => param("mm", "Shaft boss outside diameter",
                "", "Metal design!C127")
                .range(12.0, 40.0, 0.1),
            hardware_g: f64 = 6.0 => param("g", "Keys / screws / lock tab mass allowance",
                "", "Metal design!C128")
                .range(0.0, 30.0, 0.5),
            max_large_dia_axial_mm: f64 = 20.0 => param("mm", "Maximum large-diameter axial region",
                "User supplied.", "Metal design!C129")
                .range(5.0, 60.0, 0.5),
            max_overall_axial_mm: f64 = 35.0 => param("mm", "Maximum overall axial length",
                "User supplied.", "Metal design!C130")
                .range(10.0, 100.0, 0.5),
            max_diameter_mm: f64 = 43.0 => param("mm", "Maximum rotating coupling diameter",
                "User supplied.", "Metal design!C131")
                .range(20.0, 80.0, 0.5),
            cap_axial_mm: f64 = 0.8 => param("mm", "Front cap axial addition",
                "", "Metal design!C133")
                .range(0.2, 5.0, 0.1),
            cap_od_mm: f64 = 42.8 => param("mm", "Threaded cap OD", "", "Metal design!C166")
                .range(20.0, 80.0, 0.1),
            cap_thread_engagement_mm: f64 = 2.0 => param("mm", "Cap thread engagement length",
                "", "Metal design!C168")
                .range(0.5, 8.0, 0.1),
            cap_thread_dia_mm: f64 = 41.0 => param("mm", "Cap thread nominal diameter",
                "M41 × 0.5 concept.", "Metal design!C169")
                .range(20.0, 80.0, 0.1),
            front_endplate_mm: f64 = 0.5 => param("mm", "Inner front endplate thickness",
                "", "Metal design!C170")
                .range(0.1, 3.0, 0.05),
            rear_endplate_mm: f64 = 1.0 => param("mm", "Inner rear endplate thickness",
                "", "Metal design!C171")
                .range(0.1, 3.0, 0.05),
            retainer_span_mm: f64 = 14.5 => param("mm", "Nominal retainer axial span",
                "", "Metal design!C172")
                .range(3.0, 50.0, 0.1),
            sleeve_bedding_mm: f64 = 0.025 => param("mm", "Inner sleeve minimum bedding clearance",
                "", "Metal design!C173")
                .range(0.0, 0.2, 0.005),
            liner_bedding_mm: f64 = 0.025 => param("mm", "Outer liner minimum bedding clearance",
                "", "Metal design!C174")
                .range(0.0, 0.2, 0.005),
            rear_endplate_hole_mm: f64 = 4.5 => param("mm", "Rear endplate screw clearance diameter",
                "", "Metal design!C182")
                .range(0.0, 12.0, 0.1),
            adapter_flange_dia_mm: f64 = 30.0 => param("mm", "Optional adapter flange diameter",
                "", "Metal design!C183")
                .range(10.0, 60.0, 0.5),
            adapter_flange_mm: f64 = 4.5 => param("mm", "Optional adapter flange thickness",
                "", "Metal design!C184")
                .range(0.5, 15.0, 0.1),
            adapter_pilot_dia_mm: f64 = 18.0 => param("mm", "Optional adapter pilot diameter",
                "", "Metal design!C185")
                .range(5.0, 40.0, 0.1),
            adapter_pilot_mm: f64 = 2.0 => param("mm", "Optional adapter pilot length",
                "", "Metal design!C186")
                .range(0.0, 10.0, 0.1),
            adapter_boss_mm: f64 = 10.0 => param("mm", "Optional adapter boss extension",
                "", "Metal design!C187")
                .range(0.0, 30.0, 0.1),
            adapter_hardware_g: f64 = 2.0 => param("g", "Optional joint extra hardware allowance",
                "", "Metal design!C190")
                .range(0.0, 20.0, 0.5),
        }
    }
}

/// Text of Metal design!C91 and C92 until a bench drag torque is entered.
pub const NOT_MEASURED: &str = "not measured";

/// Open validation items from the sheet: (label, status, what to do). A GUI checklist.
pub const VALIDATION_ITEMS: [(&str, &str, &str); 12] = [
    (
        "Prototype metrology",
        "Open",
        "Record corner gap, flat gap, radii, overlap, magnet orientation and magnet temperature.",
    ),
    (
        "Torque repeatability",
        "Open",
        "Slow torque-angle tests in both directions and multiple positions.",
    ),
    (
        "Metal A/B test",
        "Open",
        "Test the metal candidate at the same measured face gap and temperature.",
    ),
    (
        "Hot / cold torque",
        "Open",
        "Verify minimum 2.5 N·m and gearbox-safe torque at service temperature limits.",
    ),
    (
        "Slip loss test",
        "Open",
        "Measure drag at the design slip speed; add laminations only if needed.",
    ),
    (
        "Retention qualification",
        "Open",
        "Magnetic forces, hoop stress, liner buckling, attachment fatigue, bond-loss capture.",
    ),
    (
        "Lifetime cycling",
        "Open",
        "Convert slip events to magnetic load cycles; test reversals, shock, vibration, dwell.",
    ),
    (
        "Environmental test",
        "Open",
        "Thermal cycling, humidity/condensation and ingress, then recheck torque.",
    ),
    (
        "Robot integration",
        "Open",
        "Record gearbox speed and torque, bus voltage, swing tracking, re-engagement.",
    ),
    (
        "Fault handling",
        "Open",
        "Detect sustained clutch slip using wheel/vehicle feedback.",
    ),
    (
        "Control tuning",
        "Open",
        "Measure resonance and slip ripple; tune the traction loop.",
    ),
    (
        "Production acceptance",
        "Open",
        "Measure each assembly or establish process capability.",
    ),
];

results! {
    /// Sleeve, liner, endplate and cap results (Metal design!C45:C46, C175:C181).
    pub struct RetainerResults {
        fields {
            retainer_span_mm: f64 => out("mm", "Proposed retainer axial span", "", "Metal design!C45"),
            retainers_g: f64 => out("g", "Approximate sleeve / liner mass", "", "Metal design!C46"),
            sleeve_id_mm: f64 => out("mm", "Inner sleeve nominal ID", "", "Metal design!C175"),
            sleeve_od_mm: f64 => out("mm", "Inner sleeve nominal OD", "", "Metal design!C176"),
            liner_od_mm: f64 => out("mm", "Outer liner nominal OD", "", "Metal design!C177"),
            liner_id_mm: f64 => out("mm", "Outer liner nominal ID", "", "Metal design!C178"),
            endplate_od_mm: f64 => out("mm", "Inner endplate nominal OD", "", "Metal design!C179"),
            cap_g: f64 => out("g", "Aluminium cap estimated gross mass", "", "Metal design!C180"),
            endplates_g: f64 => out("g", "Two inner endplates estimated mass", "", "Metal design!C181"),
        }
    }
}

results! {
    /// Metal design sheet results (Metal design!C5:C192).
    pub struct MetalDesignResults {
        fields {
            torque_op_Nm: f64 => out("N·m", "Predicted candidate torque at operating temperature", "", "Metal design!C5"),
            torque_20C_Nm: f64 => out("N·m", "Predicted candidate torque at 20 °C", "", "Metal design!C6"),
            torque_cold_Nm: f64 => out("N·m", "Predicted candidate torque at cold temperature", "", "Metal design!C8"),
            torque_hot_low_Nm: f64 => out("N·m", "Hot-side low torque including assumed variation", "", "Metal design!C9"),
            torque_cold_high_Nm: f64 => out("N·m", "Cold torque with assumed positive variation", "", "Metal design!C10"),
            hot_min_check: String => out("", "Hot minimum with variation allowance", "", "Metal design!C11"),
            running_clearance_mm: f64 => out("mm", "Remaining radial clearance, screening estimate", "", "Metal design!C12"),
            op_temp_C: f64 => out("°C", "Operating / maximum magnet temperature", "", "Metal design!C15"),
            alpha_br_per_C: f64 => out("1/°C", "Br temperature coefficient", "", "Metal design!C17"),
            required_20C_Nm: f64 => out("N·m", "Required nominal 20 °C torque for hot minimum", "", "Metal design!C19"),
            hot_margin: f64 => out("fraction", "Nominal hot margin above the requirement", "", "Metal design!C20"),
            corner_gap_mm: f64 => out("mm", "Magnet corner-to-opposing-face gap", "", "Metal design!C24"),
            sleeve_liner_clearance_mm: f64 => out("mm", "Nominal sleeve-to-liner radial clearance", "", "Metal design!C27"),
            adverse_movement_mm: f64 => out("mm", "Total adverse radial movement", "", "Metal design!C34"),
            min_running_clearance_mm: f64 => out("mm", "Minimum running clearance after allowances", "", "Metal design!C35"),
            clearance_check: String => out("", "Clearance screening", "", "Metal design!C37"),
            rotating_mass_g: f64 => out("g", "Modeled rotating coupling mass", "", "Metal design!C47"),
            slip_freq_Hz: f64 => out("Hz", "Pole-pair slip frequency", "", "Metal design!C86"),
            magnetic_cycles: f64 => out("cycles", "Approximate magnetic cycles over life", "", "Metal design!C89"),
            slip_loss_W: NumOrText => out("W", "Slip-loss power", "'not measured' until a bench drag torque is entered.", "Metal design!C91"),
            slip_energy_J: NumOrText => out("J", "Slip-loss energy per event", "", "Metal design!C92"),
            required_20C_zero_scatter_Nm: f64 => out("N·m", "20 °C torque needed at hot limit, zero scatter", "", "Metal design!C111"),
            torque_cold_zero_var_Nm: f64 => out("N·m", "Predicted cold torque, zero variation", "", "Metal design!C112"),
            axial_stack_mm: f64 => out("mm", "Proposed total axial stack", "", "Metal design!C134"),
            rotating_od_mm: f64 => out("mm", "Proposed rotating outside diameter", "", "Metal design!C135"),
            diameter_reserve_mm: f64 => out("mm", "Diameter reserve", "", "Metal design!C136"),
            large_dia_stack_mm: f64 => out("mm", "Proposed large-diameter axial stack", "", "Metal design!C137"),
            large_dia_reserve_mm: f64 => out("mm", "Large-diameter axial reserve", "", "Metal design!C138"),
            axial_reserve_mm: f64 => out("mm", "Overall axial reserve", "", "Metal design!C139"),
            installed_magnets: f64 => out("count", "Installed magnets", "", "Metal design!C140"),
            assembled_face_gap_mm: f64 => out("mm", "Assembled face gap", "", "Metal design!C141"),
            corner_clearance_mm: f64 => out("mm", "Corner clearance before sleeves", "", "Metal design!C142"),
            nominal_sleeve_liner_mm: f64 => out("mm", "Nominal sleeve-to-liner clearance", "", "Metal design!C143"),
            allowed_radial_disp_mm: f64 => out("mm", "Allowed relative radial displacement", "", "Metal design!C144"),
            steel_cup_mass_g: f64 => out("g", "Selected one-piece steel-cup mass",
                "The back-iron material at back iron = 1; with no back iron (correction E9) the body material: the workbook's aluminium, or a non-ferromagnetic back-iron pick (Addendum A5).", "Metal design!C147"),
            adapter_variant_mass_g: f64 => out("g", "Optional aluminium-adapter variant mass", "", "Metal design!C148"),
            adapter_mass_saved_g: f64 => out("g", "Mass saved by optional aluminium adapter", "", "Metal design!C149"),
            retainers_mass_g: f64 => out("g", "Estimated mass of both thin retainers", "", "Metal design!C150"),
            noiron_baseline_hot_Nm: f64 => out("N·m", "No-back-iron baseline hot torque", "Measured prototype, temperature-scaled.", "Metal design!C151"),
            cold_for_hot_min_Nm: f64 => out("N·m", "Cold torque corresponding to the hot minimum", "", "Metal design!C155"),
            cold_for_hot_min_input_Nm: f64 => out("N·m", "Input-equivalent torque for value above", "", "Metal design!C156"),
            cold_high_Nm: f64 => out("N·m", "Predicted candidate cold high, with allowance", "", "Metal design!C157"),
            cold_high_input_Nm: f64 => out("N·m", "Candidate cold high, input equivalent", "", "Metal design!C158"),
            cup_body_od_mm: f64 => out("mm", "Steel cup body OD", "", "Metal design!C165"),
            cap_face_mm: f64 => out("mm", "Front cap face thickness", "", "Metal design!C167"),
            adapter_g: f64 => out("g", "Optional aluminium adapter gross mass", "", "Metal design!C188"),
            adapter_steel_removed_g: f64 => out("g", "Steel removed for optional larger pilot bore",
                "Priced at the cup's density: the back-iron material at back iron = 1; with no back iron the body material: the workbook's aluminium, or a non-ferromagnetic back-iron pick (corrections E9 and E16, Addendum A5).",
                "Metal design!C189"),
            hybrid_mass_g: f64 => out("g", "Optional hybrid gross mass", "", "Metal design!C191"),
            hybrid_length_mm: f64 => out("mm", "Optional hybrid overall length", "", "Metal design!C192"),
        }
    }
}

/// Sleeve, liner, endplate and cap geometry and mass (Metal design rows 45-46, 175-181).
/// `inner_corner_radius_mm` is Calculator!C55; only E8 reads it (Python has no such parameter).
#[allow(clippy::too_many_arguments)] // Python signature plus C55 (E8): 8 parameters
pub fn retainers(
    md: &MetalDesignInputs,
    inner_back_apothem_mm: f64,
    inner_thickness_mm: f64,
    inner_width_mm: f64,
    outer_face_apothem_mm: f64,
    bore_mm: f64,
    inner_corner_radius_mm: f64,
    dev: Deviations,
) -> RetainerResults {
    // E8: the sleeve clears the inner corner radius C55 (the face radius for arcs).
    let sleeve_id = if dev.is_on(DeviationId::E8) {
        2.0 * (inner_corner_radius_mm + md.sleeve_bedding_mm)
    } else {
        2.0 * (corner_radius(inner_back_apothem_mm + inner_thickness_mm, inner_width_mm)
            + md.sleeve_bedding_mm)
    };
    let sleeve_od = sleeve_id + 2.0 * md.sleeve_mm;
    let liner_od = 2.0 * (outer_face_apothem_mm - md.liner_bedding_mm);
    let liner_id = liner_od - 2.0 * md.liner_mm;
    let span = md.retainer_span_mm;
    let m_ret = PI / 4.0
        * (sleeve_od.powi(2) - sleeve_id.powi(2) + liner_od.powi(2) - liner_id.powi(2))
        * span
        * md.sleeve_density_g_mm3;
    let cap = (PI / 4.0 * (md.cap_od_mm.powi(2) - liner_id.powi(2)) * md.cap_axial_mm
        + PI / 4.0
            * (md.cap_od_mm.powi(2) - md.cap_thread_dia_mm.powi(2))
            * md.cap_thread_engagement_mm)
        * md.al_density_g_mm3;
    let endplate_od = sleeve_id;
    let endplates = PI / 4.0
        * ((endplate_od.powi(2) - bore_mm.powi(2)) * md.front_endplate_mm
            + (endplate_od.powi(2) - md.rear_endplate_hole_mm.powi(2)) * md.rear_endplate_mm)
        * md.sleeve_density_g_mm3;
    RetainerResults {
        retainer_span_mm: span,
        retainers_g: m_ret,
        sleeve_id_mm: sleeve_id,
        sleeve_od_mm: sleeve_od,
        liner_od_mm: liner_od,
        liner_id_mm: liner_id,
        endplate_od_mm: endplate_od,
        cap_g: cap,
        endplates_g: endplates,
    }
}

/// The Metal design sheet (Python `metal_design.compute`): torque at the hot and
/// cold limits, the radial clearance stack, duty, the axial stack, the optional
/// aluminium adapter and the hybrid mass. `ret` is [`retainers`]' result;
/// `cup_density_g_mm3` is the density the mass model gives the web
/// (`model::cup_boss_density`), read only by E16. `alpha_br` is the calculator's single
/// alpha (C17; the prototype's baseline C151); `alpha_inner` and `alpha_outer` are each
/// ring's (decision A2-7: a grade-mode ring's grade, else `alpha_br`), and the torques go with
/// their product.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn compute(
    md: &MetalDesignInputs,
    torque_op_Nm: f64,
    torque_20C_Nm: f64,
    op_temp_C: f64,
    alpha_br: f64,
    alpha_inner: f64,
    alpha_outer: f64,
    corner_gap_mm: f64,
    face_gap_mm: f64,
    cup_od_mm: f64,
    npole: i64,
    bore_mm: f64,
    gear_ratio: f64,
    gear_eff: f64,
    mass_total_g: f64,
    boss_mass_g: f64,
    ret: &RetainerResults,
    proto_measured_Nm: f64,
    proto_test_temp_C: f64,
    cup_density_g_mm3: f64,
    dev: Deviations,
) -> MetalDesignResults {
    let th = |T: f64| br_factor(alpha_br, T); // Python lambda th
    // Decision A2-7: torque ~ Br_i(T) Br_o(T), each ring with its coefficient; with one
    // coefficient these are th(T)**2 and (th(a) / th(b))**2, bit for bit.
    let th2 = |T: f64| ring_pair_factor(alpha_inner, alpha_outer, T);
    let ratio2 = |a: f64, b: f64| {
        (br_factor(alpha_inner, a) / br_factor(alpha_inner, b))
            * (br_factor(alpha_outer, a) / br_factor(alpha_outer, b))
    };
    let cold = torque_20C_Nm * th2(md.min_temp_C);
    let hot_low = torque_op_Nm * (1.0 - md.variation);
    let cold_high = cold * (1.0 + md.variation);
    let clearance = (ret.liner_id_mm - ret.sleeve_od_mm) / 2.0;
    let adverse = md.shaft_displacement_mm
        + md.runout_mm
        + md.deflection_mm
        + md.thermal_mm
        + md.sleeve_form_mm
        + md.magnet_position_mm;
    let min_run = clearance - adverse;
    let slip_f = npole as f64 / 2.0 * md.slip_rpm / 60.0;
    let (loss, energy) = match md.measured_drag_Nm {
        Some(drag) => {
            let loss = drag * 2.0 * PI * md.slip_rpm / 60.0;
            (NumOrText::Num(loss), NumOrText::Num(loss * md.slip_event_s))
        }
        None => (NumOrText::Text(NOT_MEASURED), NumOrText::Text(NOT_MEASURED)),
    };
    let stack = md.cap_axial_mm + md.cup_depth_mm + md.web_mm + md.boss_length_mm;
    let rot_od = py_max(cup_od_mm, md.cap_od_mm);
    let large = md.cap_axial_mm + md.cup_depth_mm + md.web_mm;
    let adapter = PI / 4.0
        * ((md.adapter_flange_dia_mm.powi(2) - bore_mm.powi(2)) * md.adapter_flange_mm
            + (md.boss_od_mm.powi(2) - bore_mm.powi(2)) * md.adapter_boss_mm
            + (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2)) * md.adapter_pilot_mm)
        * md.al_density_g_mm3;
    // E16: the disc lies in the web, so it is priced at the web's density (E9 makes it
    // aluminium with no back iron); the workbook always uses steel.
    let web_density = if dev.is_on(DeviationId::E16) {
        cup_density_g_mm3
    } else {
        md.steel_density_g_mm3
    };
    let removed =
        PI / 4.0 * (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2)) * md.web_mm * web_density;
    let hybrid = mass_total_g - boss_mass_g - removed + adapter + md.adapter_hardware_g;
    let cold_for_min = md.required_min_Nm * ratio2(md.min_temp_C, op_temp_C);
    MetalDesignResults {
        torque_op_Nm,
        torque_20C_Nm,
        torque_cold_Nm: cold,
        torque_hot_low_Nm: hot_low,
        torque_cold_high_Nm: cold_high,
        hot_min_check: if hot_low < md.required_min_Nm {
            "Below hot minimum"
        } else {
            "Estimate covers hot min"
        }
        .to_owned(),
        running_clearance_mm: min_run,
        op_temp_C,
        alpha_br_per_C: alpha_br,
        required_20C_Nm: md.required_min_Nm / (th2(op_temp_C) * (1.0 - md.variation)),
        hot_margin: torque_op_Nm / md.required_min_Nm - 1.0,
        corner_gap_mm,
        sleeve_liner_clearance_mm: clearance,
        adverse_movement_mm: adverse,
        min_running_clearance_mm: min_run,
        clearance_check: if min_run < md.residual_target_mm {
            "Below target"
        } else {
            "Meets assumed target"
        }
        .to_owned(),
        rotating_mass_g: mass_total_g,
        slip_freq_Hz: slip_f,
        magnetic_cycles: slip_f * md.slip_event_s * md.life_events,
        slip_loss_W: loss,
        slip_energy_J: energy,
        required_20C_zero_scatter_Nm: md.required_min_Nm / th2(op_temp_C),
        torque_cold_zero_var_Nm: cold,
        axial_stack_mm: stack,
        rotating_od_mm: rot_od,
        diameter_reserve_mm: md.max_diameter_mm - rot_od,
        large_dia_stack_mm: large,
        large_dia_reserve_mm: md.max_large_dia_axial_mm - large,
        axial_reserve_mm: md.max_overall_axial_mm - stack,
        installed_magnets: 2.0 * npole as f64,
        assembled_face_gap_mm: face_gap_mm,
        corner_clearance_mm: corner_gap_mm,
        nominal_sleeve_liner_mm: clearance,
        allowed_radial_disp_mm: clearance
            - (md.runout_mm
                + md.deflection_mm
                + md.thermal_mm
                + md.sleeve_form_mm
                + md.magnet_position_mm)
            - md.residual_target_mm,
        steel_cup_mass_g: mass_total_g,
        adapter_variant_mass_g: hybrid,
        adapter_mass_saved_g: mass_total_g - hybrid,
        retainers_mass_g: ret.retainers_g,
        noiron_baseline_hot_Nm: proto_measured_Nm * (th(op_temp_C) / th(proto_test_temp_C)).powi(2),
        cold_for_hot_min_Nm: cold_for_min,
        cold_for_hot_min_input_Nm: cold_for_min / (gear_ratio * gear_eff),
        cold_high_Nm: cold_high,
        cold_high_input_Nm: cold_high / (gear_ratio * gear_eff),
        cup_body_od_mm: cup_od_mm,
        cap_face_mm: md.cap_axial_mm,
        adapter_g: adapter,
        adapter_steel_removed_g: removed,
        hybrid_mass_g: hybrid,
        hybrid_length_mm: md.cap_axial_mm
            + md.cup_depth_mm
            + md.web_mm
            + md.adapter_flange_mm
            + md.adapter_boss_mm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_ring_scales_the_torque_with_its_own_coefficient() {
        // Decision A2-7: torque goes with Br_inner * Br_outer, each ring with its coefficient;
        // with one coefficient the products are the workbook's squares, bit for bit.
        let ret = RetainerResults {
            retainer_span_mm: 14.5,
            retainers_g: 3.131,
            sleeve_id_mm: 27.436,
            sleeve_od_mm: 27.636,
            liner_od_mm: 29.39,
            liner_id_mm: 28.99,
            endplate_od_mm: 27.436,
            cap_g: 2.322,
            endplates_g: 6.653,
        };
        let run = |alpha_inner: f64, alpha_outer: f64| {
            compute(
                &MetalDesignInputs::default(),
                2.6473,
                2.8487,
                50.0,
                -0.0012,
                alpha_inner,
                alpha_outer,
                1.0268,
                1.4,
                41.2,
                10,
                10.0,
                5.0,
                0.95,
                173.79,
                30.778,
                &ret,
                0.9,
                20.0,
                0.00785,
                Deviations::NONE,
            )
        };
        let md = MetalDesignInputs::default();
        let th = |alpha: f64, t: f64| br_factor(alpha, t);
        let one = run(-0.0012, -0.0012);
        assert_eq!(
            one.torque_cold_Nm,
            2.8487 * th(-0.0012, md.min_temp_C).powi(2)
        );
        assert_eq!(
            one.cold_for_hot_min_Nm,
            md.required_min_Nm * (th(-0.0012, md.min_temp_C) / th(-0.0012, 50.0)).powi(2)
        );
        let mixed = run(-0.002, -0.0012);
        assert_eq!(
            mixed.torque_cold_Nm,
            2.8487 * (th(-0.002, md.min_temp_C) * th(-0.0012, md.min_temp_C))
        );
        assert_eq!(
            mixed.required_20C_zero_scatter_Nm,
            md.required_min_Nm / (th(-0.002, 50.0) * th(-0.0012, 50.0))
        );
        // The prototype's baseline (C151) keeps the calculator's alpha: B842SH rings.
        assert_eq!(mixed.noiron_baseline_hot_Nm, one.noiron_baseline_hot_Nm);
        assert_eq!(mixed.alpha_br_per_C, -0.0012);
    }

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Neither left-hand side (hot_low, min_run) reads its
        // right-hand side, so one run supplies it and a second run puts the check at equality.
        // The default design's retainers, rounded:
        let ret = RetainerResults {
            retainer_span_mm: 14.5,
            retainers_g: 3.131,
            sleeve_id_mm: 27.436,
            sleeve_od_mm: 27.636,
            liner_od_mm: 29.39,
            liner_id_mm: 28.99,
            endplate_od_mm: 27.436,
            cap_g: 2.322,
            endplates_g: 6.653,
        };
        let run = |md: &MetalDesignInputs| {
            compute(
                md,
                2.6473,
                2.8487,
                50.0,
                -0.0012,
                -0.0012,
                -0.0012,
                1.0268,
                1.4,
                41.2,
                10,
                10.0,
                5.0,
                0.95,
                173.79,
                30.778,
                &ret,
                0.9,
                20.0,
                0.00785,
                Deviations::NONE,
            )
        };
        let base = run(&MetalDesignInputs::default());
        let md = MetalDesignInputs {
            required_min_Nm: base.torque_hot_low_Nm,
            residual_target_mm: base.min_running_clearance_mm,
            ..MetalDesignInputs::default()
        };
        let r = run(&md);
        assert_eq!(
            (r.torque_hot_low_Nm, r.min_running_clearance_mm),
            (md.required_min_Nm, md.residual_target_mm)
        );
        assert_eq!(r.hot_min_check, "Estimate covers hot min"); // `hot_low < required` is strict
        assert_eq!(r.clearance_check, "Meets assumed target"); // `min_run < target` is strict
    }
}
