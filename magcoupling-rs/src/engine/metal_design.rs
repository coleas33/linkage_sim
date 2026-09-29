//! Metal design sheet ('Metal design'): inputs only so far.
//!
//! Port of `reference/magcoupling-py/magcoupling/metal_design.py`. This module
//! holds [`MetalDesignInputs`] (48 fields); the retainers, the sheet's results
//! and `VALIDATION_ITEMS` follow with their ports.
//!
//! Planned deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): E8 (Metal design!C175).

use super::meta::{inputs, param};

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
