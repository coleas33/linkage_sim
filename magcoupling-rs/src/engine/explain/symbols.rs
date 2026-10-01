//! Display symbols of the terms that have no record of their own: every input a formula
//! reads, and any result shown with its workbook cell only. A record's target takes the
//! record's symbol. The registry refuses a term without a symbol, an entry no formula
//! reads, an entry for a path with a record, and two paths with one symbol.
//!
//! Conventions (plan A-3): T torque, σ shear stress, ϑ temperature, φ electrical
//! angle, τ_p pole pitch; `^{cal}` marks the Calibration prototype; a selector is an upright
//! word (`faceted`), which the typesetter writes with the choice's label in conditions.

/// (path, symbol markup), grouped by input group.
pub const SYMBOLS: &[(&str, &str)] = &[
    // coupling
    ("coupling.npole", "N"),
    ("coupling.backiron", "backiron"),
    ("coupling.faceted", "faceted"),
    ("coupling.inner_back_apothem_mm", "a_i"),
    ("coupling.op_temp_C", "ϑ_{op}"),
    ("coupling.c_end", "c_{end}"),
    ("coupling.max_harmonic", "N_h"),
    ("coupling.mu0", "μ_0"),
    ("coupling.gear_ratio", "i_g"),
    ("coupling.gear_efficiency", "η_g"),
    ("coupling.magnets.part_inner", "part_i"),
    ("coupling.magnets.part_outer", "part_o"),
    ("coupling.magnets.grade_inner", "grade_i"),
    ("coupling.magnets.grade_outer", "grade_o"),
    ("coupling.magnets.axial_length_mm", "L_{ax}"),
    ("coupling.magnets.manual_inner_length_mm", "L_{i,man}"),
    ("coupling.magnets.manual_outer_length_mm", "L_{o,man}"),
    ("coupling.magnets.manual_inner_width_mm", "w_{i,man}"),
    ("coupling.magnets.manual_outer_width_mm", "w_{o,man}"),
    ("coupling.magnets.manual_inner_thickness_mm", "t_{i,man}"),
    ("coupling.magnets.manual_outer_thickness_mm", "t_{o,man}"),
    ("coupling.magnets.manual_inner_br_T", "B_{r,i,man}"),
    ("coupling.magnets.manual_outer_br_T", "B_{r,o,man}"),
    // metal
    ("metal.face_gap_mm", "g_{face}"),
    ("metal.min_temp_C", "ϑ_{min}"),
    ("metal.variation", "v"),
    ("metal.required_min_Nm", "T_{req}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
    ("calibration.spacing_mm", "s^{cal}"),
    ("calibration.gap_definition", "gapdef"),
    ("calibration.test_temp_C", "ϑ_{test}"),
    ("calibration.apothem_mm", "a^{cal}"),
    ("calibration.magnet_length_mm", "L^{cal}"),
    ("calibration.magnet_width_mm", "w^{cal}"),
    ("calibration.magnet_thickness_mm", "t^{cal}"),
    ("calibration.br_T", "B_r^{cal}"),
    ("calibration.alpha_br_per_C", "α"),
    ("calibration.c_end", "c_{end}^{cal}"),
    ("calibration.f_cal_original", "f_{cal,0}"),
    ("calibration.mu0", "μ_0^{cal}"),
    ("calibration.fea_torque1_Nm", "T_{3D,1}"),
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
];
