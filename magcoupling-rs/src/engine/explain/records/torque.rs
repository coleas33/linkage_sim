//! Torque chain (plan A-3 batch 1, the tracer): the harmonics (τ1 to τ11 as the shear
//! stresses σ_n), the total shear stress, end effect, calibration factor, pull-out, and the
//! hot and cold torques, plus the geometry, remanence and magnet-resolution records their
//! terms need, so every term drills down to an input.
//!
//! Each formula is transcribed from the engine function named beside it, with every approved
//! correction on (the explorer shows what users see). `tests/explain.rs` evaluates each one
//! over the engine's own term values at the defaults and every differential and augmented
//! input set: a transcription error fails there, never silently.
//!
//! Symbols: σ shear stress, T torque, ϑ temperature, φ electrical angle, τ_p pole pitch
//! (plan A-3; `symbols.rs` lists the conventions).

use crate::engine::deviations::DeviationId::{E3, E7, E8};
use crate::engine::explain::record::{Family, Record, family, record};
use crate::engine::model::ODD_HARMONICS;

/// The harmonic families: one record per odd harmonic up to 11, summed or not.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // model::shear_stress: k = n (N/2) / R_g.
    family(record("model.k#", "k_{#}",
        "frac(n * {coupling.npole}, 2 * {model.gap_radius_mm|m})"), &ODD_HARMONICS),
    // model::harmonic_amplitude: Br 4/(nπ) sin(n α π/2), square-wave magnetization.
    family(record("model.b_i#", "B_{i,#}",
        "{model.br_inner_T_op} · frac(4, n * π) · sin(frac(n * π * {model.fill_inner}, 2))"), &ODD_HARMONICS),
    family(record("model.b_o#", "B_{o,#}",
        "{model.br_outer_T_op} · frac(4, n * π) · sin(frac(n * π * {model.fill_outer}, 2))"), &ODD_HARMONICS),
    // model::geometry_factor, steel-backed circuit.
    family(record("model.s#_iron", "S_{#}^{iron}",
        "frac(sinh({model.k#} * {model.inner_thickness_mm|m}) * sinh({model.k#} * {model.outer_thickness_mm|m}), \
         sinh({model.k#} * ({model.inner_thickness_mm|m} + {model.outer_thickness_mm|m} + {model.face_gap_mm|m})))"),
        &ODD_HARMONICS),
    // model::geometry_factor, free-space rings.
    family(record("model.s#_free", "S_{#}^{free}",
        "frac((1 - exp(-{model.k#} * {model.inner_thickness_mm|m})) * (1 - exp(-{model.k#} * {model.outer_thickness_mm|m})) \
         * exp(-{model.k#} * {model.face_gap_mm|m}), 2)"),
        &ODD_HARMONICS),
    // model::Harmonic::amplitude, in the circuit in effect (A5: a non-ferromagnetic back iron is free space).
    family(record("model.amp#_Pa", "A_{#}",
        "frac({model.b_i#} * {model.b_o#}, 2 * {coupling.mu0}) · \
         cases({materials.circuit_backiron} = 1 => {model.s#_iron}; else => {model.s#_free})"),
        &ODD_HARMONICS),
    // model::compute via at_pull_out and harmonic_slot: summed harmonics at the E7 angle; 0 when left out.
    family(record("model.tau#_Pa", "σ_{#}",
        "cases(n <= {coupling.max_harmonic} => {model.amp#_Pa} · sin(n * {model.pullout_angle_rad}); else => 0)")
        .corrected(&[E7]), &ODD_HARMONICS),
    // calibration::compute amp_n: the prototype, identical free-space rings.
    family(record("calibration.amp#_Pa", "A_{#}^{cal}",
        "frac((4 * {calibration.br_test_T} / (n * π))^2 * sin(frac(n * π * {calibration.fill_inner}, 2)) \
         * sin(frac(n * π * {calibration.fill_outer}, 2)), 2 * {calibration.mu0}) \
         · frac((1 - exp(-[k_{#}^{cal}] * {calibration.magnet_thickness_mm|m}))^2 * exp(-[k_{#}^{cal}] * {calibration.flat_gap_mm|m}), 2) \
         where [k_{#}^{cal}] = frac(n * {calibration.poles_per_ring}, 2 * {calibration.gap_radius_mm|m})"),
        &ODD_HARMONICS),
    family(record("calibration.tau#_Pa", "σ_{#}^{cal}",
        "cases(n <= {coupling.max_harmonic} => {calibration.amp#_Pa} · sin(n * {calibration.pullout_angle_rad}); else => 0)")
        .corrected(&[E7]), &ODD_HARMONICS),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Calculator: shear stress and pull-out (model::compute) ---
    record("model.tau_Pa", "σ", "sum(n in H: {model.tau#_Pa})"),
    record("model.pullout_angle_rad", "φ_{pull}", "peak(n in H: {model.amp#_Pa})").corrected(&[E7]),
    record("model.iron_circuit_angle_rad", "φ_{iron}",
        "peak(n in H: {model.b_i#} * {model.b_o#} * {model.s#_iron})").corrected(&[E7]),
    record("model.free_circuit_angle_rad", "φ_{free}",
        "peak(n in H: {model.b_i#} * {model.b_o#} * {model.s#_free})").corrected(&[E7]),
    record("model.area_lever_m3", "A_L", "2 * π * {model.gap_radius_mm|m}^2 * {model.active_length_mm|m}"),
    record("model.torque_2d_Nm", "T_{2D}", "{model.tau_Pa} * {model.area_lever_m3}"),
    record("model.f_end", "f_{end}", "1 - {coupling.c_end} · frac({model.pole_pitch_mm}, {model.active_length_mm})"),
    record("model.pullout_Nm", "T_{pull}", "{model.torque_2d_Nm} * {model.f_end} * {model.f_cal}"),
    record("model.pullout_20C_Nm", "T_{pull,20}",
        "{model.pullout_Nm} · frac({model.inner_br_T} * {model.outer_br_T}, {model.br_inner_T_op} * {model.br_outer_T_op})"),
    record("model.pullout_iron_Nm", "T_{iron}",
        "frac(sum(n in H: {model.b_i#} * {model.b_o#} * {model.s#_iron} * sin(n * {model.iron_circuit_angle_rad})), 2 * {coupling.mu0}) \
         * {model.area_lever_m3} * {model.f_end} * {calibration.f_cal_original}").corrected(&[E7]),
    record("model.pullout_noiron_Nm", "T_{free}",
        "frac(sum(n in H: {model.b_i#} * {model.b_o#} * {model.s#_free} * sin(n * {model.free_circuit_angle_rad})), 2 * {coupling.mu0}) \
         * {model.area_lever_m3} * {model.f_end} * {model.f_cal}").corrected(&[E7]),
    // model::select_calibration_factor: the measured correction only for the prototype's own circuit.
    record("model.f_cal", "f_{cal}",
        r#"cases({materials.circuit_backiron} = 0 and {coupling.npole} = {calibration.poles_per_ring}
               and {coupling.magnets.part_inner} = "B842SH" and {coupling.magnets.part_outer} = "B842SH"
               => {calibration.f_cal_updated};
             else => {calibration.f_cal_original})"#),

    // --- Calculator: the geometry and remanence the torque terms read ---
    record("model.gap_radius_mm", "R_g", "{model.inner_face_radius_mm} + {model.face_gap_mm} / 2"),
    record("model.pole_pitch_mm", "τ_p", "frac(2 * π * {model.gap_radius_mm}, {coupling.npole})"),
    // model::pitch_share, clamped at 1 (C66, C67).
    record("model.fill_inner", "λ_i",
        "min(1, frac({model.inner_width_mm}, 2 * π * ({coupling.inner_back_apothem_mm} + {model.inner_thickness_mm} / 2) / {coupling.npole}))"),
    record("model.fill_outer", "λ_o",
        "min(1, frac({model.outer_width_mm}, 2 * π * ({model.outer_face_apothem_mm} + {model.outer_thickness_mm} / 2) / {coupling.npole}))"),
    // model::br_factor, each ring with its own coefficient (decision A2-7).
    record("model.br_inner_T_op", "B_{r,i,op}",
        "{model.inner_br_T} * (1 + {model.inner_alpha_br_per_C} * ({coupling.op_temp_C} - 20))"),
    record("model.br_outer_T_op", "B_{r,o,op}",
        "{model.outer_br_T} * (1 + {model.outer_alpha_br_per_C} * ({coupling.op_temp_C} - 20))"),
    record("model.active_length_mm", "L", "min({model.inner_length_mm}, {model.outer_length_mm})"),
    record("model.face_gap_mm", "g", "{model.outer_face_apothem_mm} - {model.inner_face_radius_mm}"),
    record("model.inner_face_radius_mm", "r_{face,i}", "{coupling.inner_back_apothem_mm} + {model.inner_thickness_mm}"),
    record("model.outer_face_apothem_mm", "A_o", "{model.inner_corner_radius_mm} + {model.corner_gap_mm}"),
    // model::corner_radius for flat blocks; the face radius for arcs.
    record("model.inner_corner_radius_mm", "r_{corner,i}",
        "cases({coupling.faceted} = 1 => sqrt({model.inner_face_radius_mm}^2 + ({model.inner_width_mm} / 2)^2);
               else => {model.inner_face_radius_mm})"),
    // E8: the corner gap from the inner corner radius C55 (the face radius for arcs).
    record("model.corner_gap_mm", "g_c",
        "{metal.face_gap_mm} - ({model.inner_corner_radius_mm} - {model.inner_face_radius_mm})").corrected(&[E8]),

    // --- Calculator: the magnets in effect (model::resolve_magnets) ---
    // The axial override wins; else the library part; else the manual value (a grade-mode
    // ring keeps the manual dimensions).
    record("model.inner_length_mm", "L_i",
        r#"cases({coupling.magnets.axial_length_mm} != none => {coupling.magnets.axial_length_mm};
               [L_{lib}] != none => [L_{lib}];
               else => {coupling.magnets.manual_inner_length_mm})
           where [L_{lib}] = table("magnets", {coupling.magnets.part_inner}, "length_mm")"#),
    record("model.outer_length_mm", "L_o",
        r#"cases({coupling.magnets.axial_length_mm} != none => {coupling.magnets.axial_length_mm};
               [L_{lib}] != none => [L_{lib}];
               else => {coupling.magnets.manual_outer_length_mm})
           where [L_{lib}] = table("magnets", {coupling.magnets.part_outer}, "length_mm")"#),
    record("model.inner_width_mm", "w_i",
        r#"cases([w_{lib}] != none => [w_{lib}]; else => {coupling.magnets.manual_inner_width_mm})
           where [w_{lib}] = table("magnets", {coupling.magnets.part_inner}, "width_mm")"#),
    record("model.outer_width_mm", "w_o",
        r#"cases([w_{lib}] != none => [w_{lib}]; else => {coupling.magnets.manual_outer_width_mm})
           where [w_{lib}] = table("magnets", {coupling.magnets.part_outer}, "width_mm")"#),
    record("model.inner_thickness_mm", "t_i",
        r#"cases([t_{lib}] != none => [t_{lib}]; else => {coupling.magnets.manual_inner_thickness_mm})
           where [t_{lib}] = table("magnets", {coupling.magnets.part_inner}, "thickness_mm")"#),
    record("model.outer_thickness_mm", "t_o",
        r#"cases([t_{lib}] != none => [t_{lib}]; else => {coupling.magnets.manual_outer_thickness_mm})
           where [t_{lib}] = table("magnets", {coupling.magnets.part_outer}, "thickness_mm")"#),
    // E3: a library part's Br is its grade's (N42SH 1.30 T); a grade-mode ring takes its grade's.
    record("model.inner_br_T", "B_{r,i}",
        r#"cases([B_{lib}] != none => [B_{lib}]; [B_{grade}] != none => [B_{grade}];
               else => {coupling.magnets.manual_inner_br_T})
           where [B_{lib}] = table("magnets", {coupling.magnets.part_inner}, "br_T"),
                 [B_{grade}] = table("grades", {coupling.magnets.grade_inner}, "br_T")"#).corrected(&[E3]),
    record("model.outer_br_T", "B_{r,o}",
        r#"cases([B_{lib}] != none => [B_{lib}]; [B_{grade}] != none => [B_{grade}];
               else => {coupling.magnets.manual_outer_br_T})
           where [B_{lib}] = table("magnets", {coupling.magnets.part_outer}, "br_T"),
                 [B_{grade}] = table("grades", {coupling.magnets.grade_outer}, "br_T")"#).corrected(&[E3]),
    // model::ResolvedMagnet::alpha_br: the grade's only in the grade mode (decision A2-7).
    record("model.inner_alpha_br_per_C", "α_i",
        r#"cases(table("magnets", {coupling.magnets.part_inner}, "br_T") = none and [α_{grade}] != none => [α_{grade}];
               else => {calibration.alpha_br_per_C})
           where [α_{grade}] = table("grades", {coupling.magnets.grade_inner}, "alpha_br_per_C")"#),
    record("model.outer_alpha_br_per_C", "α_o",
        r#"cases(table("magnets", {coupling.magnets.part_outer}, "br_T") = none and [α_{grade}] != none => [α_{grade}];
               else => {calibration.alpha_br_per_C})
           where [α_{grade}] = table("grades", {coupling.magnets.grade_outer}, "alpha_br_per_C")"#),
    // material_library::resolve: a non-ferromagnetic back iron selects the free-space circuit (A5).
    record("materials.circuit_backiron", "circuit",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => 0; else => {coupling.backiron})"#),

    // --- Calibration: the prototype model and the one-point correction (calibration::compute) ---
    record("calibration.poles_per_ring", "N^{cal}", "{calibration.total_magnets} / 2"),
    record("calibration.face_radius_mm", "r_{face}^{cal}", "{calibration.apothem_mm} + {calibration.magnet_thickness_mm}"),
    record("calibration.corner_radius_mm", "r_{corner}^{cal}",
        "sqrt(({calibration.face_radius_mm})^2 + ({calibration.magnet_width_mm} / 2)^2)"),
    record("calibration.corner_gap_mm", "g_c^{cal}",
        "cases({calibration.gap_definition} = 0 => {calibration.spacing_mm};
               else => {calibration.spacing_mm} - ({calibration.corner_radius_mm} - {calibration.face_radius_mm}))"),
    record("calibration.outer_face_apothem_mm", "A_o^{cal}", "{calibration.corner_radius_mm} + {calibration.corner_gap_mm}"),
    record("calibration.flat_gap_mm", "g^{cal}", "{calibration.outer_face_apothem_mm} - {calibration.face_radius_mm}"),
    record("calibration.gap_radius_mm", "R_g^{cal}", "{calibration.face_radius_mm} + {calibration.flat_gap_mm} / 2"),
    record("calibration.fill_inner", "λ_i^{cal}",
        "min(1, frac({calibration.magnet_width_mm}, 2 * π * ({calibration.apothem_mm} + {calibration.magnet_thickness_mm} / 2) / {calibration.poles_per_ring}))"),
    record("calibration.fill_outer", "λ_o^{cal}",
        "min(1, frac({calibration.magnet_width_mm}, 2 * π * ({calibration.outer_face_apothem_mm} + {calibration.magnet_thickness_mm} / 2) / {calibration.poles_per_ring}))"),
    record("calibration.br_test_T", "B_{r,test}",
        "{calibration.br_T} * (1 + {calibration.alpha_br_per_C} * ({calibration.test_temp_C} - 20))"),
    record("calibration.pole_pitch_mm", "τ_p^{cal}", "frac(2 * π * {calibration.gap_radius_mm}, {calibration.poles_per_ring})"),
    record("calibration.f_end", "f_{end}^{cal}",
        "1 - {calibration.c_end} · frac({calibration.pole_pitch_mm}, {calibration.magnet_length_mm})"),
    record("calibration.pullout_angle_rad", "φ^{cal}", "peak(n in H: {calibration.amp#_Pa})").corrected(&[E7]),
    record("calibration.torque_2d_Nm", "T_{2D}^{cal}",
        "(sum(n in H: {calibration.tau#_Pa})) * 2 * π * ({calibration.gap_radius_mm|m})^2 * {calibration.magnet_length_mm|m}"),
    record("calibration.original_model_Nm", "T_{orig}^{cal}",
        "{calibration.torque_2d_Nm} * {calibration.f_end} * {calibration.f_cal_original}"),
    record("calibration.model_torque_Nm", "T_{model}^{cal}", "{calibration.original_model_Nm}"),
    record("calibration.measured_over_model", "r^{cal}", "frac({calibration.measured_torque_Nm}, {calibration.model_torque_Nm})"),
    record("calibration.f_cal_updated", "f_{cal,1}", "{calibration.f_cal_original} * {calibration.measured_over_model}"),
    record("calibration.model_error", "ε^{cal}", "frac({calibration.model_torque_Nm}, {calibration.measured_torque_Nm}) - 1"),
    record("calibration.fea_interp_Nm", "T_{3D}",
        r#"cases({calibration.corner_gap_mm} >= 1 and {calibration.corner_gap_mm} <= 1.5
               => {calibration.fea_torque1_Nm} + ({calibration.fea_torque2_Nm} - {calibration.fea_torque1_Nm}) · frac({calibration.corner_gap_mm} - 1, 0.5);
             else => "outside range")"#),
    record("calibration.fea_interp_error", "ε_{3D}",
        r#"cases({calibration.fea_interp_Nm} != "outside range" => frac({calibration.fea_interp_Nm}, {calibration.measured_torque_Nm}) - 1;
             else => "n.a.")"#),

    // --- Metal design: hot and cold torque (metal_design::compute) ---
    record("metal.op_temp_C", "ϑ_{op,MD}", "{coupling.op_temp_C}"),
    record("metal.alpha_br_per_C", "α_{MD}", "{calibration.alpha_br_per_C}"),
    record("metal.torque_op_Nm", "T_{op}", "{model.pullout_Nm}"),
    record("metal.torque_20C_Nm", "T_{20}", "{model.pullout_20C_Nm}"),
    record("metal.torque_cold_Nm", "T_{cold}",
        "{metal.torque_20C_Nm} * (1 + {model.inner_alpha_br_per_C} * ({metal.min_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.min_temp_C} - 20))"),
    record("metal.torque_hot_low_Nm", "T_{hot,low}", "{metal.torque_op_Nm} * (1 - {metal.variation})"),
    record("metal.torque_cold_high_Nm", "T_{cold,high}", "{metal.torque_cold_Nm} * (1 + {metal.variation})"),
    record("metal.hot_min_check", "C_{hot}",
        r#"cases({metal.torque_hot_low_Nm} < {metal.required_min_Nm} => "Below hot minimum"; else => "Estimate covers hot min")"#),
    record("metal.required_20C_Nm", "T_{req,20}",
        "frac({metal.required_min_Nm}, (1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20)) * (1 - {metal.variation}))"),
    record("metal.hot_margin", "m_{hot}", "frac({metal.torque_op_Nm}, {metal.required_min_Nm}) - 1"),
    record("metal.required_20C_zero_scatter_Nm", "T_{req,20,v=0}",
        "frac({metal.required_min_Nm}, (1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20)))"),
    record("metal.torque_cold_zero_var_Nm", "T_{cold,v=0}", "{metal.torque_cold_Nm}"),
    record("metal.noiron_baseline_hot_Nm", "T_{proto,hot}",
        "{calibration.measured_torque_Nm} * (frac(1 + {metal.alpha_br_per_C} * ({metal.op_temp_C} - 20), \
         1 + {metal.alpha_br_per_C} * ({calibration.test_temp_C} - 20)))^2"),
    record("metal.cold_for_hot_min_Nm", "T_{cold,req}",
        "{metal.required_min_Nm} · frac(1 + {model.inner_alpha_br_per_C} * ({metal.min_temp_C} - 20), 1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         · frac(1 + {model.outer_alpha_br_per_C} * ({metal.min_temp_C} - 20), 1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20))"),
    record("metal.cold_for_hot_min_input_Nm", "T_{cold,req,in}",
        "frac({metal.cold_for_hot_min_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),
    record("metal.cold_high_Nm", "T_{cold,high,MD}", "{metal.torque_cold_high_Nm}"),
    record("metal.cold_high_input_Nm", "T_{cold,high,in}",
        "frac({metal.cold_high_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),
];
