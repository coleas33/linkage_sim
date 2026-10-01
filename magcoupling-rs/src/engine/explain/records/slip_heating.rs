//! Slip heating (plan A-3 batch 3): the eddy-current slip losses (Temperature design C109 to
//! C134, corrections E5, E17), the heat capacity and thermal time constant (C137 to C161, E15),
//! the time and drag to the limit (E12), the metal-design slip rows, and what they need to
//! reach inputs: the materials in effect (Addendum A5), the mass model (C110 to C114, E8, E9)
//! with the axial housing (decision A2-8), the sleeve, liner, cap and endplates (Metal design
//! C175 to C181, E8), and the peak magnet temperature the demagnetization margins read.
//!
//! Each formula is transcribed from the engine function named beside it, with every approved
//! correction on (`temperature.rs`, `model.rs`, `metal_design.rs`, `housing.rs`,
//! `material_library.rs`).
//!
//! Symbols: P power, ω_e the field's angular frequency (p ω_s), ω_s the slip speed, σ conductivity, ρ density, c specific heat, m mass.

use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17};
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The materials in effect (material_library::resolve) ---
    // A ferromagnetic pick other than the default supplies the steel values; the default and a
    // non-ferromagnetic pick keep the Materials inputs.
    record("temperature.slip_loss.steel_sigma_S_m", "σ_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "sigma_S_m");
               else => {materials.steel.conductivity_S_m})"#),
    record("temperature.slip_loss.steel_mu_r", "μ_r", "{materials.steel.mu_r_incremental}"),
    record("temperature.thermal.steel_c", "c_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "cp_J_kgK");
               else => {materials.steel.specific_heat_J_kgK})"#),
    record("materials.steel_density_g_mm3", "ρ_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "density_g_mm3");
               else => {metal.steel_density_g_mm3})"#),
    // Without back iron the hub, cup and boss are the workbook's 6061-T6, or a
    // non-ferromagnetic pick (A5: the pick becomes that material).
    record("materials.body_sigma_S_m", "σ_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "sigma_S_m");
               else => table("aluminium", "6061-T6", "conductivity_S_m"))"#),
    record("materials.body_density_g_mm3", "ρ_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "density_g_mm3");
               else => {metal.al_density_g_mm3})"#),
    record("materials.body_c_J_kgK", "c_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "cp_J_kgK");
               else => {temperature.thermal.c_aluminium})"#),
    record("materials.sleeve_sigma_S_m", "σ_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {temperature.slip_loss.sigma_316_S_m};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "sigma_S_m"))"#),
    record("materials.sleeve_density_g_mm3", "ρ_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {metal.sleeve_density_g_mm3};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "density_g_mm3"))"#),
    record("materials.sleeve_c_J_kgK", "c_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {temperature.thermal.c_316};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "cp_J_kgK"))"#),
    record("temperature.slip_loss.cap_sigma_S_m", "σ_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => table("aluminium", "6061-T6", "conductivity_S_m");
               else => table("cap_housing", {materials.parts.cap_housing}, "sigma_S_m"))"#),
    record("materials.cap_density_g_mm3", "ρ_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => {metal.al_density_g_mm3};
               else => table("cap_housing", {materials.parts.cap_housing}, "density_g_mm3"))"#),
    record("materials.cap_c_J_kgK", "c_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => {temperature.thermal.c_aluminium};
               else => table("cap_housing", {materials.parts.cap_housing}, "cp_J_kgK"))"#),

    // --- Duty (temperature::compute) ---
    record("temperature.duty.slip_rpm", "n_{slip}", "{metal.slip_rpm}"),
    record("temperature.duty.slip_rad_s", "ω_s", "{temperature.duty.slip_rpm} * 2 * π / 60"),
    record("temperature.duty.pole_pairs", "p", "{coupling.npole} / 2"),
    // The opposite ring's field sweeps each part p times per slip revolution.
    record("temperature.duty.field_freq_Hz", "f_e", "{temperature.duty.pole_pairs} * {temperature.duty.slip_rpm} / 60"),
    record("temperature.duty.field_omega_rad_s", "ω_e", "2 * π * {temperature.duty.field_freq_Hz}"),

    // --- Geometry the losses read ---
    record("model.hub_wall_mm", "t_{hub}", "{coupling.inner_back_apothem_mm} - {metal.bond_inner_mm} - {coupling.bore_mm} / 2"),
    record("model.outer_back_apothem_mm", "A_{back}", "{model.outer_face_apothem_mm} + {model.outer_thickness_mm}"),
    record("temperature.adhesive.inner_mid_radius_mm", "r_{mid}", "{coupling.inner_back_apothem_mm} + {model.inner_thickness_mm} / 2"),
    // metal_design::retainers: E8, the sleeve clears the inner corner radius C55.
    record("retainers.sleeve_id_mm", "D_{sl,i}", "2 * ({model.inner_corner_radius_mm} + {metal.sleeve_bedding_mm})").corrected(&[E8]),
    record("retainers.sleeve_od_mm", "D_{sl,o}", "{retainers.sleeve_id_mm} + 2 * {metal.sleeve_mm}"),
    record("retainers.liner_od_mm", "D_{ln,o}", "2 * ({model.outer_face_apothem_mm} - {metal.liner_bedding_mm})"),
    record("retainers.liner_id_mm", "D_{ln,i}", "{retainers.liner_od_mm} - 2 * {metal.liner_mm}"),
    record("retainers.endplate_od_mm", "D_{ep}", "{retainers.sleeve_id_mm}"),

    // --- Slip losses (temperature::compute) ---
    // Steel skin depth at the field frequency: δ = √(2 / (ω_e μ0 μ_r σ)).
    record("temperature.slip_loss.skin_depth_mm", "δ",
        "sqrt(frac(2, {temperature.duty.field_omega_rad_s} * {coupling.mu0} * {temperature.slip_loss.steel_mu_r} * {temperature.slip_loss.steel_sigma_S_m})) * 1000"),
    // Solid steel is skin-limited; with no back iron the hub is aluminium (resistance-limited,
    // E17's low-Reynolds form T1 with the free-space field and the thin-conductor end factor).
    // The engine's third branch, half the steel-circuit field for an aluminium hub in a steel
    // cup, needs E9 off: with every correction on the cup is aluminium whenever the hub is.
    record("temperature.slip_loss.hub_W", "P_{hub}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2
                         * {temperature.slip_loss.b_hub_free_T}^2, 2 * [k]^2) * frac(1 - exp(-2 * [k] * {model.hub_wall_mm|m}), 2 * [k])
                    * 2 * π * [r] * {model.active_length_mm|m};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_hub_T}^2
                         * {temperature.slip_loss.skin_depth_mm|m}, 4 * [k]^2) * 2 * π * [r] * {model.active_length_mm|m})
         where [r] = {coupling.inner_back_apothem_mm|m} - {metal.bond_inner_mm|m},
               [k] = {temperature.duty.pole_pairs} / [r]").corrected(&[E17]),
    record("temperature.slip_loss.cup_W", "P_{cup}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2
                         * {temperature.slip_loss.b_cup_free_T}^2, 2 * [k]^2) * frac(1 - exp(-2 * [k] * {metal.cup_wall_corner_mm|m}), 2 * [k])
                    * 2 * π * [r] * {model.active_length_mm|m};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_cup_T}^2
                         * {temperature.slip_loss.skin_depth_mm|m}, 4 * [k]^2) * 2 * π * [r] * {model.active_length_mm|m})
         where [r] = {model.outer_back_apothem_mm|m} + {metal.bond_outer_mm|m},
               [k] = {temperature.duty.pole_pairs} / [r]").corrected(&[E17]),
    // The rear web sees the end field ∫B² dA (E5: the steel-surface value); (r_mid / p)² replaces 1/k².
    record("temperature.slip_loss.web_W", "P_{web}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2, 2)
                    * ({temperature.adhesive.inner_mid_radius_mm|m} / {temperature.duty.pole_pairs})^2
                    * frac(1 - exp(-2 * [k] * {metal.web_mm|m}), 2 * [k]) * {temperature.slip_loss.web_integral_free_T2m2};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.skin_depth_mm|m}, 4)
                    * ({temperature.adhesive.inner_mid_radius_mm|m} / {temperature.duty.pole_pairs})^2 * {temperature.slip_loss.web_integral_T2m2})
         where [k] = {temperature.duty.pole_pairs} / {temperature.adhesive.inner_mid_radius_mm|m}").corrected(&[E5, E17]),
    // Thin shells: the loss per volume σ (ω_s r B)² / 2, times the end factor.
    record("temperature.slip_loss.sleeve_W", "P_{sl}",
        "frac({temperature.slip_loss.end_factor} * {materials.sleeve_sigma_S_m} * {metal.sleeve_mm|m} * ({temperature.duty.slip_rad_s} * [r])^2
              * {temperature.slip_loss.b_sleeve_T}^2, 2) * 2 * π * [r] * {model.active_length_mm|m}
         where [r] = frac({retainers.sleeve_id_mm|m} + {retainers.sleeve_od_mm|m}, 4)"),
    record("temperature.slip_loss.liner_W", "P_{ln}",
        "frac({temperature.slip_loss.end_factor} * {materials.sleeve_sigma_S_m} * {metal.liner_mm|m} * ({temperature.duty.slip_rad_s} * [r])^2
              * {temperature.slip_loss.b_liner_T}^2, 2) * 2 * π * [r] * {model.active_length_mm|m}
         where [r] = frac({retainers.liner_od_mm|m} + {retainers.liner_id_mm|m}, 4)"),
    record("temperature.slip_loss.cap_W", "P_{cap}",
        "{temperature.slip_loss.end_factor} * {temperature.slip_loss.cap_sigma_S_m} * {metal.cap_axial_mm|m}
         * {temperature.duty.slip_rad_s}^2 * {temperature.slip_loss.cap_integral_T2m4}"),
    // Eddy currents inside the blocks: σ ω_e² B² w² / 24 per volume, both rings.
    record("temperature.slip_loss.magnets_W", "P_{mag}",
        "frac({temperature.slip_loss.sigma_ndfeb_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_magnet_T}^2
              * {model.inner_width_mm|m}^2, 24) * ({model.inner_length_mm|m} * {model.inner_width_mm|m} * {model.inner_thickness_mm|m})
         * 2 * {coupling.npole}"),
    record("temperature.slip_loss.total_W", "P_{tot}",
        "{temperature.slip_loss.hub_W} + {temperature.slip_loss.cup_W} + {temperature.slip_loss.web_W} + {temperature.slip_loss.sleeve_W}
         + {temperature.slip_loss.liner_W} + {temperature.slip_loss.cap_W} + {temperature.slip_loss.magnets_W}"),
    record("temperature.slip_loss.drag_Nm", "T_{drag}", "frac({temperature.slip_loss.total_W}, {temperature.duty.slip_rad_s})"),
    // A bench drag replaces the estimate (and the high case) once entered.
    record("temperature.slip_loss.used_W", "P_{use}",
        "cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * {temperature.duty.slip_rad_s};
               else => {temperature.slip_loss.total_W})"),
    record("temperature.slip_loss.high_W", "P_{hi}",
        "cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * {temperature.duty.slip_rad_s};
               else => {temperature.slip_loss.total_W} * {temperature.slip_loss.high_multiplier})"),

    // --- The mass model the heat capacity reads (model::mass_estimate) ---
    // Decision A2-7: a grade-mode ring's density, else NdFeB's 7.5 g/cm³.
    record("model.inner_magnet_density_g_mm3", "ρ_{mag,i}",
        r#"cases(table("magnets", {coupling.magnets.part_inner}, "br_T") = none and [ρ_{grade}] != none => [ρ_{grade}]; else => 0.0075)
           where [ρ_{grade}] = table("grades", {coupling.magnets.grade_inner}, "density_g_mm3")"#),
    record("model.outer_magnet_density_g_mm3", "ρ_{mag,o}",
        r#"cases(table("magnets", {coupling.magnets.part_outer}, "br_T") = none and [ρ_{grade}] != none => [ρ_{grade}]; else => 0.0075)
           where [ρ_{grade}] = table("grades", {coupling.magnets.grade_outer}, "density_g_mm3")"#),
    record("mass.magnets_g", "m_{mag}",
        "cases({model.inner_magnet_density_g_mm3} = {model.outer_magnet_density_g_mm3}
                 => {coupling.npole} * ([V_i] + [V_o]) * {model.inner_magnet_density_g_mm3};
               else => {coupling.npole} * ([V_i] * {model.inner_magnet_density_g_mm3} + [V_o] * {model.outer_magnet_density_g_mm3}))
         where [V_i] = {model.inner_length_mm} * {model.inner_width_mm} * {model.inner_thickness_mm},
               [V_o] = {model.outer_length_mm} * {model.outer_width_mm} * {model.outer_thickness_mm}"),
    record("model.pocket_corner_radius_mm", "r_{pocket}",
        "cases({coupling.faceted} = 1 => frac({model.outer_back_apothem_mm} + {metal.bond_outer_mm}, cos(π / {coupling.npole}));
               else => {model.outer_back_apothem_mm} + {metal.bond_outer_mm})"),
    record("model.cup_od_mm", "D_{cup}", "2 * ({model.pocket_corner_radius_mm} + {metal.cup_wall_corner_mm})"),
    // E9: with no back iron the cup and boss are aluminium (the body material), as the hub is.
    record("mass.cup_g", "m_{cup}",
        "((π * ({model.cup_od_mm} / 2)^2 - [A_{cav}]) * {housing.cup_depth_mm}
          + π * (({model.cup_od_mm} / 2)^2 - ({coupling.bore_mm} / 2)^2) * {metal.web_mm}) * [ρ]
         where [A_{cav}] = cases({coupling.faceted} != 1 => π * ({model.outer_back_apothem_mm} + {metal.bond_outer_mm})^2;
                                 else => {coupling.npole} * ({model.outer_back_apothem_mm} + {metal.bond_outer_mm})^2 * tan(π / {coupling.npole})),
               [ρ] = cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})")
        .corrected(&[E8, E9]),
    record("mass.hub_g", "m_{hub}",
        "([A_{hub}] - π * ({coupling.bore_mm} / 2)^2) * {housing.hub_length_mm} * [ρ]
         where [A_{hub}] = cases({coupling.faceted} = 1 => {coupling.npole} * ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm})^2 * tan(π / {coupling.npole});
                                 else => π * ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm})^2),
               [ρ] = cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})"),
    record("mass.boss_g", "m_{boss}",
        "π * (({metal.boss_od_mm} / 2)^2 - ({coupling.bore_mm} / 2)^2) * {metal.boss_length_mm}
         * cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})")
        .corrected(&[E9]),
    // housing::axial_housing (decision A2-8): with the axial length override set, a dimension
    // that bounds a ring follows the ring's length change, and is never shorter than the ring.
    record("housing.hub_length_mm", "L_{hub}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.hub_length_mm} + ({coupling.magnets.axial_length_mm} - [L_{i,own}]), {coupling.magnets.axial_length_mm});
               else => {metal.hub_length_mm})
           where [L_{i,own}] = cases(table("magnets", {coupling.magnets.part_inner}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_inner}, "length_mm");
                                     else => {coupling.magnets.manual_inner_length_mm})"#),
    record("housing.cup_depth_mm", "L_{cav}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.cup_depth_mm} + ({coupling.magnets.axial_length_mm} - [L_{o,own}]), {coupling.magnets.axial_length_mm});
               else => {metal.cup_depth_mm})
           where [L_{o,own}] = cases(table("magnets", {coupling.magnets.part_outer}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_outer}, "length_mm");
                                     else => {coupling.magnets.manual_outer_length_mm})"#),
    record("housing.retainer_span_mm", "L_{ret}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.retainer_span_mm} + ({coupling.magnets.axial_length_mm} - max([L_{i,own}], [L_{o,own}])), {coupling.magnets.axial_length_mm});
               else => {metal.retainer_span_mm})
           where [L_{i,own}] = cases(table("magnets", {coupling.magnets.part_inner}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_inner}, "length_mm");
                                     else => {coupling.magnets.manual_inner_length_mm}),
                 [L_{o,own}] = cases(table("magnets", {coupling.magnets.part_outer}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_outer}, "length_mm");
                                     else => {coupling.magnets.manual_outer_length_mm})"#),
    record("retainers.retainers_g", "m_{ret}",
        "frac(π, 4) * ({retainers.sleeve_od_mm}^2 - {retainers.sleeve_id_mm}^2 + {retainers.liner_od_mm}^2 - {retainers.liner_id_mm}^2)
         * {housing.retainer_span_mm} * {materials.sleeve_density_g_mm3}"),
    record("retainers.cap_g", "m_{cap}",
        "(frac(π, 4) * ({metal.cap_od_mm}^2 - {retainers.liner_id_mm}^2) * {metal.cap_axial_mm}
          + frac(π, 4) * ({metal.cap_od_mm}^2 - {metal.cap_thread_dia_mm}^2) * {metal.cap_thread_engagement_mm}) * {materials.cap_density_g_mm3}"),
    record("retainers.endplates_g", "m_{ep}",
        "frac(π, 4) * (({retainers.endplate_od_mm}^2 - {coupling.bore_mm}^2) * {metal.front_endplate_mm}
                  + ({retainers.endplate_od_mm}^2 - {metal.rear_endplate_hole_mm}^2) * {metal.rear_endplate_mm}) * {materials.sleeve_density_g_mm3}"),

    // --- Thermal network: one lumped heat capacity and one conductance ---
    // E15: an aluminium hub, cup and boss at the body material's specific heat; hardware stays steel.
    record("temperature.thermal.heat_capacity_J_K", "C",
        "cases({materials.circuit_backiron} != 1
                 => ({mass.magnets_g} * {temperature.thermal.c_ndfeb} + ({mass.cup_g} + {mass.boss_g}) * {materials.body_c_J_kgK}
                     + {mass.hub_g} * {materials.body_c_J_kgK} + {metal.hardware_g} * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000;
               else => ({mass.magnets_g} * {temperature.thermal.c_ndfeb}
                     + ({mass.cup_g} + {mass.hub_g} + {mass.boss_g} + {metal.hardware_g}) * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000)")
        .corrected(&[E9, E15]),
    record("temperature.thermal.time_constant_s", "τ_{th}", "frac({temperature.thermal.heat_capacity_J_K}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.t95_s", "t_{95}", "3 * {temperature.thermal.time_constant_s}"),
    record("temperature.thermal.rev_per_tau", "N_{τ}", "{temperature.thermal.time_constant_s} * ({temperature.duty.slip_rpm} / 60)"),
    record("temperature.thermal.rev95", "N_{95}", "3 * {temperature.thermal.time_constant_s} * ({temperature.duty.slip_rpm} / 60)"),
    record("temperature.thermal.steady_rise_est_C", "Δϑ_{ss,est}", "frac({temperature.slip_loss.used_W}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.steady_rise_high_C", "Δϑ_{ss,hi}", "frac({temperature.slip_loss.high_W}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.steady_est_C", "ϑ_{ss,est}", "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_est_C}"),
    record("temperature.thermal.steady_high_C", "ϑ_{ss,hi}", "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_high_C}"),
    // First-order heating toward the steady temperature: ϑ(t) = ϑ_hot + Δϑ_ss (1 − e^(−t/τ)).
    // E12: a start at or above the limit reaches it at once.
    record("temperature.thermal.time_to_limit_high", "t_{lim,hi}",
        r#"cases({temperature.duty.hot_day_start_C} >= {temperature.summary.governing_limit_C} => 0;
               {temperature.thermal.steady_high_C} <= {temperature.summary.governing_limit_C} => "never: steady state stays below the limit";
               else => -{temperature.thermal.time_constant_s}
                       * ln(1 - frac({temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}, {temperature.thermal.steady_rise_high_C})))"#)
        .corrected(&[E12]),
    record("temperature.thermal.time_to_limit_est", "t_{lim,est}",
        r#"cases({temperature.duty.hot_day_start_C} >= {temperature.summary.governing_limit_C} => 0;
               {temperature.thermal.steady_est_C} <= {temperature.summary.governing_limit_C} => "never: steady state stays below the limit";
               else => -{temperature.thermal.time_constant_s}
                       * ln(1 - frac({temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}, {temperature.thermal.steady_rise_est_C})))"#)
        .corrected(&[E12]),
    record("temperature.thermal.rotations_to_limit_high", "N_{lim,hi}",
        r#"cases({temperature.thermal.time_to_limit_high} != "never: steady state stays below the limit"
                   => {temperature.thermal.time_to_limit_high} * ({temperature.duty.slip_rpm} / 60);
               else => "never")"#),
    // E12: slip never cools the magnets, so the critical drag is not negative.
    record("temperature.thermal.critical_drag_Nm", "T_{crit}",
        "max(0, {temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}) * {temperature.thermal.conductance_W_K}
         / {temperature.duty.slip_rad_s}").corrected(&[E12]),
    record("temperature.summary.time_to_limit_high", "t_{lim,hi}^{sum}", "{temperature.thermal.time_to_limit_high}"),
    record("temperature.summary.critical_drag_Nm", "T_{crit}^{sum}", "{temperature.thermal.critical_drag_Nm}"),
    record("temperature.thermal.temp_at_fault_C", "ϑ_{fault}",
        "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_high_C}
         * (1 - exp(-{temperature.duty.fault_trip_s} / {temperature.thermal.time_constant_s}))"),

    // --- Slip life and the peak magnet temperature ---
    record("temperature.slip_life.rise_per_event_high_C", "Δϑ_{ev,hi}",
        "{temperature.slip_loss.high_W} * {metal.slip_event_s} / {temperature.thermal.heat_capacity_J_K}"),
    record("temperature.slip_life.slip_hours", "t_{slip}", "{metal.life_events} * {metal.slip_event_s} / 3600"),
    record("temperature.slip_life.slip_duty", "D_{slip}", "frac({temperature.slip_life.slip_hours}, {temperature.duty.life_hours})"),
    // The hot day, plus the fault-limited slip or one event, whichever is hotter, plus the
    // average heating of the life slip duty.
    record("temperature.magnet_life.peak_C", "ϑ_{peak}",
        "max({temperature.thermal.temp_at_fault_C}, {temperature.duty.hot_day_start_C} + {temperature.slip_life.rise_per_event_high_C})
         + {temperature.slip_life.slip_duty} * {temperature.thermal.steady_rise_high_C}"),

    // --- Metal design slip rows (metal_design::compute) ---
    record("metal.slip_freq_Hz", "f_{e,MD}", "frac({coupling.npole}, 2) * frac({metal.slip_rpm}, 60)"),
    record("metal.slip_loss_W", "P_{slip,MD}",
        r#"cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * 2 * π * {metal.slip_rpm} / 60; else => "not measured")"#),
    record("metal.slip_energy_J", "E_{slip,MD}",
        r#"cases({metal.measured_drag_Nm} != none => {metal.slip_loss_W} * {metal.slip_event_s}; else => "not measured")"#),

    // --- The demagnetization margins and the summary rows (temperature::compute) ---
    record("temperature.summary.service_max_C", "ϑ_{svc}", "{coupling.op_temp_C}"),
    record("temperature.summary.onset_aligned_C", "ϑ_{al}^{sum}", "{temperature.demag.onset_aligned_C}"),
    record("temperature.summary.onset_pullout_C", "ϑ_{po}^{sum}", "{temperature.demag.onset_pullout_C}"),
    record("temperature.summary.onset_skipping_C", "ϑ_{sk}^{sum}", "{temperature.demag.onset_skipping_C}"),
    record("temperature.summary.margin_service_C", "Δϑ_{svc}",
        "{temperature.summary.governing_limit_C} - {temperature.summary.service_max_C}"),
    record("temperature.summary.margin_hot_day_C", "Δϑ_{hot}",
        "{temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}"),
    record("temperature.summary.cure_margin_C", "Δϑ_{cure}", "{temperature.demag.onset_single_ring_C} - {temperature.adhesive.cure_C}"),
    record("temperature.magnet_life.margin_onset_C", "Δϑ_{sk}", "{temperature.demag.onset_skipping_C} - {temperature.magnet_life.peak_C}"),
    record("temperature.magnet_life.margin_limit_C", "Δϑ_{mag}", "{temperature.demag.magnet_limit_C} - {temperature.magnet_life.peak_C}"),
];
