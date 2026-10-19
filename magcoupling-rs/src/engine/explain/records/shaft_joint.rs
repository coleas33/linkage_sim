//! Shaft joints (BL-064): the keyed shrink fits that hold the inner hub on the wheel shaft and
//! the cup on the gearbox output shaft. One record per numeric result of the check (23); the
//! verdict `shaft_joint.check` is a text and has none.
//!
//! Each formula is transcribed from `shaft_joint::compute` (its `joint_specs`, `joint`,
//! `hub_fit_allowed` and `shaft_shrink_um`) and the part material from
//! `material_library::PartProperties::hub`, every approved correction on. Plane-stress Lamé
//! thick cylinders on a solid shaft: the contact pressure per µm of effective diametral
//! interference, the holding torque with the key's arc taken off the contact, the bore's von
//! Mises stress; the interference's change at the two temperature ends; the fitting
//! temperature with the shaft's shrink; the key alone. A joint with no wall around its bore
//! (the part's outer diameter at most the bore, a part typed at or below zero included, or a
//! keyway as deep as the wall) has NaN Lamé rows.
//!
//! The engine's named constants appear as numbers (the drift guard holds each to the engine):
//! `SHAFT_MODULUS_GPA` (205 GPa, written 205000 MPa), `POISSON` 0.3, `SHAFT_CTE_PER_C`
//! 11.5e-6 /°C, `ROOM_C` 20 °C, `YIELD_FACTOR` 1.2, `NIST_FE_2_25NI` (the five coefficients)
//! and `NIST_FIT_TOP_K` 265 K, with 0 °C at 273.15 K, and `temperature::CURE_MARGIN_MIN_C`,
//! 10 °C. The workbook's aluminium body (no back iron and no non-magnetic pick) is 6061-T6:
//! E18's 68.9 GPa and 23.6e-6 /°C (`temperature::AL_HUB_MODULUS_GPA`, `AL_HUB_CTE_PER_C`) and
//! the library's 241 MPa yield.
//!
//! The two joints' records repeat each other with their own outer diameter, length, band,
//! allowed temperature and shaft, as the inner and outer magnet-limit records do: the inner
//! hub's outer diameter is its flats' incircle under the bondline, 2 (a_i − b_i), its length
//! the hub's in effect; the cup's outer diameter is its boss, its length the boss and the web
//! the bore runs through.
//!
//! Locals: [D] the part's outer diameter, [Q] bore over outer diameter, [k_θ] the bore's hoop
//! factor (1 + Q²)/(1 − Q²), [p_1] the contact pressure per µm, [b] the key's arc, [T_1] the
//! holding torque per µm, [σ_1] the bore stress per µm, [Δ_c] and [Δ_h] the interference's
//! change at the cold and the hot end, [δ_{lo}] and [δ_{hi}] the loosest and tightest of them,
//! [l] and [k] the key's bearing length and contact height.

use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- What both joints share (shaft_joint::joint_specs, PartProperties::hub) ---
    record("shaft_joint.required_Nm", "T_{fit,req}", "{shaft_joint.safety_factor} * {metal.torque_cold_high_Nm}"),
    // The part material in effect: on the steel circuit the steel's values (the Materials
    // inputs for the default pick, else the pick's), else the body's (a non-magnetic pick's,
    // else the workbook's 6061-T6); the yield is the library's for the metal in effect.
    record("shaft_joint.part_modulus_GPa", "E_{part}",
        r#"cases({materials.circuit_backiron} = 1 and {materials.parts.back_iron} = 1 => {materials.steel.modulus_GPa};
               {materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "modulus_GPa");
               else => 68.9)"#),
    record("shaft_joint.part_cte_per_C", "α_{part}",
        r#"cases({materials.circuit_backiron} = 1 and {materials.parts.back_iron} = 1 => {materials.steel.cte_per_C};
               {materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "cte_per_C");
               else => 23.6e-6)"#),
    record("shaft_joint.part_yield_MPa", "σ_{y,part}",
        r#"cases({materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "yield_MPa");
               else => 241)"#),
    // YIELD_FACTOR: the part stays elastic with a margin.
    record("shaft_joint.stress_limit_MPa", "σ_{lim}", "frac({shaft_joint.part_yield_MPa}, 1.2)"),

    // --- The inner hub on the wheel shaft (shaft_joint::joint) ---
    // The fit's holding torque at its loosest: friction × contact pressure × contact arc ×
    // length × radius per µm, times the effective minimum (the measured minimum less smoothing,
    // at the end that loosens it more, floored at 0).
    record("shaft_joint.hub_holding_Nm", "T_{hold,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [T_1] · max(0, {shaft_joint.hub_interference_min_um} - {shaft_joint.smoothing_um} + [δ_{lo}]))
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · {housing.hub_length_mm} · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    // The smallest measured minimum that holds the required torque.
    record("shaft_joint.hub_needed_um", "δ_{need,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.required_Nm}, [T_1]) + {shaft_joint.smoothing_um} - [δ_{lo}])
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · {housing.hub_length_mm} · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    // The bore's von Mises stress at the tightest fit: the stress per µm, √(k_θ² + k_θ + 1) p_1,
    // times the effective maximum (the measured maximum less smoothing, at the end that
    // tightens it more, floored at 0).
    record("shaft_joint.hub_bore_stress_MPa", "σ_{vM,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [σ_1] · max(0, {shaft_joint.hub_interference_max_um} - {shaft_joint.smoothing_um} + [δ_{hi}]))
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    // The largest measured maximum within the stress limit.
    record("shaft_joint.hub_elastic_max_um", "δ_{el,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.stress_limit_MPa}, [σ_1]) + {shaft_joint.smoothing_um} - [δ_{hi}])
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    // The key alone at the required torque (form A: bearing length L − b), its contact capped
    // at the keyway depth: 0 with no keyway, +inf with no bearing length or contact.
    record("shaft_joint.hub_key_pressure_MPa", "p_{key,hub}",
        r#"cases({coupling.keyway_depth_mm} > 0 and [l] > 0 and [k] > 0 => frac(2 · {shaft_joint.required_Nm} · 1000, {coupling.bore_mm} · [k] · [l]);
               {coupling.keyway_depth_mm} > 0 => inf;
               else => 0)
           where [l] = {housing.hub_length_mm} - {shaft_joint.key_width_mm},
                 [k] = min({shaft_joint.key_contact_mm}, {coupling.keyway_depth_mm})"#),
    // The wheel shaft's diametral shrink from 20 °C: the linear expansion from 265 K up; below
    // it NIST's Fe-2.25Ni fit, y = (L − L293)/L293 × 10⁵, anchored at 265 K.
    record("shaft_joint.wheel_shaft_shrink_um", "Δd_{wheel}",
        "cases([ϑ_K] >= 265 => 11.5e-6 · (20 - {shaft_joint.wheel_shaft_fit_C}) · {coupling.bore_mm} · 1000;
               else => (11.5e-6 · (20 - (265 - 273.15)) + ([y_{265}] - [y]) · 1e-5) · {coupling.bore_mm} · 1000)
           where [ϑ_K] = {shaft_joint.wheel_shaft_fit_C} + 273.15,
                 [y_{265}] = -210.4 - 0.05699 · 265 + 5.072e-3 · 265^2 - 1.381e-5 · 265^3 + 1.897e-8 · 265^4,
                 [y] = -210.4 - 0.05699 · [ϑ_K] + 5.072e-3 · [ϑ_K]^2 - 1.381e-5 · [ϑ_K]^3 + 1.897e-8 · [ϑ_K]^4"),
    // What the allowed temperature and the shaft's shrink open, less the joining clearance.
    record("shaft_joint.hub_fittable_max_um", "δ_{fit,hub}",
        "{shaft_joint.part_cte_per_C} · ({shaft_joint.hub_fit_allowed_C} - 20) · {coupling.bore_mm} · 1000
         + {shaft_joint.wheel_shaft_shrink_um} - {shaft_joint.joining_clearance_um}"),
    // The part's temperature the fit would need on its shaft as set (the measured maximum and
    // the joining clearance, less the shaft's shrink).
    record("shaft_joint.hub_fit_temp_C", "ϑ_{need,hub}",
        "20 + frac({shaft_joint.hub_interference_max_um} + {shaft_joint.joining_clearance_um} - {shaft_joint.wheel_shaft_shrink_um},
                   {shaft_joint.part_cte_per_C} · {coupling.bore_mm} · 1000)"),
    // Owner R-15: the lowest of your limit for the bonded inner hub, the selected adhesive's
    // design limit and the inner ring's own single-ring onset less the cure margin
    // (CURE_MARGIN_MIN_C, 10 °C), ties to the earlier, as `hub_fit_allowed` names them. Each arm
    // compares its value with both others, so a NaN in any of the three fails every arm: not a
    // number (the markup's min would carry a NaN only in its first argument).
    record("shaft_joint.hub_fit_allowed_C", "ϑ_{allow,hub}",
        "cases({shaft_joint.hub_fit_max_C} <= {temperature.adhesive.design_limit_C} and {shaft_joint.hub_fit_max_C} <= [ϑ_{cap}]
                   => {shaft_joint.hub_fit_max_C};
               {temperature.adhesive.design_limit_C} <= {shaft_joint.hub_fit_max_C} and {temperature.adhesive.design_limit_C} <= [ϑ_{cap}]
                   => {temperature.adhesive.design_limit_C};
               [ϑ_{cap}] <= {shaft_joint.hub_fit_max_C} and [ϑ_{cap}] <= {temperature.adhesive.design_limit_C} => [ϑ_{cap}];
               else => nan)
           where [ϑ_{cap}] = {temperature.demag.inner_onset_single_ring_C} - 10"),

    // --- The cup on the gearbox output shaft (shaft_joint::joint): its boss is the outer
    // diameter, its boss and web the length; otherwise as the inner hub's ---
    record("shaft_joint.cup_holding_Nm", "T_{hold,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [T_1] · max(0, {shaft_joint.cup_interference_min_um} - {shaft_joint.smoothing_um} + [δ_{lo}]))
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · ({metal.boss_length_mm} + {metal.web_mm}) · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    record("shaft_joint.cup_needed_um", "δ_{need,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.required_Nm}, [T_1]) + {shaft_joint.smoothing_um} - [δ_{lo}])
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · ({metal.boss_length_mm} + {metal.web_mm}) · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    record("shaft_joint.cup_bore_stress_MPa", "σ_{vM,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [σ_1] · max(0, {shaft_joint.cup_interference_max_um} - {shaft_joint.smoothing_um} + [δ_{hi}]))
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    record("shaft_joint.cup_elastic_max_um", "δ_{el,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.stress_limit_MPa}, [σ_1]) + {shaft_joint.smoothing_um} - [δ_{hi}])
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    record("shaft_joint.cup_key_pressure_MPa", "p_{key,cup}",
        r#"cases({coupling.keyway_depth_mm} > 0 and [l] > 0 and [k] > 0 => frac(2 · {shaft_joint.required_Nm} · 1000, {coupling.bore_mm} · [k] · [l]);
               {coupling.keyway_depth_mm} > 0 => inf;
               else => 0)
           where [l] = {metal.boss_length_mm} + {metal.web_mm} - {shaft_joint.key_width_mm},
                 [k] = min({shaft_joint.key_contact_mm}, {coupling.keyway_depth_mm})"#),
    record("shaft_joint.output_shaft_shrink_um", "Δd_{gbx}",
        "cases([ϑ_K] >= 265 => 11.5e-6 · (20 - {shaft_joint.output_shaft_fit_C}) · {coupling.bore_mm} · 1000;
               else => (11.5e-6 · (20 - (265 - 273.15)) + ([y_{265}] - [y]) · 1e-5) · {coupling.bore_mm} · 1000)
           where [ϑ_K] = {shaft_joint.output_shaft_fit_C} + 273.15,
                 [y_{265}] = -210.4 - 0.05699 · 265 + 5.072e-3 · 265^2 - 1.381e-5 · 265^3 + 1.897e-8 · 265^4,
                 [y] = -210.4 - 0.05699 · [ϑ_K] + 5.072e-3 · [ϑ_K]^2 - 1.381e-5 · [ϑ_K]^3 + 1.897e-8 · [ϑ_K]^4"),
    record("shaft_joint.cup_fittable_max_um", "δ_{fit,cup}",
        "{shaft_joint.part_cte_per_C} · ({shaft_joint.cup_fit_allowed_C} - 20) · {coupling.bore_mm} · 1000
         + {shaft_joint.output_shaft_shrink_um} - {shaft_joint.joining_clearance_um}"),
    record("shaft_joint.cup_fit_temp_C", "ϑ_{need,cup}",
        "20 + frac({shaft_joint.cup_interference_max_um} + {shaft_joint.joining_clearance_um} - {shaft_joint.output_shaft_shrink_um},
                   {shaft_joint.part_cte_per_C} · {coupling.bore_mm} · 1000)"),
    // The bare cup is fitted before its magnets are bonded: your limit alone.
    record("shaft_joint.cup_fit_allowed_C", "ϑ_{allow,cup}", "{shaft_joint.cup_fit_max_C}"),
];
