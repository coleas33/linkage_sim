//! Demagnetization (plan A-3 batch 2): the Temperature design block C42 to C62 for both rings
//! (correction E20: each ring against its own grade, Br and rating; the ring with the lower
//! magnet limit governs and the block shows it), the positive-beta cold side of hard ferrite,
//! and the governing temperature limit with the adhesive limit and the hot-day start it is
//! compared with.
//!
//! Each formula is transcribed from the engine function named beside it (`temperature.rs`),
//! with every approved correction on. The onsets are the workbook's linear crossing of the
//! reverse field (scaling with Br) and the knee (scaling with Hcj):
//! ϑ = 20 + (H_k − H) / (H_k |β| − H |α|) − Δϑ_cal, H_k = k_knee · H_cj.
//!
//! Symbols: ϑ temperature, H field [kA/m], β the Hcj coefficient, α the Br coefficient.

use crate::engine::deviations::DeviationId::{E19, E20};
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The rings' grades and ratings (model::resolve_magnets, E19 on library parts) ---
    record("model.inner_grade", "grade_{i,used}",
        r#"cases([g_{lib}] != none => [g_{lib}]; [g_{in}] != none => [g_{in}]; else => "")
           where [g_{lib}] = table("magnets", {coupling.magnets.part_inner}, "grade"),
                 [g_{in}] = table("grades", {coupling.magnets.grade_inner}, "id")"#).corrected(&[E19]),
    record("model.outer_grade", "grade_{o,used}",
        r#"cases([g_{lib}] != none => [g_{lib}]; [g_{in}] != none => [g_{in}]; else => "")
           where [g_{lib}] = table("magnets", {coupling.magnets.part_outer}, "grade"),
                 [g_{in}] = table("grades", {coupling.magnets.grade_outer}, "id")"#).corrected(&[E19]),
    record("model.inner_tmax_C", "ϑ_{max,i}",
        r#"cases([ϑ_{lib}] != none => [ϑ_{lib}]; [ϑ_{grade}] != none => [ϑ_{grade}]; else => "n/a")
           where [ϑ_{lib}] = table("magnets", {coupling.magnets.part_inner}, "tmax_C"),
                 [ϑ_{grade}] = table("grades", {coupling.magnets.grade_inner}, "tmax_C")"#).corrected(&[E19]),
    record("model.outer_tmax_C", "ϑ_{max,o}",
        r#"cases([ϑ_{lib}] != none => [ϑ_{lib}]; [ϑ_{grade}] != none => [ϑ_{grade}]; else => "n/a")
           where [ϑ_{lib}] = table("magnets", {coupling.magnets.part_outer}, "tmax_C"),
                 [ϑ_{grade}] = table("grades", {coupling.magnets.grade_outer}, "tmax_C")"#).corrected(&[E19]),

    // --- Each ring's coercivity and limits (temperature::ring_demag, E20) ---
    // The ring's grade supplies Hcj and beta unless the coercivity source is the inputs (0);
    // a magnet without a grade always uses the inputs.
    record("temperature.demag.inner_hcj20_kA_m", "H_{cj,i}",
        r#"cases({temperature.demag.coercivity_source} = 1 and [H_{grade}] != none => [H_{grade}];
               else => {temperature.demag.hcj20_kA_m})
           where [H_{grade}] = table("grades", {model.inner_grade}, "hcj20_kA_m")"#).corrected(&[E20]),
    record("temperature.demag.outer_hcj20_kA_m", "H_{cj,o}",
        r#"cases({temperature.demag.coercivity_source} = 1 and [H_{grade}] != none => [H_{grade}];
               else => {temperature.demag.hcj20_kA_m})
           where [H_{grade}] = table("grades", {model.outer_grade}, "hcj20_kA_m")"#).corrected(&[E20]),
    record("temperature.demag.inner_beta_per_C", "β_i",
        r#"cases({temperature.demag.coercivity_source} = 1 and [β_{grade}] != none => [β_{grade}];
               else => {temperature.demag.beta_hcj_per_C})
           where [β_{grade}] = table("grades", {model.inner_grade}, "beta_hcj_per_C")"#).corrected(&[E20]),
    record("temperature.demag.outer_beta_per_C", "β_o",
        r#"cases({temperature.demag.coercivity_source} = 1 and [β_{grade}] != none => [β_{grade}];
               else => {temperature.demag.beta_hcj_per_C})
           where [β_{grade}] = table("grades", {model.outer_grade}, "beta_hcj_per_C")"#).corrected(&[E20]),
    // A ring's magnet limit: its skipping onset minus the margin, the onset calibrated so the
    // reference magnet (permeance coefficient 1) reaches the knee at the rating; with a positive
    // beta (hard ferrite) the knee is never reached on heating and the rating is the limit.
    record("temperature.demag.inner_magnet_limit_C", "ϑ_{lim,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0
                   => cases({model.inner_tmax_C} != "n/a" => {model.inner_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - [Δϑ_{cal,i}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000,
                 [Δϑ_{cal,i}] = cases({model.inner_tmax_C} != "n/a"
                     => 20 + frac([H_k] - [H_{ref,i}], [H_k] · abs({temperature.demag.inner_beta_per_C}) - [H_{ref,i}] · abs({model.inner_alpha_br_per_C}))
                        - {model.inner_tmax_C};
                     else => 0)"#).corrected(&[E20]),
    record("temperature.demag.outer_magnet_limit_C", "ϑ_{lim,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0
                   => cases({model.outer_tmax_C} != "n/a" => {model.outer_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.outer_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.outer_alpha_br_per_C}))
                       - [Δϑ_{cal,o}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m},
                 [H_{ref,o}] = frac({model.outer_br_T}, 2 * {coupling.mu0}) / 1000,
                 [Δϑ_{cal,o}] = cases({model.outer_tmax_C} != "n/a"
                     => 20 + frac([H_k] - [H_{ref,o}], [H_k] · abs({temperature.demag.outer_beta_per_C}) - [H_{ref,o}] · abs({model.outer_alpha_br_per_C}))
                        - {model.outer_tmax_C};
                     else => 0)"#).corrected(&[E20]),
    // temperature::cold_onset_C: with beta > 0 the knee falls as the magnet cools; the ring's
    // cold limit is its skipping cold onset plus the margin.
    record("temperature.demag.inner_cold_limit_C", "ϑ_{cold,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0
                   => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                {temperature.demag.h_rev_likepole_kA_m} · {model.inner_alpha_br_per_C} - [H_k] · {temperature.demag.inner_beta_per_C})
                      + {temperature.demag.design_margin_C};
               else => "n/a")
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m}"#).corrected(&[E20]),
    record("temperature.demag.outer_cold_limit_C", "ϑ_{cold,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0
                   => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                {temperature.demag.h_rev_likepole_kA_m} · {model.outer_alpha_br_per_C} - [H_k] · {temperature.demag.outer_beta_per_C})
                      + {temperature.demag.design_margin_C};
               else => "n/a")
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}"#).corrected(&[E20]),

    // --- The block the sheet shows: the governing ring (temperature::compute) ---
    record("temperature.demag.demag_ring", "ring_{hot}",
        r#"cases({temperature.demag.outer_magnet_limit_C} < {temperature.demag.inner_magnet_limit_C} => "outer"; else => "inner")"#)
        .corrected(&[E20]),
    record("temperature.demag.br20_T", "B_{r,ring}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_br_T}; else => {model.inner_br_T})"#).corrected(&[E20]),
    record("temperature.demag.alpha_br", "α_{ring}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C})"#)
        .corrected(&[E20]),
    record("temperature.demag.tmax_lib_C", "ϑ_{max}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_tmax_C}; else => {model.inner_tmax_C})"#).corrected(&[E20]),
    record("temperature.demag.hcj20_used_kA_m", "H_{cj}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m}; else => {temperature.demag.inner_hcj20_kA_m})"#)
        .corrected(&[E20]),
    record("temperature.demag.beta_used_per_C", "β",
        r#"cases({temperature.demag.demag_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C})"#)
        .corrected(&[E20]),
    // The reverse field of a magnet at permeance coefficient 1: Br / (2 μ0), in kA/m.
    record("temperature.demag.h_ref_kA_m", "H_{ref}", "frac({temperature.demag.br20_T}, 2 * {coupling.mu0}) / 1000"),
    record("temperature.demag.t_ref_model_C", "ϑ_{ref}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_ref_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_ref_kA_m} · abs({temperature.demag.alpha_br})))
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.calibration_offset_C", "Δϑ_{cal}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 => 0;
               {temperature.demag.tmax_lib_C} != "n/a" => {temperature.demag.t_ref_model_C} - {temperature.demag.tmax_lib_C};
               else => 0)"#).corrected(&[E20]),
    record("temperature.demag.onset_aligned_C", "ϑ_{al}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_aligned_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_aligned_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_pullout_C", "ϑ_{po}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_pullout_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_pullout_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_skipping_C", "ϑ_{sk}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_single_ring_C", "ϑ_{sr}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_single_ring_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.magnet_limit_C", "ϑ_{mag}",
        r#"cases({temperature.demag.beta_used_per_C} > 0
                   => cases({temperature.demag.tmax_lib_C} != "n/a" => {temperature.demag.tmax_lib_C}; else => inf);
               else => {temperature.demag.onset_skipping_C} - {temperature.demag.design_margin_C})"#).corrected(&[E20]),

    // --- The cold side (E20, positive beta): the ring with the higher cold limit ---
    record("temperature.demag.cold_ring", "ring_{cold}",
        r#"cases({temperature.demag.outer_cold_limit_C} != "n/a" and {temperature.demag.inner_cold_limit_C} != "n/a"
                   and {temperature.demag.outer_cold_limit_C} > {temperature.demag.inner_cold_limit_C} => "outer";
               {temperature.demag.outer_cold_limit_C} != "n/a" and {temperature.demag.inner_cold_limit_C} = "n/a" => "outer";
               else => "inner")"#).corrected(&[E20]),
    record("temperature.demag.cold_limit_C", "ϑ_{cold}",
        r#"cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_cold_limit_C};
               else => {temperature.demag.inner_cold_limit_C})"#).corrected(&[E20]),
    // Both rings' checks, as the engine's: the inner ring passes, then the outer ring; a ring
    // with no cold limit passes. Not the higher limit's check alone: an undefined (NaN) limit
    // never holds the cold side, yet its ring fails.
    record("temperature.demag.cold_check", "C_{cold}",
        r#"cases({temperature.demag.cold_limit_C} = "n/a" => "n/a (coercivity rises as the magnet cools)";
               {temperature.demag.inner_cold_limit_C} = "n/a" or {metal.min_temp_C} >= {temperature.demag.inner_cold_limit_C}
                   => cases({temperature.demag.outer_cold_limit_C} = "n/a" or {metal.min_temp_C} >= {temperature.demag.outer_cold_limit_C} => "OK";
                            else => "Below the cold demagnetization limit");
               else => "Below the cold demagnetization limit")"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_aligned_C", "ϑ_{cold,al}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_aligned_kA_m}, {temperature.demag.h_rev_aligned_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_pullout_C", "ϑ_{cold,po}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_pullout_kA_m}, {temperature.demag.h_rev_pullout_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_skipping_C", "ϑ_{cold,sk}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m}, {temperature.demag.h_rev_likepole_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_single_ring_C", "ϑ_{cold,sr}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m}, {temperature.demag.h_rev_single_ring_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),

    // --- Torque at the limits (decision A2-7: each ring's Br with its own coefficient) ---
    // With a positive beta and no rating there is no hot limit (+inf) and no torque at it.
    record("temperature.demag.torque_at_limit_Nm", "T_{mag}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 and {temperature.demag.tmax_lib_C} = "n/a" => nan;
               else => {model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.demag.magnet_limit_C} - 20))
                                              * (1 + {model.outer_alpha_br_per_C} * ({temperature.demag.magnet_limit_C} - 20))))"#).corrected(&[E20]),
    record("temperature.demag.torque_at_service_Nm", "T_{svc}", "{model.pullout_Nm}"),

    // --- The governing limit and what it is compared with (temperature::compute) ---
    record("temperature.adhesive.design_limit_C", "ϑ_{adh}",
        r#"table("adhesives", {temperature.adhesive.selected}, "design_limit_C")"#),
    record("temperature.adhesive.cure_C", "ϑ_{cure}", r#"table("adhesives", {temperature.adhesive.selected}, "cure_C")"#),
    record("temperature.summary.governing_limit_C", "ϑ_{gov}",
        "min({temperature.demag.magnet_limit_C}, {temperature.adhesive.design_limit_C})"),
    record("temperature.duty.hot_day_start_C", "ϑ_{hot}", "{temperature.duty.hot_ambient_C} + {temperature.duty.driving_rise_C}"),
];
