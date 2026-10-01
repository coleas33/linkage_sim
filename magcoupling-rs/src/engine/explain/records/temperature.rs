//! Temperature (plan A-3 batch 4): Br(T) and torque against temperature (torque ∝ Br_i Br_o,
//! each ring with its own coefficient, decision A2-7), the magnet ratings and their checks,
//! and the temperature summary: the limits, which one governs, the margins and the verdict
//! (Temperature design C6 to C25, C180 to C192; Calculator C35, C107, C108).
//!
//! The Br(T) factor 1 + α (ϑ − 20) appears in each torque record because each record is one
//! workbook cell and the factor is what the student should see (plan A-3).
//! Each formula is transcribed from the engine function named beside it, corrections on.

use crate::engine::deviations::DeviationId::E20;
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Calculator: the coefficient and the rating checks (model::compute) ---
    record("model.alpha_br_per_C", "α_{calc}", "{calibration.alpha_br_per_C}"),
    record("model.inner_temp_check", "C_{ϑ,i}",
        r#"cases({model.inner_tmax_C} = "n/a" => "unknown"; {coupling.op_temp_C} <= {model.inner_tmax_C} => "OK";
               else => "OVER the magnet rating")"#),
    record("model.outer_temp_check", "C_{ϑ,o}",
        r#"cases({model.outer_tmax_C} = "n/a" => "unknown"; {coupling.op_temp_C} <= {model.outer_tmax_C} => "OK";
               else => "OVER the magnet rating")"#),

    // --- Torque against temperature (temperature::compute: magnet life, adhesive life) ---
    // Reversible: T(ϑ) = T_20 · (1 + α_i (ϑ − 20)) (1 + α_o (ϑ − 20)).
    record("temperature.magnet_life.torque_hot_day_Nm", "T_{hot}",
        "{model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.duty.hot_day_start_C} - 20))
                                * (1 + {model.outer_alpha_br_per_C} * ({temperature.duty.hot_day_start_C} - 20)))"),
    record("temperature.magnet_life.torque_hot_day_check", "C_{T,hot}",
        r#"cases({temperature.magnet_life.torque_hot_day_Nm} >= {metal.required_min_Nm} => "Meets it nominally (no variation allowance)";
               else => "Below it")"#),
    record("temperature.magnet_life.torque_peak_Nm", "T_{peak}",
        "{model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.magnet_life.peak_C} - 20))
                                * (1 + {model.outer_alpha_br_per_C} * ({temperature.magnet_life.peak_C} - 20)))"),
    record("temperature.adhesive_life.torque_peak_var_Nm", "T_{peak,var}",
        "{temperature.magnet_life.torque_peak_Nm} * (1 + {metal.variation})"),
    record("temperature.adhesive_life.margin_C", "Δϑ_{adh}", "{temperature.adhesive.design_limit_C} - {temperature.magnet_life.peak_C}"),
    record("temperature.mismatch.cold_limit_C", "ϑ_{min,mm}", "{metal.min_temp_C}"),

    // --- The summary block (temperature::compute) ---
    record("temperature.summary.magnet_limit_C", "ϑ_{mag}^{sum}", "{temperature.demag.magnet_limit_C}"),
    record("temperature.summary.adhesive_limit_C", "ϑ_{adh}^{sum}", "{temperature.adhesive.design_limit_C}"),
    record("temperature.summary.governing_note", "C_{gov}",
        r#"cases({temperature.demag.magnet_limit_C} <= {temperature.adhesive.design_limit_C} => "Magnets govern (skipping case).";
               else => "Adhesive governs.")"#),
    record("temperature.summary.hot_day_start_C", "ϑ_{hot}^{sum}", "{temperature.duty.hot_day_start_C}"),
    record("temperature.summary.torque_hot_day_Nm", "T_{hot}^{sum}", "{temperature.magnet_life.torque_hot_day_Nm}"),
    record("temperature.summary.torque_hot_day_note", "C_{T,hot}^{sum}", "{temperature.magnet_life.torque_hot_day_check}"),
    // OK needs margin at the hot-day start, both limits above the peak, 10 °C of cure margin,
    // and (E20) no ring below its cold limit.
    record("temperature.summary.verdict", "V_{temp}",
        r#"cases({temperature.summary.margin_hot_day_C} > 0 and {temperature.magnet_life.margin_limit_C} > 0
                   and {temperature.adhesive_life.margin_C} > 0 and {temperature.summary.cure_margin_C} >= 10
                   and {temperature.demag.cold_check} != "Below the cold demagnetization limit"
                   => "OK on temperature. Confirm drag torque and thermal cycling by test.";
               else => "CHECK: see the rows above.")"#).corrected(&[E20]),
];
