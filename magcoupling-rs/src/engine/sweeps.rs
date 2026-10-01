//! Design-space sweeps ('Gap sweep' and 'Pole sweep' sheets).
//!
//! Port of `reference/magcoupling-py/magcoupling/sweeps.py`. Both sweeps rerun
//! the Calculator's 2D harmonic model with one variable changed:
//!
//! - gap sweep: the inner-block-corner to outer-face gap (current hub apothem);
//!   uses the Calculator's calibration factor.
//! - pole sweep: the number of poles per ring, with the smallest inner apothem
//!   that fits the block width (+0.05 mm) and the keyed bore wall; uses the
//!   original 0.95 factor (the measured correction does not transfer).
//!
//! Each row also gets a fit/status flag in the workbook's priority order: inner
//! flat too narrow, outer flat too narrow, outside OD envelope, below hot
//! minimum, nominal: test needed.
//!
//! This module holds the sweep variables ([`GAP_SWEEP_CORNER_GAPS_MM`],
//! [`POLE_SWEEP_POLES`]), the sweep row ([`SweepRow`], 26 fields: 338 Gap sweep
//! and 156 Pole sweep cells), the borrowed values ([`SweepContext`]) and the two
//! sweeps ([`gap_sweep`], [`pole_sweep`]). The rows reuse
//! [`super::model::shear_stress`].
//!
//! Deviations touching these sheets (see
//! [`crate::engine::deviations::REGISTRY`]): E4 is applied (Pole sweep C6:C11,
//! the keyed-bore wall adds the inner bondline), and so is E7 (columns N, Q, T
//! and U, and through them V to AA: pull-out at the maximum over angle, through
//! the model's `at_pull_out`). Addendum A3: a row sums the Calculator's harmonic
//! set (`coupling.max_harmonic`); column U includes harmonics 7 to 11 when the set
//! does, which have no column of their own. A row whose `f_end` (column W) is 0 or
//! below is outside the end-effect model (audit M9): the GUI flags it with
//! `model::end_effect_in_range`, as the Calculator's `end_effect_check` does.

use std::f64::consts::PI;

use super::compat::{py_max, py_min};
use super::deviations::{DeviationId, Deviations};
use super::meta::{col, rows};
use super::model::{
    at_pull_out, corner_radius, harmonic_count, harmonic_slot, harmonic_sum, shear_stress,
};

/// Corner gaps of the gap sweep [mm] (rows 6-18).
pub const GAP_SWEEP_CORNER_GAPS_MM: [f64; 13] = [
    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0,
];
/// Pole counts of the pole sweep (rows 6-11).
pub const POLE_SWEEP_POLES: [i64; 6] = [6, 8, 10, 12, 14, 16];

rows! {
    /// One row of the Gap sweep or Pole sweep sheet (columns B-AA). Labels and
    /// units are the workbook headers (rows 4 and 5).
    pub struct SweepRow {
        variable: f64 => col("", "Swept variable",
            "Corner gap [mm] in the gap sweep (workbook 'Corner gap (mm)'); poles per ring in the pole sweep ('Poles').", "B"),
        inner_apothem_mm: f64 => col("mm", "Inner apothem", "", "C"),
        corner_gap_mm: f64 => col("mm", "Corner gap", "", "D"),
        centre_gap_mm: f64 => col("mm", "Centre gap", "", "E"),
        outer_face_apothem_mm: f64 => col("mm", "Outer face apothem", "", "F"),
        cup_od_mm: f64 => col("mm", "Cup OD", "", "G"),
        gap_radius_mm: f64 => col("mm", "R_g", "", "H"),
        pole_pitch_mm: f64 => col("mm", "Pole pitch", "", "I"),
        fill_inner: f64 => col("-", "Fill in", "", "J"),
        fill_outer: f64 => col("-", "Fill out", "", "K"),
        k1: f64 => col("1/m", "k1", "", "L"),
        s1: f64 => col("-", "S1", "", "M"),
        tau1_Pa: f64 => col("Pa", "tau1", "", "N"),
        k3: f64 => col("1/m", "k3", "", "O"),
        s3: f64 => col("-", "S3", "", "P"),
        tau3_Pa: f64 => col("Pa", "tau3", "", "Q"),
        k5: f64 => col("1/m", "k5", "", "R"),
        s5: f64 => col("-", "S5", "", "S"),
        tau5_Pa: f64 => col("Pa", "tau5", "", "T"),
        tau_Pa: f64 => col("Pa", "tau", "", "U"),
        torque_2d_Nm: f64 => col("Nm", "T2D", "", "V"),
        f_end: f64 => col("-", "f_end", "", "W"),
        pullout_op_Nm: f64 => col("Nm", "Pull-out at T_op", "", "X"),
        pullout_20C_Nm: f64 => col("Nm", "Pull-out at 20 C", "", "Y"),
        gearbox_input_Nm: f64 => col("Nm", "Gearbox input at slip", "", "Z"),
        status: String => col("", "Hot nominal / fit", "", "AA"),
    }
}

/// Values the sweeps borrow from the Calculator and Metal design (Python `SweepContext`).
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct SweepContext {
    pub faceted: i64,
    pub backiron: i64,
    pub t_i: f64,
    pub w_i: f64,
    pub t_o: f64,
    pub w_o: f64,
    pub L: f64,
    pub br_i20: f64,
    pub br_o20: f64,
    pub br_i_op: f64,
    pub br_o_op: f64,
    pub bond_outer: f64,
    /// Inner magnet bondline [mm] (Metal design C120), read only by correction E4.
    pub bond_inner: f64,
    pub cup_wall_corner: f64,
    pub c_end: f64,
    pub mu0: f64,
    pub gear_ratio: f64,
    pub gear_eff: f64,
    pub required_floor_Nm: f64,
    pub max_diameter_mm: f64,
    /// The Calculator's highest harmonic (`coupling.max_harmonic`, Addendum A3).
    pub max_harmonic: i64,
}

/// One sweep row (Python `_row`), columns C-AA with the workbook's status priority.
#[allow(non_snake_case)] // Python names (F, E, G, H, I, J, K, U, V, W, X, Y, Z)
fn row(
    ctx: &SweepContext,
    variable: f64,
    npole: i64,
    a_i: f64,
    corner_gap: f64,
    factor: f64,
    dev: Deviations,
) -> SweepRow {
    let n = npole as f64;
    let r_face = a_i + ctx.t_i;
    let F = (if ctx.faceted == 1 {
        corner_radius(r_face, ctx.w_i)
    } else {
        r_face
    }) + corner_gap;
    let E = F - r_face;
    let back = F + ctx.t_o + ctx.bond_outer;
    let G = 2.0
        * ((if ctx.faceted == 1 {
            back / (PI / n).cos()
        } else {
            back
        }) + ctx.cup_wall_corner);
    let H = r_face + E / 2.0;
    let I = 2.0 * PI * H / n;
    let J = py_min(1.0, ctx.w_i / (2.0 * PI * (a_i + ctx.t_i / 2.0) / n));
    let K = py_min(1.0, ctx.w_o / (2.0 * PI * (F + ctx.t_o / 2.0) / n));
    let h = shear_stress(
        ctx.br_i_op,
        ctx.br_o_op,
        J,
        K,
        npole,
        H,
        ctx.t_i,
        ctx.t_o,
        E,
        ctx.backiron,
        ctx.mu0,
    );
    let s = h.map(|x| x.s(ctx.backiron));
    // Addendum A3: the Calculator's harmonic set. E7: every harmonic at the true pull-out
    // angle when half a pitch is not the maximum (as the model).
    let count = harmonic_count(ctx.max_harmonic);
    let h_pull = at_pull_out(&h[..count.unwrap_or(0)], ctx.backiron, ctx.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let U = harmonic_sum(count, taus.iter().copied());
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
    let W = 1.0 - ctx.c_end * I / ctx.L;
    let X = V * W * factor;
    let Y = X * (ctx.br_i20 * ctx.br_o20) / (ctx.br_i_op * ctx.br_o_op);
    let Z = X / (ctx.gear_ratio * ctx.gear_eff);
    let status = if 2.0 * a_i * (PI / n).tan() < ctx.w_i {
        "inner flat too narrow"
    } else if 2.0 * F * (PI / n).tan() < ctx.w_o {
        "outer flat too narrow"
    } else if G > ctx.max_diameter_mm {
        "outside OD envelope"
    } else if X < ctx.required_floor_Nm {
        "below hot minimum"
    } else {
        "nominal: test needed"
    };
    let [h1, h3, h5, ..] = h;
    SweepRow {
        variable,
        inner_apothem_mm: a_i,
        corner_gap_mm: corner_gap,
        centre_gap_mm: E,
        outer_face_apothem_mm: F,
        cup_od_mm: G,
        gap_radius_mm: H,
        pole_pitch_mm: I,
        fill_inner: J,
        fill_outer: K,
        k1: h1.k,
        s1: s[0],
        tau1_Pa: harmonic_slot(count, &taus, 0),
        k3: h3.k,
        s3: s[1],
        tau3_Pa: harmonic_slot(count, &taus, 1),
        k5: h5.k,
        s5: s[2],
        tau5_Pa: harmonic_slot(count, &taus, 2),
        tau_Pa: U,
        torque_2d_Nm: V,
        f_end: W,
        pullout_op_Nm: X,
        pullout_20C_Nm: Y,
        gearbox_input_Nm: Z,
        status: status.to_owned(),
    }
}

/// Pull-out vs corner gap for the current layout (Calculator calibration factor).
pub fn gap_sweep(
    ctx: &SweepContext,
    npole: i64,
    a_i: f64,
    f_cal: f64,
    dev: Deviations,
) -> Vec<SweepRow> {
    GAP_SWEEP_CORNER_GAPS_MM
        .iter()
        .map(|&g| row(ctx, g, npole, a_i, g, f_cal, dev))
        .collect()
}

/// Pull-out vs poles; smallest apothem that fits the block (+0.05 mm) and the keyed-bore wall (2.5 mm;
/// with E4 the wall is 2.5 mm of steel under the inner bondline).
pub fn pole_sweep(
    ctx: &SweepContext,
    corner_gap_mm: f64,
    bore_mm: f64,
    keyway_depth_mm: f64,
    f_cal_original: f64,
    dev: Deviations,
) -> Vec<SweepRow> {
    POLE_SWEEP_POLES
        .iter()
        .map(|&n| {
            // E4: the 2.5 mm keyed-bore wall is steel; the inner bondline sits on top of it.
            let keyed_wall = if dev.is_on(DeviationId::E4) {
                bore_mm / 2.0 + keyway_depth_mm + 2.5 + ctx.bond_inner
            } else {
                bore_mm / 2.0 + keyway_depth_mm + 2.5
            };
            let a_i = py_max(ctx.w_i / (2.0 * (PI / n as f64).tan()) + 0.05, keyed_wall);
            row(ctx, n as f64, n, a_i, corner_gap_mm, f_cal_original, dev)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::WORKBOOK_MAX_HARMONIC;

    /// The default design's sweep context (`api::compute` at the workbook defaults).
    fn ctx() -> SweepContext {
        SweepContext {
            faceted: 1,
            backiron: 1,
            t_i: 3.17,
            w_i: 6.35,
            t_o: 3.17,
            w_o: 6.35,
            L: 12.7,
            br_i20: 1.29,
            br_o20: 1.29,
            br_i_op: 1.24356,
            br_o_op: 1.24356,
            bond_outer: 0.05,
            bond_inner: 0.05,
            cup_wall_corner: 1.8,
            c_end: 0.15,
            mu0: 1.256637e-6,
            gear_ratio: 5.0,
            gear_eff: 0.95,
            required_floor_Nm: 2.5,
            max_diameter_mm: 43.0,
            max_harmonic: WORKBOOK_MAX_HARMONIC,
        }
    }

    #[test]
    fn the_harmonic_set_reaches_every_row() {
        // Addendum A3: a row sums the harmonics the Calculator sums; a left-out one reads 0,
        // and a code outside the choices gives NaN (decision D3).
        let run = |max_harmonic| {
            let c = SweepContext {
                max_harmonic,
                ..ctx()
            };
            row(&c, 1.25, 10, 10.15, 1.25, 0.95, Deviations::NONE)
        };
        let (one, workbook, eleven) = (run(1), run(WORKBOOK_MAX_HARMONIC), run(11));
        assert_eq!((one.tau3_Pa, one.tau5_Pa), (0.0, 0.0));
        assert_eq!(one.tau_Pa, one.tau1_Pa);
        assert_eq!(
            workbook.tau_Pa,
            0.0 + workbook.tau1_Pa + workbook.tau3_Pa + workbook.tau5_Pa
        );
        assert!(eleven.tau_Pa != workbook.tau_Pa);
        assert_eq!(
            (eleven.tau1_Pa, eleven.s3, eleven.k5),
            (workbook.tau1_Pa, workbook.s3, workbook.k5)
        );
        let invalid = run(4);
        assert!(invalid.tau_Pa.is_nan() && invalid.pullout_op_Nm.is_nan());
    }

    #[test]
    fn status_checks_are_strict_at_exact_equality() {
        // Architecture section 7 step 8. All four status comparisons at exact equality: the
        // strict `<` and `>` of sweeps.py fall through every check to "nominal: test needed".
        // Each right-hand side is a context value its left-hand side does not read.
        let (npole, a_i, gap) = (10, 10.15, 1.25);
        let n = npole as f64;
        let run = |c: &SweepContext| row(c, gap, npole, a_i, gap, 0.95, Deviations::NONE);
        let mut c = ctx();
        c.w_i = 2.0 * a_i * (PI / n).tan(); // the inner-flat expression of `row`, bit for bit
        c.w_o = 2.0 * run(&c).outer_face_apothem_mm * (PI / n).tan(); // F reads w_i, not w_o
        let second = run(&c);
        c.max_diameter_mm = second.cup_od_mm; // G reads neither limit
        c.required_floor_Nm = second.pullout_op_Nm; // X reads w_o (outer fill), already final
        let r = run(&c);
        assert_eq!(
            (r.cup_od_mm, r.pullout_op_Nm),
            (c.max_diameter_mm, c.required_floor_Nm)
        );
        assert_eq!(r.status, "nominal: test needed");
    }
}
