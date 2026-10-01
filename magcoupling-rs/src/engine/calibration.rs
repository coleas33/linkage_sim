//! Prototype measurement and model calibration ('Calibration' sheet).
//!
//! Port of `reference/magcoupling-py/magcoupling/calibration.py`.
//!
//! Purpose: reproduce the original analytical model for the 3D-printed,
//! no-back-iron prototype (20 × B842SH, 1.4 mm flat-face gap), compare it with
//! the 1.8 N·m bench measurement, and derive a one-point correction.
//!
//! That correction applies ONLY to the same magnet rings without intentional
//! back iron. The steel-backed candidate keeps the original 0.95 factor (the
//! model's calibration-factor selection, ported with `model`).
//!
//! Port notes:
//! - Python's `fea_gap1_mm = 1.0` and `fea_gap2_mm = 1.5` are plain dataclass
//!   fields without metadata, so they are not inputs (`input_schema()` omits
//!   them) and `compute` never reads them: it uses the literals 1, 1.5 and 0.5.
//!   They are omitted here, and the literals are kept.
//! - Deviations touching this sheet: E3 is applied (the Calibration!C21 default
//!   follows the corrected N42SH remanence, 1.30 T), and so is E7 (the τ_n
//!   harmonic sum, C40-C42, at the maximum over angle), see
//!   [`crate::engine::deviations::REGISTRY`].

use std::f64::consts::PI;

use super::compat::py_min;
use super::constants::MU0;
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, param, results};
use super::model::{br_factor, corner_radius, peak_angle, tau_at};

inputs! {
    /// Prototype inputs (Calibration!C5:C25, C47:C48).
    pub struct CalibrationInputs {
        fields {
            measured_torque_Nm: f64 = 1.8 => param("N·m", "Measured pull-out torque",
                "User bench result. Method, repeatability and temperature not supplied.", "Calibration!C5")
                .range(0.1, 10.0, 0.01),
            total_magnets: i64 = 20 => param("count", "Total installed magnets",
                "Two equal rings, alternating polarity.", "Calibration!C12")
                .range(4.0, 40.0, 2.0),
            spacing_mm: f64 = 1.4 => param("mm", "Reported spacing",
                "Between the opposing flat faces.", "Calibration!C14")
                .range(0.2, 5.0, 0.01),
            gap_definition: i64 = 1 => param("selector", "Gap definition",
                "0 = corner, 1 = flat centre.", "Calibration!C15")
                .choices(&[(0, "corner"), (1, "flat centre")]),
            test_temp_C: f64 = 20.0 => param("°C", "Assumed test magnet temperature",
                "ASSUMED; replace with the recorded value.", "Calibration!C16")
                .range(-40.0, 150.0, 0.5),
            apothem_mm: f64 = 9.85 => param("mm", "Prototype inner hub apothem", "", "Calibration!C17")
                .range(4.0, 30.0, 0.01),
            magnet_length_mm: f64 = 12.7 => param("mm", "Prototype magnet axial length", "", "Calibration!C18")
                .range(2.0, 50.8, 0.01),
            magnet_width_mm: f64 = 6.35 => param("mm", "Prototype magnet tangential width", "", "Calibration!C19")
                .range(1.0, 25.4, 0.01),
            magnet_thickness_mm: f64 = 3.17 => param("mm", "Prototype magnet radial thickness", "", "Calibration!C20")
                .range(0.5, 10.0, 0.01),
            br_T: f64 = 1.30 => param("T", "Prototype remanence at 20 °C", "", "Calibration!C21")
                .range(0.2, 1.5, 0.001),
            alpha_br_per_C: f64 = -0.0012 => param("1/°C", "Reversible Br temperature coefficient",
                "Used by every sheet. Torque scales with the square of the Br ratio.", "Calibration!C22")
                .range(-0.0025, 0.0, 0.00001)
                .assumption(),
            c_end: f64 = 0.15 => param("-", "Original end-effect coefficient", "", "Calibration!C23")
                .range(0.0, 0.5, 0.005)
                .assumption(),
            f_cal_original: f64 = 0.95 => param("-", "Original calibration coefficient",
                "Kept separate from the measured correction.", "Calibration!C24")
                .range(0.5, 1.5, 0.005)
                .assumption(),
            mu0: f64 = MU0 => param("H/m", "Vacuum permeability", "", "Calibration!C25")
                .range(1.2566e-6, 1.2567e-6, 1e-11),
            fea_torque1_Nm: f64 = 2.06 => param("N·m", "3D result at 1.0 mm corner gap",
                "Supplied context, not rerun.", "Calibration!C47")
                .range(0.0, 5.0, 0.01),
            fea_torque2_Nm: f64 = 1.7 => param("N·m", "3D result at 1.5 mm corner gap", "", "Calibration!C48")
                .range(0.0, 5.0, 0.01),
        }
    }
}

results! {
    /// Calibration sheet results (Calibration!C6:C50).
    pub struct CalibrationResults {
        fields {
            model_torque_Nm: f64 => out("N·m", "Original model at the assumed test conditions", "", "Calibration!C6"),
            model_error: f64 => out("fraction", "Model error relative to measured torque",
                "Negative means underprediction.", "Calibration!C7"),
            measured_over_model: f64 => out("factor", "Measured / original model",
                "One-point correction.", "Calibration!C8"),
            f_cal_updated: f64 => out("factor", "Updated model calibration coefficient",
                "Applies to the no-iron prototype only.", "Calibration!C9"),
            poles_per_ring: f64 => out("count", "Poles per ring", "", "Calibration!C13"),
            face_radius_mm: f64 => out("mm", "Inner magnet flat-face radius", "", "Calibration!C28"),
            corner_radius_mm: f64 => out("mm", "Inner magnet corner radius", "", "Calibration!C29"),
            corner_gap_mm: f64 => out("mm", "Equivalent minimum corner gap", "", "Calibration!C30"),
            outer_face_apothem_mm: f64 => out("mm", "Outer magnet face apothem", "", "Calibration!C31"),
            flat_gap_mm: f64 => out("mm", "Effective flat-centre gap", "", "Calibration!C32"),
            gap_radius_mm: f64 => out("mm", "Mean gap radius", "", "Calibration!C33"),
            fill_inner: f64 => out("-", "Inner magnet fill factor", "", "Calibration!C34"),
            fill_outer: f64 => out("-", "Outer magnet fill factor", "", "Calibration!C35"),
            br_test_T: f64 => out("T", "Br at assumed test temperature", "", "Calibration!C36"),
            pole_pitch_mm: f64 => out("mm", "Pole pitch", "", "Calibration!C37"),
            f_end: f64 => out("-", "End-effect factor", "", "Calibration!C38"),
            tau1_Pa: f64 => out("Pa", "Shear stress, harmonic 1", "", "Calibration!C40"),
            tau3_Pa: f64 => out("Pa", "Shear stress, harmonic 3", "", "Calibration!C41"),
            tau5_Pa: f64 => out("Pa", "Shear stress, harmonic 5", "", "Calibration!C42"),
            torque_2d_Nm: f64 => out("N·m", "2D torque before end and calibration factors", "", "Calibration!C43"),
            original_model_Nm: f64 => out("N·m", "Original model pull-out torque",
                "Same value as model_torque_Nm.", "Calibration!C44"),
            fea_interp_Nm: NumOrText => out("N·m", "3D interpolation at the prototype corner gap",
                "Linear; 'outside range' if off the 1.0–1.5 mm span.", "Calibration!C49"),
            fea_interp_error: NumOrText => out("fraction", "3D interpolation error versus measured", "", "Calibration!C50"),
        }
    }
}

/// Text of Calibration!C49 when the corner gap is off the 1.0–1.5 mm span.
pub const OUTSIDE_RANGE: &str = "outside range";
/// Text of Calibration!C50 when C49 is [`OUTSIDE_RANGE`].
pub const NOT_APPLICABLE: &str = "n.a.";

/// The Calibration sheet. Line by line the Python `calibration.compute`, with
/// E7 (the τ_n harmonic sum at the maximum over angle) when `dev` has it on.
pub fn compute(c: &CalibrationInputs, dev: Deviations) -> CalibrationResults {
    let poles = c.total_magnets as f64 / 2.0;
    let r_face = c.apothem_mm + c.magnet_thickness_mm;
    let r_corner = corner_radius(r_face, c.magnet_width_mm);
    let corner_gap = if c.gap_definition == 0 {
        c.spacing_mm
    } else {
        c.spacing_mm - (r_corner - r_face)
    };
    let a_o = r_corner + corner_gap;
    let g = a_o - r_face;
    let r_g = r_face + g / 2.0;
    let fill_i = py_min(
        1.0,
        c.magnet_width_mm / (2.0 * PI * (c.apothem_mm + c.magnet_thickness_mm / 2.0) / poles),
    );
    let fill_o = py_min(
        1.0,
        c.magnet_width_mm / (2.0 * PI * (a_o + c.magnet_thickness_mm / 2.0) / poles),
    );
    let br_t = c.br_T * br_factor(c.alpha_br_per_C, c.test_temp_C);
    let tau_p = 2.0 * PI * r_g / poles;
    let f_end = 1.0 - c.c_end * tau_p / c.magnet_length_mm;

    // Python's tau_n up to its last factor, sin(n pi/2): the harmonic's amplitude.
    let amp_n = |n: u32| -> f64 {
        let n = f64::from(n);
        let k = n * (poles / 2.0) / (r_g / 1000.0);
        (br_t * 4.0 / (n * PI)).powi(2)
            * (n * fill_i * PI / 2.0).sin()
            * (n * fill_o * PI / 2.0).sin()
            / (2.0 * c.mu0)
            * (1.0 - (-k * c.magnet_thickness_mm / 1000.0).exp()).powi(2)
            * (-k * g / 1000.0).exp()
            / 2.0
    };

    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    let amps = [amp_n(1), amp_n(3), amp_n(5)];
    let peak = peak_angle(&amps, dev);
    let tau_n = |a: f64, n: u32| match peak {
        Some(x) => tau_at(a, n, x),
        None => a * (f64::from(n) * PI / 2.0).sin(), // the workbook expression, bit for bit
    };
    let (t1, t3, t5) = (tau_n(amps[0], 1), tau_n(amps[1], 3), tau_n(amps[2], 5));
    let t2d = (t1 + t3 + t5) * 2.0 * PI * (r_g / 1000.0).powi(2) * (c.magnet_length_mm / 1000.0);
    let model = t2d * f_end * c.f_cal_original;
    let (interp, interp_err) = if (1.0..=1.5).contains(&corner_gap) {
        let interp =
            c.fea_torque1_Nm + (c.fea_torque2_Nm - c.fea_torque1_Nm) * (corner_gap - 1.0) / 0.5;
        (
            NumOrText::Num(interp),
            NumOrText::Num(interp / c.measured_torque_Nm - 1.0),
        )
    } else {
        (
            NumOrText::Text(OUTSIDE_RANGE),
            NumOrText::Text(NOT_APPLICABLE),
        )
    };
    let ratio = c.measured_torque_Nm / model;

    CalibrationResults {
        model_torque_Nm: model,
        model_error: model / c.measured_torque_Nm - 1.0,
        measured_over_model: ratio,
        f_cal_updated: c.f_cal_original * ratio,
        poles_per_ring: poles,
        face_radius_mm: r_face,
        corner_radius_mm: r_corner,
        corner_gap_mm: corner_gap,
        outer_face_apothem_mm: a_o,
        flat_gap_mm: g,
        gap_radius_mm: r_g,
        fill_inner: fill_i,
        fill_outer: fill_o,
        br_test_T: br_t,
        pole_pitch_mm: tau_p,
        f_end,
        tau1_Pa: t1,
        tau3_Pa: t3,
        tau5_Pa: t5,
        torque_2d_Nm: t2d,
        original_model_Nm: model,
        fea_interp_Nm: interp,
        fea_interp_error: interp_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_design_reproduces_the_bench_correction() {
        // Workbook values (tests/data/reference_values.json); the full check is tests/parity.rs.
        // The workbook's Br (1.29 T) is restored by hand: E3 corrects the declared default.
        let workbook = CalibrationInputs {
            br_T: 1.29,
            ..CalibrationInputs::default()
        };
        let r = compute(&workbook, Deviations::NONE);
        assert!((r.model_torque_Nm - 1.6044531397852).abs() < 1e-12);
        assert!((r.f_cal_updated - 1.06578369763353).abs() < 1e-12);
        assert_eq!(r.poles_per_ring, 10.0);
        match r.fea_interp_Nm {
            NumOrText::Num(x) => assert!((x - 2.04670210123201).abs() < 1e-12),
            other => panic!("the default corner gap is inside the 3D span: {other:?}"),
        }
    }

    #[test]
    fn corner_definition_uses_the_spacing_as_the_corner_gap() {
        let c = CalibrationInputs {
            gap_definition: 0,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, Deviations::ALL);
        assert_eq!(r.corner_gap_mm, c.spacing_mm);
        assert!(
            r.flat_gap_mm > c.spacing_mm,
            "the flat gap exceeds the corner gap"
        );
    }

    #[test]
    fn interpolation_span_is_inclusive_at_both_ends() {
        for (spacing, inside) in [
            (0.999, false),
            (1.0, true),
            (1.25, true),
            (1.5, true),
            (1.501, false),
        ] {
            let c = CalibrationInputs {
                gap_definition: 0,
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            let r = compute(&c, Deviations::ALL);
            match (inside, r.fea_interp_Nm, r.fea_interp_error) {
                (true, NumOrText::Num(_), NumOrText::Num(_)) => {}
                (false, NumOrText::Text(OUTSIDE_RANGE), NumOrText::Text(NOT_APPLICABLE)) => {}
                other => panic!("spacing {spacing}: {other:?}"),
            }
        }
    }

    #[test]
    fn interpolation_hits_the_supplied_points_at_the_span_ends() {
        let at = |spacing| {
            let c = CalibrationInputs {
                gap_definition: 0,
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            compute(&c, Deviations::ALL).fea_interp_Nm
        };
        assert_eq!(at(1.0), NumOrText::Num(2.06));
        assert_eq!(at(1.5), NumOrText::Num(1.7));
    }

    #[test]
    fn fill_factor_is_capped_at_one() {
        let c = CalibrationInputs {
            magnet_width_mm: 25.4,
            total_magnets: 40,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, Deviations::ALL);
        assert_eq!((r.fill_inner, r.fill_outer), (1.0, 1.0));
    }
}
