//! Inputs no parity or differential case holds (the plan's Review Focus): a
//! selector code outside its choices set on the struct, a measured drag of
//! exactly zero, extreme typed values. The engine must never panic.

use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::NumOrText;

#[test]
fn an_invalid_adhesive_code_selects_no_adhesive() {
    for code in [0, 5, -1, i64::MIN, i64::MAX] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.temperature.adhesive.selected = code; // bypasses set(), which refuses it
        let t = compute_all_with(&inputs, Deviations::NONE).temperature;
        // Python's candidates[selected - 1] would silently take DP460 for code 0.
        assert_eq!(t.adhesive.selected_name, "#N/A", "{code}");
        assert!(
            t.adhesive.lap_shear_MPa.is_nan() && t.summary.adhesive_limit_C.is_nan(),
            "{code}"
        );
    }
}

#[test]
fn zero_measured_drag_does_not_panic() {
    // Python raises ZeroDivisionError (rotations per degree, C156/C157); the limit is +inf.
    for dev in [Deviations::NONE, Deviations::ALL] {
        let mut inputs = DesignInputs::defaults_with(dev);
        inputs.metal.measured_drag_Nm = Some(0.0);
        let t = compute_all_with(&inputs, dev).temperature;
        assert_eq!(t.thermal.rev_per_C_est, f64::INFINITY);
        assert_eq!(t.thermal.rev_per_C_high, f64::INFINITY);
        assert!(t.thermal.heat_capacity_J_K.is_finite() && t.summary.governing_limit_C.is_finite());
        assert_eq!(
            t.thermal.time_to_limit_high,
            NumOrText::Text("never: steady state stays below the limit")
        );
    }
}

#[test]
fn an_invalid_screw_class_is_nan_not_a_panic() {
    for code in [0, 4, i64::MIN] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.clamps.screw_class = code; // bypasses set()
        let c = compute_all_with(&inputs, Deviations::NONE).clamps;
        assert!(c.screw_proof_MPa.is_nan(), "{code}");
        assert!(c.table.iter().all(|r| r.preload_N.is_nan()), "{code}");
        assert!(
            c.recommended.ends_with("class #N/A") || c.recommended.starts_with("None"),
            "{code}: {}",
            c.recommended
        );
    }
}
