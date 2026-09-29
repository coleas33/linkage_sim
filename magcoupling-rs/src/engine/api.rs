//! Public entry point: one inputs object in, one results object out.
//!
//! Port of `reference/magcoupling-py/magcoupling/api.py`, growing module by
//! module. Calculation order mirrors the workbook's dependencies, as the Python
//! `compute_all` does:
//!
//! Calibration → Calculator (model) → Metal design retainers → Calculator mass
//! → Metal design → Materials → Temperature design → Shaft clamps → sweeps.
//!
//! Ported so far: Calibration.
//!
//! Python API mapping: `compute_all(inp)` is [`compute_all`];
//! `input_schema(inp)` and `result_schema(res)` are
//! [`crate::engine::meta::input_rows`] and [`crate::engine::meta::result_rows`];
//! `set_input(inp, path, value)` (returns a modified copy) is
//! [`crate::engine::meta::InputSet::set`] (modifies in place; clone first to keep
//! the original).

use super::calibration::{self, CalibrationInputs, CalibrationResults};
use super::deviations::Deviations;
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::meta::{inputs, results};

inputs! {
    /// Every editable input, grouped as the Python `DesignInputs`.
    pub struct DesignInputs {
        fields {}
        groups {
            calibration: CalibrationInputs,
        }
    }
}

results! {
    /// Every computed value, grouped as the Python `DesignResults`.
    pub struct DesignResults {
        fields {}
        groups {
            calibration: CalibrationResults,
        }
    }
}

/// Computes every result from the inputs, with all approved corrections.
///
/// Pure: no I/O, no global state; cheap enough to call on every GUI frame.
pub fn compute_all(inputs: &DesignInputs) -> DesignResults {
    compute(inputs, Deviations::ALL)
}

/// TEST-ONLY. [`compute_all`] with a chosen set of corrections;
/// `Deviations::NONE` reproduces the workbook and the Python engine exactly.
#[cfg(feature = "workbook-parity")]
pub fn compute_all_with(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    compute(inputs, dev)
}

fn compute(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    let calibration = calibration::compute(&inputs.calibration, dev);
    DesignResults { calibration }
}

#[cfg(feature = "workbook-parity")]
impl DesignInputs {
    /// TEST-ONLY. The default inputs as they are with the corrections in `dev`:
    /// `DesignInputs::defaults_with(Deviations::NONE)` gives the workbook's
    /// defaults, even where an applied deviation corrects a default.
    pub fn defaults_with(dev: Deviations) -> Self {
        let mut inputs = Self::default();
        restore_workbook_defaults(&mut inputs, dev, REGISTRY);
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::{InputSet, Value, input_rows, result_rows};

    #[test]
    fn paths_match_the_python_api() {
        let inputs = DesignInputs::default();
        assert_eq!(inputs.get("calibration.br_T"), Some(Value::Num(1.29)));
        let rows = result_rows(&compute_all(&inputs));
        assert!(rows.iter().any(
            |r| r.path == "calibration.f_cal_updated" && r.meta.cell == Some("Calibration!C9")
        ));
        assert!(
            input_rows(&inputs)
                .iter()
                .all(|r| r.path.starts_with("calibration."))
        );
    }

    #[test]
    fn changing_an_input_changes_results() {
        let base = compute_all(&DesignInputs::default());
        let mut hotter = DesignInputs::default();
        hotter
            .set("calibration.test_temp_C", Value::Num(80.0))
            .unwrap();
        let hot = compute_all(&hotter);
        assert!(hot.calibration.model_torque_Nm < base.calibration.model_torque_Nm);
    }

    #[test]
    fn workbook_defaults_equal_the_defaults_while_no_default_is_corrected() {
        assert_eq!(
            DesignInputs::defaults_with(Deviations::NONE),
            DesignInputs::default()
        );
        assert_eq!(
            DesignInputs::defaults_with(Deviations::ALL),
            DesignInputs::default()
        );
    }
}
