//! Public entry point: one inputs object in, one results object out.
//!
//! Port of `reference/magcoupling-py/magcoupling/api.py`, growing module by
//! module. Calculation order mirrors the workbook's dependencies, as the Python
//! `compute_all` does:
//!
//! Calibration → Calculator (model) → Metal design retainers → Calculator mass
//! → Metal design → Materials → Temperature design → Shaft clamps → sweeps.
//!
//! Ported so far: Calibration, Calculator (model), Metal design retainers; the
//! rest of Metal design and Materials contribute their inputs only.
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
use super::materials::MaterialsInputs;
use super::meta::{inputs, results};
use super::metal_design::{self, MetalDesignInputs, RetainerResults};
use super::model::{self, CouplingInputs, ModelResults};

inputs! {
    /// Every editable input, grouped as the Python `DesignInputs` (same order).
    pub struct DesignInputs {
        fields {}
        groups {
            coupling: CouplingInputs,
            metal: MetalDesignInputs,
            calibration: CalibrationInputs,
            materials: MaterialsInputs,
        }
    }
}

results! {
    /// Every computed value, grouped as the Python `DesignResults` (same order).
    pub struct DesignResults {
        fields {}
        groups {
            calibration: CalibrationResults,
            model: ModelResults,
            retainers: RetainerResults,
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

// Python api.compute_all lines 52-99; Python local names.
fn compute(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    let (ci, md, cal_in, mat_in) = (
        &inputs.coupling,
        &inputs.metal,
        &inputs.calibration,
        &inputs.materials,
    );
    let cal = calibration::compute(cal_in, dev);
    let f_cal = model::select_calibration_factor(
        ci.backiron,
        ci.npole,
        &ci.magnets.part_inner,
        &ci.magnets.part_outer,
        cal.poles_per_ring,
        cal.f_cal_updated,
        cal_in.f_cal_original,
    );
    let m = model::compute(
        ci,
        md.face_gap_mm,
        md.bond_inner_mm,
        md.bond_outer_mm,
        md.cup_wall_corner_mm,
        cal_in.alpha_br_per_C,
        mat_in.steel.bsat_T,
        f_cal,
        cal_in.f_cal_original,
        md.slip_rpm,
        md.required_min_Nm,
        dev,
    );
    let ret = metal_design::retainers(
        md,
        ci.inner_back_apothem_mm,
        m.inner_thickness_mm,
        m.inner_width_mm,
        m.outer_face_apothem_mm,
        ci.bore_mm,
        dev,
    );
    DesignResults {
        calibration: cal,
        model: m,
        retainers: ret,
    }
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
        assert_eq!(input_rows(&inputs)[0].path, "coupling.npole");
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
