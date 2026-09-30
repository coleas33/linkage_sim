//! Public entry point: one inputs object in, one results object out.
//!
//! Port of `reference/magcoupling-py/magcoupling/api.py`, growing module by
//! module. Calculation order mirrors the workbook's dependencies, as the Python
//! `compute_all` does:
//!
//! Calibration → Calculator (model) → Metal design retainers → Calculator mass
//! → Metal design → Materials → Temperature design → Shaft clamps → sweeps.
//!
//! Ported: every module except `fields3d` (M3): every sheet of the Python
//! `compute_all` (Calibration, Calculator (model), Metal design retainers,
//! Calculator mass, Metal design, Materials, Temperature design, Shaft clamps,
//! Gap sweep, Pole sweep), `headline` and the input check.
//!
//! Python API mapping: `compute_all(inp)` is [`compute_all`];
//! `input_schema(inp)` and `result_schema(res)` are
//! [`crate::engine::meta::input_rows`] and [`crate::engine::meta::result_rows`];
//! `set_input(inp, path, value)` (returns a modified copy) is
//! [`crate::engine::meta::InputSet::set`] (modifies in place; clone first to keep
//! the original); `headline(res)` is [`headline`] (a list of pairs in Python's
//! order, not a dict). Rust-only: [`DesignInputs::validate`], for inputs that
//! bypass `set`. `to_dict` is not ported: JSON export is M4.

use super::calibration::{self, CalibrationInputs, CalibrationResults};
use super::clamps::{self, ClampInputs, ClampResults};
use super::deviations::Deviations;
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::materials::{self, MaterialsInputs, MaterialsResults};
use super::meta::{SetError, TableLayout, Value, inputs, result_rows, results, validate};
use super::metal_design::{self, MetalDesignInputs, MetalDesignResults, RetainerResults};
use super::model::{self, CouplingInputs, MassResults, ModelResults};
use super::sweeps::{self, SweepRow};
use super::temperature::{self, TemperatureInputs, TemperatureResults};

inputs! {
    /// Every editable input, grouped as the Python `DesignInputs` (same order).
    pub struct DesignInputs {
        fields {}
        groups {
            coupling: CouplingInputs,
            metal: MetalDesignInputs,
            calibration: CalibrationInputs,
            materials: MaterialsInputs,
            temperature: TemperatureInputs,
            clamps: ClampInputs,
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
            mass: MassResults,
            retainers: RetainerResults,
            metal: MetalDesignResults,
            materials: MaterialsResults,
            temperature: TemperatureResults,
            clamps: ClampResults,
        }
        tables {
            gap_sweep: SweepRow => TableLayout::RowsDown { sheet: "Gap sweep", first_row: 6 },
            pole_sweep: SweepRow => TableLayout::RowsDown { sheet: "Pole sweep", first_row: 6 },
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

/// The numbers a dashboard shows first: (Python key, result path), in the order
/// of Python's `headline()` (`api.py` lines 131-150).
pub const HEADLINE: [(&str, &str); 15] = [
    ("pullout_at_op_temp_Nm", "model.pullout_Nm"),
    ("pullout_at_20C_Nm", "model.pullout_20C_Nm"),
    ("hot_low_with_variation_Nm", "metal.torque_hot_low_Nm"),
    ("hot_min_check", "metal.hot_min_check"),
    ("cold_high_with_variation_Nm", "metal.torque_cold_high_Nm"),
    ("gearbox_input_ripple_Nm", "model.gearbox_input_ripple_Nm"),
    ("cup_od_mm", "model.cup_od_mm"),
    ("rotating_mass_g", "mass.total_g"),
    ("running_clearance_mm", "metal.min_running_clearance_mm"),
    ("clearance_check", "metal.clearance_check"),
    ("cup_wall_check", "materials.cup_wall_check"),
    (
        "governing_temp_limit_C",
        "temperature.summary.governing_limit_C",
    ),
    ("hot_day_margin_C", "temperature.summary.margin_hot_day_C"),
    ("temperature_verdict", "temperature.summary.verdict"),
    ("clamp_screw", "clamps.recommended"),
];

/// Python `headline(res)`: the dashboard numbers, keyed and ordered as in Python.
pub fn headline(results: &DesignResults) -> Vec<(&'static str, Value)> {
    let rows = result_rows(results);
    HEADLINE
        .iter()
        .map(|&(key, path)| {
            (
                key,
                rows.iter()
                    .find(|r| r.path == path)
                    .map_or(Value::None, |r| r.value.clone()),
            )
        })
        .collect()
}

impl DesignInputs {
    /// Every selector code among its choices and every number finite (see
    /// [`crate::engine::meta::validate`]). Decision D3: `compute_all` never
    /// panics on invalid inputs; call this where inputs enter (design file, share link).
    pub fn validate(&self) -> Result<(), Vec<SetError>> {
        validate(self)
    }
}

// Python api.compute_all lines 52-99; Python local names.
#[allow(non_snake_case)] // SweepContext has a field `L`
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
        m.inner_corner_radius_mm,
        dev,
    );
    let mass = model::mass_estimate(
        ci,
        &m,
        md.bond_inner_mm,
        md.bond_outer_mm,
        md.cup_depth_mm,
        md.web_mm,
        md.hub_length_mm,
        md.boss_length_mm,
        md.boss_od_mm,
        md.steel_density_g_mm3,
        md.al_density_g_mm3,
        ret.retainers_g,
        md.hardware_g,
        ret.cap_g,
        ret.endplates_g,
        dev,
    );
    let mdr = metal_design::compute(
        md,
        m.pullout_Nm,
        m.pullout_20C_Nm,
        ci.op_temp_C,
        cal_in.alpha_br_per_C,
        m.corner_gap_mm,
        m.face_gap_mm,
        m.cup_od_mm,
        ci.npole,
        ci.bore_mm,
        ci.gear_ratio,
        ci.gear_efficiency,
        mass.total_g,
        mass.boss_g,
        &ret,
        cal_in.measured_torque_Nm,
        cal_in.test_temp_C,
        dev,
    );
    let matr = materials::compute(mat_in, m.backiron_needed_mm, md.cup_wall_corner_mm, dev);
    let links = temperature::TemperatureLinks {
        op_temp_C: ci.op_temp_C,
        npole: ci.npole,
        br20_T: m.inner_br_T,
        alpha_br: cal_in.alpha_br_per_C,
        tmax_lib_C: m.inner_tmax_C,
        mu0: ci.mu0,
        pullout_op_Nm: m.pullout_Nm,
        pullout_20C_Nm: m.pullout_20C_Nm,
        inner_back_apothem_mm: ci.inner_back_apothem_mm,
        inner_length_mm: m.inner_length_mm,
        inner_width_mm: m.inner_width_mm,
        inner_thickness_mm: m.inner_thickness_mm,
        hub_wall_mm: m.hub_wall_mm,
        active_length_mm: m.active_length_mm,
        outer_back_apothem_mm: m.outer_back_apothem_mm,
        mass_magnets_g: mass.magnets_g,
        mass_cup_g: mass.cup_g,
        mass_hub_g: mass.hub_g,
        mass_boss_g: mass.boss_g,
        slip_rpm: md.slip_rpm,
        slip_event_s: md.slip_event_s,
        life_events: md.life_events,
        measured_drag_Nm: md.measured_drag_Nm,
        cold_high_Nm: mdr.torque_cold_high_Nm,
        required_min_Nm: md.required_min_Nm,
        variation: md.variation,
        min_temp_C: md.min_temp_C,
        magnetic_cycles: mdr.magnetic_cycles,
        bond_inner_mm: md.bond_inner_mm,
        bond_outer_mm: md.bond_outer_mm,
        sleeve_mm: md.sleeve_mm,
        liner_mm: md.liner_mm,
        sleeve_id_mm: ret.sleeve_id_mm,
        sleeve_od_mm: ret.sleeve_od_mm,
        liner_od_mm: ret.liner_od_mm,
        liner_id_mm: ret.liner_id_mm,
        cap_face_mm: md.cap_axial_mm,
        hardware_g: md.hardware_g,
        retainers_g: ret.retainers_g,
        cap_g: ret.cap_g,
        endplates_g: ret.endplates_g,
        steel_sigma_S_m: mat_in.steel.conductivity_S_m,
        steel_mu_r: mat_in.steel.mu_r_incremental,
        steel_c: mat_in.steel.specific_heat_J_kgK,
        steel_cte: mat_in.steel.cte_per_C,
        steel_E_GPa: mat_in.steel.modulus_GPa,
        al6061_sigma_S_m: materials::AL6061.conductivity_S_m,
    };
    let temp = temperature::compute(&inputs.temperature, &links, dev);

    let alloy = if inputs.clamps.alloy == 1 {
        &materials::AL7075
    } else {
        &materials::AL6061
    };
    let clr = clamps::compute(
        &inputs.clamps,
        ci.bore_mm,
        mdr.torque_cold_high_Nm,
        alloy,
        mat_in.screws.proof(inputs.clamps.screw_class),
        dev,
    );
    let ctx = sweeps::SweepContext {
        faceted: ci.faceted,
        backiron: ci.backiron,
        t_i: m.inner_thickness_mm,
        w_i: m.inner_width_mm,
        t_o: m.outer_thickness_mm,
        w_o: m.outer_width_mm,
        L: m.active_length_mm,
        br_i20: m.inner_br_T,
        br_o20: m.outer_br_T,
        br_i_op: m.br_inner_T_op,
        br_o_op: m.br_outer_T_op,
        bond_outer: md.bond_outer_mm,
        bond_inner: md.bond_inner_mm,
        cup_wall_corner: md.cup_wall_corner_mm,
        c_end: ci.c_end,
        mu0: ci.mu0,
        gear_ratio: ci.gear_ratio,
        gear_eff: ci.gear_efficiency,
        required_floor_Nm: m.required_floor_Nm,
        max_diameter_mm: md.max_diameter_mm,
    };
    let gap = sweeps::gap_sweep(&ctx, ci.npole, ci.inner_back_apothem_mm, f_cal, dev);
    let pole = sweeps::pole_sweep(
        &ctx,
        m.corner_gap_mm,
        ci.bore_mm,
        ci.keyway_depth_mm,
        cal_in.f_cal_original,
        dev,
    );
    DesignResults {
        calibration: cal,
        model: m,
        mass,
        retainers: ret,
        metal: mdr,
        materials: matr,
        temperature: temp,
        clamps: clr,
        gap_sweep: gap,
        pole_sweep: pole,
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
        let workbook = DesignInputs::defaults_with(Deviations::NONE);
        assert_eq!(workbook.get("calibration.br_T"), Some(Value::Num(1.29)));
        let inputs = DesignInputs::default();
        assert_eq!(
            inputs.get("calibration.br_T"),
            Some(Value::Num(1.30)),
            "E3 corrects the default"
        );
        let rows = result_rows(&compute_all(&inputs));
        assert!(rows.iter().any(
            |r| r.path == "calibration.f_cal_updated" && r.meta.cell == Some("Calibration!C9")
        ));
        assert_eq!(input_rows(&inputs)[0].path, "coupling.npole");
    }

    #[test]
    fn headline_names_existing_results() {
        assert!(
            headline(&compute_all(&DesignInputs::default()))
                .iter()
                .all(|(_, v)| *v != Value::None)
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
    fn workbook_defaults_put_back_every_corrected_default() {
        assert_eq!(
            DesignInputs::defaults_with(Deviations::ALL),
            DesignInputs::default()
        );
        let workbook = DesignInputs::defaults_with(Deviations::NONE);
        let corrected = DesignInputs::default();
        // E1: the adhesive shear modulus.
        assert_eq!(
            workbook.temperature.mismatch.adhesive_shear_modulus_GPa,
            0.55
        );
        assert_eq!(
            corrected.temperature.mismatch.adhesive_shear_modulus_GPa,
            0.107
        );
        // E3: the manual and calibration remanence follow the N42SH library rows.
        let br = |inputs: &DesignInputs| {
            [
                inputs.coupling.magnets.manual_inner_br_T,
                inputs.coupling.magnets.manual_outer_br_T,
                inputs.calibration.br_T,
            ]
        };
        assert_eq!(br(&workbook), [1.29; 3]);
        assert_eq!(br(&corrected), [1.30; 3]);
        // E5: the rear-web field integral is the field at the steel surface.
        assert_eq!(workbook.temperature.slip_loss.web_integral_T2m2, 1.035e-5);
        assert_eq!(corrected.temperature.slip_loss.web_integral_T2m2, 4.14e-5);
    }
}
