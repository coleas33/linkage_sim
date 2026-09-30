//! Magnetic slip coupling design calculator.
//!
//! Rust port of the `magcoupling` 1.0.0 Python package, itself a port of
//! `magnetic_coupling_torque_calculator.xlsx`. The vendored Python package in
//! `reference/magcoupling-py/` is the oracle: every result must match the
//! workbook snapshot and the Python engine, except where an approved
//! correction from the M1 math audit is registered in
//! [`engine::deviations::REGISTRY`].
//!
//! Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`.
//!
//! ```
//! use magcoupling::{DesignInputs, compute_all, headline};
//! let res = compute_all(&DesignInputs::default());
//! // The production-variation allowance lowers the hot-side torque below the nominal pull-out.
//! assert!(res.metal.torque_hot_low_Nm < res.model.pullout_Nm);
//! assert_eq!(headline(&res)[0].0, "pullout_at_op_temp_Nm");
//! ```

pub mod engine;

pub use engine::api::{DesignInputs, DesignResults, compute_all, headline};
