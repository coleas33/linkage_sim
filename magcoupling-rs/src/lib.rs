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
//! use magcoupling::{DesignInputs, compute_all};
//! let res = compute_all(&DesignInputs::default());
//! assert!((res.calibration.f_cal_updated - 1.0658).abs() < 1e-4);
//! ```

pub mod engine;

pub use engine::api::{DesignInputs, DesignResults, compute_all};
