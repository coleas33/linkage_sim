//! Magnetic slip coupling design calculator.
//!
//! Rust port of the `magcoupling` 1.0.0 Python package, itself a port of
//! `magnetic_coupling_torque_calculator.xlsx`. The vendored Python package in
//! `reference/magcoupling-py/` is the oracle: every result must match the
//! workbook snapshot and the Python engine, except where an approved
//! correction (the M1 math audit's E1 to E14, the Addendum A verification's E15
//! to E20) is registered in [`engine::deviations::REGISTRY`]. The Addendum A
//! inputs the Python engine does not have (part materials, grades, the
//! coercivity source, the E17 free-space fields, the harmonic set, the axial
//! length override) are Rust-only and default to the ported behaviour. Beyond
//! the forward calculation: [`engine::sizing::solve`] (inverse sizing, Addendum
//! A1) and [`engine::assumptions`] (the A3 assumptions panel).
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
