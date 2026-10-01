//! The calculation engine: pure functions and data, no GUI dependencies.
//!
//! One Rust module per Python module of `reference/magcoupling-py/magcoupling/`
//! (the port grows module by module; see `magcoupling-rs/README.md`), plus the
//! infrastructure every module uses:
//!
//! - [`meta`]: field metadata for inputs and results (`param`/`out`, the
//!   `inputs!`/`results!` macros, dynamic [`meta::Value`] access by path);
//! - [`compat`]: the Python and Excel semantics the port reproduces exactly;
//! - [`deviations`]: the registry of approved corrections to the workbook.

pub mod api;
pub mod assumptions;
pub mod calibration;
pub mod clamps;
pub mod compat;
pub mod constants;
pub mod deviations;
pub mod grades;
pub mod library;
pub mod material_library;
pub mod materials;
pub mod meta;
pub mod metal_design;
pub mod model;
pub mod sizing;
pub mod sweeps;
pub mod temperature;
pub mod warnings;
