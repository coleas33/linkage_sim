//! Physical constants used by the workbook (kept identical for parity).
//!
//! Port of `reference/magcoupling-py/magcoupling/constants.py`.

/// Vacuum permeability [T·m/A], the workbook's rounded value (not 4π·1e-7).
pub const MU0: f64 = 1.256637e-06;

/// NdFeB density [g/mm³] (7.5 g/cm³), used for magnet mass.
pub const NDFEB_DENSITY_G_MM3: f64 = 0.0075;
