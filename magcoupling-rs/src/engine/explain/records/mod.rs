//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod clamps;
pub mod dashboard;
pub mod demagnetization;
pub mod geometry;
pub mod slip_heating;
pub mod temperature;
pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[
    torque::RECORDS,
    demagnetization::RECORDS,
    slip_heating::RECORDS,
    temperature::RECORDS,
    clamps::RECORDS,
    dashboard::RECORDS,
    geometry::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES, clamps::FAMILIES, dashboard::FAMILIES];
