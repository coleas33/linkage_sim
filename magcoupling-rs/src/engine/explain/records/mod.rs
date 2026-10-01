//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod demagnetization;
pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[torque::RECORDS, demagnetization::RECORDS];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
