//! The authoring form of an equation record: what a records file declares, as `const` data.
//!
//! A record says what one result IS, one step from its terms: `σ = Σ_{n∈H} σ_n`, never the
//! chain behind the terms. Each term is an input or result path, so its value comes from the
//! engine, the hover key is the path, and a click drills into that term's own record. The
//! label, unit and workbook cell are not repeated here: the registry reads them from the
//! result's metadata by path.

use std::fmt;

use super::eval::{EvalError, TermSource};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::Value;

/// A Rust evaluation over a record's terms (it sees only the formula's terms).
pub type CustomEval = fn(&dyn TermSource) -> Result<Value, EvalError>;

/// How the drift guard computes a record's value.
#[derive(Clone, Copy)]
pub enum Eval {
    /// The formula markup itself: the guard proves the displayed formula. The default.
    Markup,
    /// Escape hatch for what the markup cannot state (for example a verdict that formats a
    /// number into its text). The display markup is still parsed for its terms, and the
    /// function sees only those; the registry counts these records and the physics review
    /// checks each display against its function by hand.
    Custom(CustomEval),
}

impl fmt::Debug for Eval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Eval::Markup => f.write_str("Markup"),
            Eval::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// One explained result.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    /// The result path it explains (in a [`Family`], `#` for the index).
    pub target: &'static str,
    /// Its display symbol, in symbol markup (in a family, `#` for the index, braced: `σ_{#}`).
    pub symbol: &'static str,
    /// Its formula, in formula markup (see [`super::markup`]).
    pub formula: &'static str,
    pub eval: Eval,
    /// The approved corrections this formula embodies (E-ids): the GUI's "corrected vs
    /// workbook" marker names them, and the traceability of a correction starts here.
    pub corrections: &'static [DeviationId],
}

/// A record whose value is its formula (the usual case).
pub const fn record(target: &'static str, symbol: &'static str, formula: &'static str) -> Record {
    Record {
        target,
        symbol,
        formula,
        eval: Eval::Markup,
        corrections: &[],
    }
}

impl Record {
    /// Names the approved corrections the formula embodies.
    pub const fn corrected(self, corrections: &'static [DeviationId]) -> Self {
        Self {
            corrections,
            ..self
        }
    }

    /// Replaces the markup evaluation by a Rust function (see [`Eval::Custom`]).
    pub const fn custom(self, eval: CustomEval) -> Self {
        Self {
            eval: Eval::Custom(eval),
            ..self
        }
    }
}

/// One record per index (the harmonics): `target`, `symbol` and the formula's term paths
/// carry `#`, and the formula's `n` is the index.
#[derive(Clone, Copy, Debug)]
pub struct Family {
    pub record: Record,
    pub indices: &'static [u32],
}

/// A family of records, one per index.
pub const fn family(record: Record, indices: &'static [u32]) -> Family {
    Family { record, indices }
}
