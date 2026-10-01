//! Addendum A2 equation explorer, engine side: the explanation layer.
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron).

pub mod eval;
pub mod markup;
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
