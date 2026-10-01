//! Addendum A2 equation explorer, engine side: the explanation layer.
//!
//! Each explained result has an equation record ([`record::Record`]): its target path, a
//! display symbol, a formula in a small markup ([`markup`]) whose terms ARE input or result
//! paths, and the approved corrections it embodies. Unit, label and workbook cell come from
//! the result's metadata. The formula is both what the panel typesets and what the drift
//! guard evaluates ([`eval`]), so the equation shown is provably the one that produced the
//! number: `tests/explain.rs` evaluates every record over the engine's own term values at
//! the defaults and at every differential and augmented input set, corrections on, and
//! requires 1e-9 relative agreement (the parity rule).
//!
//! The engine code stays as ported. Where a formula needs a term the engine computed but did
//! not expose (the harmonics 7 to 11 parts, the torque-angle amplitudes, the E7 angles), the
//! engine exposes it as a Rust-only result: a pure read, no formula change.
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`notes`]: the A4 teaching notes, the "start here" order and the accuracy gate;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).

pub mod eval;
pub mod markup;
pub mod notes;
pub mod record;
pub mod records;
pub mod registry;
pub mod render;
pub mod scope;
pub mod symbols;
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
pub use registry::{Design, Equation, Registry, TermKind, TermRow, TermStyle};
