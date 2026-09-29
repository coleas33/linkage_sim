//! Field metadata for inputs and results.
//!
//! Every input and result carries the metadata the Python package attaches with
//! `param()` and `out()` (`reference/magcoupling-py/magcoupling/_fields.py`):
//! unit, label, help, workbook cell, and choices for selectors. Inputs also
//! carry what the Python has no place for: a slider range (min, max, step,
//! logarithmic flag) defined by hand from physical bounds, and an `assumption`
//! flag marking model assumptions (spec Addendum A3) apart from design inputs.
//!
//! # Design: the metadata sits on the field's own line
//!
//! A module declares its input and result structs with [`inputs!`] and
//! [`results!`]. One declaration per field gives the struct field, its default,
//! its metadata row and its by-path accessors:
//!
//! ```ignore
//! inputs! {
//!     pub struct CalibrationInputs {
//!         fields {
//!             measured_torque_Nm: f64 = 1.8 => param("N·m", "Measured pull-out torque",
//!                 "User bench result. ...", "Calibration!C5").range(0.1, 10.0, 0.01),
//!         }
//!     }
//! }
//! ```
//!
//! Why this shape:
//!
//! - **No drift.** The field, default, metadata and accessors come from one
//!   declaration, so they cannot disagree.
//! - **Mechanical transcription.** The builders take Python's positional order:
//!   `name: float = param(default, unit, label, help, cell)` becomes
//!   `name: f64 = default => param(unit, label, help, cell)`, and
//!   `out(unit, label, help, cell)` stays `out(unit, label, help, cell)`. A
//!   reviewer can read a port against the Python line by line.
//! - **Python names verbatim.** Field names keep the Python spelling
//!   (`measured_torque_Nm`), so dotted paths (`calibration.measured_torque_Nm`)
//!   equal the Python `input_schema()` paths used by design files, the
//!   differential data and the GUI. The macros allow `non_snake_case` for this.
//! - **Plain `const` data.** Metadata tables are `const`: no start-up
//!   registration, no global state, nothing extra on wasm32.
//!
//! Rejected: a derive proc-macro (a second crate and a build dependency for what
//! `macro_rules!` does here), separate const tables (metadata drifts away from
//! the struct), serde attributes (no place for units, cells or ranges).
//!
//! Selector inputs stay `i64` workbook codes (`backiron`: 1 steel, 0 none), not
//! Rust enums, because the engine mirrors the Python branching on those codes;
//! [`InputSet::set`] rejects codes outside `choices`.
//!
//! # Tables
//!
//! The screw table and the two sweeps are lists of rows whose workbook cells
//! follow a layout, not one declared cell per value. A row struct is declared
//! with [`rows!`] (each field carries [`col`], [`at_row`] or [`uncelled_col`]
//! metadata naming only its column letter or its sheet row); a [`results!`] struct
//! lists its tables in a `tables { name: Row => layout }` section, where the
//! [`TableLayout`] says whether the rows run down the sheet or across it. The
//! field is a `Vec<Row>` and its paths are `name[i].field`, as in the Python
//! schema. Every value gets a synthesized workbook cell (`Gap sweep!N8`), which
//! the visit callback hands out like a scalar's `meta.cell`.

use std::fmt;

/// A field value in dynamic form: what [`InputSet::get`] returns, what
/// [`InputSet::set`] accepts, and what schema walks yield.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A float (Python `float`).
    Num(f64),
    /// An integer (Python `int`): counts and selector codes.
    Int(i64),
    /// Text (Python `str`): part names, verdicts, sentinels such as "n/a".
    Text(String),
    /// No value (Python `None`), e.g. a drag torque that was not measured.
    None,
}

/// The Rust type behind a field, as recorded in its metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    /// `f64`
    F64,
    /// `i64`: counts and selector codes
    I64,
    /// `Option<f64>`: a number or "not entered"
    OptF64,
    /// `String`
    Text,
    /// [`NumOrText`]: a number, or one of the workbook's text sentinels
    NumOrText,
}

impl FieldType {
    /// Stable name used in exported schemas (`tests/data/input_schema.json`).
    pub const fn name(self) -> &'static str {
        match self {
            FieldType::F64 => "f64",
            FieldType::I64 => "i64",
            FieldType::OptF64 => "opt_f64",
            FieldType::Text => "text",
            FieldType::NumOrText => "num_or_text",
        }
    }
}

/// A result that is a number, or a fixed text sentinel where the workbook
/// shows text instead (Python fields typed `object`, e.g. "outside range").
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumOrText {
    /// The numeric case.
    Num(f64),
    /// The text case; always one of the workbook's fixed strings.
    Text(&'static str),
}

/// Slider range of an input, defined by hand from physical bounds.
///
/// `step` is the snapping increment (and the arrow-key nudge); `log` asks for a
/// logarithmic slider. The range bounds the slider and the differential test
/// inputs; it does not reject typed values (the Python engine has no ranges).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderRange {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub log: bool,
}

/// Metadata of one input field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputMeta {
    /// Field name (the Python name), the last segment of its path.
    pub name: &'static str,
    pub ty: FieldType,
    pub unit: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    /// Workbook cell, e.g. `"Calibration!C5"`.
    pub cell: Option<&'static str>,
    /// Selector codes and their texts; empty for non-selectors.
    pub choices: &'static [(i64, &'static str)],
    pub range: Option<SliderRange>,
    /// A model assumption (Addendum A3) rather than a design input.
    pub assumption: bool,
}

/// Declares an input's metadata, in the order of Python's
/// `param(default, unit, label, help, cell)` (the default goes on the field).
pub const fn param(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
    cell: &'static str,
) -> InputMeta {
    InputMeta {
        name: "",
        ty: FieldType::F64,
        unit,
        label,
        help,
        cell: Some(cell),
        choices: &[],
        range: None,
        assumption: false,
    }
}

impl InputMeta {
    /// Selector codes and texts, like Python's `choices={code: text}`.
    pub const fn choices(self, choices: &'static [(i64, &'static str)]) -> Self {
        Self { choices, ..self }
    }

    /// Linear slider range.
    pub const fn range(self, min: f64, max: f64, step: f64) -> Self {
        Self {
            range: Some(SliderRange {
                min,
                max,
                step,
                log: false,
            }),
            ..self
        }
    }

    /// Makes the slider logarithmic. Call after [`InputMeta::range`].
    pub const fn log(self) -> Self {
        match self.range {
            Some(r) => Self {
                range: Some(SliderRange { log: true, ..r }),
                ..self
            },
            None => panic!("log() needs range() first"),
        }
    }

    /// Marks the input as a model assumption (Addendum A3).
    pub const fn assumption(self) -> Self {
        Self {
            assumption: true,
            ..self
        }
    }

    /// Completes the metadata with the field's name and type (used by `inputs!`).
    #[doc(hidden)]
    pub const fn bind(self, name: &'static str, ty: FieldType) -> Self {
        Self { name, ty, ..self }
    }
}

/// Metadata of one result field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultMeta {
    /// Field name (the Python name), the last segment of its path.
    pub name: &'static str,
    pub ty: FieldType,
    pub unit: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    /// Workbook cell, e.g. `"Calibration!C6"`; `None` for the rare result the
    /// workbook does not show.
    pub cell: Option<&'static str>,
}

/// Declares a result's metadata, in the order of Python's
/// `out(unit, label, help, cell)`.
pub const fn out(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
    cell: &'static str,
) -> ResultMeta {
    ResultMeta {
        name: "",
        ty: FieldType::F64,
        unit,
        label,
        help,
        cell: Some(cell),
    }
}

/// Declares a result without a workbook cell (Python `out(...)` with no cell).
pub const fn out_uncelled(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
) -> ResultMeta {
    ResultMeta {
        name: "",
        ty: FieldType::F64,
        unit,
        label,
        help,
        cell: None,
    }
}

impl ResultMeta {
    /// Completes the metadata with the field's name and type (used by `results!`).
    #[doc(hidden)]
    pub const fn bind(self, name: &'static str, ty: FieldType) -> Self {
        Self { name, ty, ..self }
    }
}

/// The callback of [`ResultSet::visit`] and [`RowSet::visit_row`]:
/// `f(path, meta, cell, value)`, where `cell` is the workbook cell (`meta.cell`
/// for a scalar, synthesized for a table row).
pub type ResultVisitor<'a> = dyn FnMut(&str, &'static ResultMeta, Option<&str>, Value) + 'a;

/// Where a table field sits on its sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAxis {
    /// A column letter: rows run down the sheet (the sweeps).
    Column(&'static str),
    /// A row number: rows run across the sheet (the screw table).
    Row(u32),
    /// Not a workbook cell (e.g. the screw size name, a header).
    None,
}

/// Metadata of one table field: the result metadata plus its place on the sheet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnMeta {
    pub meta: ResultMeta,
    pub axis: CellAxis,
}

const fn column_meta(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
    axis: CellAxis,
) -> ColumnMeta {
    ColumnMeta {
        meta: ResultMeta {
            name: "",
            ty: FieldType::F64,
            unit,
            label,
            help,
            cell: None,
        },
        axis,
    }
}

/// A table field that sits in a sheet column (the sweeps: `B` .. `AA`).
pub const fn col(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
    column: &'static str,
) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::Column(column))
}

/// A table field that sits in a sheet row (the screw table: rows 6 .. 38).
pub const fn at_row(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
    row: u32,
) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::Row(row))
}

/// A table field with no workbook cell.
pub const fn uncelled_col(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::None)
}

impl ColumnMeta {
    /// Completes the metadata with the field's name and type (used by `rows!`).
    #[doc(hidden)]
    pub const fn bind(self, name: &'static str, ty: FieldType) -> Self {
        Self {
            meta: self.meta.bind(name, ty),
            ..self
        }
    }
}

/// How the rows of a table map to workbook cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TableLayout {
    /// Row `i` of the table is sheet row `first_row + i`; fields name their column.
    RowsDown { sheet: &'static str, first_row: u32 },
    /// Row `i` of the table is sheet column `columns[i]`; fields name their row.
    ColumnsAcross {
        sheet: &'static str,
        columns: &'static [&'static str],
    },
}

impl TableLayout {
    /// The cell of field `axis` in table row `index`; `None` when the axis does
    /// not fit the layout or the row has no column.
    pub fn cell(&self, axis: CellAxis, index: usize) -> Option<String> {
        match (*self, axis) {
            (TableLayout::RowsDown { sheet, first_row }, CellAxis::Column(column)) => {
                Some(format!("{sheet}!{column}{}", first_row as usize + index))
            }
            (TableLayout::ColumnsAcross { sheet, columns }, CellAxis::Row(row)) => columns
                .get(index)
                .map(|column| format!("{sheet}!{column}{row}")),
            _ => None,
        }
    }
}

/// A table row declared with [`rows!`].
pub trait RowSet {
    /// Field metadata, in declaration order.
    const COLUMNS: &'static [ColumnMeta];

    /// Calls `f(path, meta, cell, value)` for every field of this row, row
    /// `index` of a table laid out by `layout`. `prefix` ends with `[index].`.
    fn visit_row(
        &self,
        prefix: &str,
        layout: &TableLayout,
        index: usize,
        f: &mut ResultVisitor<'_>,
    );
}

/// Why [`InputSet::set`] refused a value.
#[derive(Clone, Debug, PartialEq)]
pub enum SetErrorKind {
    /// No input has this path.
    UnknownPath,
    /// The value has the wrong type for the field.
    TypeMismatch { expected: FieldType, got: Value },
    /// NaN or an infinity.
    NotFinite,
    /// A selector code that is not one of the field's `choices`.
    NotAChoice { code: i64 },
}

/// An error from [`InputSet::set`], naming the full dotted path.
#[derive(Clone, Debug, PartialEq)]
pub struct SetError {
    pub path: String,
    pub kind: SetErrorKind,
}

impl SetError {
    /// An unknown-path error.
    pub fn unknown_path(path: &str) -> Self {
        Self {
            path: path.to_owned(),
            kind: SetErrorKind::UnknownPath,
        }
    }

    /// Prefixes the path with the enclosing group, so nested errors name the full path.
    pub fn within(self, group: &str) -> Self {
        Self {
            path: format!("{group}.{}", self.path),
            ..self
        }
    }
}

impl fmt::Display for SetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            SetErrorKind::UnknownPath => write!(f, "{}: no such input", self.path),
            SetErrorKind::TypeMismatch { expected, got } => {
                write!(
                    f,
                    "{}: expected {}, got {got:?}",
                    self.path,
                    expected.name()
                )
            }
            SetErrorKind::NotFinite => write!(f, "{}: value is not finite", self.path),
            SetErrorKind::NotAChoice { code } => {
                write!(f, "{}: {code} is not one of the choices", self.path)
            }
        }
    }
}

impl std::error::Error for SetError {}

/// A Rust type usable as a field: its [`FieldType`] and its dynamic [`Value`].
pub trait FieldValue {
    const TYPE: FieldType;
    fn to_value(&self) -> Value;
}

/// A [`FieldValue`] that inputs can take from a dynamic [`Value`].
pub trait InputValue: FieldValue + Sized {
    fn from_value(value: Value) -> Result<Self, SetErrorKind>;
}

impl FieldValue for f64 {
    const TYPE: FieldType = FieldType::F64;
    fn to_value(&self) -> Value {
        Value::Num(*self)
    }
}

impl InputValue for f64 {
    /// Accepts a float or an integer (Python float inputs often have int
    /// defaults, e.g. `test_temp_C = 20`); rejects NaN and infinities.
    fn from_value(value: Value) -> Result<Self, SetErrorKind> {
        match value {
            Value::Num(x) if x.is_finite() => Ok(x),
            Value::Num(_) => Err(SetErrorKind::NotFinite),
            Value::Int(i) => Ok(i as f64),
            got => Err(SetErrorKind::TypeMismatch {
                expected: FieldType::F64,
                got,
            }),
        }
    }
}

impl FieldValue for i64 {
    const TYPE: FieldType = FieldType::I64;
    fn to_value(&self) -> Value {
        Value::Int(*self)
    }
}

impl InputValue for i64 {
    /// Accepts integers only: a count or a selector code is never fractional.
    fn from_value(value: Value) -> Result<Self, SetErrorKind> {
        match value {
            Value::Int(i) => Ok(i),
            got => Err(SetErrorKind::TypeMismatch {
                expected: FieldType::I64,
                got,
            }),
        }
    }
}

impl FieldValue for Option<f64> {
    const TYPE: FieldType = FieldType::OptF64;
    fn to_value(&self) -> Value {
        match self {
            Some(x) => Value::Num(*x),
            None => Value::None,
        }
    }
}

impl InputValue for Option<f64> {
    fn from_value(value: Value) -> Result<Self, SetErrorKind> {
        match value {
            Value::None => Ok(None),
            number @ (Value::Num(_) | Value::Int(_)) => f64::from_value(number).map(Some),
            got => Err(SetErrorKind::TypeMismatch {
                expected: FieldType::OptF64,
                got,
            }),
        }
    }
}

impl FieldValue for String {
    const TYPE: FieldType = FieldType::Text;
    fn to_value(&self) -> Value {
        Value::Text(self.clone())
    }
}

impl InputValue for String {
    fn from_value(value: Value) -> Result<Self, SetErrorKind> {
        match value {
            Value::Text(s) => Ok(s),
            got => Err(SetErrorKind::TypeMismatch {
                expected: FieldType::Text,
                got,
            }),
        }
    }
}

impl FieldValue for NumOrText {
    const TYPE: FieldType = FieldType::NumOrText;
    fn to_value(&self) -> Value {
        match self {
            NumOrText::Num(x) => Value::Num(*x),
            NumOrText::Text(s) => Value::Text((*s).to_owned()),
        }
    }
}

/// A struct of inputs declared with [`inputs!`], possibly with nested groups.
pub trait InputSet {
    /// Leaf fields declared on this struct, in declaration order.
    const FIELDS: &'static [InputMeta];
    /// Names of the nested groups declared on this struct, in declaration order.
    const GROUPS: &'static [&'static str];

    /// The value at a dotted path relative to this struct, or `None` if no
    /// input has that path.
    fn get(&self, path: &str) -> Option<Value>;

    /// Sets the input at a dotted path relative to this struct. Rejects unknown
    /// paths, wrong types, non-finite numbers and selector codes outside the
    /// field's choices; leaves the struct unchanged on error.
    fn set(&mut self, path: &str, value: Value) -> Result<(), SetError>;

    /// Calls `f(path, meta, value)` for every leaf input, fields before groups,
    /// each in declaration order. `prefix` is prepended to every path.
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &'static InputMeta, Value));
}

/// A struct of results declared with [`results!`], possibly with nested groups.
pub trait ResultSet {
    /// Leaf fields declared on this struct, in declaration order.
    const FIELDS: &'static [ResultMeta];
    /// Names of the nested groups declared on this struct, in declaration order.
    const GROUPS: &'static [&'static str];

    /// Calls `f(path, meta, cell, value)` for every leaf result, fields before
    /// groups before tables, each in declaration order. `prefix` is prepended to
    /// every path; `cell` is the field's workbook cell: `meta.cell` for a scalar,
    /// synthesized for a table row.
    fn visit(&self, prefix: &str, f: &mut ResultVisitor<'_>);
}

/// One input with its full path, metadata and current value.
#[derive(Clone, Debug, PartialEq)]
pub struct InputRow {
    pub path: String,
    pub meta: &'static InputMeta,
    pub value: Value,
}

/// One result with its full path, metadata and value.
#[derive(Clone, Debug, PartialEq)]
pub struct ResultRow {
    pub path: String,
    pub meta: &'static ResultMeta,
    /// The workbook cell, synthesized for table rows.
    pub cell: Option<String>,
    pub value: Value,
}

/// Every leaf input with path, metadata and value (Python `input_schema()`).
pub fn input_rows<T: InputSet>(inputs: &T) -> Vec<InputRow> {
    let mut rows = Vec::new();
    inputs.visit("", &mut |path, meta, value| {
        rows.push(InputRow {
            path: path.to_owned(),
            meta,
            value,
        })
    });
    rows
}

/// Every leaf result with path, metadata and value (Python `result_schema()`).
pub fn result_rows<T: ResultSet>(results: &T) -> Vec<ResultRow> {
    let mut rows = Vec::new();
    results.visit("", &mut |path, meta, cell, value| {
        rows.push(ResultRow {
            path: path.to_owned(),
            meta,
            cell: cell.map(str::to_owned),
            value,
        })
    });
    rows
}

/// Converts and validates a value for the input described by `meta` (used by `inputs!`).
#[doc(hidden)]
pub fn convert_input<T: InputValue>(meta: &InputMeta, value: Value) -> Result<T, SetError> {
    let error = |kind| SetError {
        path: meta.name.to_owned(),
        kind,
    };
    if let Value::Int(code) = value
        && !meta.choices.is_empty()
        && !meta.choices.iter().any(|&(c, _)| c == code)
    {
        return Err(error(SetErrorKind::NotAChoice { code }));
    }
    T::from_value(value).map_err(error)
}

/// The metadata row of a named leaf field (used by `inputs!`; the name always
/// exists because the macro generated both).
#[doc(hidden)]
pub fn field_meta(fields: &'static [InputMeta], name: &str) -> &'static InputMeta {
    fields
        .iter()
        .find(|m| m.name == name)
        .expect("inputs! generates a metadata row for every field it matches")
}

/// Declares an input struct: fields with defaults and metadata, then optional
/// nested groups. See the module docs for the field syntax.
macro_rules! inputs {
    (
        $(#[$sattr:meta])*
        pub struct $Name:ident {
            fields {
                $(
                    $(#[$fattr:meta])*
                    $field:ident : $ty:ty = $default:expr => $meta:expr
                ),* $(,)?
            }
            $(
                groups {
                    $( $(#[$gattr:meta])* $group:ident : $Group:ty ),* $(,)?
                }
            )?
        }
    ) => {
        $(#[$sattr])*
        #[derive(Clone, Debug, PartialEq)]
        #[allow(non_snake_case)]
        pub struct $Name {
            $( $(#[$fattr])* pub $field: $ty, )*
            $( $( $(#[$gattr])* pub $group: $Group, )* )?
        }

        impl ::core::default::Default for $Name {
            #[allow(clippy::useless_conversion)]
            fn default() -> Self {
                Self {
                    $( $field: <$ty as ::core::convert::From<_>>::from($default), )*
                    $( $( $group: <$Group as ::core::default::Default>::default(), )* )?
                }
            }
        }

        impl $crate::engine::meta::InputSet for $Name {
            const FIELDS: &'static [$crate::engine::meta::InputMeta] = &[
                $( ($meta).bind(
                    stringify!($field),
                    <$ty as $crate::engine::meta::FieldValue>::TYPE,
                ), )*
            ];
            const GROUPS: &'static [&'static str] = &[ $( $( stringify!($group), )* )? ];

            #[allow(unused_variables, clippy::match_single_binding)]
            fn get(&self, path: &str) -> ::core::option::Option<$crate::engine::meta::Value> {
                match path.split_once('.') {
                    ::core::option::Option::Some((head, rest)) => match head {
                        $( $( stringify!($group) =>
                            $crate::engine::meta::InputSet::get(&self.$group, rest), )* )?
                        _ => ::core::option::Option::None,
                    },
                    ::core::option::Option::None => match path {
                        $( stringify!($field) => ::core::option::Option::Some(
                            $crate::engine::meta::FieldValue::to_value(&self.$field),
                        ), )*
                        _ => ::core::option::Option::None,
                    },
                }
            }

            #[allow(unused_variables, clippy::match_single_binding)]
            fn set(
                &mut self,
                path: &str,
                value: $crate::engine::meta::Value,
            ) -> ::core::result::Result<(), $crate::engine::meta::SetError> {
                match path.split_once('.') {
                    ::core::option::Option::Some((head, rest)) => match head {
                        $( $( stringify!($group) =>
                            $crate::engine::meta::InputSet::set(&mut self.$group, rest, value)
                                .map_err(|e| e.within(stringify!($group))), )* )?
                        _ => ::core::result::Result::Err(
                            $crate::engine::meta::SetError::unknown_path(path),
                        ),
                    },
                    ::core::option::Option::None => match path {
                        $( stringify!($field) => {
                            let meta = $crate::engine::meta::field_meta(
                                <Self as $crate::engine::meta::InputSet>::FIELDS,
                                stringify!($field),
                            );
                            self.$field = $crate::engine::meta::convert_input::<$ty>(meta, value)?;
                            ::core::result::Result::Ok(())
                        } )*
                        _ => ::core::result::Result::Err(
                            $crate::engine::meta::SetError::unknown_path(path),
                        ),
                    },
                }
            }

            fn visit(
                &self,
                prefix: &str,
                f: &mut dyn FnMut(&str, &'static $crate::engine::meta::InputMeta, $crate::engine::meta::Value),
            ) {
                let values = [ $( $crate::engine::meta::FieldValue::to_value(&self.$field), )* ];
                for (meta, value) in <Self as $crate::engine::meta::InputSet>::FIELDS.iter().zip(values) {
                    f(&::std::format!("{prefix}{}", meta.name), meta, value);
                }
                $( $(
                    $crate::engine::meta::InputSet::visit(
                        &self.$group,
                        &::std::format!("{prefix}{}.", stringify!($group)),
                        f,
                    );
                )* )?
            }
        }
    };
}
pub(crate) use inputs;

/// Declares a result struct: fields with metadata, then optional nested
/// groups and tables. See the module docs for the field syntax (no defaults:
/// results are built by the module's `compute`, which must set every field). A
/// `tables { name: Row => layout }` entry is a `Vec<Row>` field (`Row` declared
/// with [`rows!`]) whose values get workbook cells from the [`TableLayout`].
macro_rules! results {
    (
        $(#[$sattr:meta])*
        pub struct $Name:ident {
            fields {
                $(
                    $(#[$fattr:meta])*
                    $field:ident : $ty:ty => $meta:expr
                ),* $(,)?
            }
            $(
                groups {
                    $( $(#[$gattr:meta])* $group:ident : $Group:ty ),* $(,)?
                }
            )?
            $(
                tables {
                    $( $(#[$tattr:meta])* $table:ident : $Row:ty => $layout:expr ),* $(,)?
                }
            )?
        }
    ) => {
        $(#[$sattr])*
        #[derive(Clone, Debug, PartialEq)]
        #[allow(non_snake_case)]
        pub struct $Name {
            $( $(#[$fattr])* pub $field: $ty, )*
            $( $( $(#[$gattr])* pub $group: $Group, )* )?
            $( $( $(#[$tattr])* pub $table: ::std::vec::Vec<$Row>, )* )?
        }

        impl $crate::engine::meta::ResultSet for $Name {
            const FIELDS: &'static [$crate::engine::meta::ResultMeta] = &[
                $( ($meta).bind(
                    stringify!($field),
                    <$ty as $crate::engine::meta::FieldValue>::TYPE,
                ), )*
            ];
            const GROUPS: &'static [&'static str] = &[ $( $( stringify!($group), )* )? ];

            fn visit(
                &self,
                prefix: &str,
                f: &mut $crate::engine::meta::ResultVisitor<'_>,
            ) {
                let values = [ $( $crate::engine::meta::FieldValue::to_value(&self.$field), )* ];
                for (meta, value) in <Self as $crate::engine::meta::ResultSet>::FIELDS.iter().zip(values) {
                    f(&::std::format!("{prefix}{}", meta.name), meta, meta.cell, value);
                }
                $( $(
                    $crate::engine::meta::ResultSet::visit(
                        &self.$group,
                        &::std::format!("{prefix}{}.", stringify!($group)),
                        f,
                    );
                )* )?
                $( $(
                    for (index, row) in self.$table.iter().enumerate() {
                        $crate::engine::meta::RowSet::visit_row(
                            row,
                            &::std::format!("{prefix}{}[{index}].", stringify!($table)),
                            &$layout,
                            index,
                            f,
                        );
                    }
                )* )?
            }
        }
    };
}
pub(crate) use results;

/// Declares a table row struct: fields with `col`/`at_row`/`uncelled_col` metadata.
/// Tables are declared in a `results!` struct's `tables { .. }` section, which
/// names the layout; the row names only each field's column or row.
#[allow(unused_macros)] // first used by clamps (Task 10), which removes this allow
macro_rules! rows {
    (
        $(#[$sattr:meta])*
        pub struct $Name:ident {
            $(
                $(#[$fattr:meta])*
                $field:ident : $ty:ty => $meta:expr
            ),* $(,)?
        }
    ) => {
        $(#[$sattr])*
        #[derive(Clone, Debug, PartialEq)]
        #[allow(non_snake_case)]
        pub struct $Name {
            $( $(#[$fattr])* pub $field: $ty, )*
        }

        impl $crate::engine::meta::RowSet for $Name {
            const COLUMNS: &'static [$crate::engine::meta::ColumnMeta] = &[
                $( ($meta).bind(
                    stringify!($field),
                    <$ty as $crate::engine::meta::FieldValue>::TYPE,
                ), )*
            ];

            fn visit_row(
                &self,
                prefix: &str,
                layout: &$crate::engine::meta::TableLayout,
                index: usize,
                f: &mut $crate::engine::meta::ResultVisitor<'_>,
            ) {
                let values = [ $( $crate::engine::meta::FieldValue::to_value(&self.$field), )* ];
                for (column, value) in <Self as $crate::engine::meta::RowSet>::COLUMNS.iter().zip(values) {
                    let cell = layout.cell(column.axis, index);
                    f(&::std::format!("{prefix}{}", column.meta.name), &column.meta, cell.as_deref(), value);
                }
            }
        }
    };
}
#[allow(unused_imports)] // first used by clamps (Task 10), which removes this allow
pub(crate) use rows;

#[cfg(test)]
mod tests {
    use super::*;

    inputs! {
        /// A leaf group with one field of every input type.
        pub struct Leaf {
            fields {
                a_mm: f64 = 1.5 => param("mm", "A", "help a", "S!C1").range(0.0, 10.0, 0.1),
                code: i64 = 1 => param("-", "Code", "", "S!C2").choices(&[(0, "zero"), (1, "one")]),
                part: String = "X1" => param("-", "Part", "", "S!C3"),
                drag_Nm: Option<f64> = None => param("N·m", "Drag", "", "S!C4").range(0.001, 1.0, 0.001).log(),
            }
        }
    }

    inputs! {
        /// A top level with a field and a nested group.
        pub struct Top {
            fields {
                n: i64 = 10 => param("-", "N", "", "S!C5").range(2.0, 20.0, 2.0).assumption(),
            }
            groups {
                leaf: Leaf,
            }
        }
    }

    results! {
        pub struct LeafOut {
            fields {
                x_mm: f64 => out("mm", "X", "", "S!D1"),
                verdict: String => out("", "Verdict", "", "S!D2"),
                mixed: NumOrText => out_uncelled("N·m", "Mixed", "help m"),
            }
        }
    }

    results! {
        pub struct TopOut {
            fields {}
            groups {
                leaf: LeafOut,
            }
        }
    }

    #[test]
    fn defaults_come_from_the_declarations() {
        let t = Top::default();
        assert_eq!(t.n, 10);
        assert_eq!(t.leaf.a_mm, 1.5);
        assert_eq!(t.leaf.code, 1);
        assert_eq!(t.leaf.part, "X1");
        assert_eq!(t.leaf.drag_Nm, None);
    }

    #[test]
    fn metadata_is_bound_to_name_and_type() {
        let names: Vec<_> = Leaf::FIELDS.iter().map(|m| (m.name, m.ty)).collect();
        assert_eq!(
            names,
            [
                ("a_mm", FieldType::F64),
                ("code", FieldType::I64),
                ("part", FieldType::Text),
                ("drag_Nm", FieldType::OptF64),
            ]
        );
        let a = &Leaf::FIELDS[0];
        assert_eq!(
            (a.unit, a.label, a.help, a.cell),
            ("mm", "A", "help a", Some("S!C1"))
        );
        assert_eq!(
            a.range,
            Some(SliderRange {
                min: 0.0,
                max: 10.0,
                step: 0.1,
                log: false
            })
        );
        assert!(!a.assumption);
        assert!(Leaf::FIELDS[3].range.unwrap().log);
        assert!(Top::FIELDS[0].assumption);
        assert_eq!(Leaf::FIELDS[1].choices, &[(0, "zero"), (1, "one")]);
        assert_eq!(Top::GROUPS, &["leaf"]);
        assert_eq!(Leaf::GROUPS, &[] as &[&str]);
        let m = &LeafOut::FIELDS[2];
        assert_eq!(
            (m.name, m.ty, m.cell, m.help),
            ("mixed", FieldType::NumOrText, None, "help m")
        );
    }

    #[test]
    fn get_reads_leaf_and_nested_paths() {
        let t = Top::default();
        assert_eq!(t.get("n"), Some(Value::Int(10)));
        assert_eq!(t.get("leaf.a_mm"), Some(Value::Num(1.5)));
        assert_eq!(t.get("leaf.part"), Some(Value::Text("X1".into())));
        assert_eq!(t.get("leaf.drag_Nm"), Some(Value::None));
        assert_eq!(t.get("leaf"), None, "a group is not a leaf input");
        assert_eq!(t.get("nope"), None);
        assert_eq!(t.get("leaf.nope"), None);
        assert_eq!(t.get("leaf.a_mm.deeper"), None);
    }

    #[test]
    fn set_writes_every_type() {
        let mut t = Top::default();
        t.set("n", Value::Int(12)).unwrap();
        t.set("leaf.a_mm", Value::Num(2.25)).unwrap();
        t.set("leaf.code", Value::Int(0)).unwrap();
        t.set("leaf.part", Value::Text("Y2".into())).unwrap();
        t.set("leaf.drag_Nm", Value::Num(0.05)).unwrap();
        assert_eq!((t.n, t.leaf.a_mm, t.leaf.code), (12, 2.25, 0));
        assert_eq!((t.leaf.part.as_str(), t.leaf.drag_Nm), ("Y2", Some(0.05)));
        t.set("leaf.drag_Nm", Value::None).unwrap();
        assert_eq!(t.leaf.drag_Nm, None);
    }

    #[test]
    fn set_accepts_an_integer_for_a_float_field() {
        let mut t = Top::default();
        t.set("leaf.a_mm", Value::Int(3)).unwrap();
        assert_eq!(t.leaf.a_mm, 3.0);
        t.set("leaf.drag_Nm", Value::Int(1)).unwrap();
        assert_eq!(t.leaf.drag_Nm, Some(1.0));
    }

    #[test]
    fn set_rejects_bad_values_and_names_the_full_path() {
        let mut t = Top::default();
        let before = t.clone();
        let kind = |r: Result<(), SetError>| r.unwrap_err();

        let e = kind(t.set("leaf.nope", Value::Num(1.0)));
        assert_eq!(
            (e.path.as_str(), &e.kind),
            ("leaf.nope", &SetErrorKind::UnknownPath)
        );
        let e = kind(t.set("nope.a_mm", Value::Num(1.0)));
        assert_eq!(
            (e.path.as_str(), &e.kind),
            ("nope.a_mm", &SetErrorKind::UnknownPath)
        );
        let e = kind(t.set("leaf", Value::Num(1.0)));
        assert_eq!(e.kind, SetErrorKind::UnknownPath);

        let e = kind(t.set("leaf.a_mm", Value::Text("1".into())));
        assert_eq!(e.path, "leaf.a_mm");
        assert!(matches!(
            e.kind,
            SetErrorKind::TypeMismatch {
                expected: FieldType::F64,
                ..
            }
        ));
        let e = kind(t.set("n", Value::Num(12.0)));
        assert!(matches!(
            e.kind,
            SetErrorKind::TypeMismatch {
                expected: FieldType::I64,
                ..
            }
        ));
        let e = kind(t.set("leaf.part", Value::None));
        assert!(matches!(
            e.kind,
            SetErrorKind::TypeMismatch {
                expected: FieldType::Text,
                ..
            }
        ));

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let e = kind(t.set("leaf.a_mm", Value::Num(bad)));
            assert_eq!(
                (e.path.as_str(), &e.kind),
                ("leaf.a_mm", &SetErrorKind::NotFinite)
            );
            let e = kind(t.set("leaf.drag_Nm", Value::Num(bad)));
            assert_eq!(e.kind, SetErrorKind::NotFinite);
        }

        let e = kind(t.set("leaf.code", Value::Int(2)));
        assert_eq!(
            (e.path.as_str(), &e.kind),
            ("leaf.code", &SetErrorKind::NotAChoice { code: 2 })
        );
        assert_eq!(e.to_string(), "leaf.code: 2 is not one of the choices");

        assert_eq!(t, before, "a refused set leaves the inputs unchanged");
    }

    #[test]
    fn visit_yields_full_paths_fields_before_groups() {
        let rows = input_rows(&Top::default());
        let paths: Vec<_> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(
            paths,
            ["n", "leaf.a_mm", "leaf.code", "leaf.part", "leaf.drag_Nm"]
        );
        assert_eq!(rows[1].meta.name, "a_mm");
        assert_eq!(rows[1].value, Value::Num(1.5));
    }

    #[test]
    fn visit_prefix_is_prepended() {
        let mut paths = Vec::new();
        Leaf::default().visit("top.leaf.", &mut |p, _, _| paths.push(p.to_owned()));
        assert_eq!(paths[0], "top.leaf.a_mm");
    }

    #[test]
    fn result_rows_walk_nested_groups() {
        let out = TopOut {
            leaf: LeafOut {
                x_mm: 2.0,
                verdict: "OK".into(),
                mixed: NumOrText::Text("n.a."),
            },
        };
        let rows = result_rows(&out);
        let got: Vec<_> = rows
            .iter()
            .map(|r| (r.path.as_str(), r.value.clone()))
            .collect();
        assert_eq!(
            got,
            [
                ("leaf.x_mm", Value::Num(2.0)),
                ("leaf.verdict", Value::Text("OK".into())),
                ("leaf.mixed", Value::Text("n.a.".into())),
            ]
        );
        assert_eq!(NumOrText::Num(1.5).to_value(), Value::Num(1.5));
    }

    rows! {
        /// A toy screw-table row: one sheet column per row, fields name their sheet row.
        pub struct ToyRow {
            size: String => uncelled_col("", "Size", ""),
            d_mm: f64 => at_row("mm", "Diameter", "", 6),
            ok: i64 => at_row("-", "Fits", "", 7),
        }
    }

    rows! {
        /// A toy sweep row: one sheet row per row, fields name their column.
        pub struct SweepToy {
            x: f64 => col("mm", "X", "", "B"),
            status: String => col("", "Status", "", "AA"),
        }
    }

    const TOY_COLUMNS: [&str; 2] = ["C", "D"];

    results! {
        pub struct ToyOut {
            fields {
                n: f64 => out("-", "N", "", "S!C1"),
            }
            tables {
                table: ToyRow => TableLayout::ColumnsAcross { sheet: "Toy", columns: &TOY_COLUMNS },
                sweep: SweepToy => TableLayout::RowsDown { sheet: "Sweep", first_row: 6 },
            }
        }
    }

    #[test]
    fn table_rows_get_synthesized_cells() {
        let out = ToyOut {
            n: 1.0,
            table: vec![
                ToyRow {
                    size: "M3".into(),
                    d_mm: 3.0,
                    ok: 1,
                },
                ToyRow {
                    size: "M4".into(),
                    d_mm: 4.0,
                    ok: 0,
                },
            ],
            sweep: vec![
                SweepToy {
                    x: 0.5,
                    status: "a".into(),
                },
                SweepToy {
                    x: 0.75,
                    status: "b".into(),
                },
            ],
        };
        let rows: Vec<(String, Option<String>, Value)> = result_rows(&out)
            .into_iter()
            .map(|r| (r.path, r.cell, r.value))
            .collect();
        let s = |x: &str| x.to_owned();
        assert_eq!(
            rows,
            vec![
                (s("n"), Some(s("S!C1")), Value::Num(1.0)),
                (s("table[0].size"), None, Value::Text(s("M3"))),
                (s("table[0].d_mm"), Some(s("Toy!C6")), Value::Num(3.0)),
                (s("table[0].ok"), Some(s("Toy!C7")), Value::Int(1)),
                (s("table[1].size"), None, Value::Text(s("M4"))),
                (s("table[1].d_mm"), Some(s("Toy!D6")), Value::Num(4.0)),
                (s("table[1].ok"), Some(s("Toy!D7")), Value::Int(0)),
                (s("sweep[0].x"), Some(s("Sweep!B6")), Value::Num(0.5)),
                (
                    s("sweep[0].status"),
                    Some(s("Sweep!AA6")),
                    Value::Text(s("a"))
                ),
                (s("sweep[1].x"), Some(s("Sweep!B7")), Value::Num(0.75)),
                (
                    s("sweep[1].status"),
                    Some(s("Sweep!AA7")),
                    Value::Text(s("b"))
                ),
            ]
        );
        assert_eq!(ToyRow::COLUMNS[1].meta.name, "d_mm");
        assert_eq!(ToyRow::COLUMNS[2].meta.ty, FieldType::I64);
    }

    #[test]
    fn a_layout_gives_no_cell_to_a_mismatched_axis_or_a_row_past_its_columns() {
        let across = TableLayout::ColumnsAcross {
            sheet: "T",
            columns: &TOY_COLUMNS,
        };
        assert_eq!(across.cell(CellAxis::Row(6), 1), Some("T!D6".to_owned()));
        assert_eq!(across.cell(CellAxis::Column("B"), 0), None);
        assert_eq!(across.cell(CellAxis::Row(6), 2), None, "only two columns");
        let down = TableLayout::RowsDown {
            sheet: "S",
            first_row: 6,
        };
        assert_eq!(
            down.cell(CellAxis::Column("AA"), 12),
            Some("S!AA18".to_owned())
        );
        assert_eq!(down.cell(CellAxis::Row(6), 0), None);
        assert_eq!(down.cell(CellAxis::None, 0), None);
    }
}
